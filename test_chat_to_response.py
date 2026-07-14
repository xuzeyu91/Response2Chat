"""Contract tests for the Chat -> Responses channel mode."""

import asyncio
import json
import os
import shutil
import tempfile
import unittest

import httpx
from fastapi.testclient import TestClient


TEMP_DB_DIR = tempfile.mkdtemp()
os.environ["DATABASE_PATH"] = os.path.join(TEMP_DB_DIR, "reverse-mode.db")
os.environ["DEFAULT_INSTRUCTIONS"] = ""

import main  # noqa: E402
from channel_store import CHANNEL_TYPE_CHAT_TO_RESPONSE, CHANNEL_TYPE_RESPONSE_TO_CHAT  # noqa: E402


def replace_http_client(app, handler):
    original_client = app.state.http_client
    asyncio.run(original_client.aclose())
    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.http_client = mock_client
    return original_client, mock_client


def parse_sse_events(payload: str):
    events = []
    for block in payload.split("\n\n"):
        if not block.strip():
            continue
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if event_type and data:
            events.append((event_type, data))
    return events


class ChatToResponseTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEMP_DB_DIR, ignore_errors=True)

    def create_reverse_channel(self):
        return main.app.state.settings_store.create_channel(
            name=f"reverse-{os.urandom(4).hex()}",
            base_url="https://upstream.example/v1",
            upstream_api_key="upstream-secret",
            protocol_type=CHANNEL_TYPE_CHAT_TO_RESPONSE,
        )

    def test_channel_type_defaults_to_existing_response_to_chat_mode(self):
        with TestClient(main.app):
            channel = main.app.state.settings_store.create_channel(
                name=f"default-{os.urandom(4).hex()}",
                base_url="https://upstream.example/v1",
                upstream_api_key="upstream-secret",
            )
            self.assertEqual(channel["protocol_type"], CHANNEL_TYPE_RESPONSE_TO_CHAT)

    def test_admin_form_persists_and_renders_reverse_channel_type(self):
        with TestClient(main.app) as client:
            login = client.post(
                "/admin/login",
                data={"username": "admin", "password": "admin123456"},
                follow_redirects=False,
            )
            self.assertEqual(login.status_code, 303)
            channel_name = f"admin-reverse-{os.urandom(4).hex()}"
            created = client.post(
                "/admin/channels",
                data={
                    "name": channel_name,
                    "upstream_base_url": "https://upstream.example/v1",
                    "upstream_api_key": "upstream-secret",
                    "protocol_type": CHANNEL_TYPE_CHAT_TO_RESPONSE,
                },
                follow_redirects=False,
            )
            self.assertEqual(created.status_code, 303)
            channel = next(
                channel
                for channel in main.app.state.settings_store.list_channels()
                if channel["name"] == channel_name
            )
            self.assertEqual(channel["protocol_type"], CHANNEL_TYPE_CHAT_TO_RESPONSE)
            dashboard = client.get("/admin")
            detail = client.get(f"/admin/channels/{channel['id']}")
            self.assertIn("Chat → Responses", dashboard.text)
            self.assertIn('value="chat_to_response" selected', detail.text)

    def test_non_stream_converts_input_image_tools_and_result(self):
        with TestClient(main.app) as client:
            channel = self.create_reverse_channel()
            access_key = channel["access_key"]

            def handler(request: httpx.Request):
                self.assertEqual(str(request.url), "https://upstream.example/v1/chat/completions")
                self.assertEqual(request.headers["Authorization"], "Bearer upstream-secret")
                body = json.loads(request.content)
                self.assertFalse(body["stream"])
                self.assertEqual(body["max_completion_tokens"], 64)
                self.assertEqual(body["messages"][0], {"role": "developer", "content": "Be concise."})
                parts = body["messages"][1]["content"]
                self.assertEqual(parts[0], {"type": "text", "text": "What is shown?"})
                self.assertEqual(
                    parts[1],
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/image.png", "detail": "low"},
                    },
                )
                self.assertEqual(body["tools"][0]["function"]["name"], "describe_image")
                self.assertEqual(body["tool_choice"], {"type": "function", "function": {"name": "describe_image"}})
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl_test",
                        "object": "chat.completion",
                        "created": 1730000000,
                        "model": "gpt-5.4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "I can inspect it.",
                                    "tool_calls": [
                                        {
                                            "id": "call_image",
                                            "type": "function",
                                            "function": {
                                                "name": "describe_image",
                                                "arguments": '{"style":"brief"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                            "prompt_tokens_details": {"cached_tokens": 2},
                            "completion_tokens_details": {"reasoning_tokens": 1},
                        },
                    },
                )

            original_client, mock_client = replace_http_client(main.app, handler)
            try:
                response = client.post(
                    "/v1/responses",
                    headers={"Authorization": f"Bearer {access_key}"},
                    json={
                        "model": "gpt-5.4",
                        "instructions": "Be concise.",
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "What is shown?"},
                                    {
                                        "type": "input_image",
                                        "image_url": "https://example.test/image.png",
                                        "detail": "low",
                                    },
                                ],
                            }
                        ],
                        "max_output_tokens": 64,
                        "tools": [
                            {
                                "type": "function",
                                "name": "describe_image",
                                "description": "Describe an image",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                        "tool_choice": {"type": "function", "name": "describe_image"},
                    },
                )
            finally:
                asyncio.run(mock_client.aclose())
                main.app.state.http_client = original_client

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["object"], "response")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["usage"]["input_tokens"], 11)
            self.assertEqual(payload["usage"]["output_tokens_details"]["reasoning_tokens"], 1)
            message = next(item for item in payload["output"] if item["type"] == "message")
            self.assertEqual(message["content"][0]["text"], "I can inspect it.")
            function_call = next(item for item in payload["output"] if item["type"] == "function_call")
            self.assertEqual(function_call["call_id"], "call_image")
            self.assertEqual(function_call["name"], "describe_image")
            self.assertEqual(function_call["arguments"], '{"style":"brief"}')

    def test_non_stream_retries_gateway_tool_choice_compatibility_shape(self):
        with TestClient(main.app) as client:
            channel = self.create_reverse_channel()
            access_key = channel["access_key"]
            attempts = []

            def handler(request: httpx.Request):
                body = json.loads(request.content)
                attempts.append(body["tool_choice"])
                if len(attempts) == 1:
                    self.assertEqual(
                        body["tool_choice"],
                        {"type": "function", "function": {"name": "lookup"}},
                    )
                    return httpx.Response(
                        400,
                        json={
                            "error": {
                                "message": "Missing required parameter: 'tool_choice.name'.",
                                "type": "invalid_request_error",
                            }
                        },
                    )
                self.assertEqual(body["tool_choice"], {"type": "function", "name": "lookup"})
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl_retry",
                        "created": 1730000002,
                        "model": "gpt-5.4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_retry",
                                            "type": "function",
                                            "function": {"name": "lookup", "arguments": "{}"},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                )

            original_client, mock_client = replace_http_client(main.app, handler)
            try:
                response = client.post(
                    "/v1/responses",
                    headers={"Authorization": f"Bearer {access_key}"},
                    json={
                        "model": "gpt-5.4",
                        "input": "Call lookup.",
                        "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
                        "tool_choice": {"type": "function", "name": "lookup"},
                    },
                )
            finally:
                asyncio.run(mock_client.aclose())
                main.app.state.http_client = original_client

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(attempts), 2)
            function_call = next(item for item in response.json()["output"] if item["type"] == "function_call")
            self.assertEqual(function_call["name"], "lookup")

    def test_stream_converts_text_and_function_call_events(self):
        with TestClient(main.app) as client:
            channel = self.create_reverse_channel()
            access_key = channel["access_key"]

            chunks = [
                {
                    "id": "chatcmpl_stream",
                    "object": "chat.completion.chunk",
                    "created": 1730000001,
                    "model": "gpt-5.4",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl_stream",
                    "object": "chat.completion.chunk",
                    "created": 1730000001,
                    "model": "gpt-5.4",
                    "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl_stream",
                    "object": "chat.completion.chunk",
                    "created": 1730000001,
                    "model": "gpt-5.4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_stream",
                                        "type": "function",
                                        "function": {"name": "lookup", "arguments": '{"q":"'},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_stream",
                    "object": "chat.completion.chunk",
                    "created": 1730000001,
                    "model": "gpt-5.4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ping\"}"}}]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
                {
                    "id": "chatcmpl_stream",
                    "object": "chat.completion.chunk",
                    "created": 1730000001,
                    "model": "gpt-5.4",
                    "choices": [],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                },
            ]
            sse_body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

            def handler(request: httpx.Request):
                body = json.loads(request.content)
                self.assertTrue(body["stream"])
                return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

            original_client, mock_client = replace_http_client(main.app, handler)
            try:
                response = client.post(
                    "/v1/responses",
                    headers={"Authorization": f"Bearer {access_key}"},
                    json={"model": "gpt-5.4", "input": "hello", "stream": True},
                )
            finally:
                asyncio.run(mock_client.aclose())
                main.app.state.http_client = original_client

            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("text/event-stream", response.headers["content-type"])
            events = parse_sse_events(response.text)
            event_types = [event_type for event_type, _ in events]
            self.assertIn("response.created", event_types)
            self.assertIn("response.output_text.delta", event_types)
            self.assertIn("response.function_call_arguments.delta", event_types)
            self.assertIn("response.function_call_arguments.done", event_types)
            completed = next(data["response"] for event_type, data in events if event_type == "response.completed")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["usage"]["total_tokens"], 9)
            function_call = next(item for item in completed["output"] if item["type"] == "function_call")
            self.assertEqual(function_call["call_id"], "call_stream")
            self.assertEqual(function_call["name"], "lookup")
            self.assertEqual(function_call["arguments"], '{"q":"ping"}')


if __name__ == "__main__":
    unittest.main()
