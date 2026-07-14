"""
Response API to Chat API 转发服务
将 OpenAI Response 协议接口转发为 Chat 协议接口
"""

import os
import json
import time
import uuid
import asyncio
import logging
import traceback
from functools import lru_cache
from html import escape
from pathlib import Path
from string import Template
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, quote

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse, Response, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from channel_store import (
    AdminSessionManager,
    CHANNEL_TYPE_CHAT_TO_RESPONSE,
    CHANNEL_TYPE_RESPONSE_TO_CHAT,
    SettingsStore,
    mask_secret,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HTTPX_LOG_LEVEL = os.getenv("HTTPX_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("response2chat")
logging.getLogger("httpx").setLevel(getattr(logging, HTTPX_LOG_LEVEL, logging.WARNING))

# ==================== 配置 ====================
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "300"))
POOL_TIMEOUT = float(os.getenv("POOL_TIMEOUT", "10"))
STREAM_READ_TIMEOUT = float(os.getenv("STREAM_READ_TIMEOUT", "120"))
STREAM_MAX_DURATION = int(os.getenv("STREAM_MAX_DURATION", "0"))  # 0 表示不限制
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/response2chat.db")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123456")
ADMIN_SESSION_TTL_SECONDS = int(os.getenv("ADMIN_SESSION_TTL_SECONDS", str(12 * 60 * 60)))
ADMIN_SESSION_COOKIE_NAME = os.getenv("ADMIN_SESSION_COOKIE_NAME", "response2chat_admin_session")
ADMIN_COOKIE_SECURE = os.getenv("ADMIN_COOKIE_SECURE", "false").lower() == "true"
ADMIN_TEST_MODEL = "gpt-5.4"
ADMIN_TEST_INPUT = "ping"
UPSTREAM_USER_AGENT = "Codex Desktop/0.131.0-alpha.9 (Windows 10.0.26200; x86_64) unknown (Codex Desktop; 26.513.40821)"

# 连接池配置 - 防止连接泄漏和资源耗尽
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))  # 最大连接数
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "30"))  # 保持活跃的连接数
KEEPALIVE_EXPIRY = int(os.getenv("KEEPALIVE_EXPIRY", "60"))  # 连接保持时间(秒)

# 默认系统提示词配置
# 当请求中没有 system 消息时，会使用此默认提示词
# 设置为空字符串可禁用默认提示词
DEFAULT_INSTRUCTIONS = os.getenv("DEFAULT_INSTRUCTIONS", "").strip()
# 是否强制使用默认提示词（即使请求中有 system 消息也会添加）
FORCE_DEFAULT_INSTRUCTIONS = os.getenv("FORCE_DEFAULT_INSTRUCTIONS", "false").lower() == "true"

# ==================== Pydantic 模型定义 ====================

# Chat API 请求模型
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # 允许为 None，当有 tool_calls 时可能为空
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    reasoning_effort: Optional[str] = None

# Chat API 响应模型
class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[str] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None


def convert_response_usage_to_chat_usage(response_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    将 Response API 的 usage 格式转换为 Chat API 的 usage 格式
    
    Response API 格式:
    {
        "input_tokens": 17254,
        "input_tokens_details": {"cached_tokens": 7936},
        "output_tokens": 336,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 17590
    }
    
    Chat API 格式:
    {
        "prompt_tokens": 12709,
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 12032
        },
        "completion_tokens": 322,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
        },
        "total_tokens": 13031
    }
    """
    if response_usage is None:
        return None
    
    # 基本字段转换: input_tokens -> prompt_tokens, output_tokens -> completion_tokens
    chat_usage = {
        "prompt_tokens": response_usage.get("input_tokens", 0),
        "completion_tokens": response_usage.get("output_tokens", 0),
        "total_tokens": response_usage.get("total_tokens", 0)
    }
    
    # 转换 input_tokens_details -> prompt_tokens_details
    input_details = response_usage.get("input_tokens_details")
    if input_details:
        chat_usage["prompt_tokens_details"] = {
            "audio_tokens": input_details.get("audio_tokens", 0),
            "cached_tokens": input_details.get("cached_tokens", 0)
        }
    
    # 转换 output_tokens_details -> completion_tokens_details
    output_details = response_usage.get("output_tokens_details")
    if output_details:
        chat_usage["completion_tokens_details"] = {
            "accepted_prediction_tokens": output_details.get("accepted_prediction_tokens", 0),
            "audio_tokens": output_details.get("audio_tokens", 0),
            "reasoning_tokens": output_details.get("reasoning_tokens", 0),
            "rejected_prediction_tokens": output_details.get("rejected_prediction_tokens", 0)
        }
    
    return chat_usage

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None

# ==================== 转换函数 ====================

def convert_chat_to_response_request(chat_request: ChatCompletionRequest) -> Dict[str, Any]:
    """将 Chat API 请求转换为 Response API 请求"""
    
    # 构建 input 数组，包含所有消息
    # 注意：Response API 不支持 system 角色，将其转换为 developer 角色
    # Response API 不支持 tool 角色，需要转换为 function_call_output 类型
    input_items = []
    
    # 预处理：为空的 tool_call.id 和对应的 tool.tool_call_id 建立映射
    # 按顺序匹配：每个 assistant 的 tool_calls 后面紧跟着对应数量的 tool 消息
    generated_call_ids: Dict[int, str] = {}  # 消息索引 -> 生成的 call_id
    tool_call_id_mapping: Dict[int, List[str]] = {}  # assistant 消息索引 -> 该消息生成的 call_ids 列表
    
    # 第一遍扫描：识别需要生成 call_id 的 tool_calls，并建立映射
    pending_call_ids: List[str] = []  # 待匹配的 call_ids 队列
    for i, msg in enumerate(chat_request.messages):
        if msg.role == "assistant" and msg.tool_calls:
            tool_call_id_mapping[i] = []
            for tool_call in msg.tool_calls:
                original_id = tool_call.get("id")
                if original_id:
                    # 有原始 id，直接使用
                    pending_call_ids.append(original_id)
                    tool_call_id_mapping[i].append(original_id)
                else:
                    # 没有原始 id，生成一个新的
                    new_id = f"call_{uuid.uuid4().hex[:24]}"
                    pending_call_ids.append(new_id)
                    tool_call_id_mapping[i].append(new_id)
                    logger.warning(f"tool_call 的 id 为空，自动生成: {new_id}")
        elif msg.role == "tool":
            # tool 消息需要匹配 call_id
            if msg.tool_call_id:
                # 有原始 tool_call_id，直接使用
                generated_call_ids[i] = msg.tool_call_id
            elif pending_call_ids:
                # 没有 tool_call_id，从队列中取一个
                generated_call_ids[i] = pending_call_ids.pop(0)
                logger.warning(f"tool 消息的 tool_call_id 为空，使用匹配的 call_id: {generated_call_ids[i]}")
            else:
                # 队列为空，生成一个新的（这种情况不应该发生）
                generated_call_ids[i] = f"call_{uuid.uuid4().hex[:24]}"
                logger.warning(f"tool 消息的 tool_call_id 为空且无法匹配，自动生成: {generated_call_ids[i]}")
    
    # 重置 pending_call_ids 用于第二遍
    pending_call_ids_iter = iter([])
    current_assistant_idx = -1
    current_tool_call_idx = 0
    
    # 第二遍：实际构建 input_items
    for i, msg in enumerate(chat_request.messages):
        # 特殊处理 tool 角色 - 转换为 function_call_output 类型
        if msg.role == "tool":
            # Chat API tool 消息格式:
            # {"role": "tool", "tool_call_id": "xxx", "content": "result"}
            # -> Response API 格式:
            # {"type": "function_call_output", "call_id": "xxx", "output": "result"}
            call_id = generated_call_ids.get(i, msg.tool_call_id or f"call_{uuid.uuid4().hex[:24]}")
            tool_output_item = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": msg.content if isinstance(msg.content, str) else json.dumps(msg.content) if msg.content else ""
            }
            input_items.append(tool_output_item)
            continue
        
        # 特殊处理 assistant 消息中的 tool_calls
        if msg.role == "assistant" and msg.tool_calls:
            # 先添加 assistant 消息内容（如果有）
            if msg.content:
                content = msg.content if isinstance(msg.content, str) else msg.content
                item = {
                    "type": "message",
                    "role": "assistant",
                    "content": content
                }
                input_items.append(item)
            
            # 获取预先生成的 call_ids
            pre_generated_ids = tool_call_id_mapping.get(i, [])
            
            # 然后添加 function_call 类型的项
            # Chat API tool_calls 格式:
            # [{"id": "call_xxx", "type": "function", "function": {"name": "xxx", "arguments": "{...}"}}]
            # -> Response API 格式:
            # {"type": "function_call", "call_id": "call_xxx", "name": "xxx", "arguments": "{...}"}
            for j, tool_call in enumerate(msg.tool_calls):
                # 处理 type 为 function 或 None 的情况（某些客户端可能不发送 type 字段）
                tool_type = tool_call.get("type")
                if tool_type == "function" or tool_type is None:
                    func = tool_call.get("function", {})
                    # 使用预先生成的 call_id
                    if j < len(pre_generated_ids):
                        call_id = pre_generated_ids[j]
                    else:
                        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                    
                    # 如果 name 为空，尝试从 arguments 中推断工具名称
                    func_name = func.get("name", "")
                    if not func_name:
                        # 尝试从 arguments 推断工具名称
                        args_str = func.get("arguments", "{}")
                        try:
                            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                            # 常见工具参数到工具名的映射
                            if "thought" in args_dict:
                                func_name = "think"
                            elif "code" in args_dict and "file_name" in args_dict:
                                func_name = "save_to_file_and_run"
                            else:
                                func_name = f"unknown_function_{uuid.uuid4().hex[:8]}"
                        except:
                            func_name = f"unknown_function_{uuid.uuid4().hex[:8]}"
                        logger.warning(f"tool_call 的 name 为空，推断为: {func_name}")
                    
                    func_call_item = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": func_name,
                        "arguments": func.get("arguments", "{}")
                    }
                    input_items.append(func_call_item)
            continue
        
        # 处理 content 字段，转换多模态内容格式
        if msg.content is None:
            # content 为空（通常在 assistant 消息有 tool_calls 时）
            converted_content = ""
        elif isinstance(msg.content, str):
            # 纯文本内容
            converted_content = msg.content
        elif isinstance(msg.content, list):
            # 多模态内容，需要转换格式
            converted_content = []
            for part in msg.content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "text":
                        # 文本部分: Chat 格式 {"type": "text", "text": "..."} 
                        # -> Response 格式 {"type": "input_text", "text": "..."}
                        converted_content.append({
                            "type": "input_text",
                            "text": part.get("text", "")
                        })
                    elif part_type == "image_url":
                        # 图片部分: Chat 格式 {"type": "image_url", "image_url": {"url": "..."}}
                        # -> Response 格式 {"type": "input_image", "image_url": "..."}
                        image_url_obj = part.get("image_url", {})
                        if isinstance(image_url_obj, dict):
                            image_url = image_url_obj.get("url", "")
                        else:
                            image_url = str(image_url_obj)
                        converted_content.append({
                            "type": "input_image",
                            "image_url": image_url
                        })
                    else:
                        # 其他类型直接保留
                        converted_content.append(part)
                else:
                    converted_content.append(part)
        else:
            converted_content = msg.content
        
        # 检查角色类型
        # Response API 不支持 system 角色，将其转换为 developer 角色
        role = msg.role
        if role == "system":
            role = "developer"
        
        item = {
            "type": "message",
            "role": role,
            "content": converted_content
        }
        input_items.append(item)
    
    response_request = {
        "model": chat_request.model,
        "input": input_items,
        "stream": True,  # Response API 始终使用 stream
    }
    
    # 处理 instructions 参数
    # 检查请求中是否已有 system 消息
    has_system_message = any(msg.role == "system" for msg in chat_request.messages)
    
    if DEFAULT_INSTRUCTIONS:
        if FORCE_DEFAULT_INSTRUCTIONS or not has_system_message:
            # 使用配置的默认 instructions
            response_request["instructions"] = DEFAULT_INSTRUCTIONS
            logger.debug(f"使用默认 instructions: {DEFAULT_INSTRUCTIONS[:50]}...")
    
    # 可选参数映射 - 只添加 Response API 支持的参数
    # 注意：某些 Response API 可能不支持 temperature, top_p, max_output_tokens 等参数
    # 根据实际 API 支持情况调整
    # max_output_tokens 参数已注释，因为某些上游 API (如 api.routin.ai) 不支持此参数
    # 如需启用，取消以下注释：
    # if chat_request.max_tokens is not None:
    #     response_request["max_output_tokens"] = chat_request.max_tokens
    # if chat_request.max_completion_tokens is not None:
    #     response_request["max_output_tokens"] = chat_request.max_completion_tokens
    
    # tools 格式转换
    # Chat API 格式: {"type": "function", "function": {"name": "xxx", "description": "xxx", "parameters": {...}}}
    # Response API 格式: {"type": "function", "name": "xxx", "description": "xxx", "parameters": {...}}
    if chat_request.tools is not None:
        converted_tools = []
        for tool in chat_request.tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                converted_tool = {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                }
                if "parameters" in func:
                    converted_tool["parameters"] = func["parameters"]
                converted_tools.append(converted_tool)
            else:
                # 其他类型直接保留
                converted_tools.append(tool)
        response_request["tools"] = converted_tools
    
    if chat_request.tool_choice is not None:
        response_request["tool_choice"] = chat_request.tool_choice
    
    # reasoning_effort 用于推理模型
    if chat_request.reasoning_effort is not None:
        response_request["reasoning"] = {"effort": chat_request.reasoning_effort}
    
    # response_format 支持 (如 json_object, json_schema)
    if chat_request.response_format is not None:
        # Response API 可能使用不同的格式，尝试转换
        fmt_type = chat_request.response_format.get("type")
        if fmt_type == "json_object":
            response_request["text"] = {"format": {"type": "json_object"}}
        elif fmt_type == "json_schema":
            # Chat API json_schema 格式:
            # {"type": "json_schema", "json_schema": {"name": "xxx", "schema": {...}, "strict": true}}
            # Response API 格式:
            # {"format": {"type": "json_schema", "name": "xxx", "schema": {...}, "strict": true}}
            json_schema_obj = chat_request.response_format.get("json_schema", {})
            response_format = {
                "type": "json_schema",
                "name": json_schema_obj.get("name", "response_schema"),
                "schema": json_schema_obj.get("schema", {}),
            }
            # 只有在 strict 存在时才添加
            if "strict" in json_schema_obj:
                response_format["strict"] = json_schema_obj.get("strict")
            response_request["text"] = {"format": response_format}
    
    # 以下参数某些 Response API 可能不支持，根据需要启用
    # if chat_request.temperature is not None and chat_request.temperature != 1:
    #     response_request["temperature"] = chat_request.temperature
    # if chat_request.top_p is not None and chat_request.top_p != 1:
    #     response_request["top_p"] = chat_request.top_p
    # if chat_request.stop is not None:
    #     response_request["stop"] = chat_request.stop
    # if chat_request.presence_penalty is not None and chat_request.presence_penalty != 0:
    #     response_request["presence_penalty"] = chat_request.presence_penalty
    # if chat_request.frequency_penalty is not None and chat_request.frequency_penalty != 0:
    #     response_request["frequency_penalty"] = chat_request.frequency_penalty
    
    return response_request


def serialize_content_as_text(value: Any) -> str:
    """Convert a Responses tool output to the string form required by Chat."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def convert_response_content_to_chat(content: Any) -> Any:
    """Map Responses message content parts to the Chat Completions equivalent."""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        return content

    converted_content: List[Any] = []
    for part in content:
        if not isinstance(part, dict):
            converted_content.append(part)
            continue

        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            converted_content.append({"type": "text", "text": part.get("text", "")})
        elif part_type == "input_image":
            image_url = part.get("image_url") or part.get("url") or ""
            image_part: Dict[str, Any] = {"url": image_url}
            if part.get("detail") is not None:
                image_part["detail"] = part["detail"]
            converted_content.append({"type": "image_url", "image_url": image_part})
        else:
            # Preserve vendor extension parts.  This makes the proxy forward
            # compatible while explicitly converting the OpenAI text/image
            # formats it owns.
            converted_content.append(part)
    return converted_content


def convert_response_input_to_chat_messages(response_input: Any) -> List[Dict[str, Any]]:
    """Convert Responses `input` (including prior tool turns) to Chat messages."""
    if isinstance(response_input, str):
        return [{"role": "user", "content": response_input}]
    if not isinstance(response_input, list):
        raise ValueError("Responses input must be a string or an array")

    messages: List[Dict[str, Any]] = []
    for item in response_input:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ValueError("Responses input array contains an invalid item")

        item_type = item.get("type", "message")
        if item_type == "message":
            role = item.get("role", "user")
            messages.append(
                {
                    "role": role,
                    "content": convert_response_content_to_chat(item.get("content")),
                }
            )
        elif item_type == "function_call":
            # A Responses function_call represents an assistant tool-call
            # message in Chat.  Consecutive function calls belong to one
            # assistant turn whenever possible.
            if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                assistant_message = messages[-1]
            else:
                assistant_message = {"role": "assistant", "content": None, "tool_calls": []}
                messages.append(assistant_message)
            arguments = item.get("arguments", "{}")
            assistant_message["tool_calls"].append(
                {
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": serialize_content_as_text(item.get("output")),
                }
            )
        else:
            raise ValueError(f"Unsupported Responses input item type: {item_type}")
    return messages


def convert_response_to_chat_request(response_request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Responses request into a Chat Completions request."""
    model = response_request.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Responses request must include a non-empty model")
    if "input" not in response_request:
        raise ValueError("Responses request must include input")

    messages = convert_response_input_to_chat_messages(response_request["input"])
    instructions = response_request.get("instructions")
    if instructions:
        messages.insert(0, {"role": "developer", "content": instructions})

    chat_request: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(response_request.get("stream", False)),
    }

    if response_request.get("max_output_tokens") is not None:
        chat_request["max_completion_tokens"] = response_request["max_output_tokens"]
    if response_request.get("temperature") is not None:
        chat_request["temperature"] = response_request["temperature"]
    if response_request.get("top_p") is not None:
        chat_request["top_p"] = response_request["top_p"]
    if response_request.get("user") is not None:
        chat_request["user"] = response_request["user"]
    if response_request.get("parallel_tool_calls") is not None:
        chat_request["parallel_tool_calls"] = response_request["parallel_tool_calls"]

    reasoning = response_request.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        chat_request["reasoning_effort"] = reasoning["effort"]

    text = response_request.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        response_format = dict(text["format"])
        if response_format.get("type") == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    key: value
                    for key, value in response_format.items()
                    if key != "type"
                },
            }
        chat_request["response_format"] = response_format

    if response_request.get("tools") is not None:
        converted_tools: List[Dict[str, Any]] = []
        for tool in response_request["tools"]:
            if isinstance(tool, dict) and tool.get("type") == "function" and "function" not in tool:
                function = {
                    key: value
                    for key, value in tool.items()
                    if key in {"name", "description", "parameters", "strict"}
                }
                converted_tools.append({"type": "function", "function": function})
            else:
                converted_tools.append(tool)
        chat_request["tools"] = converted_tools

    tool_choice = response_request.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "function" not in tool_choice:
        chat_request["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    elif tool_choice is not None:
        chat_request["tool_choice"] = tool_choice

    return chat_request


def convert_chat_usage_to_response_usage(chat_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Map Chat usage accounting to the equivalent Responses fields."""
    if not isinstance(chat_usage, dict):
        return None

    response_usage: Dict[str, Any] = {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "total_tokens": chat_usage.get("total_tokens", 0),
    }
    prompt_details = chat_usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        response_usage["input_tokens_details"] = {
            key: value
            for key, value in prompt_details.items()
            if key in {"cached_tokens", "audio_tokens"}
        }
    completion_details = chat_usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        response_usage["output_tokens_details"] = {
            key: value
            for key, value in completion_details.items()
            if key in {
                "reasoning_tokens",
                "audio_tokens",
                "accepted_prediction_tokens",
                "rejected_prediction_tokens",
            }
        }
    return response_usage


def build_flat_tool_choice_fallback(chat_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a retry body for providers that use Responses-style tool_choice in Chat.

    OpenAI Chat Completions uses ``tool_choice.function.name``.  A few
    OpenAI-compatible gateways instead expose Chat at the same path while
    requiring the Responses shape (``tool_choice.name``).  We only use this
    body after that exact validation error, never as the default.
    """
    tool_choice = chat_request.get("tool_choice")
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, dict) or not function.get("name"):
        return None
    fallback = dict(chat_request)
    fallback["tool_choice"] = {"type": "function", "name": function["name"]}
    return fallback


def upstream_requires_flat_tool_choice(response: httpx.Response) -> bool:
    """Return true only for the documented nonstandard tool_choice error."""
    if response.status_code != 400:
        return False
    return "tool_choice.name" in response.text and "missing" in response.text.lower()


def convert_chat_message_to_response_output(message: Dict[str, Any], output_index: int) -> List[Dict[str, Any]]:
    """Convert one completed Chat assistant message to Responses output items."""
    output: List[Dict[str, Any]] = []
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        output.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_content}],
            }
        )

    content = message.get("content")
    refusal = message.get("refusal")
    if content is not None or refusal:
        content_parts: List[Dict[str, Any]] = []
        if isinstance(content, str):
            content_parts.append({"type": "output_text", "text": content, "annotations": []})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                    content_parts.append({"type": "output_text", "text": part.get("text", ""), "annotations": []})
                elif isinstance(part, dict):
                    content_parts.append(part)
        elif content:
            content_parts.append({"type": "output_text", "text": serialize_content_as_text(content), "annotations": []})
        if refusal:
            content_parts.append({"type": "refusal", "refusal": refusal})
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": content_parts,
            }
        )

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        arguments = function.get("arguments", "{}")
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "name": function.get("name", ""),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
            }
        )
    return output


def convert_chat_to_response_payload(chat_response: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    """Convert a non-stream Chat Completions result to a Responses object."""
    output: List[Dict[str, Any]] = []
    finish_reasons: List[str] = []
    for choice in chat_response.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            output.extend(convert_chat_message_to_response_output(message, choice.get("index", len(output))))
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reasons.append(finish_reason)

    status = "incomplete" if "length" in finish_reasons else "completed"
    response_payload: Dict[str, Any] = {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(chat_response.get("created") or time.time()),
        "status": status,
        "model": chat_response.get("model") or requested_model,
        "output": output,
        "parallel_tool_calls": True,
    }
    if status == "incomplete":
        response_payload["incomplete_details"] = {"reason": "max_output_tokens"}
    usage = convert_chat_usage_to_response_usage(chat_response.get("usage"))
    if usage is not None:
        response_payload["usage"] = usage
    return response_payload


def generate_chat_id() -> str:
    """生成 Chat Completion ID"""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def create_chat_stream_chunk(
    chunk_id: str,
    model: str,
    delta: Dict[str, Any],
    index: int = 0,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None
) -> str:
    """创建流式响应的 chunk"""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ==================== 流式响应处理 ====================

class ResponseStreamProcessor:
    """处理 Response API 的流式响应"""
    
    def __init__(self, chat_id: str, model: str, include_usage: bool = False):
        self.chat_id = chat_id
        self.model = model
        self.include_usage = include_usage
        self.accumulated_content = ""
        self.accumulated_reasoning = ""
        self.usage = None
        self.is_first_content = True
        self.current_output_index = None
        self.tool_calls = []
        self.current_tool_call = None
    
    def process_event(self, event_type: str, event_data: Dict[str, Any]) -> List[str]:
        """处理单个 SSE 事件，返回要发送的 Chat chunks"""
        chunks = []
        
        if event_type == "response.created":
            # 发送开始的 role delta
            chunks.append(create_chat_stream_chunk(
                self.chat_id, self.model,
                {"role": "assistant", "content": ""}
            ))
        
        elif event_type == "response.output_item.added":
            # 新的输出项开始
            item = event_data.get("item", {})
            self.current_output_index = event_data.get("output_index", 0)
            if item.get("type") == "function_call":
                # 工具调用开始
                self.current_tool_call = {
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": ""
                    }
                }
        
        elif event_type == "response.output_text.delta":
            # 文本增量
            delta_text = event_data.get("delta", "")
            if delta_text:
                self.accumulated_content += delta_text
                chunks.append(create_chat_stream_chunk(
                    self.chat_id, self.model,
                    {"content": delta_text}
                ))
        
        elif event_type == "response.content_part.delta":
            # 内容部分增量（另一种格式）
            delta = event_data.get("delta", {})
            if delta.get("type") == "text_delta":
                delta_text = delta.get("text", "")
                if delta_text:
                    self.accumulated_content += delta_text
                    chunks.append(create_chat_stream_chunk(
                        self.chat_id, self.model,
                        {"content": delta_text}
                    ))
        
        elif event_type == "response.reasoning_summary_text.delta":
            # 推理内容增量
            delta_text = event_data.get("delta", "")
            if delta_text:
                self.accumulated_reasoning += delta_text
                # 推理内容作为 reasoning_content 字段
                chunks.append(create_chat_stream_chunk(
                    self.chat_id, self.model,
                    {"reasoning_content": delta_text}
                ))
        
        elif event_type == "response.function_call_arguments.delta":
            # 函数调用参数增量
            delta_args = event_data.get("delta", "")
            if self.current_tool_call and delta_args:
                self.current_tool_call["function"]["arguments"] += delta_args
                # 发送工具调用增量
                tool_call_delta = {
                    "tool_calls": [{
                        "index": len(self.tool_calls),
                        "function": {"arguments": delta_args}
                    }]
                }
                chunks.append(create_chat_stream_chunk(
                    self.chat_id, self.model,
                    tool_call_delta
                ))
        
        elif event_type == "response.function_call_arguments.done":
            # 函数调用完成
            if self.current_tool_call:
                self.tool_calls.append(self.current_tool_call)
                self.current_tool_call = None
        
        elif event_type == "response.output_item.done":
            # 单个输出项完成
            pass
        
        elif event_type == "response.completed":
            # 响应完成
            response_data = event_data.get("response", {})
            self.usage = response_data.get("usage")
        
        elif event_type == "response.done":
            # 所有响应完成 (兼容不同的事件名)
            if "usage" in event_data:
                self.usage = event_data.get("usage")
        
        return chunks
    
    def get_final_chunks(self) -> List[str]:
        """获取最终的 chunks（完成信号和使用统计）"""
        chunks = []
        
        # 转换 usage 格式: Response API -> Chat API
        chat_usage = convert_response_usage_to_chat_usage(self.usage) if self.include_usage else None
        
        # 发送完成信号
        finish_chunk = create_chat_stream_chunk(
            self.chat_id, self.model,
            {},
            finish_reason="stop",
            usage=chat_usage
        )
        chunks.append(finish_chunk)
        chunks.append("data: [DONE]\n\n")
        
        return chunks
    
    def get_accumulated_response(self) -> Dict[str, Any]:
        """获取累积的完整响应（用于非流式模式）"""
        message = {
            "role": "assistant",
            "content": self.accumulated_content
        }
        
        if self.accumulated_reasoning:
            message["reasoning_content"] = self.accumulated_reasoning
        
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        
        # 转换 usage 格式: Response API -> Chat API
        chat_usage = convert_response_usage_to_chat_usage(self.usage)
        
        return {
            "id": self.chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop" if not self.tool_calls else "tool_calls"
                }
            ],
            "usage": chat_usage
        }


def create_response_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format one OpenAI Responses SSE event."""
    payload = dict(data)
    payload.setdefault("type", event_type)
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ChatStreamToResponseProcessor:
    """Turn Chat Completions SSE deltas into OpenAI Responses SSE events."""

    def __init__(self, model: str):
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.created_at = int(time.time())
        self.output_items: List[Dict[str, Any]] = []
        self.message_items: Dict[int, Dict[str, Any]] = {}
        self.tool_items: Dict[tuple[int, int], Dict[str, Any]] = {}
        self.usage: Optional[Dict[str, Any]] = None
        self.finish_reasons: List[str] = []
        self.created_emitted = False

    def _response_snapshot(self, status: str = "in_progress") -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": [self._public_item(item) for item in self.output_items],
            "parallel_tool_calls": True,
        }
        usage = convert_chat_usage_to_response_usage(self.usage)
        if usage is not None:
            response["usage"] = usage
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        return response

    def start_events(self) -> List[str]:
        if self.created_emitted:
            return []
        self.created_emitted = True
        return [
            create_response_sse_event(
                "response.created",
                {"response": self._response_snapshot()},
            )
        ]

    def _ensure_message(self, choice_index: int) -> tuple[Dict[str, Any], List[str]]:
        existing = self.message_items.get(choice_index)
        if existing is not None:
            return existing, []

        item = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
            "_output_index": len(self.output_items),
            "_content_started": False,
            "_text": "",
        }
        self.message_items[choice_index] = item
        self.output_items.append(item)
        return item, [
            create_response_sse_event(
                "response.output_item.added",
                {
                    "output_index": item["_output_index"],
                    "item": self._public_item(item),
                },
            )
        ]

    def _ensure_message_content_part(self, item: Dict[str, Any]) -> List[str]:
        if item["_content_started"]:
            return []
        item["_content_started"] = True
        part = {"type": "output_text", "text": "", "annotations": []}
        item["content"].append(part)
        return [
            create_response_sse_event(
                "response.content_part.added",
                {
                    "item_id": item["id"],
                    "output_index": item["_output_index"],
                    "content_index": 0,
                    "part": part,
                },
            )
        ]

    def _ensure_tool(self, choice_index: int, tool_index: int, delta: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        key = (choice_index, tool_index)
        existing = self.tool_items.get(key)
        if existing is not None:
            return existing, []

        function = delta.get("function") if isinstance(delta.get("function"), dict) else {}
        item = {
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "type": "function_call",
            "status": "in_progress",
            "call_id": delta.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "name": function.get("name", ""),
            "arguments": "",
            "_output_index": len(self.output_items),
        }
        self.tool_items[key] = item
        self.output_items.append(item)
        return item, [
            create_response_sse_event(
                "response.output_item.added",
                {
                    "output_index": item["_output_index"],
                    "item": self._public_item(item),
                },
            )
        ]

    @staticmethod
    def _public_item(item: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in item.items() if not key.startswith("_")}

    def process_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        events = self.start_events()
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = usage

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            choice_index = int(choice.get("index", 0))
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}

            content = delta.get("content")
            if content is not None:
                item, item_events = self._ensure_message(choice_index)
                events.extend(item_events)
                events.extend(self._ensure_message_content_part(item))
                if content:
                    item["_text"] += content
                    item["content"][0]["text"] = item["_text"]
                    events.append(
                        create_response_sse_event(
                            "response.output_text.delta",
                            {
                                "item_id": item["id"],
                                "output_index": item["_output_index"],
                                "content_index": 0,
                                "delta": content,
                            },
                        )
                    )

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for fallback_index, tool_delta in enumerate(tool_calls):
                    if not isinstance(tool_delta, dict):
                        continue
                    tool_index = int(tool_delta.get("index", fallback_index))
                    item, item_events = self._ensure_tool(choice_index, tool_index, tool_delta)
                    events.extend(item_events)
                    function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
                    if function.get("name"):
                        item["name"] = function["name"]
                    arguments_delta = function.get("arguments")
                    if arguments_delta:
                        item["arguments"] += arguments_delta
                        events.append(
                            create_response_sse_event(
                                "response.function_call_arguments.delta",
                                {
                                    "item_id": item["id"],
                                    "output_index": item["_output_index"],
                                    "delta": arguments_delta,
                                },
                            )
                        )

            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self.finish_reasons.append(finish_reason)
        return events

    def final_events(self) -> List[str]:
        events = self.start_events()
        for item in self.output_items:
            output_index = item["_output_index"]
            if item["type"] == "message":
                if item["_content_started"]:
                    part = item["content"][0]
                    events.append(
                        create_response_sse_event(
                            "response.output_text.done",
                            {
                                "item_id": item["id"],
                                "output_index": output_index,
                                "content_index": 0,
                                "text": part["text"],
                            },
                        )
                    )
                    events.append(
                        create_response_sse_event(
                            "response.content_part.done",
                            {
                                "item_id": item["id"],
                                "output_index": output_index,
                                "content_index": 0,
                                "part": part,
                            },
                        )
                    )
                item["status"] = "completed"
            elif item["type"] == "function_call":
                item["status"] = "completed"
                events.append(
                    create_response_sse_event(
                        "response.function_call_arguments.done",
                        {
                            "item_id": item["id"],
                            "output_index": output_index,
                            "arguments": item["arguments"],
                        },
                    )
                )
            events.append(
                create_response_sse_event(
                    "response.output_item.done",
                    {
                        "output_index": output_index,
                        "item": self._public_item(item),
                    },
                )
            )

        status = "incomplete" if "length" in self.finish_reasons else "completed"
        events.append(
            create_response_sse_event(
                "response.completed",
                {"response": self._response_snapshot(status)},
            )
        )
        return events


async def parse_sse_line(line: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """解析 SSE 行，返回 (event_type, event_data)"""
    if not line or line.startswith(":"):
        return None, None
    
    if line.startswith("event:"):
        return line[6:].strip(), None
    
    if line.startswith("data:"):
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return "done", None
        try:
            return None, json.loads(data_str)
        except json.JSONDecodeError:
            return None, None
    
    return None, None


def format_admin_time(value: str) -> str:
    if not value:
        return "-"
    return value.replace("T", " ").replace("+00:00", " UTC")


def build_admin_notice(message: str, level: str) -> str:
    if not message:
        return ""

    level_class = {
        "success": "notice-success",
        "error": "notice-error",
        "warning": "notice-warning",
    }.get(level, "notice-success")

    return f'<div class="notice {level_class}">{escape(message)}</div>'


@lru_cache(maxsize=None)
def load_html_template(template_name: str) -> Template:
    template_path = TEMPLATE_DIR / template_name
    return Template(template_path.read_text(encoding="utf-8"))


def render_html_template(template_name: str, **context: Any) -> str:
    normalized_context = {
        key: "" if value is None else str(value)
        for key, value in context.items()
    }
    return load_html_template(template_name).safe_substitute(normalized_context)


def render_admin_layout(
    title: str,
    content: str,
    username: Optional[str] = None,
    notice: str = "",
    level: str = "success",
) -> str:
    nav_html = ""
    if username:
        nav_html = f"""
        <div class="topbar-actions">
            <span class="badge">管理员 {escape(username)}</span>
            <a class="ghost-link" href="/admin">控制台</a>
            <form method="post" action="/admin/logout">
                <button class="ghost-button" type="submit">退出登录</button>
            </form>
        </div>
        """

    return render_html_template(
        "admin_layout.html",
        title=escape(title),
        nav_html=nav_html,
        notice_html=build_admin_notice(notice, level),
        content_html=content,
    )


def render_login_page(error_message: str = "", username: str = "", next_path: str = "/admin") -> str:
    body = render_html_template(
        "admin_login.html",
        next_path=escape(next_path),
        username=escape(username),
    )
    return render_admin_layout("管理后台登录", body, notice=error_message, level="error" if error_message else "success")


def render_dashboard_page(
    request: Request,
    username: str,
    channels: List[Dict[str, Any]],
    stats: Dict[str, int],
    notice: str = "",
    level: str = "success",
) -> str:
    channel_cards = []

    if channels:
        for channel in channels:
            toggle_label = "停用" if channel["enabled"] else "启用"
            toggle_target = "0" if channel["enabled"] else "1"
            state_class = "pill-enabled" if channel["enabled"] else "pill-disabled"
            state_text = "启用中" if channel["enabled"] else "已停用"
            protocol_type_label = (
                "Chat → Responses"
                if channel.get("protocol_type") == CHANNEL_TYPE_CHAT_TO_RESPONSE
                else "Responses → Chat"
            )
            description = escape(channel["description"] or "未填写描述")
            channel_cards.append(
                render_html_template(
                    "admin_channel_card.html",
                    channel_name=escape(channel["name"]),
                    description=description,
                    state_class=state_class,
                    state_text=state_text,
                    upstream_base_url=escape(channel["upstream_base_url"]),
                    upstream_api_key_masked=escape(mask_secret(channel["upstream_api_key"])),
                    protocol_type_label=protocol_type_label,
                    access_key=escape(channel["access_key"]),
                    updated_at=escape(format_admin_time(channel["updated_at"])),
                    channel_id=channel["id"],
                    toggle_target=toggle_target,
                    toggle_label=toggle_label,
                )
            )

    body = render_html_template(
        "admin_dashboard.html",
        stats_total=stats["total"],
        stats_enabled=stats["enabled"],
        stats_disabled=stats["disabled"],
        channel_cards_html="".join(channel_cards) if channel_cards else '<div class="channel-item"><p class="muted">当前还没有渠道，请先创建一个。</p></div>',
    )

    return render_admin_layout("多渠道控制台", body, username=username, notice=notice, level=level)


def render_channel_detail_page(
    request: Request,
    username: str,
    channel: Dict[str, Any],
    notice: str = "",
    level: str = "success",
) -> str:
    checked = "checked" if channel["enabled"] else ""
    body = render_html_template(
        "admin_channel_detail.html",
        channel_id=channel["id"],
        channel_name=escape(channel["name"]),
        upstream_base_url=escape(channel["upstream_base_url"]),
        description=escape(channel["description"]),
        enabled_checked=checked,
        access_key=escape(channel["access_key"]),
        upstream_api_key_masked=escape(mask_secret(channel["upstream_api_key"])),
        response_to_chat_selected=(
            "selected" if channel.get("protocol_type") != CHANNEL_TYPE_CHAT_TO_RESPONSE else ""
        ),
        chat_to_response_selected=(
            "selected" if channel.get("protocol_type") == CHANNEL_TYPE_CHAT_TO_RESPONSE else ""
        ),
        created_at=escape(format_admin_time(channel["created_at"])),
        updated_at=escape(format_admin_time(channel["updated_at"])),
    )

    return render_admin_layout(
        f"渠道详情 - {channel['name']}",
        body,
        username=username,
        notice=notice,
        level=level,
    )


async def parse_form_body(request: Request) -> Dict[str, str]:
    raw_body = await request.body()
    parsed = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def get_authenticated_admin(request: Request) -> Optional[str]:
    session_manager: AdminSessionManager = request.app.state.admin_sessions
    return session_manager.get_username(request.cookies.get(ADMIN_SESSION_COOKIE_NAME))


def build_admin_redirect(path: str, message: str = "", level: str = "success") -> RedirectResponse:
    if message:
        separator = "&" if "?" in path else "?"
        path = f"{path}{separator}notice={quote(message)}&level={quote(level)}"
    return RedirectResponse(url=path, status_code=303)


def wants_json_response(request: Request) -> bool:
    accept = request.headers.get("accept", "").lower()
    requested_with = request.headers.get("x-requested-with", "").lower()
    return requested_with == "xmlhttprequest" or "application/json" in accept


def build_admin_feedback_response(
    request: Request,
    path: str,
    message: str,
    level: str = "success",
    status_code: int = 200,
) -> Response:
    if wants_json_response(request):
        return JSONResponse(
            {
                "ok": level != "error",
                "level": level,
                "message": message,
            },
            status_code=status_code,
        )
    return build_admin_redirect(path, message, level)


def build_login_redirect(next_path: str = "/admin") -> RedirectResponse:
    return RedirectResponse(url=f"/admin/login?next={quote(next_path)}", status_code=303)


def normalize_next_path(next_path: Optional[str]) -> str:
    if next_path and next_path.startswith("/admin"):
        return next_path
    return "/admin"


async def resolve_channel_from_request(request: Request, authorization: Optional[str]) -> Dict[str, Any]:
    access_key = extract_bearer_token(authorization)
    store: SettingsStore = request.app.state.settings_store
    channel = store.get_channel_by_access_key(access_key)

    if not channel:
        logger.warning("无效的渠道访问 key")
        raise HTTPException(status_code=401, detail="Invalid channel access key")

    if not channel["enabled"]:
        logger.warning(f"渠道已停用: id={channel['id']}, name={channel['name']}")
        raise HTTPException(status_code=403, detail="Channel is disabled")

    return channel


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    app.state.settings_store = SettingsStore(
        database_path=DATABASE_PATH,
        default_admin_username=ADMIN_USERNAME,
        default_admin_password=ADMIN_PASSWORD,
    )
    await asyncio.to_thread(app.state.settings_store.initialize)
    app.state.admin_sessions = AdminSessionManager(ADMIN_SESSION_TTL_SECONDS)

    # 配置连接池限制，防止长时间运行后连接泄漏
    limits = httpx.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=KEEPALIVE_EXPIRY
    )
    # 配置超时：连接超时、读取超时、写入超时、连接池获取超时
    timeout = httpx.Timeout(
        connect=30.0,      # 连接超时
        read=DEFAULT_TIMEOUT,  # 读取超时
        write=30.0,        # 写入超时  
        pool=POOL_TIMEOUT          # 从连接池获取连接的超时
    )
    app.state.http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=True  # 启用 HTTP/2 提升长连接性能
    )
    logger.info(f"配置存储初始化完成: database={DATABASE_PATH}")
    logger.info(f"HTTP 客户端初始化: max_connections={MAX_CONNECTIONS}, keepalive={MAX_KEEPALIVE_CONNECTIONS}, expiry={KEEPALIVE_EXPIRY}s")
    yield
    await app.state.http_client.aclose()
    logger.info("HTTP 客户端已关闭")

app = FastAPI(
    title="Response to Chat API Proxy",
    description="将 OpenAI Response 协议接口转发为 Chat 协议接口",
    version="1.0.0",
    lifespan=lifespan
)


PASSTHROUGH_REQUEST_EXCLUDED_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "transfer-encoding",
}

PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def extract_bearer_token(authorization: Optional[str]) -> str:
    """Extract bearer token from Authorization header."""
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    return authorization.replace("Bearer ", "", 1) if authorization.startswith("Bearer ") else authorization


def build_passthrough_request_headers(request: Request, token: str) -> Dict[str, str]:
    """Copy client headers for upstream passthrough, excluding hop-by-hop headers."""
    headers: Dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in PASSTHROUGH_REQUEST_EXCLUDED_HEADERS:
            continue
        headers[key] = value

    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers.pop("Authorization", None)
    headers["User-Agent"] = UPSTREAM_USER_AGENT
    return headers


def build_channel_upstream_headers(
    channel: Dict[str, Any],
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": UPSTREAM_USER_AGENT,
    }
    if channel.get("upstream_api_key"):
        headers["Authorization"] = f"Bearer {channel['upstream_api_key']}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


def build_passthrough_response_headers(headers: httpx.Headers) -> Dict[str, str]:
    """Filter upstream response headers for downstream responses."""
    filtered_headers: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS:
            continue
        filtered_headers[key] = value
    return filtered_headers


def build_upstream_error_response(status_code: int, error_text: str) -> JSONResponse:
    """Normalize upstream non-200 responses for downstream clients."""
    should_return_500 = False
    error_output: Dict[str, Any]

    try:
        error_output = json.loads(error_text)
        error_obj = error_output.get("error", {})
        if isinstance(error_obj, dict):
            error_code = error_obj.get("code")
            error_message = error_obj.get("message", "")
        else:
            error_code = None
            error_message = str(error_obj)

        if error_code == 503 or \
           error_code == "plan_quota_exceeded" or \
           "账户池都无可用" in error_message or \
           status_code == 402:
            should_return_500 = True
    except json.JSONDecodeError:
        if "账户池都无可用" in error_text:
            should_return_500 = True
        error_output = {
            "error": {
                "message": error_text,
                "type": "upstream_error",
                "code": str(status_code)
            }
        }

    if status_code == 402:
        should_return_500 = True

    if should_return_500 and "error" in error_output:
        error_output["error"]["upstream_status_code"] = status_code
        error_output["error"]["gateway_status_code"] = 500

    return JSONResponse(
        status_code=500 if should_return_500 else status_code,
        content=error_output
    )


def summarize_upstream_response(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        text = " ".join(response.text.split())
        if not text:
            return ""
        return text[:120] + ("..." if len(text) > 120 else "")

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            code = str(error.get("code") or "").strip()
            if message and code:
                return f"{message} (code: {code})"
            if message:
                return message
            if code:
                return f"code: {code}"

        data = payload.get("data")
        if isinstance(data, list):
            return f"已返回 {len(data)} 个模型"

    return ""


async def handle_chat_to_response_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request_body: Dict[str, Any],
    model: str,
) -> Response:
    """Call a Chat upstream as SSE and expose an equivalent Responses stream."""
    processor = ChatStreamToResponseProcessor(model)
    start_time = time.monotonic()
    stream_context = client.stream(
        "POST",
        url,
        headers=headers,
        json=request_body,
        timeout=httpx.Timeout(
            connect=30.0,
            read=STREAM_READ_TIMEOUT,
            write=30.0,
            pool=POOL_TIMEOUT,
        ),
    )
    upstream_response: Optional[httpx.Response] = None
    close_stream_context = True
    fallback_request_body = build_flat_tool_choice_fallback(request_body)

    try:
        upstream_response = await stream_context.__aenter__()
        if not upstream_response.is_success:
            error_body = await upstream_response.aread()
            if fallback_request_body and upstream_requires_flat_tool_choice(upstream_response):
                logger.info("Retrying Chat stream with flat tool_choice.name compatibility format")
                await stream_context.__aexit__(None, None, None)
                stream_context = client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=fallback_request_body,
                    timeout=httpx.Timeout(
                        connect=30.0,
                        read=STREAM_READ_TIMEOUT,
                        write=30.0,
                        pool=POOL_TIMEOUT,
                    ),
                )
                upstream_response = await stream_context.__aenter__()
                if not upstream_response.is_success:
                    error_body = await upstream_response.aread()
                    return build_upstream_error_response(
                        upstream_response.status_code,
                        error_body.decode("utf-8", errors="ignore"),
                    )
                close_stream_context = False
            else:
                return build_upstream_error_response(
                    upstream_response.status_code,
                    error_body.decode("utf-8", errors="ignore"),
                )
        else:
            close_stream_context = False
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Request timeout", "type": "timeout_error"}},
        )
    except Exception as exc:
        logger.error("Chat to Responses stream initialization failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error"}},
        )
    finally:
        if close_stream_context:
            await stream_context.__aexit__(None, None, None)

    async def stream_generator():
        emitted_final_events = False
        try:
            initial_events = processor.start_events()
            for event in initial_events:
                yield event

            async for line in upstream_response.aiter_lines():
                if STREAM_MAX_DURATION > 0 and (time.monotonic() - start_time) > STREAM_MAX_DURATION:
                    yield create_response_sse_event(
                        "error",
                        {"message": "Stream max duration exceeded", "code": "timeout_error"},
                    )
                    return
                line = line.strip()
                if not line or line.startswith("event:"):
                    continue
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    for event in processor.final_events():
                        yield event
                    emitted_final_events = True
                    return
                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in upstream Chat SSE: %s", data_str[:200])
                    continue

                upstream_error = event_data.get("error") if isinstance(event_data, dict) else None
                if isinstance(upstream_error, dict):
                    yield create_response_sse_event("error", upstream_error)
                    return
                if isinstance(event_data, dict):
                    for event in processor.process_chunk(event_data):
                        yield event

            for event in processor.final_events():
                yield event
            emitted_final_events = True
        except asyncio.CancelledError:
            logger.warning("Chat to Responses stream cancelled by client")
            raise
        except httpx.TimeoutException:
            yield create_response_sse_event(
                "error",
                {"message": "Request timeout", "code": "timeout_error"},
            )
        except Exception as exc:
            logger.error("Chat to Responses stream failed: %s", exc, exc_info=True)
            yield create_response_sse_event(
                "error",
                {"message": str(exc), "code": "internal_error"},
            )
        finally:
            if not emitted_final_events:
                logger.debug("Chat to Responses stream ended without a completion event")
            await stream_context.__aexit__(None, None, None)

    return StreamingResponse(
        stream_generator(),
        status_code=upstream_response.status_code,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def handle_response_to_chat_conversion(
    request: Request,
    channel: Dict[str, Any],
) -> Response:
    """Expose /v1/responses by converting it to an upstream Chat request."""
    try:
        response_request = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON: {exc}", "type": "invalid_request_error"}},
        )

    if not isinstance(response_request, dict):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Request body must be a JSON object", "type": "invalid_request_error"}},
        )

    try:
        chat_request = convert_response_to_chat_request(response_request)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    client: httpx.AsyncClient = request.app.state.http_client
    is_stream_request = bool(chat_request["stream"])
    headers = build_channel_upstream_headers(
        channel,
        {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if is_stream_request else "application/json",
        },
    )
    upstream_url = f"{channel['upstream_base_url']}/chat/completions"
    logger.debug(
        "Converted /v1/responses -> %s: model=%s, stream=%s, messages=%d",
        upstream_url,
        chat_request["model"],
        is_stream_request,
        len(chat_request["messages"]),
    )

    if is_stream_request:
        return await handle_chat_to_response_stream(
            client,
            upstream_url,
            headers,
            chat_request,
            chat_request["model"],
        )

    try:
        upstream_response = await client.post(
            upstream_url,
            headers=headers,
            json=chat_request,
            timeout=DEFAULT_TIMEOUT,
        )
        fallback_request = build_flat_tool_choice_fallback(chat_request)
        if fallback_request and upstream_requires_flat_tool_choice(upstream_response):
            logger.info("Retrying Chat request with flat tool_choice.name compatibility format")
            upstream_response = await client.post(
                upstream_url,
                headers=headers,
                json=fallback_request,
                timeout=DEFAULT_TIMEOUT,
            )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Request timeout", "type": "timeout_error"}},
        )
    except httpx.HTTPError as exc:
        logger.error("Chat to Responses non-stream request failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )

    if not upstream_response.is_success:
        return build_upstream_error_response(upstream_response.status_code, upstream_response.text)
    try:
        chat_response = upstream_response.json()
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Upstream Chat response is not valid JSON",
                    "type": "upstream_error",
                }
            },
        )
    if not isinstance(chat_response, dict):
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Upstream Chat response is invalid", "type": "upstream_error"}},
        )
    return JSONResponse(content=convert_chat_to_response_payload(chat_response, chat_request["model"]))


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/login")
async def admin_login_page(request: Request):
    username = get_authenticated_admin(request)
    if username:
        return RedirectResponse(url="/admin", status_code=303)

    next_path = normalize_next_path(request.query_params.get("next"))
    error_message = request.query_params.get("error", "")
    return HTMLResponse(
        render_login_page(
            error_message=error_message,
            username=request.query_params.get("username", ""),
            next_path=next_path,
        )
    )


@app.post("/admin/login")
async def admin_login_submit(request: Request):
    form = await parse_form_body(request)
    username = form.get("username", "").strip()
    password = form.get("password", "")
    next_path = normalize_next_path(form.get("next"))

    store: SettingsStore = request.app.state.settings_store
    is_valid = await asyncio.to_thread(store.authenticate_admin, username, password)
    if not is_valid:
        login_page = render_login_page(
            error_message="账号或密码错误",
            username=username,
            next_path=next_path,
        )
        return HTMLResponse(login_page, status_code=401)

    session_manager: AdminSessionManager = request.app.state.admin_sessions
    session_token = session_manager.create_session(username)
    response = RedirectResponse(url=next_path, status_code=303)
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE_NAME,
        value=session_token,
        max_age=ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=ADMIN_COOKIE_SECURE,
        path="/",
    )
    return response


@app.post("/admin/logout")
async def admin_logout(request: Request):
    session_manager: AdminSessionManager = request.app.state.admin_sessions
    session_manager.revoke(request.cookies.get(ADMIN_SESSION_COOKIE_NAME))
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(ADMIN_SESSION_COOKIE_NAME, path="/")
    return response


@app.get("/admin")
async def admin_dashboard(request: Request):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    store: SettingsStore = request.app.state.settings_store
    channels, stats = await asyncio.gather(
        asyncio.to_thread(store.list_channels),
        asyncio.to_thread(store.count_channels),
    )
    return HTMLResponse(
        render_dashboard_page(
            request=request,
            username=username,
            channels=channels,
            stats=stats,
            notice=request.query_params.get("notice", ""),
            level=request.query_params.get("level", "success"),
        )
    )


@app.get("/admin/channels/{channel_id}")
async def admin_channel_detail(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect(f"/admin/channels/{channel_id}")

    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.get_channel, channel_id)
    if not channel:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return HTMLResponse(
        render_channel_detail_page(
            request=request,
            username=username,
            channel=channel,
            notice=request.query_params.get("notice", ""),
            level=request.query_params.get("level", "success"),
        )
    )


@app.post("/admin/channels")
async def admin_create_channel(request: Request):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    form = await parse_form_body(request)
    store: SettingsStore = request.app.state.settings_store

    try:
        await asyncio.to_thread(
            store.create_channel,
            form.get("name", ""),
            form.get("upstream_base_url", ""),
            form.get("upstream_api_key", ""),
            form.get("description", ""),
            form.get("protocol_type", CHANNEL_TYPE_RESPONSE_TO_CHAT),
        )
        return build_admin_redirect(
            "/admin",
            "渠道已创建，系统已自动生成外部访问 key",
            "success",
        )
    except ValueError as exc:
        return build_admin_redirect("/admin", str(exc), "error")


@app.post("/admin/channels/{channel_id}")
async def admin_update_channel(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect(f"/admin/channels/{channel_id}")

    form = await parse_form_body(request)
    store: SettingsStore = request.app.state.settings_store

    try:
        channel = await asyncio.to_thread(
            store.update_channel,
            channel_id,
            form.get("name", ""),
            form.get("upstream_base_url", ""),
            form.get("upstream_api_key", ""),
            form.get("description", ""),
            form.get("enabled") == "on",
            form.get("clear_upstream_api_key") == "on",
            form.get("protocol_type", CHANNEL_TYPE_RESPONSE_TO_CHAT),
        )
        if not channel:
            return build_admin_redirect("/admin", "渠道不存在", "error")
        return build_admin_redirect("/admin", "渠道配置已更新", "success")
    except ValueError as exc:
        return build_admin_redirect(f"/admin/channels/{channel_id}", str(exc), "error")


@app.post("/admin/channels/{channel_id}/test")
async def admin_test_channel(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        if wants_json_response(request):
            next_path = f"/admin/channels/{channel_id}"
            return JSONResponse(
                {
                    "ok": False,
                    "level": "error",
                    "message": "登录状态已失效，请重新登录",
                    "redirect_to": f"/admin/login?next={quote(next_path)}",
                },
                status_code=401,
            )
        return build_login_redirect(f"/admin/channels/{channel_id}")

    form = await parse_form_body(request)
    return_path = normalize_next_path(form.get("return_to") or f"/admin/channels/{channel_id}")
    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.get_channel, channel_id)
    if not channel:
        return build_admin_feedback_response(
            request=request,
            path="/admin",
            message="渠道不存在",
            level="error",
            status_code=404,
        )

    client: httpx.AsyncClient = request.app.state.http_client
    if channel.get("protocol_type") == CHANNEL_TYPE_CHAT_TO_RESPONSE:
        test_endpoint = "/chat/completions"
        test_payload = {
            "model": ADMIN_TEST_MODEL,
            "messages": [{"role": "user", "content": ADMIN_TEST_INPUT}],
            "stream": False,
        }
    else:
        test_endpoint = "/responses"
        test_payload = {
            "model": ADMIN_TEST_MODEL,
            "input": ADMIN_TEST_INPUT,
            "stream": False,
        }
    try:
        response = await client.post(
            f"{channel['upstream_base_url']}{test_endpoint}",
            headers=build_channel_upstream_headers(
                channel,
                {
                    "Accept": "application/json",
                },
            ),
            json=test_payload,
        )
    except httpx.TimeoutException:
        return build_admin_feedback_response(
            request=request,
            path=return_path,
            message=f"渠道 {channel['name']} 联通测试超时：调用 {ADMIN_TEST_MODEL} 未在限定时间内返回",
            level="error",
            status_code=504,
        )
    except httpx.HTTPError as exc:
        return build_admin_feedback_response(
            request=request,
            path=return_path,
            message=f"渠道 {channel['name']} 联通测试失败：{exc}",
            level="error",
            status_code=502,
        )

    summary = summarize_upstream_response(response)
    if response.is_success:
        message = f"渠道 {channel['name']} 联通正常，{ADMIN_TEST_MODEL} 调用成功"
        if summary:
            message = f"{message}，{summary}"
        return build_admin_feedback_response(
            request=request,
            path=return_path,
            message=message,
            level="success",
        )

    failure_reason = f"HTTP {response.status_code}"
    if response.status_code in (401, 403):
        failure_reason = f"HTTP {response.status_code}，上游鉴权失败"
    elif response.status_code == 404:
        failure_reason = f"HTTP 404，上游未提供 {test_endpoint} 接口"
    if summary:
        failure_reason = f"{failure_reason}，{summary}"
    return build_admin_feedback_response(
        request=request,
        path=return_path,
        message=f"渠道 {channel['name']} 联通失败：{ADMIN_TEST_MODEL} 调用未通过，{failure_reason}",
        level="error",
        status_code=response.status_code,
    )


@app.post("/admin/channels/{channel_id}/toggle")
async def admin_toggle_channel(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    form = await parse_form_body(request)
    enabled = form.get("enabled") == "1"
    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.set_channel_enabled, channel_id, enabled)
    if not channel:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return build_admin_redirect(
        "/admin",
        f"渠道 {channel['name']} 已{'启用' if enabled else '停用'}",
        "success",
    )


@app.post("/admin/channels/{channel_id}/rotate-key")
async def admin_rotate_channel_key(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect(f"/admin/channels/{channel_id}")

    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.rotate_access_key, channel_id)
    if not channel:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return build_admin_redirect(
        f"/admin/channels/{channel_id}",
        "外部访问 key 已轮换，请同步更新外部调用方配置",
        "warning",
    )


@app.post("/admin/channels/{channel_id}/delete")
async def admin_delete_channel(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    store: SettingsStore = request.app.state.settings_store
    deleted = await asyncio.to_thread(store.delete_channel, channel_id)
    if not deleted:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return build_admin_redirect("/admin", "渠道已删除", "success")


@app.post("/admin/change-password")
async def admin_change_password(request: Request):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    form = await parse_form_body(request)
    new_password = form.get("new_password", "")
    confirm_password = form.get("confirm_password", "")
    if new_password != confirm_password:
        return build_admin_redirect("/admin", "两次输入的新密码不一致", "error")

    store: SettingsStore = request.app.state.settings_store
    success, message = await asyncio.to_thread(
        store.change_admin_password,
        username,
        form.get("current_password", ""),
        new_password,
    )
    return build_admin_redirect("/admin", message, "success" if success else "error")


@app.post("/v1/responses")
async def responses_passthrough(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Responses passthrough, or Responses -> Chat for reverse-mode channels."""
    channel = await resolve_channel_from_request(request, authorization)
    logger.debug("/v1/responses 命中渠道: id=%s, name=%s", channel["id"], channel["name"])

    if channel.get("protocol_type") == CHANNEL_TYPE_CHAT_TO_RESPONSE:
        return await handle_response_to_chat_conversion(request, channel)

    raw_body = await request.body()
    is_stream_request = "text/event-stream" in request.headers.get("accept", "").lower()

    if raw_body:
        try:
            request_json = json.loads(raw_body)
            is_stream_request = bool(request_json.get("stream")) or is_stream_request
            logger.debug(
                "Received /v1/responses request: stream=%s, body_bytes=%d",
                is_stream_request,
                len(raw_body),
            )
        except json.JSONDecodeError:
            logger.debug("Received /v1/responses request: <non-json body>")
    else:
        logger.debug("Received /v1/responses request: <empty body>")

    client: httpx.AsyncClient = request.app.state.http_client
    upstream_url = f"{channel['upstream_base_url']}/responses"
    upstream_headers = build_passthrough_request_headers(request, channel["upstream_api_key"])
    upstream_params = tuple(request.query_params.multi_items())
    stream_timeout = httpx.Timeout(
        connect=30.0,
        read=STREAM_READ_TIMEOUT,
        write=30.0,
        pool=POOL_TIMEOUT
    )

    logger.debug("Passthrough /v1/responses -> %s, stream=%s", upstream_url, is_stream_request)

    if is_stream_request:
        stream_context = client.stream(
            "POST",
            upstream_url,
            headers=upstream_headers,
            params=upstream_params,
            content=raw_body,
            timeout=stream_timeout
        )
        upstream_response = None

        try:
            upstream_response = await stream_context.__aenter__()
            logger.debug("Upstream /responses stream status: %s", upstream_response.status_code)
            response_headers = build_passthrough_response_headers(upstream_response.headers)

            async def stream_generator():
                try:
                    async for chunk in upstream_response.aiter_raw():
                        if chunk:
                            yield chunk
                except asyncio.CancelledError:
                    logger.warning("/v1/responses stream cancelled by client")
                    raise
                finally:
                    await stream_context.__aexit__(None, None, None)

            return StreamingResponse(
                stream_generator(),
                status_code=upstream_response.status_code,
                headers=response_headers
            )
        except httpx.TimeoutException:
            if upstream_response is not None:
                await upstream_response.aclose()
            logger.error("/v1/responses streaming request timed out")
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": "Request timeout",
                        "type": "timeout_error"
                    }
                }
            )
        except Exception:
            if upstream_response is not None:
                await upstream_response.aclose()
            raise

    try:
        upstream_response = await client.post(
            upstream_url,
            headers=upstream_headers,
            params=upstream_params,
            content=raw_body,
            timeout=DEFAULT_TIMEOUT
        )
        logger.debug("Upstream /responses non-stream status: %s", upstream_response.status_code)
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=build_passthrough_response_headers(upstream_response.headers)
        )
    except httpx.TimeoutException:
        logger.error("/v1/responses non-stream request timed out")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"/v1/responses passthrough failed: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
        )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Chat Completions 接口 - 转发到 Response API"""

    channel = await resolve_channel_from_request(request, authorization)
    logger.debug("/v1/chat/completions 命中渠道: id=%s, name=%s", channel["id"], channel["name"])

    if channel.get("protocol_type") == CHANNEL_TYPE_CHAT_TO_RESPONSE:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "This channel is configured for Chat to Responses conversion; use /v1/responses.",
                    "type": "invalid_request_error",
                }
            },
        )
    
    # 解析请求体
    try:
        body = await request.json()
        chat_request = ChatCompletionRequest(**body)
        logger.debug(
            "Received /v1/chat/completions request: model=%s, stream=%s, messages=%d",
            chat_request.model,
            chat_request.stream,
            len(chat_request.messages),
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"请求体解析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid request body: {str(e)}")
    
    # 转换为 Response API 请求
    response_request = convert_chat_to_response_request(chat_request)
    logger.debug(
        "Converted Response API request: model=%s, input_items=%d, stream=%s",
        response_request["model"],
        len(response_request["input"]),
        response_request["stream"],
    )
    
    # 生成 Chat ID
    chat_id = generate_chat_id()
    logger.debug(f"生成 Chat ID: {chat_id}")
    
    # 获取 HTTP 客户端
    client: httpx.AsyncClient = request.app.state.http_client
    
    # 准备请求头
    headers = build_channel_upstream_headers(
        channel,
        {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    
    # Response API URL
    response_url = f"{channel['upstream_base_url']}/responses"
    logger.debug("转发到: %s", response_url)
    
    if chat_request.stream:
        # 流式模式：直接转发 SSE
        logger.debug("使用流式模式处理请求")
        return await handle_stream_response(
            client, response_url, headers, response_request,
            chat_id, chat_request.model,
            bool(chat_request.stream_options.include_usage) if chat_request.stream_options else False
        )
    else:
        # 非流式模式：收集完整响应后返回
        logger.debug("使用非流式模式处理请求")
        return await handle_non_stream_response(
            client, response_url, headers, response_request,
            chat_id, chat_request.model
        )


async def handle_stream_response(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request_body: Dict[str, Any],
    chat_id: str,
    model: str,
    include_usage: bool
) -> Response:
    """处理流式响应"""

    processor = ResponseStreamProcessor(chat_id, model, include_usage)
    current_event_type = None
    start_time = time.monotonic()
    stream_context = client.stream(
        "POST",
        url,
        headers=headers,
        json=request_body,
        timeout=httpx.Timeout(
            connect=30.0,
            read=STREAM_READ_TIMEOUT,
            write=30.0,
            pool=POOL_TIMEOUT
        )
    )
    upstream_response: Optional[httpx.Response] = None
    close_stream_context = True

    try:
        logger.debug(f"开始流式请求到 {url}")
        upstream_response = await stream_context.__aenter__()
        logger.debug("上游响应状态码: %s", upstream_response.status_code)
        logger.debug(f"上游响应头: {dict(upstream_response.headers)}")

        if upstream_response.status_code != 200:
            error_body = await upstream_response.aread()
            error_msg = error_body.decode("utf-8", errors="ignore")
            logger.error(f"上游错误响应: {error_msg}")
            return build_upstream_error_response(upstream_response.status_code, error_msg)

        close_stream_context = False
    except httpx.TimeoutException:
        logger.error("请求超时")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"流式处理初始化异常: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
        )
    finally:
        if close_stream_context:
            await stream_context.__aexit__(None, None, None)

    async def stream_generator():
        nonlocal current_event_type
        try:
            if upstream_response is None:
                return

            async for line in upstream_response.aiter_lines():
                if STREAM_MAX_DURATION > 0 and (time.monotonic() - start_time) > STREAM_MAX_DURATION:
                    logger.error(f"流式请求超过最大持续时间: {STREAM_MAX_DURATION}s, chat_id={chat_id}")
                    error_chunk = {
                        "error": {
                            "message": "Stream max duration exceeded",
                            "type": "timeout_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                    return

                line = line.strip()
                if not line:
                    continue

                logger.debug(f"收到上游数据行: {line[:200]}..." if len(line) > 200 else f"收到上游数据行: {line}")

                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    logger.debug(f"事件类型: {current_event_type}")
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        logger.debug(
                            "流式响应完成: chat_id=%s, model=%s, content_chars=%d, reasoning_chars=%d, tool_calls=%d",
                            chat_id,
                            model,
                            len(processor.accumulated_content),
                            len(processor.accumulated_reasoning),
                            len(processor.tool_calls),
                        )
                        for chunk in processor.get_final_chunks():
                            yield chunk
                        return

                    try:
                        event_data = json.loads(data_str)
                        logger.debug(f"解析事件数据: type={event_data.get('type', current_event_type)}")

                        if "error" in event_data:
                            error_info = event_data.get("error", {})
                            error_code = error_info.get("code")
                            error_message = error_info.get("message", "")
                            logger.error(f"上游错误响应: {json.dumps(event_data, ensure_ascii=False)}")

                            should_return_500 = (error_code == 503 or 
                                                 error_code == "503" or 
                                                 error_code == "plan_quota_exceeded" or
                                                 "账户池都无可用" in error_message or
                                                 "quota" in error_message.lower())

                            if should_return_500:
                                event_data["error"]["gateway_status_code"] = 500

                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                            return

                        if current_event_type:
                            chunks = processor.process_event(current_event_type, event_data)
                            for chunk in chunks:
                                logger.debug(f"发送 chunk: {chunk[:100]}..." if len(chunk) > 100 else f"发送 chunk: {chunk}")
                                yield chunk
                        elif "type" in event_data:
                            chunks = processor.process_event(event_data["type"], event_data)
                            for chunk in chunks:
                                logger.debug(f"发送 chunk: {chunk[:100]}..." if len(chunk) > 100 else f"发送 chunk: {chunk}")
                                yield chunk
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {e}, 原始数据: {data_str[:100]}")
                        continue

            logger.debug(
                "流结束: chat_id=%s, model=%s, content_chars=%d, reasoning_chars=%d, tool_calls=%d",
                chat_id,
                model,
                len(processor.accumulated_content),
                len(processor.accumulated_reasoning),
                len(processor.tool_calls),
            )
            for chunk in processor.get_final_chunks():
                yield chunk

        except httpx.TimeoutException:
            logger.error("请求超时")
            error_chunk = {
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        except httpx.RemoteProtocolError as e:
            logger.error(f"远程协议错误(可能是连接被重置): {str(e)}")
            error_chunk = {
                "error": {
                    "message": f"Connection reset: {str(e)}",
                    "type": "connection_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        except httpx.ReadError as e:
            logger.error(f"读取错误: {str(e)}")
            error_chunk = {
                "error": {
                    "message": f"Read error: {str(e)}",
                    "type": "connection_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            logger.warning(f"流式请求被取消 (客户端可能断开连接): chat_id={chat_id}")
            return
        except GeneratorExit:
            logger.warning(f"生成器退出 (客户端断开): chat_id={chat_id}")
            return
        except Exception as e:
            logger.error(f"流式处理异常: {str(e)}\n{traceback.format_exc()}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        finally:
            logger.debug(f"流式生成器结束: chat_id={chat_id}")
            await stream_context.__aexit__(None, None, None)

    return StreamingResponse(
        stream_generator(),
        status_code=upstream_response.status_code,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def handle_non_stream_response(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request_body: Dict[str, Any],
    chat_id: str,
    model: str
) -> JSONResponse:
    """处理非流式响应 - 收集完整的流式响应后返回"""
    
    processor = ResponseStreamProcessor(chat_id, model, include_usage=True)
    current_event_type = None
    
    try:
        logger.debug(f"开始非流式请求到 {url}")
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=request_body,
            timeout=DEFAULT_TIMEOUT
        ) as response:
            logger.debug("上游响应状态码: %s", response.status_code)
            logger.debug(f"上游响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                logger.error(f"上游错误响应: {error_text}")
                
                # 检查是否为需要返回 500 状态码的错误（让网关触发自动禁用）
                # 包括：账户池无可用(503)、配额不足(402)
                should_return_500 = False
                error_output: Dict[str, Any]
                try:
                    error_data = json.loads(error_text)
                    error_code = error_data.get("error", {}).get("code")
                    error_message = error_data.get("error", {}).get("message", "")
                    if error_code == 503 or \
                       error_code == "plan_quota_exceeded" or \
                       "账户池都无可用" in error_message or \
                       response.status_code == 402:
                        should_return_500 = True
                    # 直接使用上游的错误响应
                    error_output = error_data
                except:
                    if "账户池都无可用" in error_text:
                        should_return_500 = True
                    # JSON 解析失败，包装成标准格式
                    error_output = {
                        "error": {
                            "message": error_text,
                            "type": "upstream_error",
                            "code": str(response.status_code)
                        }
                    }
                
                # 如果上游返回 402，也需要返回 500
                if response.status_code == 402:
                    should_return_500 = True
                
                # 添加状态码标记信息
                if should_return_500 and "error" in error_output:
                    error_output["error"]["upstream_status_code"] = response.status_code
                    error_output["error"]["gateway_status_code"] = 500
                
                return JSONResponse(
                    status_code=500 if should_return_500 else response.status_code,
                    content=error_output
                )
            
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                
                logger.debug(f"收到上游数据行: {line[:200]}..." if len(line) > 200 else f"收到上游数据行: {line}")
                
                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    logger.debug(f"事件类型: {current_event_type}")
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        logger.debug("收到 [DONE] 信号")
                        break
                    
                    try:
                        event_data = json.loads(data_str)
                        logger.debug(f"解析事件数据: type={event_data.get('type', current_event_type)}")
                        
                        # 检查是否为上游错误响应（如账户池无可用、配额不足）
                        if "error" in event_data:
                            error_info = event_data.get("error", {})
                            error_code = error_info.get("code")
                            error_message = error_info.get("message", "")
                            logger.error(f"上游错误响应: {json.dumps(event_data, ensure_ascii=False)}")
                            
                            # 检查是否为需要返回 500 的错误（让网关触发自动禁用）
                            # 包括：账户池无可用(503)、配额不足(plan_quota_exceeded)
                            should_return_500 = (error_code == 503 or 
                                                 error_code == "503" or 
                                                 error_code == "plan_quota_exceeded" or
                                                 "账户池都无可用" in error_message or
                                                 "quota" in error_message.lower())
                            
                            # 直接透传上游的错误响应，添加状态码标记
                            if should_return_500:
                                event_data["error"]["gateway_status_code"] = 500
                            
                            return JSONResponse(
                                status_code=500 if should_return_500 else 502,
                                content=event_data
                            )
                        
                        if current_event_type:
                            processor.process_event(current_event_type, event_data)
                        elif "type" in event_data:
                            processor.process_event(event_data["type"], event_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {e}, 原始数据: {data_str[:100]}")
                        continue
        
        # 返回累积的完整响应
        result = processor.get_accumulated_response()
        logger.debug(
            "返回完整响应: chat_id=%s, model=%s, content_chars=%d, reasoning_chars=%d, tool_calls=%d",
            chat_id,
            model,
            len(processor.accumulated_content),
            len(processor.accumulated_reasoning),
            len(processor.tool_calls),
        )
        return JSONResponse(content=result)
        
    except httpx.TimeoutException:
        logger.error("请求超时")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"非流式处理异常: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
        )


@app.get("/health")
async def health_check(request: Request):
    """健康检查接口 - 包含连接池状态"""
    client: httpx.AsyncClient = request.app.state.http_client
    store: SettingsStore = request.app.state.settings_store
    
    # 获取连接池统计信息
    pool_status = {}
    try:
        # httpx 的连接池信息
        if hasattr(client, '_transport') and client._transport:
            transport = client._transport
            pool = getattr(transport, '_pool', None)
            if pool is not None:
                pool_connections = getattr(pool, '_connections', None)
                pool_status = {
                    "connections_in_pool": len(pool_connections) if pool_connections is not None else "unknown"
                }
    except Exception as e:
        pool_status = {"error": str(e)}

    channel_stats = await asyncio.to_thread(store.count_channels)
    
    return {
        "status": "ok", 
        "service": "response-to-chat-proxy",
        "pool_status": pool_status,
        "database_path": DATABASE_PATH,
        "channels": channel_stats,
        "config": {
            "max_connections": MAX_CONNECTIONS,
            "max_keepalive_connections": MAX_KEEPALIVE_CONNECTIONS,
            "keepalive_expiry": KEEPALIVE_EXPIRY,
            "default_timeout": DEFAULT_TIMEOUT,
            "pool_timeout": POOL_TIMEOUT,
            "stream_read_timeout": STREAM_READ_TIMEOUT,
            "stream_max_duration": STREAM_MAX_DURATION
        }
    }


@app.get("/v1/models")
async def list_models(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """模型列表接口 - 透传到上游"""
    channel = await resolve_channel_from_request(request, authorization)
    client: httpx.AsyncClient = request.app.state.http_client
    headers = build_channel_upstream_headers(channel)
    
    try:
        response = await client.get(
            f"{channel['upstream_base_url']}/models",
            headers=headers,
            params=tuple(request.query_params.multi_items())
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=build_passthrough_response_headers(response.headers)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e)}}
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Response to Chat API Proxy on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)
