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
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("response2chat")

# ==================== 配置 ====================
RESPONSE_API_BASE = os.getenv("RESPONSE_API_BASE")
if not RESPONSE_API_BASE:
    raise ValueError("必须配置 RESPONSE_API_BASE 环境变量，请在 .env 文件中设置 Response API 的基础 URL")
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "300"))

# ==================== Pydantic 模型定义 ====================

# Chat API 请求模型
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
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
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None

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
    input_items = []
    
    for msg in chat_request.messages:
        # 处理 content 字段，转换多模态内容格式
        if isinstance(msg.content, str):
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
    
    # response_format 支持 (如 json_object)
    if chat_request.response_format is not None:
        # Response API 可能使用不同的格式，尝试转换
        fmt_type = chat_request.response_format.get("type")
        if fmt_type == "json_object":
            response_request["text"] = {"format": {"type": "json_object"}}
        elif fmt_type == "json_schema":
            response_request["text"] = {"format": chat_request.response_format}
    
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
        
        # 发送完成信号
        finish_chunk = create_chat_stream_chunk(
            self.chat_id, self.model,
            {},
            finish_reason="stop",
            usage=self.usage if self.include_usage else None
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
            "usage": self.usage
        }


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


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    app.state.http_client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    yield
    await app.state.http_client.aclose()

app = FastAPI(
    title="Response to Chat API Proxy",
    description="将 OpenAI Response 协议接口转发为 Chat 协议接口",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Chat Completions 接口 - 转发到 Response API"""
    
    # 获取认证 token
    if not authorization:
        logger.warning("请求缺少 Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    logger.debug(f"Token: {token[:10]}..." if len(token) > 10 else f"Token: {token}")
    
    # 解析请求体
    try:
        body = await request.json()
        logger.info(f"收到请求: {json.dumps(body, ensure_ascii=False, indent=2)}")
        chat_request = ChatCompletionRequest(**body)
        logger.debug(f"解析后的请求: model={chat_request.model}, stream={chat_request.stream}, messages_count={len(chat_request.messages)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"请求体解析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid request body: {str(e)}")
    
    # 转换为 Response API 请求
    response_request = convert_chat_to_response_request(chat_request)
    logger.info(f"转换后的 Response API 请求: {json.dumps(response_request, ensure_ascii=False, indent=2)}")
    
    # 生成 Chat ID
    chat_id = generate_chat_id()
    logger.debug(f"生成 Chat ID: {chat_id}")
    
    # 获取 HTTP 客户端
    client: httpx.AsyncClient = request.app.state.http_client
    
    # 准备请求头
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    # Response API URL
    response_url = f"{RESPONSE_API_BASE}/responses"
    logger.info(f"转发到: {response_url}")
    
    if chat_request.stream:
        # 流式模式：直接转发 SSE
        logger.info("使用流式模式处理请求")
        return await handle_stream_response(
            client, response_url, headers, response_request,
            chat_id, chat_request.model,
            chat_request.stream_options.include_usage if chat_request.stream_options else False
        )
    else:
        # 非流式模式：收集完整响应后返回
        logger.info("使用非流式模式处理请求")
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
) -> StreamingResponse:
    """处理流式响应"""
    
    async def stream_generator():
        processor = ResponseStreamProcessor(chat_id, model, include_usage)
        current_event_type = None
        
        try:
            logger.debug(f"开始流式请求到 {url}")
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=request_body,
                timeout=DEFAULT_TIMEOUT
            ) as response:
                logger.info(f"上游响应状态码: {response.status_code}")
                logger.debug(f"上游响应头: {dict(response.headers)}")
                
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_msg = error_body.decode("utf-8", errors="ignore")
                    logger.error(f"上游错误响应: {error_msg}")
                    error_chunk = {
                        "error": {
                            "message": f"Upstream error: {error_msg}",
                            "type": "upstream_error",
                            "code": str(response.status_code)
                        }
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                    return
                
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
                            logger.info("收到 [DONE] 信号，结束流式响应")
                            # 发送最终 chunks
                            for chunk in processor.get_final_chunks():
                                yield chunk
                            return
                        
                        try:
                            event_data = json.loads(data_str)
                            logger.debug(f"解析事件数据: type={event_data.get('type', current_event_type)}")
                            # 处理事件
                            if current_event_type:
                                chunks = processor.process_event(current_event_type, event_data)
                                for chunk in chunks:
                                    logger.debug(f"发送 chunk: {chunk[:100]}..." if len(chunk) > 100 else f"发送 chunk: {chunk}")
                                    yield chunk
                            # 也尝试从 data 中获取 type
                            elif "type" in event_data:
                                chunks = processor.process_event(event_data["type"], event_data)
                                for chunk in chunks:
                                    logger.debug(f"发送 chunk: {chunk[:100]}..." if len(chunk) > 100 else f"发送 chunk: {chunk}")
                                    yield chunk
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON 解析失败: {e}, 原始数据: {data_str[:100]}")
                            continue
                
                # 如果没有收到 [DONE]，手动发送结束
                logger.info("流结束，发送最终 chunks")
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
        except Exception as e:
            logger.error(f"流式处理异常: {str(e)}\n{traceback.format_exc()}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        stream_generator(),
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
            logger.info(f"上游响应状态码: {response.status_code}")
            logger.debug(f"上游响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                logger.error(f"上游错误响应: {error_text}")
                try:
                    error_data = json.loads(error_text)
                except:
                    error_data = {"message": error_text}
                
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": error_data}
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
                        logger.info("收到 [DONE] 信号")
                        break
                    
                    try:
                        event_data = json.loads(data_str)
                        logger.debug(f"解析事件数据: type={event_data.get('type', current_event_type)}")
                        if current_event_type:
                            processor.process_event(current_event_type, event_data)
                        elif "type" in event_data:
                            processor.process_event(event_data["type"], event_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {e}, 原始数据: {data_str[:100]}")
                        continue
        
        # 返回累积的完整响应
        result = processor.get_accumulated_response()
        logger.info(f"返回完整响应: content_length={len(result.get('choices', [{}])[0].get('message', {}).get('content', ''))}")
        logger.debug(f"完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
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
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "response-to-chat-proxy"}


@app.get("/v1/models")
async def list_models(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """模型列表接口 - 透传到上游"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    client: httpx.AsyncClient = request.app.state.http_client
    
    try:
        response = await client.get(
            f"{RESPONSE_API_BASE}/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
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
    print(f"Upstream API: {RESPONSE_API_BASE}")
    
    uvicorn.run(app, host=host, port=port)
