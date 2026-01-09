# Response2Chat API Proxy

一个将 OpenAI **Response API** 协议自动转换为 **Chat API** 协议的代理服务。

## 🎯 使用场景

当你有一个只支持 Response API 格式的上游服务，但你的客户端（如 ChatGPT 客户端、OpenAI SDK 等）只支持标准的 Chat API 格式时，可以使用本代理服务进行协议转换。

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Chat Client   │────▶│   Response2Chat     │────▶│  Response API   │
│  (OpenAI SDK)   │◀────│   Proxy (FastAPI)   │◀────│   (Upstream)    │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
        ▲                       │
        │                       ▼
   Chat API 格式           自动协议转换
```

## ✨ 功能特性

- ✅ **流式响应支持** - 完美支持 Chat API 的 stream 模式
- ✅ **非流式响应支持** - 自动收集完整响应后返回
- ✅ **工具调用转换** - 支持 Tool Calls / Function Calling
- ✅ **推理内容透传** - 支持 Reasoning Content 字段
- ✅ **多模态内容** - 支持图片等多模态输入格式转换
- ✅ **完整错误处理** - 超时控制和错误信息透传
- ✅ **使用统计** - 支持 stream_options.include_usage

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制配置模板并修改：

```bash
cp .env.example .env
```

**必须配置**：编辑 `.env` 文件，设置你的 Response API 地址：

```env
# 【必填】Response API 基础 URL
RESPONSE_API_BASE=https://your-response-api.com/v1

# 服务监听配置
HOST=0.0.0.0
PORT=8000

# 请求超时时间（秒）
DEFAULT_TIMEOUT=300
```

### 3. 启动服务

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Windows 用户也可以直接运行：

```bash
start.bat
```

## 📖 API 使用

### Chat Completions

完全兼容 OpenAI Chat API 格式：

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 流式响应

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### 流式响应（含使用统计）

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### 健康检查

```bash
curl http://localhost:8000/health
```

### 模型列表

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/v1/models
```

## ⚙️ 配置说明

| 环境变量 | 必填 | 说明 | 默认值 |
|----------|------|------|--------|
| `RESPONSE_API_BASE` | ✅ 是 | Response API 基础 URL | 无（必须配置） |
| `HOST` | 否 | 服务监听地址 | `0.0.0.0` |
| `PORT` | 否 | 服务监听端口 | `8000` |
| `DEFAULT_TIMEOUT` | 否 | 请求超时时间（秒） | `300` |

## 🔄 参数映射

| Chat API 参数 | Response API 映射 | 说明 |
|---------------|-------------------|------|
| `model` | `model` | 模型 ID |
| `messages` | `input` | 对话消息列表 |
| `max_tokens` | `max_output_tokens` | 最大生成 Token 数 |
| `max_completion_tokens` | `max_output_tokens` | 最大补全 Token 数 |
| `tools` | `tools` | 工具定义 |
| `tool_choice` | `tool_choice` | 工具选择 |
| `reasoning_effort` | `reasoning.effort` | 推理强度 |
| `response_format` | `text.format` | 响应格式 |

> 注意：`system` 角色会自动转换为 `developer` 角色（Response API 规范）

## 📝 响应格式

### 非流式响应示例

```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### 流式响应示例

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

## 📄 License

MIT License
