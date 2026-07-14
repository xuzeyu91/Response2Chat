# Response2Chat API Proxy

一个可在 OpenAI **Responses API** 与 **Chat Completions API** 之间双向转换的代理服务，支持多渠道路由和内置管理后台。

## 🎯 使用场景

当你有一个只支持 Response API 格式的上游服务，但你的客户端（如 ChatGPT 客户端、OpenAI SDK 等）只支持标准的 Chat API 格式时，可以使用本代理服务进行协议转换。

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Chat Client   │────▶│   Response2Chat     │────▶│  Response API   │
│  (OpenAI SDK)   │◀────│   Proxy (FastAPI)   │◀────│   (Upstream)    │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
        ▲                       │
        │                       ▼
        渠道 Access Key         自动协议转换 + 渠道路由
```

## ✨ 功能特性

      - ✅ **多渠道管理** - 每个渠道独立配置上游 URL、上游 Key、启停状态和说明
      - ✅ **管理后台** - 内置管理员登录、渠道新增/编辑/删除/轮换 Key
      - ✅ **渠道 Access Key** - 自动生成对外访问 Key，外部调用无需暴露真实上游 Key
- ✅ **流式响应支持** - 完美支持 Chat API 的 stream 模式
- ✅ **非流式响应支持** - 自动收集完整响应后返回
- ✅ **工具调用转换** - 支持 Tool Calls / Function Calling
- ✅ **推理内容透传** - 支持 Reasoning Content 字段
- ✅ **多模态内容** - 支持图片等多模态输入格式转换
- ✅ **双向协议转换** - 渠道可选择 `Responses → Chat`（默认）或 `Chat → Responses`
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

编辑 `.env` 文件。现在有两种初始化方式：

- 只配置管理员账号，启动后登录后台手工创建渠道
- 额外填写 `ENABLE_BOOTSTRAP_CHANNEL=true`、`RESPONSE_API_BASE` 和 `RESPONSE_API_KEY`，让系统启动时自动创建一个引导渠道

```env
# 管理员账号，数据库第一次初始化时写入
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123456

# 可选：启动时自动创建一个默认渠道，默认关闭
ENABLE_BOOTSTRAP_CHANNEL=false
RESPONSE_API_BASE=
RESPONSE_API_KEY=sk-your-upstream-key
BOOTSTRAP_CHANNEL_NAME=默认渠道

# SQLite 数据库文件
DATABASE_PATH=data/response2chat.db

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

### 4. 登录管理后台并创建渠道

服务启动后，打开：

```text
http://localhost:8000/admin/login
```

使用 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 登录。进入控制台后可配置：

- 渠道名称
- 上游基础 URL
- 上游 API Key
- 渠道描述
- 启停状态

保存后系统会为该渠道自动生成一个对外访问的 `access_key`。

### 5. 使用 Docker 部署

#### 使用 Docker 直接构建

```bash
# 构建镜像
docker build -t response2chat .

# 构建镜像（无缓存）
docker build --no-cache -t response2chat .

# 运行容器
docker run -d \
  --name response2chat \
  -p 8011:8000 \
  -v ./response2chat-data:/app/data \
  -e DEFAULT_TIMEOUT=300 \
  response2chat
```

如果你希望容器启动时自动创建引导渠道，再额外追加下面这些环境变量：

```bash
-e ENABLE_BOOTSTRAP_CHANNEL=true \
-e RESPONSE_API_BASE=https://your-response-api.com/v1 \
-e RESPONSE_API_KEY=sk-your-upstream-key
```

数据库默认写入 /app/data/response2chat.db。建议保留上面的卷挂载，这样即使容器删除后重新创建，渠道配置和管理员数据也不会丢失。

如果只是对同一个容器执行 docker restart，数据通常不会丢；更常见的数据丢失场景是删容器后重新 docker run，或者执行 docker-compose down -v。

#### 使用 Docker Compose（推荐）

```bash
# 先配置 .env 文件
cp .env.example .env
# 编辑 .env，至少设置管理员账号；如需引导渠道再设置 ENABLE_BOOTSTRAP_CHANNEL=true 和 RESPONSE_API_BASE

# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

Compose 已默认把 /app/data 挂到 named volume response2chat-data。若需要同时删除数据库数据，再显式执行 docker-compose down -v。

## 📖 API 使用

所有外部客户端都调用当前代理服务，`Authorization` 里放的是系统生成的渠道 `access_key`，不是上游厂商的真实 API Key。

### 转换类型

每个渠道都有独立的“转换类型”，在管理后台创建或编辑渠道时选择：

| 渠道类型 | 客户端调用代理 | 代理调用上游 | 默认值 |
| --- | --- | --- | --- |
| `Responses → Chat` | `/v1/chat/completions`（Chat 协议） | `/v1/responses`（Responses 协议） | 是 |
| `Chat → Responses` | `/v1/responses`（Responses 协议） | `/v1/chat/completions`（Chat 协议） | 否 |

已有渠道升级后会自动保持为 `Responses → Chat`，因此不会改变既有调用方式。对于 `Chat → Responses` 渠道，请使用 `/v1/responses`；如果误调用 `/v1/chat/completions`，代理会返回明确的 400 错误，避免静默按错误协议转发。

### Chat Completions

完全兼容 OpenAI Chat API 格式：

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer CHANNEL_ACCESS_KEY" \
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
  -H "Authorization: Bearer CHANNEL_ACCESS_KEY" \
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
  -H "Authorization: Bearer CHANNEL_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### Responses 直通透传（Responses → Chat 渠道）

当客户端直接请求 `/v1/responses` 时，代理不会再做 `chat/completions -> responses` 转换，而是将请求体、查询参数和上游响应按原样透传。

```bash
curl -X POST "http://localhost:8000/v1/responses" \
  -H "Authorization: Bearer CHANNEL_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "Hello!",
    "stream": false
  }'
```

如果请求里带 `"stream": true`，代理会直接把上游 SSE 响应流原样返回给客户端。

### Chat → Responses 调用示例

当渠道类型为 `Chat → Responses` 时，客户端按标准 Responses API 请求代理；代理会在上游使用 Chat Completions API，并将结果转换回标准 Responses 对象或 Responses SSE 事件。

```bash
curl -X POST "http://localhost:8000/v1/responses" \
  -H "Authorization: Bearer CHANNEL_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4",
    "instructions": "Answer concisely.",
    "input": [{
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this image."},
        {"type": "input_image", "image_url": "https://example.com/image.png"}
      ]
    }],
    "stream": false
  }'
```

`Chat → Responses` 的参数转换包括：`input` / `instructions`、`input_image`、函数工具定义与 `tool_choice`、`function_call` / `function_call_output` 对话续接、`max_output_tokens`、`reasoning.effort`、JSON 输出格式和 token usage。流式请求会输出 `response.created`、文本或函数参数 delta、`response.output_item.done`、`response.completed` 等标准 Responses SSE 事件。

### 健康检查

```bash
curl http://localhost:8000/health
```

### 模型列表

```bash
curl -H "Authorization: Bearer CHANNEL_ACCESS_KEY" http://localhost:8000/v1/models
```

## ⚙️ 配置说明


| 环境变量 | 必填 | 说明 | 默认值 |
| --- | --- | --- | --- |
| `ADMIN_USERNAME` | 否 | 默认管理员用户名，仅首次初始化数据库时写入 | `admin` |
| `ADMIN_PASSWORD` | 否 | 默认管理员密码，仅首次初始化数据库时写入 | `admin123456` |
| `DATABASE_PATH` | 否 | SQLite 数据库文件路径 | `data/response2chat.db` |
| `ENABLE_BOOTSTRAP_CHANNEL` | 否 | 是否在启动时自动创建引导渠道 | `false` |
| `RESPONSE_API_BASE` | 否 | 引导渠道的上游基础 URL；仅在 `ENABLE_BOOTSTRAP_CHANNEL=true` 时生效 | 空 |
| `RESPONSE_API_KEY` | 否 | 引导渠道的上游 API Key；仅在 `ENABLE_BOOTSTRAP_CHANNEL=true` 时生效 | 空 |
| `BOOTSTRAP_CHANNEL_NAME` | 否 | 引导渠道名称 | `默认渠道` |
| `ADMIN_SESSION_TTL_SECONDS` | 否 | 管理后台登录态有效期（秒） | `43200` |
| `ADMIN_SESSION_COOKIE_NAME` | 否 | 管理后台 Cookie 名称 | `response2chat_admin_session` |
| `ADMIN_COOKIE_SECURE` | 否 | 是否仅通过 HTTPS 下发管理后台 Cookie | `false` |
| `HOST` | 否 | 服务监听地址 | `0.0.0.0` |
| `PORT` | 否 | 服务监听端口 | `8000` |
| `HTTPX_LOG_LEVEL` | 否 | 上游 HTTP 客户端日志级别；高并发下建议保持 `WARNING` | `WARNING` |
| `DEFAULT_TIMEOUT` | 否 | 普通请求读取超时（秒） | `300` |
| `POOL_TIMEOUT` | 否 | 从连接池获取连接的超时（秒） | `10` |
| `STREAM_READ_TIMEOUT` | 否 | 流式读取超时（秒） | `120` |
| `STREAM_MAX_DURATION` | 否 | 单个流式请求最长持续时间，`0` 为不限制 | `0` |
| `MAX_CONNECTIONS` | 否 | HTTP 连接池最大连接数 | `100` |
| `MAX_KEEPALIVE_CONNECTIONS` | 否 | HTTP Keep-Alive 连接数 | `30` |
| `KEEPALIVE_EXPIRY` | 否 | Keep-Alive 连接保留时间（秒） | `60` |
| `DEFAULT_INSTRUCTIONS` | 否 | 默认系统提示词 | 空 |
| `FORCE_DEFAULT_INSTRUCTIONS` | 否 | 即使已有 system 消息也强制附加默认提示词 | `false` |

## 🖥️ 管理后台能力

- 登录入口：`/admin/login`
- 控制台首页：`/admin`
- 渠道管理：新增、编辑、启停、删除、轮换外部访问 Key
- 管理员密码：支持在控制台内修改

对外访问时，代理会按 `Authorization: Bearer CHANNEL_ACCESS_KEY` 查找渠道，再将请求转发到该渠道绑定的真实上游 URL，并自动附上该渠道的真实上游 API Key。


## 🔄 参数映射


| Chat API 参数             | Response API 映射     | 说明           |
| ----------------------- | ------------------- | ------------ |
| `model`                 | `model`             | 模型 ID        |
| `messages`              | `input`             | 对话消息列表       |
| `max_tokens`            | `max_output_tokens` | 最大生成 Token 数 |
| `max_completion_tokens` | `max_output_tokens` | 最大补全 Token 数 |
| `tools`                 | `tools`             | 工具定义         |
| `tool_choice`           | `tool_choice`       | 工具选择         |
| `reasoning_effort`      | `reasoning.effort`  | 推理强度         |
| `response_format`       | `text.format`       | 响应格式         |


> 注意：`system` 角色会自动转换为 `developer` 角色（Response API 规范）

### Chat → Responses 参数映射

| Responses API 参数 / 项目 | Chat Completions 映射 | 说明 |
| --- | --- | --- |
| `input` | `messages` | 文本消息、`input_text` 与 `input_image` 会转换；`instructions` 作为首个 `developer` 消息 |
| `max_output_tokens` | `max_completion_tokens` | 最大输出 Token 数 |
| `tools` | `tools[].function` | Responses 的扁平 function 定义转换为 Chat 嵌套定义 |
| `tool_choice` | `tool_choice` | 指定 function 时自动调整结构 |
| `function_call` | assistant `tool_calls` | 用于带工具历史的后续请求 |
| `function_call_output` | `tool` message | 使用 `call_id` 关联函数结果 |
| `reasoning.effort` | `reasoning_effort` | 推理强度 |
| `text.format` | `response_format` | 支持 `json_object` 和 `json_schema` |

返回时，Chat 的普通 assistant 消息会转成 Responses `message` / `output_text`，Chat function calls 会转成 Responses `function_call`；`prompt_tokens` / `completion_tokens` 同时映射为 `input_tokens` / `output_tokens`。

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
