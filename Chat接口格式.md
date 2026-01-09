请求输入
curl -X POST "https://docs.newapi.pro/v1/chat/completions" \
  -H "Authorization: Bearer " \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {
        "role": "system",
        "content": "string"
      }
    ]
  }'

响应：200
{
  "id": "string",
  "object": "chat.completion",
  "created": 0,
  "model": "string",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "system",
        "content": "string",
        "name": "string",
        "tool_calls": [
          {
            "id": "string",
            "type": "function",
            "function": {
              "name": "string",
              "arguments": "string"
            }
          }
        ],
        "tool_call_id": "string",
        "reasoning_content": "string"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "text_tokens": 0,
      "audio_tokens": 0,
      "image_tokens": 0
    },
    "completion_tokens_details": {
      "text_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0
    }
  },
  "system_fingerprint": "string"
}

响应400
{
  "error": {
    "message": "string",
    "type": "string",
    "param": "string",
    "code": "string"
  }
}

响应429
{
  "error": {
    "message": "string",
    "type": "string",
    "param": "string",
    "code": "string"
  }
}

接口参数
Authorization
BearerAuth

Authorization
Bearer <token>
使用 Bearer Token 认证。 格式: Authorization: Bearer sk-xxxxxx

In: header

Request Body
application/json

model*
string
模型 ID

messages*
array<object>
对话消息列表

temperature?
number
采样温度

Default1
Range0 <= value <= 2
top_p?
number
核采样参数

Default1
Range0 <= value <= 1
n?
integer
生成数量

Default1
Range1 <= value
stream?
boolean
是否流式响应

Defaultfalse
stream_options?
object
stop?
string
|
array<string>
停止序列

max_tokens?
integer
最大生成 Token 数

max_completion_tokens?
integer
最大补全 Token 数

presence_penalty?
number
Default0
Range-2 <= value <= 2
frequency_penalty?
number
Default0
Range-2 <= value <= 2
logit_bias?
object
user?
string
tools?
array<object>
tool_choice?
string
|
object
response_format?
object
seed?
integer
reasoning_effort?
string
推理强度 (用于支持推理的模型)

Value in"low" | "medium" | "high"
modalities?
array<string>
audio?
object