原生Response格式


curl -X POST "https://docs.newapi.pro/v1/responses" \
  -H "Authorization: Bearer " \
  -H "Content-Type: application/json" \
  -d '{
    "model": "string"
  }'

返回
{
  "id": "string",
  "object": "response",
  "created_at": 0,
  "status": "completed",
  "model": "string",
  "output": [
    {
      "type": "string",
      "id": "string",
      "status": "string",
      "role": "string",
      "content": [
        {
          "type": "string",
          "text": "string"
        }
      ]
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
  }
}


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
input?
string
|
array<object>
输入内容，可以是字符串或消息数组

instructions?
string
max_output_tokens?
integer
temperature?
number
top_p?
number
stream?
boolean
tools?
array<object>
tool_choice?
string
|
object
reasoning?
object
previous_response_id?
string
truncation?
string
Value in"auto" | "disabled"