#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:10003}"

curl -s "${SERVICE_URL}/health"
echo

curl -s "${SERVICE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "请用一句话介绍苏州。"}
    ],
    "temperature": 0.2,
    "max_tokens": 64
  }'
echo
