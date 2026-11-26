#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo ".venv 가 없습니다. 먼저 ./setup.sh 를 실행하세요." >&2
    exit 1
fi

# Claude API 키 같은 비밀 값은 .env 에 둔다 (git 에 올라가지 않는다). 예: ANTHROPIC_API_KEY=sk-ant-...
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

exec .venv/bin/python ui.py "$@"
