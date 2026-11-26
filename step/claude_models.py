"""
Claude API 로 서브에이전트와 비전 모델을 돌린다.

옵션의 에이전트 모델이나 비전 모델 이름이 claude- 로 시작하면 Ollama 대신 이쪽을 쓴다.
API 키는 ANTHROPIC_API_KEY 환경 변수에서 읽는다 (run.sh 가 .env 를 읽어 넣어 준다).
"""

import os
from pathlib import Path

import anthropic
from anthropic import beta_async_tool


# 에이전트 루프는 도구 호출이 많아 medium, 프레임 묘사는 단순해서 low 로 둔다.
AGENT_EFFORT = "medium"
VISION_EFFORT = "low"
AGENT_MAX_TOKENS = 16000
VISION_MAX_TOKENS = 4000

# 안전 분류기가 요청을 거절하면 서버가 다른 모델로 이어서 답하게 하는 기능 (이 모델들만 지원한다)
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def is_claude_model(model_name):
    return model_name.strip().startswith("claude-")


class ClaudeModel:
    def __init__(self, model_name):
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists()):
            raise RuntimeError(
                f"{model_name} 를 쓰려면 Claude API 키가 필요합니다. "
                "Mujica 폴더의 .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 를 적고 run.sh 로 다시 실행하세요."
            )
        self.model_name = model_name.strip()
        self.client = anthropic.AsyncAnthropic()
        # 이 모델로 쓴 토큰 합계 (비용 확인용)
        self.usage = {"requests": 0, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0}

    def count_usage(self, usage):
        self.usage["requests"] += 1
        self.usage["input"] += usage.input_tokens or 0
        self.usage["output"] += usage.output_tokens or 0
        self.usage["cache_write"] += usage.cache_creation_input_tokens or 0
        self.usage["cache_read"] += usage.cache_read_input_tokens or 0

    def request_options(self, effort):
        options = {"model": self.model_name, "output_config": {"effort": effort}}
        if self.model_name in FALLBACK_MODELS:
            options["betas"] = [FALLBACK_BETA]
            options["fallbacks"] = "default"
        return options

    async def ask_with_images(self, prompt, jpeg_images_base64):
        """이미지들과 질문을 보내고 글로 된 답을 돌려받는다. 이미지는 질문에 적힌 순서대로 붙는다."""
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}}
            for image in jpeg_images_base64
        ]
        content.append({"type": "text", "text": prompt})

        response = await self.client.beta.messages.create(
            max_tokens=VISION_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
            **self.request_options(VISION_EFFORT),
        )
        self.count_usage(response.usage)
        if response.stop_reason == "refusal":
            return "비전 모델이 이 프레임에 대한 답을 거절했습니다."
        return "\n".join(block.text for block in response.content if block.type == "text")

    async def run_agent(self, system_prompt, user_message, tool_functions, max_iterations):
        """tool_functions(async 함수들)를 도구로 주고, Claude 가 도구 호출을 멈출 때까지 돌린다."""
        tools = [beta_async_tool(function, strict=True) for function in tool_functions]
        runner = self.client.beta.messages.tool_runner(
            max_tokens=AGENT_MAX_TOKENS,
            system=system_prompt,
            tools=tools,
            messages=[{"role": "user", "content": user_message}],
            max_iterations=max_iterations,
            # 시스템 프롬프트·도구·앞선 대화가 매 요청 그대로 반복되므로 캐시해서 비용을 줄인다.
            cache_control={"type": "ephemeral"},
            **self.request_options(AGENT_EFFORT),
        )
        async for message in runner:
            self.count_usage(message.usage)
            if message.stop_reason == "refusal":
                break
