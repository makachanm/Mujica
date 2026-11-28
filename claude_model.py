"""
Claude API 로 서브에이전트와 비전 모델을 돌린다.

옵션의 에이전트 모델이나 비전 모델 이름이 claude- 로 시작하면 Ollama 대신 이쪽을 쓴다.
API 키는 옵션(options.json)에 적은 값을 쓰고, 비어 있으면 ANTHROPIC_API_KEY 환경 변수(.env)를 쓴다.
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


def has_api_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


class ClaudeModel:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        if api_key != "":
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            return

        if not has_api_key():
            raise RuntimeError(f"{model_name} 를 쓰려면 옵션의 Claude API 키 칸을 채워 주세요.")
        self.client = anthropic.AsyncAnthropic()

    def request_options(self, effort):
        options = {"model": self.model_name, "output_config": {"effort": effort}}
        if self.model_name in FALLBACK_MODELS:
            options["betas"] = [FALLBACK_BETA]
            options["fallbacks"] = "default"
        return options

    async def ask_with_images(self, prompt, jpeg_images_base64):
        """이미지들과 질문을 보내고 글로 된 답을 돌려받는다. 이미지는 질문에 적힌 순서대로 붙는다."""
        content = []
        for image in jpeg_images_base64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}})
        content.append({"type": "text", "text": prompt})

        response = await self.client.beta.messages.create(
            max_tokens=VISION_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
            **self.request_options(VISION_EFFORT),
        )
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
            if message.stop_reason == "refusal":
                break
