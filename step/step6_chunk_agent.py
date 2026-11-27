import base64
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from claude_model import ClaudeModel
from common import FoundSegment, format_timestamp


AGENT_STEP_LIMIT = 60

# 프롬프트에 시각을 소수 첫째 자리까지만 보여 주므로, 그만큼의 반올림 오차는 허용한다.
BOUNDARY_TOLERANCE_SECONDS = 0.1

# Ollama 기본 컨텍스트는 작아서, 받아쓰기가 길면 앞부분이 소리 없이 잘려 나간다.
OLLAMA_CONTEXT_LENGTH = 32768


SYSTEM_PROMPT = """\
당신은 긴 영상의 한 조각을 맡은 영상 편집 보조입니다.
사용자가 원하는 장면이 당신이 맡은 조각 안에서 몇 초부터 몇 초까지인지 찾아야 합니다.

받는 자료
- 영상 배경: 사용자가 적은 영상 설명과 고유명사 사전. 없을 수도 있습니다.
  어떤 영상인지, 누가 나오는지, 무슨 상황인지를 여기서 먼저 파악하고, 받아쓰기와 프레임을 이 배경에 맞춰 해석하세요.
  다만 구간을 고르는 근거는 받아쓰기와 프레임입니다. 배경에 적힌 내용만으로 구간을 정하지 마세요.
- 조각 전체를 일정 길이의 '윈도우'로 나눈 목록과, 윈도우별 음성 받아쓰기 결과.
- 받아쓰기 앞에 "(노래 구간)" 또는 "(음악·BGM 구간)"이 붙은 윈도우는 노래나 음악이 나오는 부분입니다.
  가사는 받아쓰지 않았으므로, 노래·음악과 관련된 요청이면 이 표시와 프레임으로 판단하세요.
- 윈도우마다 대표 프레임이 한 장씩 있습니다. look_at_frames 도구로 프레임 범위와 질문을 보내면
  프레임을 본 결과를 글로 돌려받습니다.

일하는 방법
1. 받아쓰기를 먼저 읽고 후보가 될 만한 윈도우를 고릅니다.
2. 화면이 중요한 요청이거나 받아쓰기만으로 확실하지 않으면 look_at_frames 로 확인합니다.
   질문은 요청과 관련해 무엇을 확인하고 싶은지 구체적으로 적습니다.
   후보 윈도우가 한 번에 볼 수 있는 프레임 수보다 많으면, 범위를 나눠 여러 번 불러서 후보를 빠짐없이 확인합니다.
3. 요청에 맞는 윈도우가 이어져 있으면 한 구간으로 묶어서 제출합니다.
   첫 윈도우의 시작부터 마지막 윈도우의 끝까지를 submit_segment 한 번으로 내고, 윈도우마다 따로 내지 않습니다.
   떨어져 있는 장면은 각각 따로 제출합니다.
   시각은 영상 전체 기준 초 단위이며, 반드시 맡은 조각의 범위 안이어야 합니다.
   제출은 submit_segment 도구를 불러야만 됩니다. 답변 글에 구간을 적기만 하면 제출되지 않습니다.
4. 맞는 구간이 없으면 아무것도 제출하지 않고 "해당 없음"이라고 답합니다.
5. 다 끝나면 무엇을 찾았는지 한두 문장으로 정리하고 마칩니다.

판단은 실제로 읽은 받아쓰기와 look_at_frames 로 돌려받은 답에만 근거합니다.
확인하지 않은 윈도우의 화면 내용을 짐작해서 말하거나, 그 짐작으로 윈도우를 빼지 마세요.
누가 무슨 말을 했는지에 관한 요청은 받아쓰기로 판단합니다. 노래 구간이나 음성이 없는 윈도우를
화면(표정, 분위기)만 보고 그 말을 한 장면이라고 짐작하지 마세요.

구간이 조각의 맨 앞이나 맨 끝에 걸쳐 있으면 조각 경계까지만 제출하세요.
이웃 조각을 맡은 에이전트의 결과와 나중에 자동으로 합쳐집니다.
"""


def create_model(model_name, ollama_url):
    """모델 이름이 claude- 로 시작하면 Claude API, 아니면 Ollama 모델을 만든다."""
    if model_name.startswith("claude-"):
        return ClaudeModel(model_name)
    return ChatOllama(model=model_name, base_url=ollama_url, num_ctx=OLLAMA_CONTEXT_LENGTH)


def describe_background(video_description, glossary_path):
    """사용자가 적은 영상 설명과 고유명사 사전을 에이전트와 비전 모델에게 보여 줄 글로 묶는다. 둘 다 없으면 빈 글."""
    parts = []
    if video_description != "":
        parts.append(f"영상 설명 (사용자가 적은 것): {video_description}")

    glossary_text = describe_glossary(glossary_path)
    if glossary_text != "":
        parts.append(glossary_text)

    return "\n".join(parts)


def describe_glossary(glossary_path):
    """
    고유명사 사전(glossary.json)을 글로 바꾼다. 파일이 없으면 빈 글.

        {
          "토아": ["토하"],                     정확한 표기: [받아쓰기에서 잘못 나오는 형태, ...]
          "캔시": [],                           잘못 나오는 형태를 모르면 비워 둔다
          "소나": {"variants": ["소나기"],      누구인지 적어 두면 에이전트가 역할을 짐작하지 않는다
                   "note": "게스트로 나온 가수"}
        }

    받아쓰기를 일괄 치환하지 않는다 ("토하다" 같은 일반 단어가 깨진다). 에이전트가 문맥을 보고 고쳐 읽게 한다.
    """
    if not Path(glossary_path).exists():
        return ""
    glossary = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
    if len(glossary) == 0:
        return ""

    lines = [
        "고유명사 사전",
        "- 아래 이름·용어는 이 영상에 나올 수 있습니다. 받아쓰기에 비슷한 소리의 단어가 나오면 문맥상 맞는 경우에만 "
        "이 표기로 고쳐 읽으세요. 일반 단어로 쓰인 경우(예: '토하다')는 고치지 마세요.",
        "- 설명(—)이 없는 이름은 누구인지 짐작하지 마세요. 특히 영상 속 주인공 본인이라고 단정하지 말고, "
        "받아쓰기에 나온 대로 이름만 쓰세요.",
    ]
    for term, entry in glossary.items():
        note = ""
        variants = entry
        if isinstance(entry, dict):
            note = entry.get("note", "")
            variants = entry.get("variants", [])

        line = f"- {term}"
        if note != "":
            line += f" — {note}"
        if len(variants) > 0:
            line += f" (받아쓰기에서 {', '.join(repr(variant) for variant in variants)}(으)로 잘못 나올 수 있음)"
        lines.append(line)

    return "\n".join(lines)


def build_briefing(chunk, chunk_count, user_request, frames_per_look, background_text):
    lines = [f"사용자 요청: {user_request}", ""]
    if background_text != "":
        lines += ["영상 배경:", background_text, ""]

    lines += [
        f"당신이 맡은 조각: {chunk.number}번 / 전체 {chunk_count}개",
        f"조각 범위: {chunk.start:.1f}s ~ {chunk.end:.1f}s "
        f"({format_timestamp(chunk.start)} ~ {format_timestamp(chunk.end)})",
        f"윈도우 수: {len(chunk.windows)}개 (윈도우 번호 1 ~ {len(chunk.windows)})",
        f"look_at_frames 로 한 번에 볼 수 있는 프레임 수: 최대 {frames_per_look}장",
    ]
    lines += ["", "윈도우별 받아쓰기:"]

    for window in chunk.windows:
        transcript = window.transcript
        if transcript == "":
            transcript = "(음성 없음)"
        time_range = f"{format_timestamp(window.start)} ~ {format_timestamp(window.end)}"
        lines.append(f"[윈도우 {window.index}] {time_range} ({window.start:.1f}s ~ {window.end:.1f}s) | {transcript}")

    return "\n".join(lines)


async def ask_vision_model(vision_model, windows, question, background_text):
    # Ollama 는 도구 결과 속 이미지를 모델에게 제대로 넘겨준다는 보장이 없어서,
    # 에이전트가 프레임을 직접 보는 대신 비전 모델에게 물어보고 글로 된 답을 받는다.
    # 비전 모델은 소리를 못 듣는다. 화면만 보고 "말하는 장면"인지 "노래하는 장면"인지 짐작하지 않도록
    # 그 시점의 받아쓰기(노래·음악 표시 포함)를 프레임마다 같이 알려 준다.
    instruction = ["아래 이미지들은 영상의 여러 시점에서 뽑은 프레임입니다."]
    for order, window in enumerate(windows, start=1):
        sound = window.transcript if window.transcript != "" else "(음성 없음)"
        instruction.append(
            f"{order}번째 이미지 = 윈도우 {window.index} ({window.start:.1f}s ~ {window.end:.1f}s) | 그때 들린 소리: {sound}"
        )
    instruction.append("")
    instruction.append(
        "'그때 들린 소리'는 그 프레임 앞뒤의 음성을 받아쓴 것입니다. "
        "\"(노래 구간)\"과 \"(음악·BGM 구간)\"은 말이 아니라 노래나 음악이 나오는 부분이라는 표시입니다."
    )
    if background_text != "":
        instruction.append("")
        instruction.append("영상 배경 (누가 누구인지, 어떤 상황인지 알아보는 데 참고하세요):")
        instruction.append(background_text)
    instruction.append("")
    instruction.append(f"질문: {question}")
    instruction.append(
        "이미지마다 몇 번째 이미지인지 밝히고 짧게 답하세요. 화면에 보이는 것과 소리로 알 수 있는 것을 구분해서 쓰고, "
        "노래 구간의 화면 자막(가사)을 사람이 한 말로 보지 마세요."
    )

    prompt = "\n".join(instruction)
    images = [base64.b64encode(Path(window.frame_path).read_bytes()).decode("ascii") for window in windows]

    if isinstance(vision_model, ClaudeModel):
        return await vision_model.ask_with_images(prompt, images)

    content = [{"type": "text", "text": prompt}]
    for encoded in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

    response = await vision_model.ainvoke([HumanMessage(content=content)])
    return response.content


def make_tool_functions(chunk, found_segments, vision_model, frames_per_look, background_text):
    # 도구 인자의 타입 힌트와 docstring 은 LangChain 과 Claude SDK 가 도구 스키마를 만드는 데 쓴다.

    async def look_at_frames(first_window: int, last_window: int, question: str) -> str:
        """first_window 번부터 last_window 번 윈도우까지(양 끝 포함)의 대표 프레임을 보고 question 에 답한다.
        한 번에 볼 수 있는 프레임 수는 처음 메시지에 적혀 있다.

        Args:
            first_window: 볼 첫 윈도우 번호 (1부터)
            last_window: 볼 마지막 윈도우 번호 (포함)
            question: 프레임을 보고 확인하고 싶은 것
        """
        if first_window < 1:
            return "윈도우 번호는 1부터 시작합니다."
        if last_window > len(chunk.windows):
            return f"이 조각의 마지막 윈도우는 {len(chunk.windows)}번입니다."
        if first_window > last_window:
            return "first_window 가 last_window 보다 클 수 없습니다."
        if last_window - first_window + 1 > frames_per_look:
            return f"한 번에 {frames_per_look}장까지만 볼 수 있습니다. 범위를 줄여 주세요."

        windows = chunk.windows[first_window - 1:last_window]
        return await ask_vision_model(vision_model, windows, question, background_text)

    async def submit_segment(start_seconds: float, end_seconds: float, reason: str) -> str:
        """요청에 해당하는 구간 하나를 제출한다. 시각은 영상 전체 기준 초 단위.
        reason 에는 이 구간이 왜 요청에 맞는지 짧게 적는다.

        Args:
            start_seconds: 구간 시작 (영상 전체 기준 초)
            end_seconds: 구간 끝 (영상 전체 기준 초)
            reason: 이 구간이 요청에 맞는 이유
        """
        # 에이전트에게는 시각을 0.1초 단위로 반올림해 보여 준다. 보여 준 조각 경계(예: 실제 105.633 → 105.6)를
        # 그대로 내면 받아들이도록, 반올림 오차만큼 벗어난 값은 조각 경계에 맞춘다.
        if chunk.start - BOUNDARY_TOLERANCE_SECONDS <= start_seconds < chunk.start:
            start_seconds = chunk.start
        if chunk.end < end_seconds <= chunk.end + BOUNDARY_TOLERANCE_SECONDS:
            end_seconds = chunk.end

        if start_seconds < chunk.start:
            return f"시작 시각은 조각 시작({chunk.start:.1f}s)보다 앞설 수 없습니다."
        if end_seconds > chunk.end:
            return f"끝 시각은 조각 끝({chunk.end:.1f}s)을 넘을 수 없습니다."
        if start_seconds >= end_seconds:
            return "끝 시각은 시작 시각보다 뒤여야 합니다."

        found_segments.append(FoundSegment(chunk.number, start_seconds, end_seconds, reason))
        return "제출했습니다."

    return [look_at_frames, submit_segment]


async def run_chunk_agent(agent_model, vision_model, chunk, chunk_count, user_request, frames_per_look,
                          background_text):
    found_segments = []
    tool_functions = make_tool_functions(chunk, found_segments, vision_model, frames_per_look, background_text)
    briefing = build_briefing(chunk, chunk_count, user_request, frames_per_look, background_text)

    if isinstance(agent_model, ClaudeModel):
        await agent_model.run_agent(SYSTEM_PROMPT, briefing, tool_functions, AGENT_STEP_LIMIT)
        return found_segments

    tools = [tool(function) for function in tool_functions]
    agent = create_agent(agent_model, tools=tools, system_prompt=SYSTEM_PROMPT)
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": briefing}]},
        config={"recursion_limit": AGENT_STEP_LIMIT},
    )
    return found_segments
