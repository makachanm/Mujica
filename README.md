# Mujica

Automated cut-editing agent for voice-intensive content. <br />
Agents transcribes audio from frame and uses information into context of agent. then return the matching parts as candidate clips.

## Requirements

- [uv](https://docs.astral.sh/uv/) and ffmpeg
- An [Ollama](https://ollama.com) server with:
  - an agent model that supports tool calling (default `qwen3:8b`)
  - a vision model that supports image input (default `qwen2.5vl:7b`)
- An OpenAI-compatible Whisper server. [speaches](https://github.com/speaches-ai/speaches)

## Usage

```bash
./setup.sh   # creates .venv and installs dependencies
./run.sh     # opens the UI at http://127.0.0.1:7860
```

### Using the Claude API instead of Ollama

Enter a model name that starts with `claude-` (for example `claude-opus-5-5`) as the agent model, the vision model, or both. Each field is independent, so the agent can run on Claude while vision stays on Ollama.

Put the API key in a `.env` file in this folder; `run.sh` loads it and git ignores it:

```
ANTHROPIC_API_KEY=sk-ant-...
```

See [docs/qwen-vs-claude.md](docs/qwen-vs-claude.md) for a small comparison of results, speed and cost.
