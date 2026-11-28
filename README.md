# Mujica

Automated cut-editing agent for voice-intensive content. <br />
Agents read the transcript and frames of a video and return the parts that match your request as candidate clips.

## Requirements

- [uv](https://docs.astral.sh/uv/) and ffmpeg
- An [Ollama](https://ollama.com) server with a tool-calling model and a vision model
- An OpenAI-compatible Whisper server, such as [speaches](https://github.com/speaches-ai/speaches)

## Usage

```bash
./setup.sh   # creates .venv and installs dependencies
./run.sh     # opens the UI at http://127.0.0.1:7860
```

To use the Claude API, enter a model name starting with `claude-` and put `ANTHROPIC_API_KEY=...` in `.env`.
See [docs/qwen-vs-claude.md](docs/qwen-vs-claude.md) for a comparison.
