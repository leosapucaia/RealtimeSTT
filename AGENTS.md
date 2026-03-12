# AGENTS.md

## Cursor Cloud specific instructions

### Overview

RealtimeSTT is a Python speech-to-text library (v0.3.104) that provides real-time audio transcription using Whisper models, with voice activity detection and wake word support. There is no database, no frontend framework, and no automated test suite (pytest, etc.).

### System dependencies

The following must be installed before `pip install`:
- `portaudio19-dev` (required for PyAudio to compile)
- `python3-dev` (required for Python.h header during PyAudio build)
- `python3.12-venv` (required to create virtual environments)
- `ffmpeg` (optional, usually pre-installed)

### Virtual environment

The project uses a venv at `/workspace/.venv`. Always activate before running:
```bash
source /workspace/.venv/bin/activate
```

### Running without a microphone

The cloud VM has **no audio input device**. To test the core library, use `use_microphone=False` and feed audio via `recorder.feed_audio(chunk)`. The file `RealtimeSTT/warmup_audio.wav` (16kHz mono, ~1.6s) is available for quick transcription tests. Example:

```python
from RealtimeSTT import AudioToTextRecorder
recorder = AudioToTextRecorder(use_microphone=False, spinner=False, model="tiny.en", device="cpu", compute_type="int8")
# Feed 16-bit PCM bytes via recorder.feed_audio(chunk), then call recorder.text()
```

### CLI entry points

After `pip install -e .`:
- `stt-server` — starts the WebSocket STT server (ports 8011/8012 by default)
- `stt` — CLI client that connects to the server

### Key gotchas

- `faster-whisper` v1.1.1 requires `requests` as a transitive dependency but does not declare it; the update script installs it explicitly.
- First run downloads Silero VAD model from GitHub and Whisper model from HuggingFace. This requires internet access.
- No GPU in the cloud VM — always use `device="cpu"` and `compute_type="int8"` for testing.
- There is no configured linter (flake8, ruff, mypy, etc.) or test runner (pytest). Syntax checking can be done with `python3 -m py_compile <file>`.
- See `README.md` for full API documentation and configuration parameters.
