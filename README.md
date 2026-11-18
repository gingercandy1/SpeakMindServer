# SpeakMind server (Rust)

Pronunciation assessment server: wav2vec2 phoneme CTC + forced alignment + GOP.
Ported from the original Python (FastAPI + PyTorch) server, with the same HTTP
API and the same scores.

## Requirements

- A recent stable Rust (tested with 1.96).
- `ffmpeg` on `PATH` (decodes the uploaded recordings).
- The espeak-ng shared library and its data. On Windows they are expected in
  `espeak/` (`espeak-ng.dll` and `espeak-ng-data/`); these local runtime files
  are not kept in version control. Elsewhere the system `libespeak-ng` is
  used. Override with `SPEAKMIND_ESPEAK_LIB` and `SPEAKMIND_ESPEAK_DATA`.
- The model exported to ONNX in `models/` (about 1.2 GB, not kept in version
  control). Create it once; this needs [uv](https://docs.astral.sh/uv/), which
  installs the Python packages into a throwaway environment:

  ```bash
  uv run --no-project --python 3.12 --with torch==2.15.0 --with transformers==5.22.2 --with onnx python tools/export_onnx.py models
  ```
- Optional, for the natural example voice: [Piper](https://github.com/rhasspy/piper)
  voices in `tts/` (about 63 MB each, not kept in version control). Without
  them the app falls back to the phone's own speech engine.

  ```bash
  mkdir -p tts && cd tts
  B=https://huggingface.co/rhasspy/piper-voices/resolve/main
  for f in en/en_US/lessac/medium/en_US-lessac-medium es/es_ES/davefx/medium/es_ES-davefx-medium; do
    curl -LO $B/$f.onnx && curl -LO $B/$f.onnx.json
  done
  ```
- Optional, for AI chat: any OpenAI-compatible chat service. See "AI chat" below.

## Languages

English (`en`, the default) and Spanish (`es`). The scoring model is
multilingual, so a language only needs an espeak voice for the reference
phonemes and, for speech, a Piper voice. Both are listed in `LANGUAGES` in
`src/config.rs`; add a row there to add a language.

## Run

```bash
cargo run --release
```

The server reads `models/` and `espeak/` relative to the working directory and
listens on `0.0.0.0:8000`.

## API

- `GET /health` returns `{"status": "ok", "model": "...", "languages": ["en", "es"],
  "tts": ["en", "es"], "chat": false}`: the languages scored, the languages
  with a voice, and whether AI chat is set up.
- `POST /assess` takes a multipart form with `audio` (a recording in any format
  ffmpeg can read), `text` (the sentence the user was asked to read) and
  optionally `language` (`en` or `es`, default `en`), and returns the scores.
  Errors are `{"detail": "message"}` with status 413, 422 or 500.
- `POST /tts` takes JSON `{"text": "...", "language": "en"}` and returns a
  16-bit mono WAV (`audio/wav`). 503 when there is no voice for the language.
- `POST /chat` takes JSON `{"messages": [{"role": "user", "content": "..."}],
  "language": "en"}` (roles `user` and `assistant`, the last one from the user)
  and returns `{"reply": "...", "tip": "..." | null, "task_done": false}`. `reply` is what the
  partner says; `tip` is a short correction in Chinese when the learner made a
  clear mistake. 503 when chat is not set up, 502 when the chat service fails.
  A level may also send `npc_prompt`, `level_id` and `task`: the persona and
  the task are merged into the system prompt, and when a task is given the
  model signals completion with a `[TASK_DONE]` marker that the server strips,
  reporting it as `task_done: true` instead.

## AI chat

The server calls an OpenAI-compatible `/chat/completions` endpoint, so the key
never reaches the app. Set `SPEAKMIND_AI_API_KEY` to use Gemini (the default
model), or also set `SPEAKMIND_AI_BASE_URL` for another service (DeepSeek, Qwen,
a local Ollama at `http://localhost:11434/v1`, ...). A local server needs no key.

```powershell
$env:SPEAKMIND_AI_API_KEY = "sk-..."
$env:SPEAKMIND_AI_MODEL = "gemini-3.5-flash-lite"   # optional
cargo run --release
```

## Settings

Environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `SPEAKMIND_HOST` | `0.0.0.0` | Address to listen on |
| `SPEAKMIND_PORT` | `8000` | Port to listen on |
| `SPEAKMIND_MODEL_DIR` | `models` | Directory with `model.onnx`, `vocab.json`, `config.json` |
| `SPEAKMIND_MODEL` | `facebook/wav2vec2-lv-60-espeak-cv-ft` | Model name reported by `/health` |
| `SPEAKMIND_TTS_DIR` | `tts` | Directory with the Piper voices |
| `SPEAKMIND_AI_API_KEY` | none | Key for the chat service; turns chat on |
| `SPEAKMIND_AI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` when a key is set | Chat service address; setting it also turns chat on |
| `SPEAKMIND_AI_MODEL` | `gemini-3.5-flash-lite` | Chat model name |
| `SPEAKMIND_MAX_TTS_CHARS` | `600` | Longest text accepted for speech |
| `SPEAKMIND_ESPEAK_LIB` | see above | Path of the espeak-ng shared library |
| `SPEAKMIND_ESPEAK_DATA` | see above | Path of `espeak-ng-data` |
| `SPEAKMIND_THREADS` | CPU count, at most 8 | Inference threads |
| `SPEAKMIND_MAX_AUDIO_SECONDS` | `30` | Longest accepted recording |
| `SPEAKMIND_MAX_UPLOAD_BYTES` | `10485760` | Largest accepted upload |
| `SPEAKMIND_MAX_TEXT_CHARS` | `300` | Longest accepted text |
| `SPEAKMIND_GOOD_THRESHOLD` | `0.7` | Score at or above this is "good" |
| `SPEAKMIND_FAIR_THRESHOLD` | `0.4` | Score at or above this is "fair" |
| `SPEAKMIND_PAUSE_SECONDS` | `0.5` | Gap between words that counts as a pause |

## Differences from the original Python server

- The model is loaded from a local ONNX file instead of being downloaded from
  Hugging Face at start-up.
- A request with a missing `audio` or `text` field gets a plain-string `detail`
  (`"audio is required"`), not FastAPI's list of validation errors.
- Inference runs one request at a time; requests arriving meanwhile wait.

## Test

```bash
cargo test
```
