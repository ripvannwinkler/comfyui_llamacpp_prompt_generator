# ComfyUI Llama.cpp Prompt Enhancer

A ComfyUI custom node that sends a text prompt to a local
[llama.cpp](https://github.com/ggml-org/llama.cpp) server running in
**router mode** (multiple models managed via `models.ini`) and returns an
LLM-enhanced version, suitable for feeding into an image-generation prompt.

## Requirements

- A llama.cpp server running in router mode, exposing the OpenAI-compatible
  API (`/v1/chat/completions`, `/v1/models`).
- Python `requests` (see `requirements.txt`).

## Install

1. Clone/copy this folder into your ComfyUI `custom_nodes/` directory.
2. Create a venv and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\pip install -r requirements.txt
   ```
   (Or install `requirements.txt` into whatever Python environment ComfyUI
   itself uses.)
3. Restart ComfyUI (or refresh node definitions from the ComfyUI menu).

## Node: "Llama.cpp Prompt Enhancer"

**Inputs**

| Name | Type | Notes |
|---|---|---|
| `prompt` | STRING (multiline) | The prompt to enhance. Required, non-empty. |
| `model` | COMBO | Populated from `GET {base_url}/v1/models` when ComfyUI loads node defs. |
| `base_url` | STRING | Default `http://127.0.0.1:8080`. |
| `temperature` | FLOAT | Default `0.7`. |
| `max_tokens` | INT | Default `512`. |
| `seed` | INT | Default `0` means "pick a fresh random seed every run" (handled internally, since ComfyUI's own control-after-generate widget isn't reliable across all frontend builds). Set a nonzero value for reproducible output. The node always re-executes rather than reusing a cached result. |
| `disable_thinking` | BOOLEAN | Default `True`. Sends `chat_template_kwargs: {"enable_thinking": false}` to suppress reasoning-model "thinking" output. |
| `extra_system_prompt` | STRING (multiline, optional) | Appended after the built-in system prompt; never replaces it. |

**Output**

| Name | Type |
|---|---|
| `enhanced_prompt` | STRING |

## Known limitations

- **Model dropdown timing**: the `model` COMBO is populated against the
  default `base_url` when ComfyUI loads the node's Python module (server
  startup or node-def refresh) — *before* you can edit the `base_url`
  widget. If your router isn't at `http://127.0.0.1:8080`, the dropdown will
  show a placeholder (`"(server unreachable - enter base_url and refresh)"`)
  until you fix `base_url` and refresh node definitions (ComfyUI menu >
  Refresh).
- **No preload/unload orchestration**: the node relies on the router's
  JIT-load behavior for the selected model. If you need explicit
  preload/unload control (e.g. for scripted multi-model pipelines), see this
  project's `llamacpp` skill instead.
- **No streaming**: the node blocks until the full completion is returned.
- **Thinking models**: if a model still returns empty `content` with
  `finish_reason: "length"` and a populated `reasoning_content` field even
  with `disable_thinking` enabled, its chat template may not honor the
  `enable_thinking` jinja variable — try increasing `max_tokens` instead.

## Out of scope (v1)

- Config file for `base_url` (it's a per-workflow node widget instead).
- Streaming output.
- Preload/unload orchestration.
