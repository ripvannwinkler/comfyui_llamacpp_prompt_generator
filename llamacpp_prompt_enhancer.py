import re

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_MODELS = 3       # seconds, for populating the COMBO at node-def time
DEFAULT_TIMEOUT_CHAT = 120       # seconds, for the blocking generation call
FALLBACK_MODEL_CHOICE = "(server unreachable - enter base_url and refresh)"

DEFAULT_SYSTEM_PROMPT = (
    "You are a prompt-engineering assistant for text-to-image diffusion models "
    "(e.g. Stable Diffusion, SDXL, Flux). Given a short or rough user prompt, "
    "rewrite and expand it into a single, detailed, vivid image-generation prompt.\n\n"
    "Guidelines:\n"
    "- Preserve the user's core subject and intent; do not change the meaning.\n"
    "- Add concrete visual detail: subject description, setting, lighting, "
    "composition, camera/lens or art-medium cues, color palette, and mood.\n"
    "- Prefer concise, comma-separated descriptive phrases over full sentences.\n"
    "- Do not add negative prompts, explanations, headers, or markdown formatting.\n"
    "- Do not wrap the output in quotes.\n"
    "- Output ONLY the enhanced prompt text, nothing else."
)


def _router_root(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def _fetch_model_list(base_url: str, timeout: float = DEFAULT_TIMEOUT_MODELS) -> list:
    root = _router_root(base_url)
    try:
        resp = requests.get(f"{root}/v1/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        ids = [m["id"] for m in data.get("data", []) if "id" in m]
        return ids if ids else [FALLBACK_MODEL_CHOICE]
    except Exception:
        return [FALLBACK_MODEL_CHOICE]


def _build_system_prompt(extra_system_prompt: str) -> str:
    extra = (extra_system_prompt or "").strip()
    if not extra:
        return DEFAULT_SYSTEM_PROMPT
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{extra}"


class LlamaCppPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        models = _fetch_model_list(DEFAULT_BASE_URL)
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (models, {}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "disable_thinking": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extra_system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "generate"
    CATEGORY = "llamacpp"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    def generate(self, prompt, model, base_url, temperature, max_tokens,
                 seed, disable_thinking, extra_system_prompt=""):
        if not prompt or not prompt.strip():
            raise ValueError("LlamaCpp Prompt Enhancer: 'prompt' input is empty.")

        if model == FALLBACK_MODEL_CHOICE:
            raise ValueError(
                "LlamaCpp Prompt Enhancer: no model selected because the server "
                f"at '{base_url}' was unreachable when the node loaded. Fix the "
                "base_url and refresh node definitions (ComfyUI menu > Refresh)."
            )

        system_prompt = _build_system_prompt(extra_system_prompt)
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_CHAT)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"LlamaCpp Prompt Enhancer: could not connect to server at "
                f"'{base_url}'. Is the llama.cpp router running? ({exc})"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"LlamaCpp Prompt Enhancer: request to '{base_url}' timed out "
                f"after {DEFAULT_TIMEOUT_CHAT}s."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            body = exc.response.text[:500] if exc.response is not None else str(exc)
            raise RuntimeError(
                f"LlamaCpp Prompt Enhancer: server returned an error: {body}"
            ) from exc

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"LlamaCpp Prompt Enhancer: server response had no 'choices'. "
                f"Raw response: {str(data)[:500]}"
            )

        message = choices[0].get("message", {})
        content = (message.get("content") or "").strip()
        finish_reason = choices[0].get("finish_reason")

        if not content:
            reasoning = message.get("reasoning_content")
            hint = ""
            if finish_reason == "length" and reasoning:
                hint = (
                    " The model produced reasoning_content but ran out of "
                    "max_tokens before emitting final content - this usually "
                    "means thinking mode is still on for this model. Try "
                    "enabling 'disable_thinking' or increasing max_tokens."
                )
            raise RuntimeError(
                f"LlamaCpp Prompt Enhancer: model returned empty content "
                f"(finish_reason={finish_reason!r}).{hint}"
            )

        return (content,)
