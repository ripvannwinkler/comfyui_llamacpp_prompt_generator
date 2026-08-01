import base64
import io
import re
import time

import numpy as np
import requests
from PIL import Image

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_MODELS = 3  # seconds, for populating the COMBO at node-def time
DEFAULT_TIMEOUT_CHAT = 120  # seconds, for the blocking generation call
MAX_RETRIES = 2  # retries on ConnectionError (e.g., model swap)
RETRY_DELAY = 1  # seconds between retries
FALLBACK_MODEL_CHOICE = "(server unreachable - enter base_url and refresh)"

DEFAULT_SYSTEM_PROMPT = (
    "You are a prompt-engineering assistant for text-to-image diffusion models "
    "(e.g. Stable Diffusion, SDXL, Flux). Given a short or rough user prompt, "
    "rewrite and expand it into a single, richly detailed natural language "
    "description suitable for image generation.\n\n"
    "Guidelines:\n"
    "- Preserve the user's core subject and intent; do not change the meaning.\n\n"
    "Format:\n"
    "- Write in natural, flowing prose using complete sentences and paragraphs.\n"
    "- Structure the description from the main subject outward, moving from "
    "foreground to background.\n"
    "- Do not add negative prompts, explanations, headers, or markdown formatting.\n"
    "- Output ONLY the enhanced prompt text, nothing else.\n"
    "- Do not wrap the output in quotes."
)

STYLE_PROMPTS = {
    "none": "",  # Empty prompt for no style modification
    "detailed": "Start the enhanced prompt with 'detailed, '. Create a hyper-detailed visual description with precise technical specifications...",
    "photorealistic": "Start the enhanced prompt with 'photorealistic, '. Create a hyperdetailed photorealistic description with sharp focus, precise technical and visual details...",
    "cinematic": "Start the enhanced prompt with 'cinematic, '. Create a cinematic shot with cinematic composition, hyperdetailed, sharp focus, dynamic lighting, precise film techniques...",
    "artistic": "Start the enhanced prompt with 'artistic, '. Create a sophisticated artwork emphasizing advanced artistic techniques...",
    "minimalist": "Start the enhanced prompt with 'minimalist, '. Create a minimalist artwork with precise reductive elements...",
    "fantasy": "Start the enhanced prompt with 'fantasy, '. Create a fantasy-themed artwork with precise magical specifications...",
    "horror": "Start the enhanced prompt with 'horror, '. Create dark horror with precise unsettling specifications...",
    "dark fantasy": "Start the enhanced prompt with 'dark fantasy, '. Create dark fantasy with precise gothic and supernatural specifications...",
    "vibrant": "Start the enhanced prompt with 'vibrant, '. Create a vibrant artwork with precise color specifications...",
    "heavenly": "Start the enhanced prompt with 'heavenly, '. Create a celestial, ethereal artwork with precise divine specifications...",
    "oil painting": "Start the enhanced prompt with 'oil painting, '. Create a classical oil painting with precise traditional specifications...",
    "watercolor": "Start the enhanced prompt with 'watercolor, '. Create a watercolor artwork with precise aqueous specifications...",
    "abstract expressionist": "Start the enhanced prompt with 'abstract expressionist, '. Create an abstract expressionist artwork with precise gestural specifications...",
    "hyperrealist": "Start the enhanced prompt with 'hyperrealist, '. Create a hyperrealistic artwork with extreme precision...",
    "cubist": "Start the enhanced prompt with 'cubist, '. Create a cubist artwork with specific geometric deconstruction...",
    "bauhaus": "Start the enhanced prompt with 'Bauhaus, '. Create a Bauhaus style artwork with specific design principles...",
    "romanticist": "Start the enhanced prompt with 'romanticist, '. Create a romanticist artwork with emotional and natural elements...",
    "dada": "Start the enhanced prompt with the word 'dada, '. Create a Dada artwork with specific anti-art elements...",
    "street art": "Start the enhanced prompt with 'street art, '. Create street art with specific urban art techniques...",
    "anime": "Start the enhanced prompt with the word 'anime, '. Create an anime-style artwork with precise animation techniques. Detail the character elements (large expressive eyes with 3-4 highlight points at 100% opacity, simplified facial features with strong emotional expressions, dynamic hair with wind physics and 20-30 distinct strands, color palette with 4-5 tonal values per element), shading techniques (cel-shading with hard edges at 85% opacity, ambient occlusion at 40% strength for depth, rim lighting at 90% intensity for edge definition), action elements (speed lines at 45-degree angles with 70% opacity, impact frames with radial blur at 25% strength, motion smears for quick movements), and background treatment (detailed establishing shots with 3-point perspective, simplified backgrounds during character focus with 20% detail retention). Include standard anime visual elements (dramatic lighting effects with stark shadows, sweat drops and anger veins for emotion, sparkles and floating petals for atmosphere), facial features (eyes at 1/3 head height, small nose and mouth with minimal detail, varied expressions from chibi to serious), and costume dynamics (flowing fabric with secondary motion, dramatic poses with foreshortening, cloth folds following form)...",
    "studio ghibli": "Start the enhanced prompt with the words 'studio ghibli, '. Create a Studio Ghibli inspired artwork with their signature animation style. Detail the environmental elements (layered clouds with cumulus structure and 30% opacity variation, grass plains with individual blade definition and wind animation patterns, trees with organic movement and dappled light effects), character design (rounded, soft features with minimal sharp angles, expressive faces with 2-3 highlight points in eyes, natural hair movement with subtle physics), color treatment (pastel base palette with 80% saturation, warm sunlight tones #FFE5B4 to #FFB347, natural color gradients with 10% steps between values), and atmospheric effects (floating particles with 2-second fade cycle, gentle wind effects at 5mph affecting foliage and fabric, dynamic skies with 3-5 cloud layers). Include signature elements (food scenes with exaggerated texture and steam effects, flying sequences with dynamic camera movements, cozy interior spaces with lived-in details), lighting techniques (soft diffused sunlight at 30-degree angle, ambient occlusion at 15% strength for depth, warm interior lighting with 2700K color temperature), and background details (European-inspired architecture with weathered textures, detailed mechanical designs with functional components, natural environments with ecological accuracy)...",
    "3d render": "Start the enhanced prompt with '3d render, '. Create a 3D render with precise technical specifications...",
    "digital art": "Start the enhanced prompt with 'digital art, '. Create digital art with precise contemporary techniques...",
    "studio photography": "Start the enhanced prompt with 'studio photography, '. Create a studio photograph with precise technical setup...",
    "concept art": "Start the enhanced prompt with 'concept art, '. Create concept art with precise production art techniques...",
    "comic book": "Start the enhanced prompt with 'comic book, '. Create a comic book illustration with precise stylistic elements. Detail the line art (bold outlines at 3-4px thickness, dynamic speed lines for motion, dramatic perspective with exaggerated foreshortening), coloring technique (flat colors with cel-shading, 4-color limited palette reminiscent of vintage comics, high contrast shadows at 80% opacity), panel composition (dramatic angles, extreme close-ups mixed with wide shots, Dutch angles for tension), and comic-specific elements (halftone dot patterns at 15-30% density, action effects like impact lines and motion blur)",
    "pixel art": "Start the enhanced prompt with pixel art. Create precise pixel art with specific technical constraints...",
    "cyberpunk": "Start the enhanced prompt with cyberpunk. Create a cyberpunk artwork with specific futuristic elements...",
    "steampunk": "Start the enhanced prompt with steampunk. Create a steampunk artwork with specific Victorian-industrial elements...",
    "gothic": "Start the enhanced prompt with 'gothic, '. Create a gothic artwork with specific architectural and atmospheric elements...",
    "art nouveau": "Start the enhanced prompt with 'art nouveau, '. Create an art nouveau artwork with specific decorative elements...",
    "art deco": "Start the enhanced prompt with 'art deco, '. Create an art deco artwork with specific geometric elements...",
    "impressionist": "Start the enhanced prompt with 'impressionist, '. Create an impressionist artwork with specific light-capturing techniques...",
    "surrealist": "Start the enhanced prompt with 'surrealist, '. Create a surrealist artwork with specific dreamlike elements...",
    "baroque": "Start the enhanced prompt with 'baroque, '. Create a baroque artwork with elaborate dramatic elements...",
    "renaissance": "Start the enhanced prompt with 'renaissance, '. Create a renaissance style artwork with precise classical elements...",
    "pop art": "Start the enhanced prompt with 'pop art, '. Create a pop art artwork with precise commercial art elements...",
    "ukiyo-e": "Start the enhanced prompt with 'ukiyo-e, '. Create a Japanese ukiyo-e style artwork with precise woodblock print elements...",
    "pencil sketch": "Start the enhanced prompt with 'pencil sketch, '. Create a detailed pencil sketch with specific traditional drawing techniques...",
    "charcoal drawing": "Start the enhanced prompt with 'charcoal drawing, '. Create a dramatic charcoal drawing with specific medium characteristics...",
    "pastel art": "Start the enhanced prompt with 'pastel art, '. Create a vibrant pastel artwork with specific medium techniques...",
    "stained glass": "Start the enhanced prompt with 'stained glass, '. Create a stained glass artwork with specific technical and design elements...",
    "mosaic": "Start the enhanced prompt with 'mosaic, '. Create a detailed mosaic artwork with specific tessellation techniques...",
    "isometric": "Start the enhanced prompt with 'isometric, '. Create a precise isometric artwork with specific technical parameters...",
    "low poly": "Start the enhanced prompt with 'low poly, '. Create a low poly artwork with specific geometric optimization techniques...",
    "vaporwave": "Start the enhanced prompt with 'vaporwave, '. Create a vaporwave aesthetic with precise retro-digital elements...",
    "retro": "Start the enhanced prompt with 'retro, '. Create a retro style artwork with precise period-specific elements...",
    "vintage": "Start the enhanced prompt with 'vintage, '. Create a vintage artwork with precise aging and period effects...",
    "sumi-e": "Start the enhanced prompt with 'sumi-e, '. Create a Japanese ink wash (sumi-e) artwork with precise traditional techniques. Detail the brushwork characteristics (bamboo brush techniques with varying pressure from 0% to 100%, four basic strokes: horizontal 'yan', vertical 'shu', diagonal 'pie', dot 'dian'), ink gradation methods (five distinct ink values: darkest 'nōboku' at 100% concentration, dark 'nōhitsu' at 80%, medium 'chūboku' at 60%, light 'usuboku' at 40%, palest 'usuhitsu' at 20%), paper interaction (washi paper with 30% cotton content, controlled water absorption rates, intentional bleeding effects), and compositional elements (asymmetrical balance with 70/30 rule, negative space 'ma' occupying 60-70% of composition, rhythmic brush movement 'keisei' with varying speeds 1-5 cm/second). Include traditional techniques (dry brush 'kasure' for texture, splashed ink 'hatsuboku' with 15-degree angle throws, pooled ink 'tamari' with 3-5mm depth), atmospheric effects (mist achieved through diluted ink at 10% concentration, rain with diagonal strokes at 75-degree angles, wind suggested through directional brushwork), and subject treatment (simplified forms with maximum 3-5 brushstrokes, captured essence 'sēshin' through minimal detail, dynamic tension through line weight variation 0.5mm to 5mm)...",
    "erotic": "Start the enhanced prompt with 'erotic, '. Create an erotic photograph with explicit content and strong sexual themes. Remove all clothes and explore the detail of the human body, including genitalia, breasts, and sexual acts. Emphasize sexual arousal, intimacy, and eroticism in the composition. Use lighting and angles to enhance the sensuality of the scene. Include explicit sexual positions. Focus on the physical and emotional aspects of sexual desire and pleasure.",
}


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


def _image_tensor_to_data_uri(image_tensor, max_dimension: int) -> str:
    try:
        frame = image_tensor[0]
        arr = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

        if max(img.width, img.height) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as exc:
        raise ValueError(
            f"LlamaCpp Prompt Enhancer: failed to encode input image: {exc}"
        ) from exc


class LlamaCppPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        models = _fetch_model_list(DEFAULT_BASE_URL)
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (models, {}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 16, "max": 8192, "step": 16},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "disable_thinking": ("BOOLEAN", {"default": True}),
                "enable_enhancement": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT},
                ),
                "style": (list(STYLE_PROMPTS.keys()), {"default": "none"}),
                "image": ("IMAGE",),
                "max_image_dimension": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "generate"
    CATEGORY = "llamacpp"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    def generate(
        self,
        prompt,
        model,
        base_url,
        temperature,
        max_tokens,
        seed,
        disable_thinking,
        enable_enhancement,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        style="none",
        image=None,
        max_image_dimension=1024,
    ):
        if not enable_enhancement:
            return (prompt,)

        if (not prompt or not prompt.strip()) and image is None:
            raise ValueError(
                "LlamaCpp Prompt Enhancer: 'prompt' is empty and no image was "
                "provided. Supply either a prompt or an image input."
            )

        if model == FALLBACK_MODEL_CHOICE:
            raise ValueError(
                "LlamaCpp Prompt Enhancer: no model selected because the server "
                f"at '{base_url}' was unreachable when the node loaded. Fix the "
                "base_url and refresh node definitions (ComfyUI menu > Refresh)."
            )

        if not system_prompt or not system_prompt.strip():
            system_prompt = DEFAULT_SYSTEM_PROMPT

        style_prompt = STYLE_PROMPTS.get(style, "").strip()
        if style_prompt:
            system_prompt = f"{system_prompt}\n\n{style_prompt}"

        if image is not None:
            data_uri = _image_tensor_to_data_uri(image, max_image_dimension)
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        else:
            user_content = prompt

        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_CHAT)
                resp.raise_for_status()
                break
            except requests.exceptions.ConnectionError as exc:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                raise RuntimeError(
                    f"LlamaCpp Prompt Enhancer: could not connect to server at "
                    f"'{base_url}' after {MAX_RETRIES} retries. Is the llama.cpp "
                    f"router running? ({exc})"
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
