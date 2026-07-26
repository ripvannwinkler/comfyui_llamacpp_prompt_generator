import base64
import io
import re

import numpy as np
import requests
from PIL import Image

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_MODELS = 3  # seconds, for populating the COMBO at node-def time
DEFAULT_TIMEOUT_CHAT = 120  # seconds, for the blocking generation call
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

STYLE_PROMPTS = {
    "none": "",  # Empty prompt for no style modification
    "detailed": "Start prompt with 'detailed, '. Convert this into a hyper-detailed visual description with precise technical specifications...",
    "photorealistic": "Start prompt with 'photorealistic, '. Convert this into a hyperdetailed photorealistic description with sharp focus, precise technical and visual details...",
    "cinematic": "Start prompt with 'cinematic, '. Convert this into a cinematic shot with cincematic composition, hyperdetailed, sharp focus, dynamic lighting, precise film techniques...",
    "artistic": "Start prompt with 'artistic, '. Convert this into a sophisticated artwork emphasizing advanced artistic techniques...",
    "minimalist": "Start prompt with 'minimalist, '. Convert this into a minimalist artwork with precise reductive elements...",
    "fantasy": "Start prompt with 'fantasy, '. Convert this into a fantasy-themed artwork with precise magical specifications...",
    "horror": "Start prompt with 'horror, '. Convert this into dark horror with precise unsettling specifications...",
    "dark fantasy": "Start prompt with 'dark fantasy, '. Convert this into dark fantasy with precise gothic and supernatural specifications...",
    "vibrant": "Start prompt with 'vibrant, '. Convert this into a vibrant artwork with precise color specifications...",
    "heavenly": "Start prompt with 'heavenly, '. Convert this into a celestial, ethereal artwork with precise divine specifications...",
    "oil painting": "Start prompt with 'oil painting, '. Convert this into a classical oil painting with precise traditional specifications...",
    "watercolor": "Start prompt with 'watercolor, '. Convert this into a watercolor artwork with precise aqueous specifications...",
    "abstract expressionist": "Start prompt with 'abstract expressionist, '. Convert this into an abstract expressionist artwork with precise gestural specifications...",
    "hyperrealist": "Start prompt with 'hyperrealist, '. Convert this into a hyperrealistic artwork with extreme precision...",
    "cubist": "Start prompt with 'cubist, '. Convert this into a cubist artwork with specific geometric deconstruction...",
    "bauhaus": "Start prompt with 'Bauhaus, '. Convert this into a Bauhaus style artwork with specific design principles...",
    "romanticist": "Start prompt with 'romanticist, '. Convert this into a romanticist artwork with emotional and natural elements...",
    "dada": "Start the prompt with the word 'dada, '. Convert this into a Dada artwork with specific anti-art elements...",
    "street art": "Start prompt with 'street art, '. Convert this into street art with specific urban art techniques...",
    "anime": "Start the prompt with the word 'anime, '. Convert this into an anime-style artwork with precise animation techniques. Detail the character elements (large expressive eyes with 3-4 highlight points at 100% opacity, simplified facial features with strong emotional expressions, dynamic hair with wind physics and 20-30 distinct strands, color palette with 4-5 tonal values per element), shading techniques (cel-shading with hard edges at 85% opacity, ambient occlusion at 40% strength for depth, rim lighting at 90% intensity for edge definition), action elements (speed lines at 45-degree angles with 70% opacity, impact frames with radial blur at 25% strength, motion smears for quick movements), and background treatment (detailed establishing shots with 3-point perspective, simplified backgrounds during character focus with 20% detail retention). Include standard anime visual elements (dramatic lighting effects with stark shadows, sweat drops and anger veins for emotion, sparkles and floating petals for atmosphere), facial features (eyes at 1/3 head height, small nose and mouth with minimal detail, varied expressions from chibi to serious), and costume dynamics (flowing fabric with secondary motion, dramatic poses with foreshortening, cloth folds following form)...",
    "studio ghibli": "Start the prompt with the words 'studio ghibli, '. Convert this into a Studio Ghibli inspired artwork with their signature animation style. Detail the environmental elements (layered clouds with cumulus structure and 30% opacity variation, grass plains with individual blade definition and wind animation patterns, trees with organic movement and dappled light effects), character design (rounded, soft features with minimal sharp angles, expressive faces with 2-3 highlight points in eyes, natural hair movement with subtle physics), color treatment (pastel base palette with 80% saturation, warm sunlight tones #FFE5B4 to #FFB347, natural color gradients with 10% steps between values), and atmospheric effects (floating particles with 2-second fade cycle, gentle wind effects at 5mph affecting foliage and fabric, dynamic skies with 3-5 cloud layers). Include signature elements (food scenes with exaggerated texture and steam effects, flying sequences with dynamic camera movements, cozy interior spaces with lived-in details), lighting techniques (soft diffused sunlight at 30-degree angle, ambient occlusion at 15% strength for depth, warm interior lighting with 2700K color temperature), and background details (European-inspired architecture with weathered textures, detailed mechanical designs with functional components, natural environments with ecological accuracy)...",
    "3d render": "Start promptwith '3d render, '. Convert this into a 3D renderwith precise technical specifications...",
    "digital art": "Start promptwith 'digital art, '. Convert this into digital artwith precise contemporary techniques...",
    "studio photography": "Start promptwith 'studio photography, '. Convert this into a studio photographwith precise technical setup...",
    "concept art": "Start promptwith 'concept art, '. Convert this into concept artwith precise production art techniques...",
    "comic book": "Start the promptwith 'comic book, '. Convert this into a comic book illustrationwith precise stylistic elements. Detail the line art (bold outlines at 3-4px thickness, dynamic speed lines for motion, dramatic perspectivewith exaggerated foreshortening), coloring technique (flat colorswith cel-shading, 4-color limited palette reminiscent of vintage comics, high contrast shadows at 80% opacity), panel composition (dramatic angles, extreme close-ups mixed wit wide shots, Dutch angles for tension), and comic-specific elements (halftone dot patterns at 15-30% density, action effects like impact lines and motion blur",
    "pixel art": "Start the prompt with pixel art. Convert this into precise pixel art with specific technical constraints...",
    "cyberpunk": "Start the prompt with cyberpunk. Convert this into a cyberpunk artwork with specific futuristic elements...",
    "steampunk": "Start the prompt with steampunk. Convert this into a steampunk artwork with specific Victorian-industrial elements...",
    "gothic": "Start the prompt with 'gothic, '. Convert this into a gothic artwork with specific architectural and atmospheric elements...",
    "art nouveau": "Start the prompt with 'art nouveau, '. Convert this into an art nouveau artwork with specific decorative elements...",
    "art deco": "Start the prompt with 'art deco, '. Convert this into an art deco artwork with specific geometric elements...",
    "impressionist": "Start the prompt with 'impressionist, '. Convert this into an impressionist artwork with specific light-capturing techniques...",
    "surrealist": "Start the prompt with 'surrealist, '. Convert this into a surrealist artwork with specific dreamlike elements...",
    "baroque": "Start the prompt with 'baroque, '. Convert this into a baroque artwork with elaborate dramatic elements...",
    "renaissance": "Start the prompt with 'renaissance, '. Convert this into a renaissance style artwork with precise classical elements...",
    "pop art": "Start the prompt with 'pop art, '. Convert this into a pop art artwork with precise commercial art elements...",
    "ukiyo-e": "Start the prompt with 'ukiyo-e, '. Convert this into a Japanese ukiyo-e style artwork with precise woodblock print elements...",
    "pencil sketch": "Start the prompt with 'pencil sketch, '. Convert this into a detailed pencil sketch with specific traditional drawing techniques...",
    "charcoal drawing": "Start the prompt with 'charcoal drawing, '. Convert this into a dramatic charcoal drawing with specific medium characteristics...",
    "pastel art": "Start the prompt with 'pastel art, '. Convert this into a vibrant pastel artwork with specific medium techniques...",
    "stained glass": "Start the prompt with 'stained glass, '. Convert this into a stained glass artwork with specific technical and design elements...",
    "mosaic": "Start the prompt with 'mosaic, '. Convert this into a detailed mosaic artwork with specific tessellation techniques...",
    "isometric": "Start the prompt with 'isometric, '. Convert this into a precise isometric artwork with specific technical parameters...",
    "low poly": "Start the prompt with 'low poly, '. Convert this into a low poly artwork with specific geometric optimization techniques...",
    "vaporwave": "Start the prompt with 'vaporwave, '. Convert this into a vaporwave aesthetic with precise retro-digital elements...",
    "retro": "Start the prompt with 'retro, '. Convert this into a retro style artwork with precise period-specific elements...",
    "vintage": "Start the prompt with 'vintage, '. Convert this into a vintage artwork with precise aging and period effects...",
    "sumi-e": "Start the prompt with 'sumi-e, '. Convert this into a Japanese ink wash (sumi-e) artwork with precise traditional techniques. Detail the brushwork characteristics (bamboo brush techniques with varying pressure from 0% to 100%, four basic strokes: horizontal 'yan', vertical 'shu', diagonal 'pie', dot 'dian'), ink gradation methods (five distinct ink values: darkest 'nōboku' at 100% concentration, dark 'nōhitsu' at 80%, medium 'chūboku' at 60%, light 'usuboku' at 40%, palest 'usuhitsu' at 20%), paper interaction (washi paper with 30% cotton content, controlled water absorption rates, intentional bleeding effects), and compositional elements (asymmetrical balance with 70/30 rule, negative space 'ma' occupying 60-70% of composition, rhythmic brush movement 'keisei' with varying speeds 1-5 cm/second). Include traditional techniques (dry brush 'kasure' for texture, splashed ink 'hatsuboku' with 15-degree angle throws, pooled ink 'tamari' with 3-5mm depth), atmospheric effects (mist achieved through diluted ink at 10% concentration, rain with diagonal strokes at 75-degree angles, wind suggested through directional brushwork), and subject treatment (simplified forms with maximum 3-5 brushstrokes, captured essence 'sēshin' through minimal detail, dynamic tension through line weight variation 0.5mm to 5mm)...",
    "erotic": "Start the prompt with 'erotic, '. Convert this into an erotic photograph with explicit content and strong sexual themes. Remove all clothes and explore the detail of the human body, including genitalia, breasts, and sexual acts. Emphasize sexual arousal, intimacy, and eroticism in the composition. Use lighting and angles to enhance the sensuality of the scene. Include explicit sexual positions. Focus on the physical and emotional aspects of sexual desire and pleasure.",
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
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        style="none",
        image=None,
        max_image_dimension=1024,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("LlamaCpp Prompt Enhancer: 'prompt' input is empty.")

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
