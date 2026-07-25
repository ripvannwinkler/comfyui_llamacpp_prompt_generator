from .llamacpp_prompt_enhancer import LlamaCppPromptEnhancer

NODE_CLASS_MAPPINGS = {
    "LlamaCppPromptEnhancer": LlamaCppPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppPromptEnhancer": "Llama.cpp Prompt Enhancer",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
