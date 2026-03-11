import logging
import os
import random
import base64
import io
import requests
import time
import re

from PIL import Image
from torch import Tensor
from .utils import images_to_pillow

# ==================================================
# Constants & Model Lists
# ==================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REPLICATE_BASE_URL = "https://api.replicate.com/v1/models/{}/predictions"

REPLICATE_MODELS = [
    "replicate/google/gemini-3-flash | $0.50/M in | $3.00/M out",
    "replicate/google/gemini-2.5-flash | $0.30/M in | $2.50/M out",
    "replicate/openai/gpt-5-mini | $0.25/M in | $2.00/M out",
]

OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-flash | $0.30/M in | $2.50/M out",
    "openrouter/google/gemini-3-flash-preview | $0.50/M in | $3/M out",
    "openrouter/x-ai/grok-4.1-fast | $0.20/M in | $0.50/M out",
    "openrouter/openai/gpt-5-mini | $0.25/M in | $2/M out",
]

AVAILABLE_MODELS = REPLICATE_MODELS + OPENROUTER_MODELS
DEFAULT_MODEL = AVAILABLE_MODELS[0]

# ==================================================
# Core Logic & Error Handling
# ==================================================

class UnifiedAPIError(Exception):
    pass

def normalize_label(label: str):
    base = label.split("|")[0].strip()
    provider, model_name = base.split("/", 1)
    return provider.strip(), model_name.strip()

def extract_prices(label: str):
    parts = label.split("|")
    if len(parts) < 3: return 0.0, 0.0
    try:
        in_p = float(parts[1].split("/")[0].replace("$", "").strip())
        out_p = float(parts[2].split("/")[0].replace("$", "").strip())
        return in_p, out_p
    except: return 0.0, 0.0

def pil_to_data_url(img: Image.Image, max_size=1024) -> str:
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ==================================================
# The Node
# ==================================================

class UnifiedCaptionNode:
    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2**31)
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (AVAILABLE_MODELS, {"default": DEFAULT_MODEL}),
                "images": ("IMAGE",), 
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "placeholder": "You are a professional image captioner..."}),
                "replicate_api_key": ("STRING", {}),
                "openrouter_api_key": ("STRING", {}),
                "retry_model": (AVAILABLE_MODELS, {"default": DEFAULT_MODEL}),
                "error_fallback_value": ("STRING", {"lazy": True}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 8192}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ask_unified"
    CATEGORY = "Unified Caption"

    def __init__(self):
        self.text_output: str | None = None
        self.logger = logging.getLogger("ComfyUI-Unified-Caption")

    def ask_unified(self, **kwargs):
        return (kwargs.get("error_fallback_value") if self.text_output is None else self.text_output,)

    # -------------------------
    # Internal Helpers
    # -------------------------

    def _display_cost(self, input_tokens, output_tokens, in_price, out_price, model):
        if input_tokens is None or output_tokens is None:
            return
        cost_usd = (input_tokens / 1_000_000.0 * in_price) + (
            output_tokens / 1_000_000.0 * out_price
        )
        self.logger.info(f"[COST] ${cost_usd:.6f} | model={model}")

    def _call_openrouter(self, key, model, prompt, sys_msg, img_url, temp, max_tokens, label):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        user_content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_url}}]
        
        messages = []
        if sys_msg: messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": user_content})

        payload = {"model": model, "messages": messages, "temperature": temp, "max_tokens": max_tokens}
        
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if r.status_code != 200: raise UnifiedAPIError(f"OpenRouter Error: {r.text}")
        
        data = r.json()
        usage = data.get("usage", {})
        
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if prompt_tokens is not None and completion_tokens is not None:
            in_price, out_price = extract_prices(label)
            self._display_cost(
                prompt_tokens,
                completion_tokens,
                in_price,
                out_price,
                model,
            )
        
        content = data["choices"][0]["message"]["content"].strip()
        if content and content[-1] not in ".!?\"'":
            self.logger.warning(f"⚠️ {model} might have cut off or triggered a safety filter.")
        return content

    def _call_replicate(self, key, model, prompt, sys_msg, img_url, temp, max_tokens, label):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        
        is_google = any(k in model.lower() for k in ["google", "gemini"])
        if is_google:
            input_data = {"prompt": prompt, "system_instruction": sys_msg, "images": [img_url], "temperature": temp, "max_output_tokens": max_tokens}
        else:
            input_data = {"prompt": prompt, "image_input": [img_url], "temperature": temp, "max_tokens": max_tokens}

        r = requests.post(REPLICATE_BASE_URL.format(model), headers=headers, json={"input": input_data}, timeout=60)
        if r.status_code not in (200, 201): raise UnifiedAPIError(f"Replicate Init Error: {r.text}")
        
        get_url = r.json()["urls"]["get"]
        start_time = time.time()
        
        while time.time() - start_time < 180:
            poll = requests.get(get_url, headers=headers, timeout=30).json()
            status = poll.get("status")
            
            if status == "succeeded":
                output = poll.get("output")
                text = "".join(output) if isinstance(output, list) else str(output)
                
                # --- Original Replicate Token Extraction Logic Restored ---
                metrics = poll.get("metrics", {})
                input_tokens = metrics.get("input_token_count", metrics.get("tokens_in", 0))
                output_tokens = metrics.get("output_token_count", metrics.get("tokens_out", 0))

                if (not input_tokens or not output_tokens) and poll.get("logs"):
                    logs_text = poll.get("logs", "")
                    in_match = re.search(r'Input token count:\s*(\d+)', logs_text, re.IGNORECASE)
                    if not in_match:
                        in_match = re.search(r'tokens_in:\s*(\d+)', logs_text, re.IGNORECASE)
                        
                    out_match = re.search(r'Output token count:\s*(\d+)', logs_text, re.IGNORECASE)
                    if not out_match:
                        out_match = re.search(r'tokens_out:\s*(\d+)', logs_text, re.IGNORECASE)

                    if in_match and not input_tokens:
                        input_tokens = int(in_match.group(1))
                    if out_match and not output_tokens:
                        output_tokens = int(out_match.group(1))

                in_price, out_price = extract_prices(label)
                self._display_cost(
                    input_tokens,
                    output_tokens,
                    in_price,
                    out_price,
                    model,
                )
                # ----------------------------------------------------------

                return text.strip()
            
            if status == "failed":
                raise UnifiedAPIError(f"Replicate Model Failed: {poll.get('error')}")
            
            time.sleep(2)
        raise UnifiedAPIError("Replicate Polling Timeout")

    # -------------------------
    # Lazy Execution Chain
    # -------------------------

    def check_lazy_status(self, prompt, model, images, **kwargs):
        self.text_output = None
        
        pil_imgs = images_to_pillow(images)
        if not pil_imgs: 
            self.logger.error("No valid image input.")
            return []
        img_url = pil_to_data_url(pil_imgs[0])

        retry = kwargs.get("retry_model")
        sequence = [model]
        if retry and retry != model: sequence.append(retry)

        for label in sequence:
            provider, actual_model = normalize_label(label)
            self.logger.info(f"Unified Node: Attempting {provider}/{actual_model}")
            
            try:
                if provider == "openrouter":
                    key = kwargs.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY")
                    if not key: raise UnifiedAPIError("OpenRouter Key Missing")
                    self.text_output = self._call_openrouter(key, actual_model, prompt, kwargs.get("system_instruction"), img_url, kwargs.get("temperature"), kwargs.get("max_tokens", 1024), label)
                else:
                    key = kwargs.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN")
                    if not key: raise UnifiedAPIError("Replicate Key Missing")
                    self.text_output = self._call_replicate(key, actual_model, prompt, kwargs.get("system_instruction"), img_url, kwargs.get("temperature"), kwargs.get("max_tokens", 1024), label)
                
                if self.text_output: 
                    self.logger.info(f"Unified Node: Success with {actual_model}")
                    return [] 
            except Exception as e:
                self.logger.warning(f"Unified Node: {actual_model} failed -> {e}")
                continue

        if kwargs.get("error_fallback_value") is not None:
            self.logger.error("All models failed. Returning fallback value.")
            return []

        return ["error_fallback_value"]

# ==================================================
# Mappings
# ==================================================
NODE_CLASS_MAPPINGS = {"Unified_Caption_Node": UnifiedCaptionNode}
NODE_DISPLAY_NAME_MAPPINGS = {"Unified_Caption_Node": "Unified Caption"}