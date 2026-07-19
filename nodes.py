import logging
import os
import random
import base64
import hashlib
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
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"

REPLICATE_MODELS = [
    "replicate/google/gemini-3-flash | $0.50/M in | $3.00/M out",
    "replicate/google/gemini-2.5-flash | $0.30/M in | $2.50/M out",
    "replicate/openai/gpt-5-mini | $0.25/M in | $2.00/M out",
]

OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-flash | $0.30/M in | $2.50/M out",
    "openrouter/google/gemini-3-flash-preview | $0.50/M in | $3/M out",
    "openrouter/x-ai/grok-4.3 | $1.25/M in | $2.50/M out",
    "openrouter/openai/gpt-5-mini | $0.25/M in | $2/M out",
]

# Cerebras: OpenAI-compatible, extremely fast (sub-2s), free key. Only gemma-4-31b is
# multimodal; gpt-oss-120b / zai-glm-4.7 return `multimodal_not_enabled` so they're
# useless for captioning and are intentionally omitted. Listed as free ($0/M).
CEREBRAS_MODELS = [
    "cerebras/gemma-4-31b | $0.00/M in | $0.00/M out",
]

AVAILABLE_MODELS = REPLICATE_MODELS + OPENROUTER_MODELS + CEREBRAS_MODELS
DEFAULT_MODEL = AVAILABLE_MODELS[0]

# Max retry attempts per model before falling through to the retry model.
# Covers transient connection drops that truncate output mid-sentence.
MAX_ATTEMPTS_PER_MODEL = 3

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
    except Exception:
        return 0.0, 0.0

def img_sig(img: Image.Image) -> str:
    """Cheap, stable signature of image content for the caption cache (downscaled thumbnail)."""
    return hashlib.md5(img.convert("RGB").resize((32, 32)).tobytes()).hexdigest()

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
                "freeze_caption": ("BOOLEAN", {"default": False,
                    "label_on": "FROZEN: reuse last caption (no API)", "label_off": "auto (re-caption on change)"}),
                "system_instruction": ("STRING", {"multiline": True, "placeholder": "You are a professional image captioner..."}),
                "replicate_api_key": ("STRING", {}),
                "openrouter_api_key": ("STRING", {}),
                "cerebras_api_key": ("STRING", {}),
                "retry_model": (AVAILABLE_MODELS, {"default": DEFAULT_MODEL}),
                "error_fallback_value": ("STRING", {"lazy": True}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 65535}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ask_unified"
    CATEGORY = "Unified Caption"

    def __init__(self):
        self.text_output: str | None = None
        # Per-instance caption cache: skips re-calling the API when the caption-relevant
        # inputs are unchanged, and backs the freeze_caption hard-lock. Survives across runs
        # within a session; cleared on ComfyUI restart.
        self._cap_key = None
        self._cap_text: str | None = None
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

    def _validate_completion(self, text: str, model: str) -> str:
        """
        Raise UnifiedAPIError if the output looks truncated.
        A complete caption ends in terminal punctuation; a connection-dropped
        or otherwise cut-off caption ends mid-word. The caller will catch the
        exception and trigger a retry with the same model.
        """
        if not text:
            raise UnifiedAPIError(f"{model} returned empty response")
        # Strip trailing whitespace and markdown emphasis characters that
        # might hide the actual terminal punctuation.
        stripped = text.rstrip().rstrip('*_`')
        if not stripped:
            raise UnifiedAPIError(f"{model} returned empty response after stripping")
        # Accepted terminators: sentence punctuation, straight/curly quotes,
        # and closing brackets/parens for parenthetical endings.
        if stripped[-1] not in '.!?"\'\u201d\u2019)]':
            raise UnifiedAPIError(
                f"{model} output appears truncated (ends: ...{stripped[-40:]!r})"
            )
        return text

    def _call_openrouter(self, key, model, prompt, sys_msg, img_url, temp, max_tokens, label):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        user_content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_url}}]
        
        messages = []
        if sys_msg: messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": user_content})

        payload = {"model": model, "messages": messages, "temperature": temp, "max_tokens": max_tokens}
        
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
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
        return self._validate_completion(content, model)

    def _call_cerebras(self, key, model, prompt, sys_msg, img_url, temp, max_tokens, label):
        # OpenAI-compatible. Two Cerebras-specific gotchas:
        #  1. Cloudflare bans the default python-requests/urllib User-Agent (403, "error code:
        #     1010") -> must send a browser-like UA.
        #  2. gemma-4-31b returns clean `.content` (no split reasoning field), so no special
        #     handling needed here; if gpt-oss/glm are ever added they'd need `.reasoning` merge.
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        user_content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_url}}]

        messages = []
        if sys_msg: messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": user_content})

        # Cerebras uses max_completion_tokens (OpenAI's newer name); max_tokens is ignored.
        payload = {"model": model, "messages": messages, "temperature": temp,
                   "max_completion_tokens": max_tokens}

        r = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=60)
        if r.status_code != 200: raise UnifiedAPIError(f"Cerebras Error: {r.text}")

        data = r.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            in_price, out_price = extract_prices(label)
            self._display_cost(prompt_tokens, completion_tokens, in_price, out_price, model)

        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        return self._validate_completion(content, model)

    def _call_replicate(self, key, model, prompt, sys_msg, img_url, temp, max_tokens, label):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        is_google = any(k in model.lower() for k in ["google", "gemini"])
        if is_google:
            input_data = {"prompt": prompt, "system_instruction": sys_msg, "images": [img_url],
                          "temperature": temp, "max_output_tokens": max_tokens}
            # On Replicate, Gemini's reasoning ("thinking") tokens are drawn from the SAME
            # max_output_tokens budget as the visible output. With thinking left on, a normal
            # caption budget gets eaten by reasoning and the response truncates mid-sentence
            # (then _validate_completion rejects it and burns retries). Captioning needs no
            # reasoning, so disable/minimize it: 2.5 supports thinking_budget=0 (fully off);
            # 3 has no off (low/high only) so use low.
            if "gemini-3" in model:
                input_data["thinking_level"] = "low"
            else:
                input_data["thinking_budget"] = 0
        else:
            input_data = {"prompt": prompt, "image_input": [img_url], "temperature": temp, "max_tokens": max_tokens}

        def _finish(poll):
            """Return caption text if the prediction is done, raise if failed, else None."""
            status = poll.get("status")
            if status == "succeeded":
                output = poll.get("output")
                text = "".join(output) if isinstance(output, list) else str(output)

                # --- Replicate Token Extraction (cost logging) ---
                metrics = poll.get("metrics", {})
                input_tokens = metrics.get("input_token_count", metrics.get("tokens_in", 0))
                output_tokens = metrics.get("output_token_count", metrics.get("tokens_out", 0))
                if (not input_tokens or not output_tokens) and poll.get("logs"):
                    logs_text = poll.get("logs", "")
                    in_match = re.search(r'Input token count:\s*(\d+)', logs_text, re.IGNORECASE) \
                        or re.search(r'tokens_in:\s*(\d+)', logs_text, re.IGNORECASE)
                    out_match = re.search(r'Output token count:\s*(\d+)', logs_text, re.IGNORECASE) \
                        or re.search(r'tokens_out:\s*(\d+)', logs_text, re.IGNORECASE)
                    if in_match and not input_tokens:
                        input_tokens = int(in_match.group(1))
                    if out_match and not output_tokens:
                        output_tokens = int(out_match.group(1))
                in_price, out_price = extract_prices(label)
                self._display_cost(input_tokens, output_tokens, in_price, out_price, model)
                # -------------------------------------------------

                return self._validate_completion(text.strip(), model)
            if status == "failed":
                raise UnifiedAPIError(f"Replicate Model Failed: {poll.get('error')}")
            return None

        # Prefer:wait holds the connection open (~60s) and frequently returns the finished
        # prediction directly, avoiding the poll loop; the timeout must exceed that hold.
        wait_headers = {**headers, "Prefer": "wait"}
        r = requests.post(REPLICATE_BASE_URL.format(model), headers=wait_headers,
                          json={"input": input_data}, timeout=70)
        if r.status_code not in (200, 201): raise UnifiedAPIError(f"Replicate Init Error: {r.text}")
        init = r.json()

        done = _finish(init)
        if done is not None:
            return done

        get_url = init["urls"]["get"]
        start_time = time.time()
        while time.time() - start_time < 180:
            poll = requests.get(get_url, headers=headers, timeout=30).json()
            done = _finish(poll)
            if done is not None:
                return done
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

        retry = kwargs.get("retry_model")

        # ---- caption cache / freeze: avoid re-calling the API when nothing relevant changed ----
        # Key on the caption-relevant inputs only (image content + prompt + model knobs + seed);
        # error_fallback_value is excluded so changing it never re-captions. seed stays in the
        # key so bumping it still forces a fresh reroll.
        cap_key = (
            img_sig(pil_imgs[0]), prompt, model, retry,
            kwargs.get("system_instruction") or "",
            round(kwargs.get("temperature") or 0.0, 3),
            kwargs.get("max_tokens", 2048), kwargs.get("seed"),
        )
        if kwargs.get("freeze_caption") and self._cap_text is not None:
            self.logger.info("Unified Node: caption FROZEN — reusing last caption (no API call)")
            self.text_output = self._cap_text
            return []
        if self._cap_text is not None and self._cap_key == cap_key:
            self.logger.info("Unified Node: inputs unchanged — reusing cached caption (no API call)")
            self.text_output = self._cap_text
            return []
        if kwargs.get("freeze_caption") and self._cap_text is None:
            self.logger.info("Unified Node: freeze_caption ON but no cached caption yet — captioning once to seed it")

        img_url = pil_to_data_url(pil_imgs[0])

        sequence = [model]
        if retry and retry != model:
            sequence.append(retry)

        for label in sequence:
            provider, actual_model = normalize_label(label)

            for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
                self.logger.info(
                    f"Unified Node: Attempting {provider}/{actual_model} "
                    f"(attempt {attempt}/{MAX_ATTEMPTS_PER_MODEL})"
                )
                try:
                    if provider == "openrouter":
                        key = kwargs.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY")
                        if not key:
                            raise UnifiedAPIError("OpenRouter Key Missing")
                        self.text_output = self._call_openrouter(
                            key, actual_model, prompt,
                            kwargs.get("system_instruction"), img_url,
                            kwargs.get("temperature"), kwargs.get("max_tokens", 2048), label
                        )
                    elif provider == "cerebras":
                        key = kwargs.get("cerebras_api_key") or os.environ.get("CEREBRAS_API_KEY")
                        if not key:
                            raise UnifiedAPIError("Cerebras Key Missing")
                        self.text_output = self._call_cerebras(
                            key, actual_model, prompt,
                            kwargs.get("system_instruction"), img_url,
                            kwargs.get("temperature"), kwargs.get("max_tokens", 2048), label
                        )
                    else:
                        key = kwargs.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN")
                        if not key:
                            raise UnifiedAPIError("Replicate Key Missing")
                        self.text_output = self._call_replicate(
                            key, actual_model, prompt,
                            kwargs.get("system_instruction"), img_url,
                            kwargs.get("temperature"), kwargs.get("max_tokens", 2048), label
                        )

                    if self.text_output:
                        self.logger.info(
                            f"Unified Node: Success with {actual_model} on attempt {attempt}"
                        )
                        self._cap_key, self._cap_text = cap_key, self.text_output  # seed the cache
                        return []

                except UnifiedAPIError as e:
                    # Non-transient failures: don't waste attempts.
                    if "Key Missing" in str(e):
                        self.logger.warning(f"Unified Node: {actual_model} -> {e}")
                        break  # skip remaining attempts for this model
                    self.logger.warning(
                        f"Unified Node: {actual_model} attempt {attempt} failed -> {e}"
                    )
                    if attempt < MAX_ATTEMPTS_PER_MODEL:
                        time.sleep(1.5 * attempt)  # backoff: 1.5s, 3s
                    continue

                except Exception as e:
                    self.logger.warning(
                        f"Unified Node: {actual_model} attempt {attempt} failed -> {e}"
                    )
                    if attempt < MAX_ATTEMPTS_PER_MODEL:
                        time.sleep(1.5 * attempt)
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
