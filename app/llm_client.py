"""Client for local Ollama LLM inference."""

from __future__ import annotations

import json
import logging

import httpx

import app.config as _cfg
from app.config import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(600.0, connect=10.0)

# Context window size for Ollama requests. None = let Ollama use whatever
# the model is currently loaded with (avoids costly KV cache reallocation).
# Set explicitly only when you need a specific context size.
NUM_CTX: int | None = None


def _build_options(temperature: float = 0.1, num_predict: int = 4096) -> dict:
    """Build the options dict, only including num_ctx when explicitly set."""
    opts: dict = {
        "temperature": temperature,
        "num_predict": num_predict,
    }
    if NUM_CTX is not None:
        opts["num_ctx"] = NUM_CTX
    return opts


async def count_tokens(text: str) -> int:
    """Count tokens in text using the model's tokenizer.

    Uses a minimal generate call (num_predict=1) and reads prompt_eval_count
    from the response. This gives exact token counts from the actual model
    tokenizer without meaningful inference cost.
    """
    payload = {
        "model": _cfg.OLLAMA_MODEL,
        "prompt": text,
        "stream": False,
        "think": False,
        "options": {"num_predict": 1},
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["prompt_eval_count"]


async def count_chat_tokens(messages: list[dict]) -> int:
    """Count input tokens for a chat-style request (system + user messages).

    Uses a minimal chat call (num_predict=1) and reads prompt_eval_count.
    This accounts for chat template overhead (special tokens, role markers).
    """
    payload = {
        "model": _cfg.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": 1},
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["prompt_eval_count"]


async def get_model_context_length() -> int:
    """Query the model's context window size from Ollama."""
    payload = {"model": _cfg.OLLAMA_MODEL}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/show", json=payload)
        resp.raise_for_status()
        data = resp.json()
    # The context length lives in model_info or parameters
    model_info = data.get("model_info", {})
    for key, value in model_info.items():
        if "context_length" in key:
            return int(value)
    # Fallback: parse from parameters string
    params = data.get("parameters", "")
    for line in params.splitlines():
        if "num_ctx" in line:
            return int(line.split()[-1])
    return 32768  # default for qwen2.5


async def check_model_ready() -> bool:
    """Check if Ollama is reachable and the model is available."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return any(m["name"].startswith(_cfg.OLLAMA_MODEL.split(":")[0]) for m in models)
    except httpx.ConnectError:
        return False
    return False


async def warm_up() -> None:
    """Preload the model and wait until ready.

    When NUM_CTX is set, Ollama may need to reallocate the KV cache if the
    model is currently loaded with a different context size. This warmup
    ensures that reallocation completes before the pipeline sends real work.

    When NUM_CTX is None, this still verifies the model is loaded and
    responsive.
    """
    # Check if model is already loaded with the right context
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for m in models:
                    if m["name"] == _cfg.OLLAMA_MODEL:
                        current_ctx = m.get("context_length")
                        if NUM_CTX is None or current_ctx == NUM_CTX:
                            logger.info(
                                f"Model ready: {_cfg.OLLAMA_MODEL} loaded with "
                                f"context_length={current_ctx}"
                            )
                            return
                        logger.info(
                            f"Model loaded with context_length={current_ctx}, "
                            f"need {NUM_CTX}. Sending warmup request..."
                        )
    except httpx.ConnectError:
        logger.warning("Ollama not reachable, warmup request may be slow")

    # Send a minimal request to trigger model load
    opts = _build_options(num_predict=1)
    payload = {
        "model": _cfg.OLLAMA_MODEL,
        "prompt": "Hi",
        "stream": False,
        "options": opts,
    }
    logger.info(f"Warming up {_cfg.OLLAMA_MODEL}...")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
    logger.info("Warm-up complete")


async def generate(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """Send a prompt to the Ollama model and return the full response text."""
    payload = {
        "model": _cfg.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "options": _build_options(temperature=temperature),
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]


async def chat(messages: list[dict], temperature: float = 0.1) -> str:
    """Send a chat-style request to the Ollama model."""
    payload = {
        "model": _cfg.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": _build_options(temperature=temperature),
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


_JSON_REPAIR_PROMPT = (
    "The following text was supposed to be valid JSON but a parser rejected it. "
    "Fix it so it becomes valid JSON. Preserve all data, remove any non-JSON "
    "artifacts (comments, markdown, stray words). Return ONLY the corrected "
    "JSON, nothing else."
)


async def _request_json_repair(broken: str, error: str) -> str:
    """Ask the LLM to fix its own broken JSON output."""
    payload = {
        "model": _cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _JSON_REPAIR_PROMPT},
            {"role": "user", "content": f"Parse error: {error}\n\n{broken}"},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": _build_options(temperature=0.0),
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


_JSON_CONSTRAINT = (
    "Your response will be passed directly to a JSON parser. "
    "Return ONLY the raw JSON object. No markdown fences, no comments, "
    "no notes, no explanations, no text outside the JSON structure."
)


async def chat_json(messages: list[dict], temperature: float = 0.1) -> dict:
    """Send a chat request and parse the response as JSON."""
    # Append a short system message reinforcing strict JSON-only output
    augmented = list(messages) + [
        {"role": "system", "content": _JSON_CONSTRAINT},
    ]
    payload = {
        "model": _cfg.OLLAMA_MODEL,
        "messages": augmented,
        "stream": False,
        "think": False,
        "format": "json",
        "options": _build_options(temperature=temperature),
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as first_err:
        logger.warning(
            "=== LLM returned invalid JSON, requesting repair ===\n%s\n===",
            content,
        )
        repaired = await _request_json_repair(content, str(first_err))
        try:
            result = json.loads(repaired)
            _count = len(
                result.get("entities", result.get("verified", result.get("descriptions",
                result.get("classifications", result.get("metrics", [])))))
            )
            logger.warning("=== Repair successful (%d items) ===\n%s\n===", _count, repaired)
            result["_repaired"] = True
            return result
        except json.JSONDecodeError:
            logger.error(
                "=== Repair also returned invalid JSON, aborting ===\n%s\n===",
                repaired,
            )
            raise SystemExit(1)
