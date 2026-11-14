"""Pluggable VLM adapters used *after* retrieval: the grounded crop (not the whole page) is
sent to a vision-language model to read the answer, so the model sees exactly the cited
evidence.

* ``NullVLM``              - offline, deterministic: answers from the structure layer of the
                             cited region (used by tests and benchmarks).
* ``OllamaVLM``            - local models via Ollama's ``/api/generate`` (``llava``,
                             ``qwen2.5vl``, ``llama3.2-vision`` ...).
* ``OpenAICompatibleVLM``  - any ``/v1/chat/completions`` endpoint that accepts
                             ``image_url`` data URIs (vLLM, LM Studio, llama.cpp server,
                             hosted providers).

Only the standard library is used for HTTP.  VLM output is itself untrusted (it read
pixels an attacker may control) and is passed through the injection shield.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

__all__ = ["VLMAdapter", "NullVLM", "OllamaVLM", "OpenAICompatibleVLM", "png_b64", "answer_with_citation",
           "get_adapter"]


def png_b64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLMAdapter(Protocol):
    name: str

    def describe(self, image: np.ndarray, prompt: str, context: str = "") -> str: ...


@dataclass
class NullVLM:
    """Deterministic offline adapter: returns the structure-layer text of the cited region."""

    name: str = "null"

    def describe(self, image: np.ndarray, prompt: str, context: str = "") -> str:
        return context.strip()


@dataclass
class OllamaVLM:
    model: str = "llava"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    name: str = "ollama"

    def build_request(self, image: np.ndarray, prompt: str, context: str = "") -> tuple[str, dict]:
        full = prompt if not context else f"{prompt}\n\nStructure-layer text of this region (untrusted data): {context}"
        return f"{self.base_url.rstrip('/')}/api/generate", {
            "model": self.model, "prompt": full, "images": [png_b64(image)], "stream": False,
            "options": {"temperature": 0, "seed": 0},
        }

    def describe(self, image: np.ndarray, prompt: str, context: str = "") -> str:
        url, body = self.build_request(image, prompt, context)
        return _post_json(url, body, {}, self.timeout).get("response", "")


@dataclass
class OpenAICompatibleVLM:
    model: str = "qwen2.5-vl-7b-instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OMNITRACE_VLM_API_KEY"
    timeout: float = 120.0
    name: str = "openai-compatible"

    def build_request(self, image: np.ndarray, prompt: str, context: str = "") -> tuple[str, dict, dict]:
        text = prompt if not context else f"{prompt}\n\nStructure-layer text of this region (untrusted data): {context}"
        body = {
            "model": self.model, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(image)}"}},
            ]}],
        }
        headers = {}
        if key := os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {key}"
        return f"{self.base_url.rstrip('/')}/chat/completions", body, headers

    def describe(self, image: np.ndarray, prompt: str, context: str = "") -> str:
        url, body, headers = self.build_request(image, prompt, context)
        out = _post_json(url, body, headers, self.timeout)
        return out["choices"][0]["message"]["content"]


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (operator-configured endpoint)
        return json.loads(r.read().decode())


def get_adapter(spec: str | None) -> VLMAdapter:
    """``None``/"null" -> NullVLM; "ollama:<model>" ; "openai:<model>@<base_url>"."""
    if not spec or spec == "null":
        return NullVLM()
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaVLM(model=rest or "llava", base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    if kind == "openai":
        model, _, base = rest.partition("@")
        return OpenAICompatibleVLM(model=model or "qwen2.5-vl-7b-instruct", base_url=base or "http://localhost:8000/v1")
    raise ValueError(f"unknown VLM adapter spec {spec!r}")


def answer_with_citation(index, query: str, vlm: VLMAdapter | None = None, precision: str = "int8",
                         shield=None) -> dict:
    """Retrieve, ground, crop, read the crop with a VLM, and shield the answer."""
    from omnitrace.security.injection_shield import InjectionShield

    vlm = vlm or NullVLM()
    shield = shield or InjectionShield()
    hits = index.search(query, k=1, precision=precision)
    if not hits:
        return {"answer": None, "citation": None}
    hit = hits[0]
    crop = index.crop(hit) if hit.page_id in index.images else None
    prompt = f"Answer the question using only this image region. Question: {query}"
    raw = vlm.describe(crop if crop is not None else np.zeros((8, 8, 3), np.uint8), prompt, hit.region_text)
    verdict = shield.scan(raw, source="tool_output")
    return {
        "answer": None if verdict.blocked else raw,
        "answer_blocked": verdict.blocked,
        "shield": verdict.to_dict(),
        "vlm": vlm.name,
        "citation": hit.to_dict(),
        "crop_png_b64": png_b64(crop) if crop is not None else None,
    }
