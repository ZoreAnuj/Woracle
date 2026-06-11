"""OpenAI-compatible HTTP VLM backend (works with OpenAI, vLLM, llama.cpp,
LM Studio, Gemini-OpenAI proxies — anything speaking /chat/completions).

Pure-stdlib HTTP (urllib) so the kernel gains no client dependency. Infra
failures (timeouts, 5xx) raise InfraError (retryable, never evidence).
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from woracle.errors import InfraError, MissingDependencyError

if TYPE_CHECKING:
    import numpy as np


def _png_b64(image: np.ndarray) -> str:
    try:
        from PIL import Image
    except ImportError as e:
        raise MissingDependencyError("encoding frames for a VLM judge", "ground") from e
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class OpenAICompatBackend:
    """``complete()`` against any OpenAI-compatible /v1/chat/completions."""

    name = "vlm.openai_compat"
    version = "0.1.0"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def complete(self, images: list[np.ndarray], prompt: str) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_png_b64(img)}"},
                }
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise InfraError(f"VLM backend HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise InfraError(f"VLM backend unreachable: {e}") from e
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise InfraError(f"VLM backend returned unexpected schema: {str(data)[:300]}") from e
