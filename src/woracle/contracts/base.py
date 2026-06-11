"""Base machinery shared by every persisted contract model.

Rules (ARCH §2, decision 10):
* Every persisted model carries ``schema_version`` and is migrated on load.
* ``extra="forbid"`` everywhere — silent key drift is schema drift.
* Arrays NEVER live inside JSON. Models reference sidecar files through
  :class:`ArtifactRef` (relative path + sha256 digest).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Annotated, Any

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class WoracleModel(BaseModel):
    """Base for all woracle contract models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(indent=kwargs.pop("indent", 2), **kwargs)


def _no_yaml_hostile_chars(s: str) -> str:
    """Reject characters YAML 1.1 silently normalizes (breaking round-trips).

    \\x85, \\u2028, \\u2029 and other C0/C1 controls are treated as line
    breaks / mangled by YAML loaders; a spec containing them would not survive
    save->load identically (found by hypothesis). Human-authored task text has
    no business containing them — fail loudly at construction.
    """
    for ch in s:
        if (unicodedata.category(ch) == "Cc" and ch not in "\n\t") or ch in "\u2028\u2029":
            raise ValueError(
                f"control character {ch!r} is not allowed in spec text "
                "(it breaks YAML round-tripping)"
            )
    return s


# Use for every human-authored text field in YAML-persisted models (TaskSpec).
Text = Annotated[str, AfterValidator(_no_yaml_hostile_chars)]


class VersionedModel(WoracleModel):
    """A model that is persisted to disk and therefore versioned + migrated."""

    schema_version: int = 1


class ArtifactRef(WoracleModel):
    """Reference to a sidecar payload file (npz / parquet / mp4 / safetensors).

    ``path`` is relative to the directory of the JSON document that holds the
    ref, so artifact bundles stay relocatable.
    """

    path: str
    sha256: str = ""
    kind: str = ""  # e.g. "frames.npz", "tracks.npz", "mask.npz"

    def resolve(self, base_dir: str) -> str:
        import os

        return os.path.normpath(os.path.join(base_dir, self.path))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing: sorted keys, no whitespace drift.

    Rejects NaN/Infinity — non-portable across JSON parsers and a silent
    cache-key instability.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(obj: Any) -> str:
    """sha256 hex digest of an object's canonical JSON form."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def digest_array(arr: np.ndarray) -> str:
    """Digest a numpy array including dtype and shape (not just raw bytes)."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class Provenance(WoracleModel):
    """Who/what produced an artifact. Embedded in every output document."""

    package_version: str = ""
    git_sha: str = ""
    created_at: str = ""  # ISO-8601, filled by the producer
    components: dict[str, str] = Field(default_factory=dict)  # name -> version
    extra: dict[str, str] = Field(default_factory=dict)
