"""Content-addressed artifact store (~200 LOC, ARCH decision 8).

Key = sha256 of canonical JSON over EXPLICIT declared fields:
``{inputs (digests), component, component_version, params}``. Never derived
from pickled closures (HF datasets' dill-fingerprint nondeterminism is the
cautionary tale). Bumping a component version or a prompt MUST miss the cache —
that is the reproducibility feature, not an inconvenience.

Layout::

    <root>/objects/<key[:2]>/<key>/
        manifest.json     # CacheManifest (what produced this, from what)
        payload/...       # arbitrary sidecar files written by the producer

Writes are atomic (tmp dir + rename) so a crashed producer never leaves a
half-entry that later reads as a cache hit.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from woracle.contracts.base import canonical_json, digest_json
from woracle.errors import StoreError

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class StoreEntry:
    key: str
    dir: str  # entry directory (contains manifest.json + payload/)
    payload_dir: str
    manifest: dict[str, Any]


class ContentStore:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(os.path.join(self.root, "objects"), exist_ok=True)

    # -- keys ---------------------------------------------------------------
    @staticmethod
    def key_for(
        *,
        stage: str,
        component: str,
        component_version: str,
        inputs: dict[str, str],
        params: dict[str, Any] | None = None,
    ) -> str:
        """Build a cache key from explicit, JSON-serializable fields only."""
        try:
            return digest_json(
                {
                    "stage": stage,
                    "component": component,
                    "component_version": component_version,
                    "inputs": inputs,
                    "params": params or {},
                }
            )
        except (TypeError, ValueError) as e:
            raise StoreError(f"cache-key fields must be canonical-JSON-able: {e}") from e

    # -- paths --------------------------------------------------------------
    def _entry_dir(self, key: str) -> str:
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise StoreError(f"malformed store key: {key!r}")
        return os.path.join(self.root, "objects", key[:2], key)

    # -- API ----------------------------------------------------------------
    def has(self, key: str) -> bool:
        d = self._entry_dir(key)
        return os.path.isfile(os.path.join(d, "manifest.json"))

    def get(self, key: str) -> StoreEntry:
        d = self._entry_dir(key)
        mpath = os.path.join(d, "manifest.json")
        if not os.path.isfile(mpath):
            raise StoreError(f"no store entry for key {key[:12]}…")
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
        return StoreEntry(key=key, dir=d, payload_dir=os.path.join(d, "payload"), manifest=manifest)

    def put(
        self,
        key: str,
        writer: Callable[[str], None],
        meta: dict[str, Any] | None = None,
    ) -> StoreEntry:
        """Create an entry atomically.

        ``writer(payload_dir)`` writes all payload files into the given dir.
        If the key already exists the existing entry wins (idempotent).
        """
        if self.has(key):
            return self.get(key)
        final_dir = self._entry_dir(key)
        parent = os.path.dirname(final_dir)
        os.makedirs(parent, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=f".tmp-{key[:8]}-", dir=parent)
        try:
            payload = os.path.join(tmp, "payload")
            os.makedirs(payload, exist_ok=True)
            writer(payload)
            manifest = {"key": key, "meta": meta or {}}
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
                f.write(canonical_json(manifest))
            try:
                os.rename(tmp, final_dir)
            except OSError:
                # Lost a race to a concurrent producer — their entry is equivalent.
                if not self.has(key):
                    raise
                shutil.rmtree(tmp, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self.get(key)

    def get_or_create(
        self,
        key: str,
        writer: Callable[[str], None],
        meta: dict[str, Any] | None = None,
    ) -> tuple[StoreEntry, bool]:
        """Return ``(entry, cache_hit)``."""
        if self.has(key):
            return self.get(key), True
        return self.put(key, writer, meta), False
