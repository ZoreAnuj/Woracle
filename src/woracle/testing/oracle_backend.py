"""TEST-ONLY VLM backend: a blobworld progress oracle.

This is test machinery, clearly and loudly: it answers the GVL prompt by
READING BLOBWORLD GEOMETRY (red-square distance to the cup interior) from the
images it is shown. It exists to test the GVL channel's MACHINERY —
shuffling, prompt formatting, reply parsing, unshuffling, VOC, reshuffle
agreement — against ground truth, with zero network and zero model weights.

It is NOT a judge, NOT registered as a component, and lives in
``woracle.testing`` so no production path can pick it up implicitly.
"""

from __future__ import annotations

import numpy as np

from woracle.testing.blobworld import INTERIOR


class BlobProgressOracleBackend:
    name = "test.blob_oracle"
    version = "0.1.0"

    def __init__(self, noise: float = 0.0, seed: int = 0, style: str = "frame_lines") -> None:
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.style = style

    @staticmethod
    def _progress_of(image: np.ndarray) -> float:
        lo, hi = np.array([150, 0, 0]), np.array([255, 90, 90])
        m = np.all((image >= lo) & (image <= hi), axis=-1)
        if not m.any():
            return 0.0
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean(), ys.mean()
        tx = (INTERIOR[0] + INTERIOR[2]) / 2.0
        ty = (INTERIOR[1] + INTERIOR[3]) / 2.0
        d = float(np.hypot(cx - tx, cy - ty))
        d0 = 110.0  # typical start distance in blobworld
        return float(np.clip(1.0 - d / d0, 0.0, 1.0))

    def complete(self, images: list[np.ndarray], prompt: str) -> str:
        assert "RANDOM order" in prompt, "oracle backend expects the GVL prompt"
        # images[0] is the anchor; the rest are the shuffled frames
        lines = []
        for i, img in enumerate(images[1:], start=1):
            p = self._progress_of(img)
            if self.noise:
                p = float(np.clip(p + self.rng.normal(0, self.noise), 0.0, 1.0))
            if self.style == "frame_lines":
                lines.append(f"Frame {i}: {round(p * 100)}%")
            else:  # messy free-text variant for parser robustness tests
                lines.append(f"I'd say roughly {round(p * 100)}% done here.")
        return "\n".join(lines)
