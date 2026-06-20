"""success.demo_match — detection-free, generalizable success verdict.

The fix for the un-groundable-object failure mode: judge task completion by
EMBEDDING SIMILARITY to labeled demonstrations, never by detecting the
manipulated object. margin = sim(rollout_tail, success_protos) -
sim(rollout_tail, fail_protos), using a frozen DINOv2 encoder (Apache-2.0).

This is the contrastive-exemplar judge from woracle's own success-judge
research, reborn as a first-class verdict-eligible channel. It needs only the
prompt + a few labeled demos the pipeline already ingests — no per-task
tuning, no object detector, no clean scene. Backed by the literature on
object-free robot success judging (GVL / TOPReward / Robometer / RoboReward).

Model-in-channel is the same pattern as the GVL progress channel: the encoder
loads lazily at call time behind the [ground] extra; the channel records
evidence-missing (never crashes) when it cannot embed.
"""

from __future__ import annotations

import numpy as np

from woracle.contracts import ChannelCaps, ChannelScore, GroundedRollout, TaskSpec
from woracle.errors import MissingDependencyError
from woracle.registry import register

_ENCODER_CACHE: dict[str, object] = {}


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_RES = 224  # 224/14 = 16 DINOv2 patches


def _load_encoder(model_id: str, device: str | None = None):
    key = f"{model_id}@{device}"
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    try:
        import torch
        from transformers import AutoModel
    except ImportError as e:
        raise MissingDependencyError("demo-match embedding", "ground") from e
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Manual ImageNet preprocessing (PIL + numpy) — avoids the torchvision
    # dependency AutoImageProcessor pulls in; portable and light.
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    enc: tuple = (model, dev, torch)
    _ENCODER_CACHE[key] = enc
    return enc


def _preprocess(frames: np.ndarray) -> np.ndarray:
    from PIL import Image

    out = np.empty((len(frames), _RES, _RES, 3), np.float32)
    for i, f in enumerate(frames):
        img = Image.fromarray(f).convert("RGB").resize((_RES, _RES), Image.Resampling.BILINEAR)
        out[i] = (np.asarray(img, np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return out.transpose(0, 3, 1, 2)  # (B,3,H,W)


def embed_frames(
    frames: np.ndarray,
    model_id: str = "facebook/dinov2-small",
    device: str | None = None,
    batch: int = 16,
) -> np.ndarray:
    """Embed (T,H,W,3) uint8 frames -> (T, D) L2-normalized CLS embeddings."""
    model, dev, torch = _load_encoder(model_id, device)  # type: ignore[misc]
    pix = _preprocess(frames)
    out = []
    with torch.no_grad():
        for i in range(0, len(pix), batch):
            t = torch.from_numpy(pix[i : i + batch]).to(dev)
            feats = model(pixel_values=t).last_hidden_state[:, 0]  # CLS token
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def _tail_embedding(
    frames: np.ndarray, tail_frac: float, max_frames: int, model_id: str, device: str | None
) -> np.ndarray:
    T = len(frames)
    lo = max(0, int(T * (1.0 - tail_frac)))
    idx = np.unique(np.linspace(lo, T - 1, min(max_frames, T - lo)).astype(int))
    emb = embed_frames(frames[idx], model_id=model_id, device=device)
    v = emb.mean(0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v


def build_demo_protos(
    demos: list[tuple[np.ndarray, bool]],
    *,
    tail_frac: float = 0.2,
    max_frames: int = 8,
    model_id: str = "facebook/dinov2-small",
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """[(frames, is_success)] -> (success_protos (Ns,D), fail_protos (Nf,D)).

    One prototype per demo = mean tail embedding. Prototype DIVERSITY matters
    (the original judge research: 1+1 protos can invert) — pass several of each.
    """
    succ, fail = [], []
    for frames, ok in demos:
        v = _tail_embedding(frames, tail_frac, max_frames, model_id, device)
        (succ if ok else fail).append(v)
    return (
        np.stack(succ) if succ else np.zeros((0, 0), np.float32),
        np.stack(fail) if fail else np.zeros((0, 0), np.float32),
    )


@register("channel", "success.demo_match")
class DemoMatchSuccessChannel:
    name = "success.demo_match"
    version = "0.1.0"
    caps = ChannelCaps(
        reference_free=False,
        needs_tracks=False,
        needs_masks=False,
        verdict_eligible=True,
        value_range=(0.0, 1.0),
    )

    def __init__(
        self,
        success_protos: np.ndarray | list | None = None,
        fail_protos: np.ndarray | list | None = None,
        tail_frac: float = 0.2,
        max_frames: int = 8,
        temp: float = 0.05,
        model_id: str = "facebook/dinov2-small",
        device: str | None = None,
    ) -> None:
        self.success_protos = (
            np.asarray(success_protos, np.float32) if success_protos is not None else None
        )
        self.fail_protos = np.asarray(fail_protos, np.float32) if fail_protos is not None else None
        self.tail_frac = float(tail_frac)
        self.max_frames = int(max_frames)
        self.temp = float(temp)
        self.model_id = model_id
        self.device = device

    @property
    def params(self) -> dict:
        # prototype CONTENT is part of the cache key (re-banking changes verdicts)
        def _digest(a):
            return (
                ""
                if a is None or a.size == 0
                else __import__("hashlib")
                .sha256(np.ascontiguousarray(a).tobytes())
                .hexdigest()[:16]
            )

        return {
            "tail_frac": self.tail_frac,
            "max_frames": self.max_frames,
            "temp": self.temp,
            "model_id": self.model_id,
            "succ_protos": _digest(self.success_protos),
            "fail_protos": _digest(self.fail_protos),
        }

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore:
        if (
            self.success_protos is None
            or self.fail_protos is None
            or self.success_protos.size == 0
            or self.fail_protos.size == 0
        ):
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="demo_match needs both success and failure prototype banks "
                "(pass success_protos/fail_protos built from labeled demos)",
            )
        from woracle.io import load_frames

        frames = load_frames(grounded.rollout)  # infra failure propagates (I-4)
        if len(frames) < 2:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason=f"rollout too short ({len(frames)} frames)",
            )
        q = _tail_embedding(frames, self.tail_frac, self.max_frames, self.model_id, self.device)
        sim_s = float(np.mean(self.success_protos @ q))
        sim_f = float(np.mean(self.fail_protos @ q))
        margin = sim_s - sim_f
        value = float(1.0 / (1.0 + np.exp(-margin / self.temp)))
        return ChannelScore(
            channel=self.name,
            value=value,
            confidence=float(min(1.0, abs(margin) / self.temp / 4.0)),
            reason="" if abs(margin) > 0.5 * self.temp else "success/fail prototypes nearly tied",
            details={
                "sim_success": round(sim_s, 4),
                "sim_fail": round(sim_f, 4),
                "margin": round(margin, 4),
            },
        )
