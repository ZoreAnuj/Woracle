"""Real open-vocabulary grounder: GroundingDINO (detect) + SAM (segment).

Models (both Apache-2.0, both transformers-native — no research-repo vendoring,
per ARCH anti-pattern 5):
* ``IDEA-Research/grounding-dino-tiny`` — zero-shot text-prompted detection
* ``facebook/sam-vit-base``            — box-prompted segmentation

Tracking is detection-based: detect on every ``stride``-th frame with ONE
multi-phrase prompt for all roles, link per role with IoU/teleport-leash logic
(pure functions in ``linking.py``), segment the chosen boxes, interpolate
tracks between samples (never beyond the observed span).

Known-limits honesty (recorded in binding reasons / signals, not hidden):
detector confidence on ABSENT objects can exceed true positives — MEASURED on
this exact stack (2026-06-11, blobworld probe): absent "purple elephant"
mean score 0.606 vs present "red block" 0.511. NO confidence threshold can
reject absent objects here; `RoleBinding.quality` is therefore detection-RATE
only and must never be read as phrase fidelity. The mitigation is appearance
consistency along the track (P2 gate signal), per the toolkit's own research
(GroundingDINO absent-object FPs outscore TPs; arXiv 2406.19057).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from woracle.contracts import (
    ArtifactRef,
    GroundedRollout,
    RoleBinding,
    RolloutRef,
    TaskSpec,
    digest_file,
)
from woracle.errors import InfraError, MissingDependencyError
from woracle.grounders.linking import (
    LinkState,
    center,
    dense_visibility,
    interpolate_track,
    motion_consistency,
    select_detection,
)
from woracle.io import load_frames
from woracle.registry import register

DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam-vit-base"


def _import_stack() -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )
    except ImportError as e:
        raise MissingDependencyError("open-vocabulary grounding", "ground") from e
    return dict(
        torch=torch,
        Image=Image,
        AutoProcessor=AutoProcessor,
        AutoModelForZeroShotObjectDetection=AutoModelForZeroShotObjectDetection,
        SamModel=SamModel,
        SamProcessor=SamProcessor,
    )


@register("grounder", "openvocab.gdino_sam")
class OpenVocabGrounder:
    """Bind spec roles to arbitrary video via open-vocab detection + SAM."""

    name = "openvocab.gdino_sam"
    version = "0.2.0"

    def __init__(
        self,
        stride: int = 5,
        det_threshold: float = 0.25,
        text_threshold: float = 0.20,
        iou_weight: float = 0.3,
        max_jump_frac: float = 0.35,
        segment_every: int = 1,  # segment every k-th SAMPLED frame
        tiles: int = 1,  # NxN tiled detection (small objects below detector floor)
        batch_size: int = 4,
        device: str | None = None,
        detector_id: str = DETECTOR_ID,
        segmenter_id: str = SEGMENTER_ID,
    ) -> None:
        self.stride = int(stride)
        self.det_threshold = float(det_threshold)
        self.text_threshold = float(text_threshold)
        self.iou_weight = float(iou_weight)
        self.max_jump_frac = float(max_jump_frac)
        self.segment_every = int(segment_every)
        self.tiles = int(tiles)
        self.batch_size = int(batch_size)
        self.device = device
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self._stack: dict[str, Any] | None = None

    # Params participate in the ground-stage cache key (store.key_for).
    @property
    def params(self) -> dict[str, Any]:
        return {
            "stride": self.stride,
            "det_threshold": self.det_threshold,
            "text_threshold": self.text_threshold,
            "iou_weight": self.iou_weight,
            "max_jump_frac": self.max_jump_frac,
            "segment_every": self.segment_every,
            "tiles": self.tiles,
            "detector": self.detector_id,
            "segmenter": self.segmenter_id,
        }

    # -- model loading (lazy, cached on the instance) -------------------------
    def _ensure_models(self) -> dict[str, Any]:
        if self._stack is not None:
            return self._stack
        st = _import_stack()
        torch = st["torch"]
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            det_proc = st["AutoProcessor"].from_pretrained(self.detector_id)
            det = (
                st["AutoModelForZeroShotObjectDetection"]
                .from_pretrained(self.detector_id)
                .to(device)
                .eval()
            )
            sam_proc = st["SamProcessor"].from_pretrained(self.segmenter_id)
            sam = st["SamModel"].from_pretrained(self.segmenter_id).to(device).eval()
        except (OSError, ValueError) as e:
            raise InfraError(f"failed to load grounding models: {e}") from e
        self._stack = dict(
            torch=torch,
            Image=st["Image"],
            device=device,
            det_proc=det_proc,
            det=det,
            sam_proc=sam_proc,
            sam=sam,
        )
        return self._stack

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _role_phrases(spec: TaskSpec) -> dict[str, list[str]]:
        return {
            r.name: [c.strip().lower() for c in (r.candidates or [r.name.replace("_", " ")])]
            for r in spec.roles
        }

    def _detect_batch(self, st: dict[str, Any], images: list, prompt: str) -> list[dict]:
        torch = st["torch"]
        inputs = st["det_proc"](images=images, text=[prompt] * len(images), return_tensors="pt")
        inputs = {k: v.to(st["device"]) for k, v in inputs.items()}
        with torch.no_grad():
            out = st["det"](**inputs)
        target_sizes = [img.size[::-1] for img in images]  # (h, w)
        return st["det_proc"].post_process_grounded_object_detection(
            out,
            inputs["input_ids"],
            threshold=self.det_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

    def _detect_frames(
        self, st: dict[str, Any], images: list, prompt: str, hw: tuple[int, int]
    ) -> list[dict]:
        """Per-frame detections; with tiles>1 each frame is split into an NxN
        grid (12% overlap), detected per tile, and boxes shifted back."""
        if self.tiles <= 1:
            out: list[dict] = []
            for b0 in range(0, len(images), self.batch_size):
                out.extend(self._detect_batch(st, images[b0 : b0 + self.batch_size], prompt))
            return out
        H, W = hw
        n = self.tiles
        oy, ox = int(H / n * 0.12), int(W / n * 0.12)
        windows: list[tuple[int, int, int, int]] = []
        for r in range(n):
            for c in range(n):
                y0 = max(0, int(r * H / n) - oy)
                y1 = min(H, int((r + 1) * H / n) + oy)
                x0 = max(0, int(c * W / n) - ox)
                x1 = min(W, int((c + 1) * W / n) + ox)
                windows.append((y0, y1, x0, x1))
        tile_imgs = []
        for img in images:
            arr = np.asarray(img)
            for y0, y1, x0, x1 in windows:
                tile_imgs.append(st["Image"].fromarray(arr[y0:y1, x0:x1]))
        flat: list[dict] = []
        for b0 in range(0, len(tile_imgs), self.batch_size):
            flat.extend(self._detect_batch(st, tile_imgs[b0 : b0 + self.batch_size], prompt))
        merged: list[dict] = []
        k = len(windows)
        for fi in range(len(images)):
            boxes: list[list[float]] = []
            scores: list[float] = []
            labels: list[str] = []
            for wi, (y0, _y1, x0, _x1) in enumerate(windows):
                det = flat[fi * k + wi]
                tl = det.get("text_labels", det.get("labels", []))
                for b, sc, lb in zip(det["boxes"], det["scores"], tl, strict=False):
                    bb = [float(v) for v in b.tolist()]
                    boxes.append([bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0])
                    scores.append(float(sc))
                    labels.append(str(lb))
            import torch as _t

            merged.append(
                {
                    "boxes": _t.tensor(boxes) if boxes else _t.zeros((0, 4)),
                    "scores": _t.tensor(scores) if scores else _t.zeros(0),
                    "text_labels": labels,
                }
            )
        return merged

    def _segment(self, st: dict[str, Any], image, box: np.ndarray) -> np.ndarray:
        torch = st["torch"]
        inputs = st["sam_proc"](image, input_boxes=[[[float(b) for b in box]]], return_tensors="pt")
        inputs = {k: v.to(st["device"]) for k, v in inputs.items()}
        with torch.no_grad():
            out = st["sam"](**inputs, multimask_output=False)
        masks = st["sam_proc"].image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return np.asarray(masks[0][0, 0].numpy(), dtype=bool)

    # -- main ------------------------------------------------------------------
    def ground(self, rollout: RolloutRef, spec: TaskSpec, out_dir: str) -> GroundedRollout:
        st = self._ensure_models()
        Image = st["Image"]
        frames = load_frames(rollout)  # (T, H, W, 3) uint8
        T, H, W = frames.shape[:3]
        diag = float(np.hypot(H, W))
        sample_idxs = np.arange(0, T, self.stride)
        images = [Image.fromarray(frames[i]) for i in sample_idxs]

        phrases = self._role_phrases(spec)
        # ONE combined prompt; GroundingDINO convention: lowercase, period-separated.
        all_phrases: list[tuple[str, str]] = [(role, p) for role, ps in phrases.items() for p in ps]
        prompt = ". ".join(p for _, p in all_phrases) + "."

        # Detect on all sampled frames (batched), optionally tiled: small
        # objects sit below the detector's resolution floor at native size
        # (binding-study F2) — tiles crop+upsample implicitly.
        detections = self._detect_frames(st, images, prompt, (H, W))

        bindings: list[RoleBinding] = []
        for role in spec.roles:
            want = set(phrases[role.name])
            S = len(sample_idxs)
            centers = np.full((S, 2), np.nan, np.float32)
            boxes_kept: list[np.ndarray | None] = [None] * S
            scores_kept = np.zeros(S, np.float32)
            state = LinkState()
            for j, det in enumerate(detections):
                labels = [str(lbl).lower() for lbl in det.get("text_labels", det.get("labels", []))]
                keep = [
                    k for k, lbl in enumerate(labels) if any(wp in lbl or lbl in wp for wp in want)
                ]
                boxes = np.zeros((0, 4), np.float32)
                scores = np.zeros(0, np.float32)
                if keep:
                    boxes = np.asarray([det["boxes"][k].tolist() for k in keep], np.float32)
                    scores = np.asarray([float(det["scores"][k]) for k in keep], np.float32)
                    pick = select_detection(
                        boxes,
                        scores,
                        state,
                        det_threshold=self.det_threshold,
                        iou_weight=self.iou_weight,
                        max_jump_frac=self.max_jump_frac,
                        image_diag=diag,
                    )
                else:
                    pick = None
                if pick is None:
                    state.misses += 1
                    continue
                k, score = pick
                state.box = boxes[k]
                state.misses = 0
                boxes_kept[j] = boxes[k]
                scores_kept[j] = score
                centers[j] = center(boxes[k])

            n_obs = int(np.isfinite(centers[:, 0]).sum())
            if n_obs == 0:
                bindings.append(
                    RoleBinding(
                        role=role.name,
                        bound=False,
                        required=role.required,
                        reason=f"no detection above {self.det_threshold} for {sorted(want)}",
                    )
                )
                continue

            # Segment the confirmed boxes (every k-th confirmed sample).
            mask_stack = np.zeros((S, H, W), np.uint8)
            seg_count = 0
            for j in range(S):
                bk = boxes_kept[j]
                if bk is None or (j % self.segment_every) != 0:
                    continue
                try:
                    mask_stack[j] = self._segment(st, images[j], bk).astype(np.uint8)
                    seg_count += 1
                except InfraError:
                    raise
                except Exception as e:
                    # Segmentation crashes are machinery failures (OOM, CUDA,
                    # library bugs) — surface as retryable InfraError; silently
                    # degrading quality would mislabel infra as evidence (I-4).
                    raise InfraError(
                        f"SAM segmentation failed at sample {j}: {type(e).__name__}: {e}"
                    ) from e

            # Motion-signature verification (binding-study F3): geometry
            # catches false latches that confidence cannot (measured inversion).
            consistent, rng_px = motion_consistency(centers, role.motion, diag)
            quality = float(n_obs / S)
            reason = f"{n_obs}/{S} samples detected, {seg_count} segmented, range={rng_px:.0f}px"
            if not consistent:
                quality *= 0.25
                reason = (
                    f"MOTION-INCONSISTENT: expected '{role.motion}' but track range "
                    f"{rng_px:.0f}px ({100 * rng_px / diag:.1f}% of diag) — likely "
                    f"false latch onto wrong object; " + reason
                )

            track = interpolate_track(sample_idxs, centers, T)
            vis = dense_visibility(sample_idxs, scores_kept, T)

            tpath = os.path.join(out_dir, f"{role.name}.track.npz")
            mpath = os.path.join(out_dir, f"{role.name}.mask.npz")
            vpath = os.path.join(out_dir, f"{role.name}.vis.npz")
            np.savez_compressed(tpath, track=track)
            np.savez_compressed(mpath, mask=mask_stack, sample_idxs=sample_idxs)
            np.savez_compressed(vpath, vis=vis)
            bindings.append(
                RoleBinding(
                    role=role.name,
                    bound=True,
                    required=role.required,
                    quality=quality,
                    reason=reason,
                    tracks=ArtifactRef(
                        path=os.path.basename(tpath), sha256=digest_file(tpath), kind="track.npz"
                    ),
                    masks=ArtifactRef(
                        path=os.path.basename(mpath), sha256=digest_file(mpath), kind="mask.npz"
                    ),
                    visibility=ArtifactRef(
                        path=os.path.basename(vpath), sha256=digest_file(vpath), kind="vis.npz"
                    ),
                )
            )

        grounded = GroundedRollout(
            rollout=rollout,
            spec_name=spec.name,
            spec_hash=spec.content_hash(),
            bindings=bindings,
            grounder=f"{self.name}@{self.version}",
            bundle_dir=out_dir,
        )
        with open(os.path.join(out_dir, "grounded.json"), "w", encoding="utf-8") as f:
            f.write(grounded.to_json())
        return grounded
