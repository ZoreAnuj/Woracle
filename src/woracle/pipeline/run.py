"""Pipeline orchestrator: ground -> gate -> grade, with caching + provenance.

Flow per rollout (ARCH §3):
1. GROUND (model stage, cached): grounder writes sidecars into the content
   store, keyed by (frames digest, spec hash, grounder@version).
2. GATE (pure): signals measure the grounded bundle; policy composes verdict.
3. GRADE (pure): channels score; verdict-eligible channels decide pass/fail;
   an ungradeable gate yields ABSTAIN with the gate's reasons.

Honesty invariants enforced here:
* abstained rollouts produce a full GradeCard (never silently dropped);
* only ``caps.verdict_eligible`` channels touch the success verdict, and a
  non-finite "value" from one is treated as missing evidence, never a pass;
* a failed rollout never kills the run silently: other rollouts complete,
  the manifest records the failure, and the run raises at the end;
* every card embeds spec hash + component versions; the run writes a manifest.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, replace

from woracle._version import __version__
from woracle.contracts import (
    GatePolicy,
    GradeCard,
    GroundedRollout,
    Provenance,
    RolloutRef,
    RunManifest,
    StageRecord,
    SuccessReport,
    TaskSpec,
    digest_file,
    migrate,
)
from woracle.errors import PluginError, StoreError
from woracle.gate import DEFAULT_POLICY, compose_gate
from woracle.registry import get as reg_get
from woracle.store import ContentStore


@dataclass
class GradeRunConfig:
    grounder: str
    signals: list[str]
    channels: list[str]
    policy: GatePolicy = field(default_factory=lambda: DEFAULT_POLICY.model_copy(deep=True))
    store_root: str = ".woracle_store"
    out_dir: str = "woracle_out"
    # Verdict mapping for verdict-eligible channel values (P3 replaces this
    # scalar with a proper Condition object; it must NOT live in channel code).
    pass_threshold: float = 0.5
    # Constructor kwargs per component name (e.g. demo tracks for the DTW
    # channel, a VLM backend for progress.gvl). NOT part of cache keys —
    # components expose cache-relevant params via their `params` property.
    component_params: dict = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)

    def fresh(self, **overrides: object) -> GradeRunConfig:
        """A copy safe to mutate (configs are treated as immutable templates).

        Explicit overrides always win; only non-overridden mutable fields are
        deep-copied from the template (NEW-3: never silently discard caller
        intent).
        """
        cfg = replace(self, **overrides)  # type: ignore[arg-type]
        if "component_params" not in overrides:
            cfg.component_params = dict(self.component_params)
        if "policy" not in overrides:
            cfg.policy = self.policy.model_copy(deep=True)
        if "signals" not in overrides:
            cfg.signals = list(self.signals)
        if "channels" not in overrides:
            cfg.channels = list(self.channels)
        if "seeds" not in overrides:
            cfg.seeds = dict(self.seeds)
        return cfg


def blob_profile() -> GradeRunConfig:
    """Fresh blobworld profile (factory — never a shared mutable global)."""
    return GradeRunConfig(
        grounder="blob.color",
        signals=["binding_health", "permanence", "motion_sanity"],
        channels=["progress.goal_distance", "success.predicates"],
    )


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


_FNAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _card_filename(rollout_id: str) -> str:
    return _FNAME_SAFE.sub("_", rollout_id) + ".json"


def _frames_digest(rollout: RolloutRef) -> str:
    """Trustworthy frames digest (C1): never an empty string.

    Prefer the recorded digest; recompute from the payload when absent. An
    empty digest must never reach the cache key — it would alias every such
    rollout onto one entry and grade rollout A with rollout B's pixels.
    """
    if rollout.frames.sha256:
        return rollout.frames.sha256
    base = rollout.meta.get("_dir", "")
    path = rollout.frames.resolve(base) if base else rollout.frames.path
    if not os.path.isfile(path):
        raise StoreError(
            f"rollout '{rollout.id}': frames digest is empty and payload "
            f"({path}) is not readable — cannot build a safe cache key"
        )
    return digest_file(path)


def _ground_cached(
    store: ContentStore, grounder, rollout: RolloutRef, spec: TaskSpec
) -> tuple[GroundedRollout, str, bool]:
    key = store.key_for(
        stage="ground",
        component=grounder.name,
        component_version=grounder.version,
        inputs={"frames": _frames_digest(rollout), "spec": spec.content_hash()},
        params=dict(getattr(grounder, "params", {})),
    )

    def writer(payload_dir: str) -> None:
        grounder.ground(rollout, spec, payload_dir)
        # Validate BEFORE the store commits the entry (I4): a grounder that
        # forgot its contract must not poison the cache permanently.
        gpath = os.path.join(payload_dir, "grounded.json")
        if not os.path.isfile(gpath):
            raise PluginError(
                f"grounder '{grounder.name}' violated its contract: no grounded.json "
                "written to out_dir (entry not committed)"
            )

    entry, hit = store.get_or_create(
        key, writer, meta={"stage": "ground", "rollout": rollout.id, "spec": spec.name}
    )
    gpath = os.path.join(entry.payload_dir, "grounded.json")
    with open(gpath, encoding="utf-8") as f:
        data = json.load(f)
    data = migrate("GroundedRollout", data)
    grounded = GroundedRollout.model_validate(data)
    grounded.bundle_dir = entry.payload_dir
    # The cached bundle was produced for a frames digest; the live RolloutRef
    # may carry fresher meta (e.g. _dir) — keep the live one.
    grounded.rollout = rollout
    return grounded, key, hit


def _success_from_channels(channels: list, scores: list, pass_threshold: float) -> SuccessReport:
    eligible = [
        (ch, sc)
        for ch, sc in zip(channels, scores, strict=True)
        if getattr(ch, "caps", None) is not None and ch.caps.verdict_eligible
    ]
    if not eligible:
        return SuccessReport(
            verdict="abstain",
            reasons=["no verdict-eligible channel produced a score"],
        )
    satisfied: list[str] = []
    violated: list[str] = []
    reasons: list[str] = []
    any_missing = False
    all_pass = True
    for _ch, sc in eligible:
        usable = sc.status == "ok" and sc.value is not None and math.isfinite(sc.value)
        if not usable:
            any_missing = True
            reasons.append(f"{sc.channel}: {sc.status} ({sc.reason or 'no usable value'})")
            continue
        for k, v in sc.details.items():
            if k.startswith("pred:"):
                (satisfied if v >= 0.5 else violated).append(k[len("pred:") :])
        if sc.value < pass_threshold:  # type: ignore[operator]
            all_pass = False
    if any_missing:
        return SuccessReport(
            verdict="abstain", satisfied=satisfied, violated=violated, reasons=reasons
        )
    return SuccessReport(
        verdict="pass" if all_pass else "fail",
        satisfied=satisfied,
        violated=violated,
        reasons=reasons,
    )


def grade_rollouts(
    rollouts: list[RolloutRef],
    spec: TaskSpec,
    config: GradeRunConfig,
) -> list[GradeCard]:
    if not rollouts:
        raise PluginError("no rollouts to grade")
    dupes = sorted({r.id for r in rollouts if sum(1 for x in rollouts if x.id == r.id) > 1})
    if dupes:
        raise PluginError(
            f"duplicate rollout ids: {', '.join(dupes)} — ids must be unique within a run "
            "(cards are keyed by id; duplicates would silently overwrite)"
        )
    seen_files: dict[str, str] = {}
    for r in rollouts:
        fn = _card_filename(r.id)
        if fn in seen_files:
            raise PluginError(
                f"rollout ids '{seen_files[fn]}' and '{r.id}' collide after filename "
                f"sanitization ('{fn}') — rename one"
            )
        seen_files[fn] = r.id
    store = ContentStore(config.store_root)
    os.makedirs(os.path.join(config.out_dir, "cards"), exist_ok=True)

    cp = config.component_params
    grounder = reg_get("grounder", config.grounder)(**cp.get(config.grounder, {}))
    signal_objs = [reg_get("gate_signal", s)(**cp.get(s, {})) for s in config.signals]
    channel_objs = [reg_get("channel", c)(**cp.get(c, {})) for c in config.channels]

    components = {
        f"grounder:{grounder.name}": grounder.version,
        **{f"signal:{s.name}": s.version for s in signal_objs},
        **{f"channel:{c.name}": c.version for c in channel_objs},
    }

    manifest = RunManifest(
        run_id=uuid.uuid4().hex[:12],
        created_at=_now_iso(),
        package_version=__version__,
        python=sys.version.split()[0],
        platform=platform.platform(),
        seeds=config.seeds,
        config={
            "grounder": config.grounder,
            "signals": config.signals,
            "channels": config.channels,
            "policy": config.policy.model_dump(),
            "pass_threshold": config.pass_threshold,
            "spec": {"name": spec.name, "hash": spec.content_hash(), "version": spec.version},
        },
    )

    cards: list[GradeCard] = []
    failures: dict[str, str] = {}
    try:
        for rollout in rollouts:
            try:
                manifest.inputs[rollout.id] = _frames_digest(rollout)
                t0 = time.perf_counter()
                grounded, key, hit = _ground_cached(store, grounder, rollout, spec)
                manifest.stages.append(
                    StageRecord(
                        stage="ground",
                        component=f"{grounder.name}@{grounder.version}",
                        cache_key=key,
                        cache_hit=hit,
                        wall_s=round(time.perf_counter() - t0, 4),
                    )
                )

                t1 = time.perf_counter()
                sig_values = [s.measure(grounded) for s in signal_objs]
                gate = compose_gate(grounded, sig_values, config.policy)
                manifest.stages.append(
                    StageRecord(
                        stage="gate",
                        component="compose_gate",
                        wall_s=round(time.perf_counter() - t1, 4),
                    )
                )

                t2 = time.perf_counter()
                if gate.verdict == "ungradeable":
                    from woracle.contracts import ChannelScore

                    scores = [
                        ChannelScore(
                            channel=c.name, status="skipped", reason="gate verdict: ungradeable"
                        )
                        for c in channel_objs
                    ]
                    success = SuccessReport(
                        verdict="abstain",
                        reasons=[f"gate: {r}" for r in gate.reasons] or ["gate: ungradeable"],
                    )
                else:
                    scores = [c.score(grounded, spec) for c in channel_objs]
                    success = _success_from_channels(channel_objs, scores, config.pass_threshold)
                manifest.stages.append(
                    StageRecord(
                        stage="grade",
                        component=",".join(config.channels),
                        wall_s=round(time.perf_counter() - t2, 4),
                    )
                )

                card = GradeCard(
                    rollout_id=rollout.id,
                    policy=rollout.policy,
                    spec_name=spec.name,
                    spec_version=spec.version,
                    spec_hash=spec.content_hash(),
                    gate=gate,
                    channels=scores,
                    success=success,
                    provenance=Provenance(
                        package_version=__version__,
                        created_at=_now_iso(),
                        components=components,
                    ),
                )
                cards.append(card)
                card_path = os.path.join(config.out_dir, "cards", _card_filename(rollout.id))
                with open(card_path, "w", encoding="utf-8") as f:
                    f.write(card.to_json())
            except Exception as e:
                failures[rollout.id] = f"{type(e).__name__}: {e}"
                manifest.notes.append(f"FAILED {rollout.id}: {failures[rollout.id]}")
    finally:
        # The manifest is written even if the run is about to raise — partial
        # results must remain auditable.
        with open(os.path.join(config.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            f.write(manifest.to_json())

    if failures:
        raise PluginError(
            f"{len(failures)}/{len(rollouts)} rollouts failed (cards for the rest were "
            f"written to {config.out_dir}/cards): "
            + "; ".join(f"{k} -> {v}" for k, v in failures.items())
        )
    return cards
