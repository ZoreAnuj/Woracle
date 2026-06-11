"""Pipeline orchestrator: ground -> gate -> grade, with caching + provenance.

Flow per rollout (ARCH §3):
1. GROUND (model stage, cached): grounder writes sidecars into the content
   store, keyed by (frames digest, spec hash, grounder@version).
2. GATE (pure): signals measure the grounded bundle; policy composes verdict.
3. GRADE (pure): channels score; verdict-eligible channels decide pass/fail;
   an ungradeable gate yields ABSTAIN with the gate's reasons.

Honesty invariants enforced here:
* abstained rollouts produce a full GradeCard (never silently dropped);
* only ``caps.verdict_eligible`` channels touch the success verdict;
* every card embeds spec hash + component versions; the run writes a manifest.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field

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
    migrate,
)
from woracle.errors import PluginError
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
    seeds: dict[str, int] = field(default_factory=dict)


BLOB_PROFILE = GradeRunConfig(
    grounder="blob.color",
    signals=["binding_health", "permanence", "motion_sanity"],
    channels=["progress.goal_distance", "success.predicates"],
)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _ground_cached(
    store: ContentStore, grounder, rollout: RolloutRef, spec: TaskSpec
) -> tuple[GroundedRollout, str, bool]:
    key = store.key_for(
        stage="ground",
        component=grounder.name,
        component_version=grounder.version,
        inputs={"frames": rollout.frames.sha256, "spec": spec.content_hash()},
        params={},
    )

    def writer(payload_dir: str) -> None:
        grounder.ground(rollout, spec, payload_dir)

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


def _success_from_channels(channels: list, scores: list) -> SuccessReport:
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
        if sc.status != "ok" or sc.value is None:
            any_missing = True
            reasons.append(f"{sc.channel}: {sc.status} ({sc.reason})")
            continue
        for k, v in sc.details.items():
            if k.startswith("pred:"):
                (satisfied if v >= 0.5 else violated).append(k[len("pred:") :])
        if sc.value < 0.5:
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
    store = ContentStore(config.store_root)
    os.makedirs(os.path.join(config.out_dir, "cards"), exist_ok=True)

    grounder = reg_get("grounder", config.grounder)()
    signal_objs = [reg_get("gate_signal", s)() for s in config.signals]
    channel_objs = [reg_get("channel", c)() for c in config.channels]

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
            "spec": {"name": spec.name, "hash": spec.content_hash(), "version": spec.version},
        },
        inputs={r.id: r.frames.sha256 for r in rollouts},
    )

    cards: list[GradeCard] = []
    for rollout in rollouts:
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
                stage="gate", component="compose_gate", wall_s=round(time.perf_counter() - t1, 4)
            )
        )

        t2 = time.perf_counter()
        if gate.verdict == "ungradeable":
            from woracle.contracts import ChannelScore

            scores = [
                ChannelScore(channel=c.name, status="skipped", reason="gate verdict: ungradeable")
                for c in channel_objs
            ]
            success = SuccessReport(
                verdict="abstain",
                reasons=[f"gate: {r}" for r in gate.reasons] or ["gate: ungradeable"],
            )
        else:
            scores = [c.score(grounded, spec) for c in channel_objs]
            success = _success_from_channels(channel_objs, scores)
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
        with open(
            os.path.join(config.out_dir, "cards", f"{rollout.id}.json"), "w", encoding="utf-8"
        ) as f:
            f.write(card.to_json())

    with open(os.path.join(config.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
    return cards
