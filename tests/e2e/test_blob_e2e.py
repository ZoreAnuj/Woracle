"""P0 exit gate: end-to-end semantics on blobworld.

success -> PASS, misses/drops -> FAIL, vanish -> ABSTAIN (gate, with reason),
success outranks failure on the progress channel, caching works, manifest +
cards land on disk and reload cleanly.
"""

from __future__ import annotations

import json
import os

import woracle
from woracle.contracts import GradeCard, RunManifest, migrate
from woracle.report import build_leaderboard, render_markdown


def _by_id(cards: list[GradeCard]) -> dict[str, GradeCard]:
    return {c.rollout_id: c for c in cards}


def test_e2e_semantics(blob_dataset: str, tmp_path) -> None:
    out = str(tmp_path / "out")
    cards = woracle.grade(blob_dataset, os.path.join(blob_dataset, "spec.yaml"), out_dir=out)
    by = _by_id(cards)
    assert len(cards) == 5

    # — verdict semantics —
    assert by["success_00"].success.verdict == "pass"
    assert by["success_00"].gate.verdict == "gradeable"
    assert by["fail_miss_00"].success.verdict == "fail"
    assert by["fail_drop_00"].success.verdict == "fail"
    assert by["random_00"].success.verdict in ("fail", "abstain")

    # — the honesty centerpiece: vanish must ABSTAIN via the permanence gate —
    vanish = by["vanish_00"]
    assert vanish.gate.verdict == "ungradeable"
    assert vanish.success.verdict == "abstain"
    assert any("permanence" in r for r in vanish.gate.reasons)
    # abstained rollouts still produce full cards with skipped channels
    assert all(s.status == "skipped" for s in vanish.channels)

    # — predicate attribution on the failure —
    miss = by["fail_miss_00"]
    assert any("contained(carried_object, receptacle)" in v for v in miss.success.violated)

    # — ranking signal: success outranks failures on progress —
    def prog(c: GradeCard) -> float:
        ch = c.channel("progress.goal_distance")
        assert ch is not None and ch.value is not None
        return ch.value

    assert prog(by["success_00"]) > prog(by["fail_miss_00"])
    assert prog(by["success_00"]) > prog(by["fail_drop_00"])
    assert prog(by["success_00"]) > 0.9

    # — artifacts on disk: cards reload, manifest is valid —
    card_files = os.listdir(os.path.join(out, "cards"))
    assert len(card_files) == 5
    with open(os.path.join(out, "cards", "success_00.json"), encoding="utf-8") as f:
        data = migrate("GradeCard", json.load(f))
    reloaded = GradeCard.model_validate(data)
    assert reloaded.spec_hash == by["success_00"].spec_hash
    assert reloaded.provenance.components  # component versions recorded

    with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
        manifest = RunManifest.model_validate(migrate("RunManifest", json.load(f)))
    assert manifest.package_version == woracle.__version__
    ground_stages = [s for s in manifest.stages if s.stage == "ground"]
    assert len(ground_stages) == 5
    assert not any(s.cache_hit for s in ground_stages)  # cold run


def test_e2e_cache_hits_on_rerun(blob_dataset: str, tmp_path) -> None:
    out1, out2 = str(tmp_path / "o1"), str(tmp_path / "o2")
    store = str(tmp_path / "shared_store")
    spec_path = os.path.join(blob_dataset, "spec.yaml")
    woracle.grade(blob_dataset, spec_path, out_dir=out1, store_root=store)
    cards2 = woracle.grade(blob_dataset, spec_path, out_dir=out2, store_root=store)
    with open(os.path.join(out2, "manifest.json"), encoding="utf-8") as f:
        manifest = RunManifest.model_validate(migrate("RunManifest", json.load(f)))
    ground_stages = [s for s in manifest.stages if s.stage == "ground"]
    assert all(s.cache_hit for s in ground_stages), "second run must hit the grounder cache"
    assert len(cards2) == 5


def test_leaderboard_is_abstain_aware(blob_dataset: str, tmp_path) -> None:
    out = str(tmp_path / "out")
    cards = woracle.grade(blob_dataset, os.path.join(blob_dataset, "spec.yaml"), out_dir=out)
    board = build_leaderboard(cards)
    rows = {p.policy: p for p in board.policies}
    assert rows["vanish"].n_abstained == 1 and rows["vanish"].pass_rate_on_graded is None
    assert rows["success"].pass_rate_on_graded == 1.0
    assert rows["fail_miss"].pass_rate_on_graded == 0.0
    assert board.policies[0].policy == "success"  # ranked first
    md = render_markdown(board)
    assert "abstained" in md and "NOT a calibrated" in "".join(board.notes)
