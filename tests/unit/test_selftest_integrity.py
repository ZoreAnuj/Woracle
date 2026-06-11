"""C-1 regression: every minted negative is evaluated as a DISTINCT episode."""

from __future__ import annotations

from woracle.compiler import compile_spec, mint_negatives
from woracle.grounders.relational import RelationalMotionGrounder
from woracle.testing.blobworld import make_episode


def test_every_negative_evaluated_distinctly(monkeypatch) -> None:
    demos = [make_episode("success", seed=s)[0] for s in (0, 1, 2)]
    n_negatives = len(mint_negatives(demos[0])) * 2  # compiler mints from 2 demos

    seen_tags = set()
    real_ground = RelationalMotionGrounder.ground

    def counting_ground(self, rollout, spec, out_dir):
        seen_tags.add(rollout.id)
        return real_ground(self, rollout, spec, out_dir)

    monkeypatch.setattr(RelationalMotionGrounder, "ground", counting_ground)
    spec = compile_spec(demos, "put the object in the container")
    neg_tags = {t for t in seen_tags if "neg" in t}
    demo_tags = {t for t in seen_tags if "demo" in t}
    assert len(neg_tags) == n_negatives, (
        f"only {len(neg_tags)}/{n_negatives} negatives were actually evaluated — "
        "provenance counts must be measured, never aggregated from aliased entries"
    )
    assert len(demo_tags) == 3
    st = spec.spec_provenance.self_test
    assert st.negatives_total == n_negatives and st.negatives_failed == n_negatives
