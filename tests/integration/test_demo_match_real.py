"""Slow lane: success.demo_match actually separates success from failure
on blobworld via DINOv2 — detection-free (downloads facebook/dinov2-small)."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.enable_socket]
pytest.importorskip("transformers", reason="requires the [ground]/[embed] extra")


def test_demo_match_separates_success_from_failure(tmp_path) -> None:
    from woracle.channels.demo_match import DemoMatchSuccessChannel, build_demo_protos
    from woracle.io import load_rollout, save_episode
    from woracle.testing.blobworld import blob_spec, make_episode
    from woracle.testing.plugins import BlobColorGrounder

    demos = []
    for s in range(3):
        demos.append((make_episode("success", seed=s)[0], True))
        demos.append((make_episode("fail_miss", seed=s)[0], False))
        demos.append((make_episode("fail_drop", seed=s)[0], False))
    sp, fp = build_demo_protos(demos, tail_frac=0.25, max_frames=6)
    assert sp.shape[0] == 3 and fp.shape[0] == 6

    chan = DemoMatchSuccessChannel(success_protos=sp, fail_protos=fp, temp=0.05)
    spec = blob_spec()

    def margin(kind: str, seed: int) -> float:
        frames, _ = make_episode(kind, seed=seed)
        ep = str(tmp_path / f"{kind}{seed}")
        save_episode(ep, f"{kind}{seed}", frames, source="blob")
        out = str(tmp_path / f"g{kind}{seed}")
        os.makedirs(out, exist_ok=True)
        g = BlobColorGrounder().ground(load_rollout(ep), spec, out)
        sc = chan.score(g, spec)
        assert sc.status == "ok"
        return sc.details["margin"]

    m_succ = margin("success", 9)
    m_miss = margin("fail_miss", 9)
    m_drop = margin("fail_drop", 9)
    # held-out success leans success (margin>0), failures lean fail (margin<0)
    assert m_succ > m_miss and m_succ > m_drop, (m_succ, m_miss, m_drop)
    assert m_succ > 0 > max(m_miss, m_drop)


def test_demo_match_without_protos_is_evidence_missing(tmp_path) -> None:
    from woracle.channels.demo_match import DemoMatchSuccessChannel
    from woracle.io import load_rollout, save_episode
    from woracle.testing.blobworld import blob_spec, make_episode
    from woracle.testing.plugins import BlobColorGrounder

    frames, _ = make_episode("success", seed=0)
    ep = str(tmp_path / "e")
    save_episode(ep, "e", frames, source="blob")
    out = str(tmp_path / "g")
    os.makedirs(out, exist_ok=True)
    g = BlobColorGrounder().ground(load_rollout(ep), blob_spec(), out)
    sc = DemoMatchSuccessChannel().score(g, blob_spec())
    assert sc.status == "evidence_missing" and "prototype" in sc.reason
