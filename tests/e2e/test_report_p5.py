"""P5 e2e: full honest report (markdown + HTML + PPI + MNAR + rank intervals)."""

from __future__ import annotations

import json
import os

import woracle


def test_full_report_with_golds_and_html(blob_dataset: str, tmp_path) -> None:
    out = str(tmp_path / "out")
    cards = woracle.grade(blob_dataset, os.path.join(blob_dataset, "spec.yaml"), out_dir=out)
    golds = {c.rollout_id: c.rollout_id.startswith("success") for c in cards}
    gpath = str(tmp_path / "golds.json")
    with open(gpath, "w") as f:
        json.dump(golds, f)

    md_path = str(tmp_path / "report.md")
    html_path = str(tmp_path / "report.html")
    board = woracle.report(out + "/cards", md_path, golds=gpath, html_path=html_path)

    with open(md_path) as f:
        md = f.read()
    assert "Abstention sensitivity" in md and "MNAR" in md
    assert "PPI-rectified" in md
    assert "ranking robust to abstention imputation" in md
    with open(html_path) as f:
        html = f.read()
    assert "<html" in html and board.spec_name in html
    assert "Grade cards" in html and "abstain" in html
    assert "polyline" in html  # sparkline rendered from snapshot series
    # vanish rollout appears with abstain verdict — never silently dropped
    assert "vanish_00" in html
