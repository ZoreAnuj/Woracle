"""Self-contained HTML report rendered FROM snapshots (evidently model).

Zero recomputation, zero JS dependencies: inline CSS + inline SVG sparklines
built from ChannelScore.series. Input = GradeCards (+ optional stats blocks);
output = one portable .html file.
"""

from __future__ import annotations

import html as _html

from woracle.contracts import GradeCard, Leaderboard

_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem;color:#222;max-width:1100px}
table{border-collapse:collapse;margin:1rem 0;width:100%}
td,th{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;font-size:.92rem}
th{background:#f5f5f5}
.pass{color:#0a7d36;font-weight:600}.fail{color:#b3261e;font-weight:600}
.abstain{color:#8a6d00;font-weight:600}
.note{background:#fff8e1;border-left:4px solid #f1c40f;padding:.5rem .8rem;margin:.6rem 0}
.small{font-size:.8rem;color:#666}
"""


def _spark(series: list[float], w: int = 140, h: int = 28) -> str:
    if len(series) < 2:
        return ""
    lo, hi = min(series), max(series)
    rng = (hi - lo) or 1.0
    pts = " ".join(
        f"{i * w / (len(series) - 1):.1f},{h - 2 - (v - lo) / rng * (h - 4):.1f}"
        for i, v in enumerate(series)
    )
    return (
        f'<svg width="{w}" height="{h}"><polyline points="{pts}" '
        'fill="none" stroke="#4078c0" stroke-width="1.5"/></svg>'
    )


def render_html(
    board: Leaderboard,
    cards: list[GradeCard],
    *,
    stats_blocks: dict[str, str] | None = None,
) -> str:
    e = _html.escape
    rows = []
    for p in board.policies:
        pr = "—" if p.pass_rate_on_graded is None else f"{p.pass_rate_on_graded:.2f}"
        chans = ", ".join(f"{k}={v:.3f}" for k, v in p.mean_channel_values.items()) or "—"
        rows.append(
            f"<tr><td>{e(p.policy)}</td><td>{p.n_rollouts}</td><td>{p.n_gradeable}</td>"
            f"<td>{p.n_abstained}</td><td>{pr}</td><td class=small>{e(chans)}</td></tr>"
        )
    card_rows = []
    for c in sorted(cards, key=lambda c: (c.policy, c.rollout_id)):
        spark = ""
        for ch in c.channels:
            if ch.series.get("progress"):
                spark = _spark(ch.series["progress"])
                break
        chans = "; ".join(
            f"{ch.channel}:{'—' if ch.value is None else f'{ch.value:.2f}'}({ch.status})"
            for ch in c.channels
        )
        reasons = "; ".join(c.success.reasons + c.gate.reasons)
        card_rows.append(
            f"<tr><td>{e(c.rollout_id)}</td><td>{e(c.policy)}</td>"
            f"<td class={c.success.verdict}>{c.success.verdict}</td>"
            f"<td>{e(c.gate.verdict)}</td><td>{spark}</td>"
            f"<td class=small>{e(chans)}</td><td class=small>{e(reasons[:240])}</td></tr>"
        )
    notes = "".join(f"<div class=note>{e(n)}</div>" for n in board.notes)
    stats_html = ""
    for title, block in (stats_blocks or {}).items():
        stats_html += f"<h2>{e(title)}</h2><pre class=small>{e(block)}</pre>"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>woracle — {e(board.spec_name)}</title><style>{_CSS}</style></head><body>
<h1>Woracle report — {e(board.spec_name)}</h1>
<p class=small>spec hash {e(board.spec_hash[:16])}… · woracle {e(board.provenance.package_version)}</p>
{notes}
<h2>Leaderboard</h2>
<table><tr><th>policy</th><th>rollouts</th><th>graded</th><th>abstained</th>
<th>pass-rate (graded only)</th><th>mean channel values</th></tr>{"".join(rows)}</table>
{stats_html}
<h2>Grade cards</h2>
<table><tr><th>rollout</th><th>policy</th><th>success</th><th>gate</th><th>progress</th>
<th>channels</th><th>reasons</th></tr>{"".join(card_rows)}</table>
</body></html>"""
