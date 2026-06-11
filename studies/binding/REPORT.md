# Binding Study ① — Open-vocab grounding on WM-generated rollouts

*2026-06-11 · woracle P1 de-risk study · 8 rollouts (2 per policy × {act_010000,
act_050000, smolvla, random}), Cosmos-3 Nano FD pipette task, 1200 frames each,
RTX 4070 laptop. Grounder: GroundingDINO-tiny + SAM-vit-base (transformers,
Apache). Runs: A = stride 8, tiles 1 (v0.1.0); B = stride 8, tiles 2 +
motion-signature verification (v0.2.0). Raw data: `out/study.json`,
`out_tiled/study.json`; every claim inspectable in `*_overlay.mp4`.*

**Context.** No published work measures detect/segment/track reliability under
GENERATIVE failure modes (drift, morphing, vanishing) — robustness suites use
photometric corruptions on real video. These rollouts begin from a REAL
observation (frame-0 anchor) and drift into fully generated frames, giving a
within-video real-vs-generated comparison with no external labels.

## Findings

**F1 — Bind rate and detector confidence are vanity metrics on WM rollouts.**
Run A: 100% bind rate on every role, every policy — including the random
policy's heavily drifted videos (quality 1.0 for tip/holder; 0.973–0.997 for
gripper). Mean per-window visibility (det confidence) stayed flat across the
2-minute horizon — carried_object 0.53–0.64, receptacle 0.32–0.44, gripper
0.24–0.36 — with real→generated drop ≈ 0 (aggregate range −0.025…+0.054). The detector "succeeds" identically on real frames,
coherent generated frames, and badly drifted frames. Confidence measures
nothing about task-relevant content here (consistent with the measured
absent-object inversion: 0.606 absent vs 0.511 present on our probe).

**F2 — The tiny carried object is a FALSE LATCH; geometry catches what
confidence cannot.** Overlay inspection (run A): the "white pipette tip"
track sits on static background clutter, jitter p90 ≈ 0.12 px — for a role
whose spec motion is `co_moves_with_effector`. Run B's motion-signature
verification flags **8/8** tip bindings MOTION-INCONSISTENT (quality 1.0 →
0.25, with the reason recorded in the binding). Downstream effect: the gate
degrades/abstains on tip-dependent verdicts instead of grading garbage —
the honest behavior the toolkit promises.

**F3 — 2× tiling did not rescue the tip.** The ~10 px tip in a 3-pane concat
at 320×180 stays below GroundingDINO-tiny's discrimination floor even
upsampled 2×; it keeps preferring background "white box" lookalikes. Tiny-
object binding needs a stronger detector, ROI re-detection around the
effector, or point-tracker seeding (P-next work) — tiling alone is not the
published cure here.

**F4 — The large static anchor binds correctly; its jitter carries drift
signal, with caveats.** The green holder binds correctly (overlay-inspected;
overlays reproducible via `run_study.py`, not tracked in git). Holder jitter
p90 per rollout (run A): act_010000 = 0.06/0.05, act_050000 = 0.07/**0.85**,
smolvla = 0.07/0.51, random = 0.14/**2.36** px. The extreme is the random
policy, but act_050000_init_001's 0.85 px outlier breaks any clean per-rollout
policy ordering — anchor jitter is a DRIFT observable, not a policy ranker.
Run B's static-role wander flag fired on 2/8 rollouts (act_050000_init_001,
random_init_000) and notably NOT on random_init_001 (the 2.36 px case — its
wander stayed under the 25%-of-diagonal flag threshold): flag coverage is
partial and thresholds need calibration (P2's job).

## Implications for woracle (already implemented)

1. `RoleBinding.quality` documents itself as detection-RATE only — never
   phrase fidelity (F1).
2. Motion-signature verification ships in the grounder (v0.2.0) and feeds
   `binding_health` → gate (F2).
3. Tiny-object strategy is an open roadmap item with measured evidence, not a
   silent failure (F3).
4. Anchor-jitter is a drift observable available to gate signals (F4;
   `track_continuity` signal).

## Honesty box

Label-free observables only: no per-frame human boxes exist for these
rollouts, so this study measures binding BEHAVIOR (rates, confidence
trajectories, motion consistency, overlay-verified latch identity on sampled
frames), not box-IoU accuracy. n=8 rollouts (deeper sweep is a rerun flag
away: `--per-policy 13` covers all 54). Frame-0 "real" windows are single-
real-frame anchors; the first ~5% window may include early coherent generated
frames at this fps.
