# The grounding bottleneck and the generalizable fix

## Diagnosis — why woracle abstained on the pipette WM test

It was an **architecture** problem, not just a perception weakness.

- woracle's only `verdict_eligible` channel was **`success.predicates`**, which
  evaluates geometric predicates over role *tracks*.
- The pipette spec's success = `co_located(carried_object, receptacle)` +
  `stationary(carried_object)` — **both need the manipulated object's track**.
- The grounder (`openvocab.gdino_sam`) reaches that object by **GroundingDINO
  text-prompting it by name** ("white pipette tip"). On a ~10px object in a
  cluttered scene this fails (absent-object confidence inversion; sub-resolution
  even at 2× tiling) — the binding study already measured this.
- With the object un-tracked there was **no other path to a verdict** → abstain
  everywhere. One fragile, appearance-keyed detector gated the entire oracle.

## What the literature says the generalizable approach is (2024–2026)

The field judges success/progress **without detecting the manipulated object**:

| Method | Signal | Evidence | Detection-free? |
|---|---|---|---|
| TOPReward (arXiv 2602.19313) | VLM token-logit progress | **0.947 VOC**, 130+ tasks, Franka/YAM/SO-100; GVL≈0 on same open model | yes |
| GVL (ICLR'25) | shuffled-frame VLM in-context value | zero-shot 300+ tasks, 0.71 success-detect | yes |
| Robometer (github, **MIT**, 4B on HF) | video+text → per-frame progress+success+preference; RBM-1M (1M trajs *with failures*) | "highly generalizable" | yes |
| RoboReward 4B/8B (2601.00675, CC-BY) | VLM reward on OXE+RoboArena, failures via counterfactual relabeling | beats Gemini-ER | yes |
| DINOv2 demo/goal-frame matching (Apache) | final-frame embedding vs demo success/fail | task-agnostic, generalizes | yes |
| TMSP (2412.19112) | success from **end-effector trajectory** + frame + instruction | open-vocab | tracks the robot, not the object |

**Principle:** never make detecting the manipulated object the *precondition* for
a verdict. Judge completion from a learned reward/progress model, demo-embedding
matching, and/or the **robust anchors** (effector + receptacle). Object grounding
becomes an optional confidence booster — not the gate.

## The fix shipped in woracle

1. **`success.demo_match`** (new, verdict-eligible, **DINOv2 Apache**, no API,
   no detection): margin = sim(rollout-tail, success protos) − sim(rollout-tail,
   fail protos), prototypes built from the few labeled demos the pipeline already
   ingests. This is woracle's own contrastive-exemplar judge, reborn as a
   first-class channel. (Validated on blobworld: held-out success margin +0.0085
   above every failure, correct sign, zero detection.)
2. **Drop-missing verdict ensemble**: a verdict-eligible channel that *cannot*
   judge (object-grounded predicates when the object is un-detectable) is
   **dropped**, not a global veto — so the detection-free channel decides. Abstain
   only if **no** channel can judge, or if those that can **disagree** (conflict →
   never guess). Single-channel behavior and verdict-isolation are preserved.
3. **`frames.passthrough` grounder**: the no-grounding entry point for the
   object-free path (binds nothing; frame-level signals + demo_match still work).
4. Roadmap (same pattern, drop-in): wrap **Robometer-4B (MIT)** / **RoboReward**
   as `success.reward_model`; promote the existing GVL channel to
   verdict-eligible-when-calibrated (optionally TOPReward token-logit upgrade);
   let specs mark the manipulated object *optional* and judge over effector +
   receptacle (TMSP).

## The re-test (what this validates)

Instead of WM rollouts, grade **real RH20T success vs failure dataset episodes**
with the object-free oracle. Expectation: `success.demo_match` separates success
from failure (AUROC) where the old tip-grounding path abstained on both. Scripts:
`export_rh20t_episodes.py` (local, rating-labeled) + `retest_real_episodes.py`.
