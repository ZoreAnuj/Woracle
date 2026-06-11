# Woracle

**The world-model oracle. Demos in, oracle out.**

World models are blind simulators: a policy rollout is just generated pixels — no object
poses, no contact flags, no `success()`. Today every WM-based policy evaluation hand-rolls a
per-task success detector (a VLM prompt, a color mask, a thousand hand labels), measured in
the literature at 65–80% accuracy, with no abstention and no statistics.

Woracle is an **evaluation compiler**: it compiles a portable, scene-invariant task spec from
a prompt and a few demonstrations, then grades policy rollouts from **any** world model
against it — with calibrated honesty (it abstains when a rollout is ungradeable, e.g. when
the WM deleted the object) and statistics that treat abstention as information.

> **Status: 0.1 pre-release.** All six stages are implemented and tested: real
> open-vocab grounding (GroundingDINO+SAM) with motion-signature verification,
> a calibratable validity gate (AUROC ≥ 0.8 on the generative-corruption
> benchmark, enforced in CI), progress/phase/trajectory/TL-DTMC channels, a
> GVL VLM judge protocol, the demos→spec COMPILER with self-test/REFUSE
> (cross-scene transfer demonstrated), and PPI/MNAR honesty statistics.
> Validated on real Cosmos-3 WM rollouts (see `studies/binding/REPORT.md`).
> Not yet on PyPI (pending); APIs may still move before 0.1.

## The shape

```
prompt + K demos ──▶ S1 COMPILE ──▶ task spec (YAML: roles, phases, predicates)
                                          │
rollouts from ANY world model ──▶ S2 GROUND ──▶ S3 GATE ──▶ S4 GRADE ──▶ S5 STATS
                                  bind roles    abstain if   channels +   honest
                                  to pixels     ungradeable  predicates   rankings
```

- **Specs store the task, never the scene** — roles are relational ("the thing that co-moves
  with the gripper"), so one spec grades pipette-into-holder *and* USB-into-port.
- **Evidence failure is data, not an exception** — "the object vanished" becomes an abstain
  with a reason, logged per policy (informative missingness), never a crash or a silent drop.
- **Models run once; everything downstream is pure** — gate/channels/stats replay on CPU from
  cached artifacts.

## Try it (blobworld, no GPU, no checkpoints)

```bash
uv sync --group test          # or: pip install -e . --group test
woracle demo --out blob_demo
woracle grade --rollouts blob_demo --spec blob_demo/spec.yaml --out out
woracle report --cards out/cards --out leaderboard.md
woracle doctor
```

`blob_demo` contains a success episode, two failure modes, a *vanish* episode (the WM
"deleted" the object — woracle must abstain, and does), and a random policy.

## Library API (the four verbs)

```python
import woracle

# the four verbs — all real
spec  = woracle.compile("demos/", "insert the tip into the holder",
                        out="specs/mytask/spec.yaml")       # self-tested, or REFUSEs
bundles = woracle.ground("rollouts/", spec)                  # bind roles to pixels
cards = woracle.grade("rollouts/", spec, out_dir="out")      # gate -> channels -> verdicts
board = woracle.report("out/cards", "leaderboard.md",
                       golds="golds.json",                   # optional: PPI rectification
                       html_path="report.html")              # abstain-aware, MNAR-bounded
```

Real-video grounding (`pip install 'woracle[ground]'`):
GroundingDINO-tiny + SAM via transformers (Apache models), detection-linked
tracking with motion-signature verification — binding-study-honest: detector
confidence is documented as content-blind on generated video; geometry checks
catch what confidence cannot (8/8 false latches in the study).

## Design & evidence

- `studies/binding/REPORT.md` — field-first measurements of open-vocab binding
  on real WM rollouts (confidence inversion, false-latch detection, anchor drift)
- `docs/PLUGINS.md` — plugin-author guide + conformance suite
- Design docs (research map ~140 works, architecture decisions with receipts)
  live in the companion workspace; ask for `WM_EVAL_TOOLKIT_PROPOSAL.md` /
  `WM_EVAL_TOOLKIT_ARCH.md`
- Kernel rule: `import woracle` pulls numpy+pydantic only (CI-enforced); heavy stacks live
  behind extras `[ground] [track] [judge] [3d]` and load lazily at call time.

## License

Apache-2.0.
