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

> ⚠️ **Status: P0 (foundations).** The kernel, contracts, gate/grading pipeline, and the
> blobworld test world are real; real-video grounders (P1), VLM channels (P3), the spec
> compiler (P4), and calibrated statistics (P5) are landing phase by phase.
> Not yet on PyPI. APIs will move.

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

spec  = woracle.load_spec("specs/blobworld-insert/spec.yaml")
cards = woracle.grade("rollouts/", spec, out_dir="out")    # grade cards (snapshots)
board = woracle.report(cards, out_path="leaderboard.md")   # abstain-aware leaderboard
# woracle.compile(demos, prompt)  ->  P4
# woracle.ground(rollouts, spec)  ->  P1
```

## Design

- `WM_EVAL_TOOLKIT_PROPOSAL.md` — research map (~140 works) + pipeline design
- `WM_EVAL_TOOLKIT_ARCH.md` — architecture decisions, with receipts
- Kernel rule: `import woracle` pulls numpy+pydantic only (CI-enforced); heavy stacks live
  behind extras `[ground] [track] [judge] [3d]` and load lazily at call time.

## License

Apache-2.0.
