# probe_eval — physics verification for the MIRA Mini compression ladder

The evaluation suite used to check that compressing MIRA Mini (1.18B teacher →
364M student, fewer diffusion steps) preserved the model's grasp of game physics,
not just its pixels. It extends the campaign in §17.1 of the
[technical report](https://alakazam.gg/mira-mini/report.pdf) with a
shared-instrument probe ladder, action-recoverability checks, long-rollout
state-drift analysis, and a multiplayer cross-view consistency probe.

## The idea

A probe (ridge regression or small MLP) learns to read the ball's position
straight out of a model's activations while the model **watches** real matches.
Then the model **imagines** the match on its own and we read again with the same
frozen probe. If imagining barely hurts the readout, the model's internal
game state survived — whatever happened to the pixels.

## Headline results

| measurement | teacher 1.18B | student 364M |
|---|---|---|
| state readout error, watching (uu, ridge) | 1846 | 2000 |
| state readout error, imagining | 2234 | 2507 |
| degradation watch → imagine | **+21%** | **+25%** |

- The student loses no more physics than its teacher: compression shrank the
  model, not its understanding of the game. (1 uu = 1 cm.)
- Before few-step self-distillation the student degraded **+49%**
  (1989 → 2964 uu); the distillation stage repaired rollout physics to teacher
  level, not just sampling speed.
- Control: a probe trained on shuffled targets reads 3906 uu (chance level);
  real activations read 1303 uu. A linear readout on the raw codec latents the
  model consumes largely fails (2461 uu) — the forward pass computes game state.
- Action recoverability (ARR): base 0.944 vs fleet 0.943; PSD teacher @1-step
  0.950; student @1-step 0.946 — no action-conditioning loss at any rung.
- Multiplayer cross-view: probed states of the four per-player views agree at
  2 s rollouts on the release freeze (context ≡ floor; no gross divergence).

## Running it

The suite runs on top of the upstream [mira-wm/mira](https://github.com/mira-wm/mira)
codebase (release `6aae8d3`) — it imports the `mira` package for data loading —
with MIRA Mini checkpoints from [Hugging Face](https://huggingface.co/alakazamworld).

```
# alongside an editable install of upstream mira and its deps
python -m probe_eval.train_probe --help      # train a state probe on activations
python -m probe_eval.rollout_eval --help     # watch-vs-imagine ladder
probe_eval/run_long_rollouts.sh              # long-horizon drift
python -m probe_eval.mp_crossview --help     # 4-player view consistency
```

Protocols follow the report's §6.2 evaluation conventions; deviations (context
length, probe generations, seed matching) are documented in the script headers.
The ARR and idle-input A/B protocols ran on the project's `physics_eval` harness,
which is archived with the campaign artifacts rather than shipped here; the raw
run data (per-clip CI ladders, trained probes, latent caches, rollout dumps) is
likewise archived in the project's diagnostics bundle.
