# alakazam-mira-mini

Play **MIRA Mini**, a neural world model of car soccer, locally on your own GPU.
Every frame is generated live by the model. After the first weight download,
everything runs on your machine; there is no cloud dependency and no account.

```
pip install alakazam-mira-mini
mira-mini play
```

Useful flags: `--interp` (2x display interpolation), `--steps N`, `--no-fast`, `--verbose`.

MIRA Mini is our from-scratch reproduction of the MIRA recipe
([General Intuition](https://www.generalintuition.com/) × [Kyutai](https://kyutai.org/),
with Epic Games), compressed until it runs on consumer hardware: fewer diffusion steps,
a smaller student model, and a compact decoder. Measurements and method are in the
[technical report](https://alakazam.gg/mira-mini).

## What you need

- An NVIDIA GPU (CUDA), or a Mac with Apple silicon (M1 or newer). CPU-only machines
  are not supported; generation is too slow to play.
- Disk for the weights, downloaded once from Hugging Face: ~5 GB for the 364M model,
  ~12 GB for the 1B.
- The weight repositories are public on Hugging Face
  ([alakazamworld](https://huggingface.co/alakazamworld)); the first run downloads them
  automatically.

## Picking a model

`mira-mini play` chooses weights for your machine: **CUDA gets the 1B**, **Apple silicon
gets the 364M laptop tier** (an MLX transformer + Core ML decoder, ~8 fps on a 2021 M1 Pro).
Override it:

```
mira-mini play --model 1b     # the 1B single-player model (needs a discrete GPU)
mira-mini play --model 364m   # the laptop tier, anywhere
```

## Options

| flag / env | effect |
|---|---|
| `--model {auto,1b,364m}` | which weights to run (default: auto, by device) |
| `--steps N` | sampler steps; 2 is the steadier default, 1 is smoother but drifts more |
| `--port N` | web UI port (default 8770) |
| `--no-browser` | don't open the browser automatically |
| `MIRA_HF_REPO` | use a custom Hugging Face weights repo |
| `MIRA_HOME` | where bundles are cached (default `~/.cache/alakazam-mira`) |
| `MIRA_DEVICE` | force `cuda` / `mps` / `cpu` |

## Weights and license

Model weights live on Hugging Face under
[alakazamworld](https://huggingface.co/alakazamworld) and are **CC BY-NC-SA 4.0**,
inherited from the training dataset
([kyutai/rocket-science](https://huggingface.co/datasets/kyutai/rocket-science), Rocket
League content used with Epic Games' permission). Non-commercial, share-alike, with
attribution. The model is a research demonstration; long rollouts drift from exact
physics.

## Credits

The architecture, training recipe, and dataset are General Intuition's and Kyutai's,
released openly with Epic Games ([mira-wm/mira](https://github.com/mira-wm/mira)).
MIRA Mini is Alakazam's independent reproduction and compression of that work. The
weights are an independent release by Alakazam: not released by, associated with, or
endorsed by General Intuition, Kyutai, or Epic Games.


## 0.1.3

Packaging fix: 0.1.1 and 0.1.2 wheels were missing the engine (`mira_vm`), the room relay, and the vendored `mira` inference runtime, so `mira-mini play` crashed with ModuleNotFoundError after downloading weights. 0.1.3 ships all of them. Thanks to the first player who reported it.

## 0.1.4

Local-play polish from a live rehearsal: the access-key prompt no longer appears (the local relay never checked it; the page now pre-seeds it), and few-step bundles (364m, psd) default to 2 diffusion steps via the engine's hard override, so `mira-mini play` hits its rated frame rate without flags.

## 0.1.5

The access-key bypass now targets the store the UI actually reads (sessionStorage) and the play URL carries `key=local`, so the prompt is gone on every browser and cache state.

## 0.1.6

The Apple fast stack is now automatic: on a Mac with the 364m bundle, `mira-mini play` wires the MLX transformer (whole-step compiled), the Core ML decoder (pipelined in a child process), and 2x display interpolation, roughly doubling the delivered frame rate; `--no-fast` restores plain torch. The CLI shows staged loading and opens the browser only when the model is ready, and the room switches to the play screen only after the first generated frame arrives. Best served by python 3.12 (coremltools has no 3.13+ bindings yet; without them the fast stack silently falls back to torch).

## 0.1.7

The play URL is a clickable terminal hyperlink (OSC 8).

## 0.1.8

The room start handshake retries with backoff instead of failing once when the engine slot is briefly held by a previous session or an idle-closed socket.

## 0.1.9

The banner shows the installed version.

## 0.1.10

Animated loading spinner with elapsed time; engine logs are hidden by default (`--verbose` restores them).
