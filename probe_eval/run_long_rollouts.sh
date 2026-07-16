#!/usr/bin/env bash
# ABOUTME: Long-rollout latent generation across the rung ladder (Hugo protocol, no image-FDD:
# ABOUTME: state-space analysis runs offline; image verdicts come from his archived JSONs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$ROOT/.venv/bin/python
OUT=$ROOT/runs/long_rollout/latents
cd "$ROOT"

echo "=== smoke: base rung 0.5 min 1 ctx ==="
PYTHONPATH=$ROOT/src:$ROOT:. $PY -m probe_eval.long_rollout_gen --rung base1b_8s --steps 10 \
  --minutes 0.5 --contexts 1 --out "$OUT.smoke"

for spec in "base1b_8s 10" "psd1b_2s 2" "student_2s 2" "student_2s 1"; do
  read -r rung steps <<< "$spec"
  echo "=== $rung @ $steps steps: 3 min x 3 contexts ==="
  PYTHONPATH=$ROOT/src:$ROOT:. $PY -m probe_eval.long_rollout_gen --rung "$rung" --steps "$steps" \
    --minutes 3 --contexts 3 --out "$OUT"
done
echo "=== long-rollout generation complete ==="
