#!/usr/bin/env bash
# End-to-end: install deps, generate data, train, sample. Override via env vars.
set -euo pipefail
cd "$(dirname "$0")"
NUM_SAMPLES=${NUM_SAMPLES:-50000}
RES=${RES:-128}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-64}
LR=${LR:-2e-4}
PY=${PY:-python3}

$PY -m pip install --upgrade torch numpy pillow opencv-python-headless
$PY generate_data.py --num_samples "$NUM_SAMPLES" --output_dir data --resolution "$RES"
$PY train.py --data_dir data --checkpoint_dir checkpoints --batch_size "$BATCH" --lr "$LR" --epochs "$EPOCHS"
$PY sample.py --prompt "a red smiley face with 'HELLO' next to it" --checkpoint checkpoints/latest.pt \
    --cfg_scale 5.0 --steps 50 --seed 0 --resolution "$RES" --out sample_red_smiley.png
$PY sample.py --prompt "a blue sad face with 'NEWS' next to it" --checkpoint checkpoints/latest.pt \
    --cfg_scale 5.0 --steps 50 --seed 1 --resolution "$RES" --out sample_blue_sad.png
