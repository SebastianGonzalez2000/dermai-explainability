#!/usr/bin/env bash
# Run the four surgical fine-tuning arms one at a time so they do not contend for the GPU.

set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "train.py --config configs/efficientnet.yaml" > /dev/null; do sleep 30; done

for arm in efficientnet_surgical_early efficientnet_surgical_late vit_surgical_early vit_surgical_late; do
    echo "=== ${arm} starting $(date +%H:%M:%S) ==="
    python -u train.py --config "configs/${arm}.yaml" > "outputs/train_${arm}.log" 2>&1
    echo "=== ${arm} exited $? at $(date +%H:%M:%S) ==="
    tail -2 "outputs/train_${arm}.log"
done
