#!/usr/bin/env bash
# End-to-end quick start on a laptop: keygen -> generate -> embed -> merge -> verify.
# ~30 min on an Apple-silicon Mac for Qwen2.5-0.5B-Instruct (see README for details).
set -euo pipefail
BASE="${BASE:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN="${RUN:-runs/quickstart}"
mkdir -p "$RUN"

ldw keygen --owner 1 --recipient 42 --out "$RUN/wm.keyfile.json" --no-registry
ldw generate --keyfile "$RUN/wm.keyfile.json" --out "$RUN/train.jsonl" --reg-samples 800
ldw embed --data "$RUN/train.jsonl" --base "$BASE" --out "$RUN/adapter" \
    --batch-size 4 --grad-accum 4 --max-len 384 --gradient-checkpointing
ldw merge --base "$BASE" --adapter "$RUN/adapter" --out "$RUN/merged"
ldw verify --keyfile "$RUN/wm.keyfile.json" --model "$RUN/merged" --paraphrase heldout --out "$RUN/verify.json"
