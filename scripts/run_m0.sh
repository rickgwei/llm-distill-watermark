#!/usr/bin/env bash
# End-to-end M0 pipeline: keygen -> generate -> embed -> merge -> verify -> controls -> attacks.
# Usage: scripts/run_m0.sh [BASE_MODEL] [RUN_DIR]
set -euo pipefail
BASE=${1:-Qwen/Qwen2.5-0.5B-Instruct}
RUN=${2:-runs/m0}
export PYTHONUNBUFFERED=1
mkdir -p "$RUN"

[ -f "$RUN/wm.keyfile.json" ] || ldw keygen --owner 1 --recipient 42 --out "$RUN/wm.keyfile.json" --registry "$RUN/registry.json"
[ -f "$RUN/train.jsonl" ]     || ldw generate --keyfile "$RUN/wm.keyfile.json" --out "$RUN/train.jsonl" --reg-samples 800

ldw embed --data "$RUN/train.jsonl" --base "$BASE" --out "$RUN/adapter" \
  --epochs 3 --batch-size 4 --grad-accum 4 --max-len 384 --gradient-checkpointing
ldw merge --base "$BASE" --adapter "$RUN/adapter" --out "$RUN/merged"

ldw verify --keyfile "$RUN/wm.keyfile.json" --model "$BASE"                    --out "$RUN/verify_base.json" --quiet
ldw verify --keyfile "$RUN/wm.keyfile.json" --model "$RUN/merged" --paraphrase heldout --out "$RUN/verify_merged_heldout.json" --quiet
ldw controls --model "$BASE"         --keyfile "$RUN/wm.keyfile.json" --out "$RUN/controls_base.json"
ldw controls --model "$RUN/merged"   --keyfile "$RUN/wm.keyfile.json" --out "$RUN/controls_merged.json"

ldw attack dilute    --keyfile "$RUN/wm.keyfile.json" --base "$BASE" --adapter "$RUN/adapter" --paraphrase heldout --out-dir "$RUN/attacks"
ldw attack sysprompt --keyfile "$RUN/wm.keyfile.json" --model "$RUN/merged" --paraphrase heldout --out-dir "$RUN/attacks"
ldw attack fakequant --keyfile "$RUN/wm.keyfile.json" --model "$RUN/merged" --paraphrase heldout --out-dir "$RUN/attacks"
ldw attack sft       --keyfile "$RUN/wm.keyfile.json" --model "$RUN/merged" --paraphrase heldout --out-dir "$RUN/attacks" --n-samples 2000 --epochs 1
ldw attack summarize "$RUN"/attacks/*.json --out "$RUN/attacks/summary.md"
echo "done -> $RUN"
