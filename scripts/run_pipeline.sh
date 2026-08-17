#!/usr/bin/env bash
# Full pipeline on one machine: keygen -> generate -> embed -> merge -> verify (+ controls)
# -> evaluation profile -> attack summary.  Idempotent: existing artefacts are reused.
#
#   scripts/run_pipeline.sh [BASE] [RUN_DIR] [PROFILE]
#   e.g. scripts/run_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct runs/m0 quick
set -euo pipefail
BASE="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN="${2:-runs/pipeline}"
PROFILE="${3:-quick}"
mkdir -p "$RUN"
export PYTHONUNBUFFERED=1

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[ -f "$RUN/wm.keyfile.json" ] || { log keygen; ldw keygen --owner 1 --recipient 42 --out "$RUN/wm.keyfile.json" --registry "$RUN/registry.json"; }
[ -f "$RUN/train.jsonl" ]     || { log generate; ldw generate --keyfile "$RUN/wm.keyfile.json" --out "$RUN/train.jsonl" --reg-samples 800; }
[ -f "$RUN/adapter/adapter_config.json" ] || { log embed; ldw embed --data "$RUN/train.jsonl" --base "$BASE" --out "$RUN/adapter" \
    --epochs 3 --batch-size 4 --grad-accum 4 --max-len 384 --gradient-checkpointing; }
[ -f "$RUN/merged/config.json" ] || { log merge; ldw merge --base "$BASE" --adapter "$RUN/adapter" --out "$RUN/merged"; }

log "verify (base control / merged held-out)"
ldw verify --keyfile "$RUN/wm.keyfile.json" --model "$BASE" --out "$RUN/verify_base.json" --quiet
ldw verify --keyfile "$RUN/wm.keyfile.json" --model "$RUN/merged" --paraphrase heldout --out "$RUN/verify_merged_heldout.json" --quiet

log "controls"
ldw controls --keyfile "$RUN/wm.keyfile.json" --model "$BASE" --out "$RUN/controls_base.json"
ldw controls --keyfile "$RUN/wm.keyfile.json" --model "$RUN/merged" --out "$RUN/controls_merged.json"

log "evaluation profile: $PROFILE"
ldw eval run --profile "$PROFILE" --keyfile "$RUN/wm.keyfile.json" --base "$BASE" --adapter "$RUN/adapter" --out-dir "$RUN"
ldw eval render --run-dir "$RUN" > /dev/null
[ -d "$RUN/eval/attacks" ] && ldw attack summarize "$RUN"/eval/attacks/*.json --out "$RUN/eval/attacks/summary.md" > /dev/null || true

log "done: $RUN/eval/eval_report.md"
