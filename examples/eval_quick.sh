#!/usr/bin/env bash
# Run the `quick` evaluation profile on an adapter and export a publishable report.
set -euo pipefail
BASE="${BASE:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:-runs/quickstart/adapter}"
KEYFILE="${KEYFILE:-runs/quickstart/wm.keyfile.json}"
OUT="${OUT:-runs/quickstart}"

ldw eval run --profile quick --keyfile "$KEYFILE" --base "$BASE" --adapter "$ADAPTER" --out-dir "$OUT"
ldw eval render --run-dir "$OUT"
ldw eval export --run-dir "$OUT" --out "$OUT/public_report.json"
echo "report: $OUT/eval/eval_report.md   public copy: $OUT/public_report.json"
