#!/usr/bin/env bash
# Kept for backwards compatibility: the M0 pipeline is now scripts/run_pipeline.sh
exec "$(dirname "$0")/run_pipeline.sh" "${1:-Qwen/Qwen2.5-0.5B-Instruct}" "${2:-runs/m0}" "${3:-quick}"
