#!/usr/bin/env bash
# Alias to clear_stage_results.sh (Safe Archive Version)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/clear_stage_results.sh" "$@"
