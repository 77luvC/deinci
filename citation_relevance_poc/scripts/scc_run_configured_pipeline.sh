#!/usr/bin/env bash
# SCC-friendly wrapper for the config-driven citation pipeline.
#
# Usage:
#   qsub/sbatch this file, or run it directly on an interactive SCC node.
#   Override CONFIG, PIPELINE_ONLY, PIPELINE_SKIP, or PIPELINE_FIELD as needed.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/pipeline_2026-07-27.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_DIR}"

# Activate your SCC environment before calling this script, or set VENV_PATH.
if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

cmd=("${PYTHON_BIN}" "scripts/run_configured_pipeline.py" "--config" "${CONFIG}")

if [[ -n "${PIPELINE_FIELD:-}" ]]; then
  for field in ${PIPELINE_FIELD}; do
    cmd+=("--field" "${field}")
  done
fi

if [[ -n "${PIPELINE_ONLY:-}" ]]; then
  for step in ${PIPELINE_ONLY}; do
    cmd+=("--only" "${step}")
  done
fi

if [[ -n "${PIPELINE_SKIP:-}" ]]; then
  for step in ${PIPELINE_SKIP}; do
    cmd+=("--skip" "${step}")
  done
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=("--dry-run")
fi

if [[ "${FORCE_RECOMPUTE:-0}" == "1" ]]; then
  cmd+=("--force-recompute")
fi

if [[ -n "${RUN_ID:-}" ]]; then
  cmd+=("--run-id" "${RUN_ID}")
fi

echo "Running from ${PROJECT_DIR}"
echo "Config: ${CONFIG}"
"${cmd[@]}"
