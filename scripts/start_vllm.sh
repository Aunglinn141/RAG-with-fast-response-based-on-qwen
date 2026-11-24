#!/usr/bin/env bash
# Run inside Linux/WSL with a dedicated vLLM environment activated.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo 'Run this script in Linux/WSL; upstream vLLM does not support native Windows.' >&2
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo 'vllm is missing. Activate the dedicated environment described in docs/vllm.md.' >&2
  exit 1
fi

task_model="${VLLM_MODEL:-Qwen/Qwen3-1.7B}"
task_name="${VLLM_SERVED_MODEL_NAME:-$task_model}"
task_args=(serve "$task_model" --served-model-name "$task_name"
  --host 127.0.0.1 --port "${VLLM_PORT:-8001}"
  --dtype auto --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-2}"
  --enable-prefix-caching --generation-config vllm)
if [[ -n "${VLLM_REVISION:-}" ]]; then
  task_args+=(--revision "$VLLM_REVISION" --tokenizer-revision "$VLLM_REVISION")
fi
export VLLM_NO_USAGE_STATS=1
exec vllm "${task_args[@]}"
