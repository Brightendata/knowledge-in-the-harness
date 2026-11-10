#!/bin/sh
# The model server of all runs: google/gemma-4-12B-it (revision 707f0a3), bf16, one data-parallel replica on each GPU
# of $GPUS behind one OpenAI-compatible API at 127.0.0.1:8001, thinking on by default. LoRA is enabled from the start,
# so the base model is served with the same settings as the rounds that serve a GRPO adapter.
#
#   GPUS=0,1,2,3 MODEL=models/gemma-4-12B-it scripts/serve_gemma.sh
#
# Ready when /v1/models answers. Adapters are loaded at run time by their path in the repository
# (/v1/load_lora_adapter, which vLLM allows with one API server only). --max-loras covers the model gate, which serves
# every checkpoint of both domains at once; --max-lora-rank is the adapters' rank ([grpo] lora_rank of
# config/grpo_config.toml).
GPUS=${GPUS:-0,1}
MODEL=${MODEL:-models/gemma-4-12B-it}
cd "$(dirname "$0")/.."
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export CUDA_VISIBLE_DEVICES=$GPUS
exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name gemma-4-12b \
  --data-parallel-size "$(echo "$GPUS" | tr , '\n' | wc -l)" \
  --api-server-count 1 \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --enable-lora \
  --max-lora-rank 32 \
  --max-loras 12
