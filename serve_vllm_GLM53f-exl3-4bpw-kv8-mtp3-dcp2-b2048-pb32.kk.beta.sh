#!/bin/bash
set -euo pipefail

# ============================================================
# GLM-5.3-Flash 4bpw (brandonmusic). 8-bit KV cache, MTP-3, GPUs split (DCP=2)
# prefill block 32
# ============================================================
# Image
# ============================================================
IMAGE="localhost/vllm-glm53f-exl3-sm120-turbo:r5"
PODNAME="vllm"
VLLM_PORT=8000

# ============================================================
# Paths
# ============================================================
# Normal locations. Export HF_CACHE / LOCAL_MODELS to override.
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
LOCAL_MODELS="${LOCAL_MODELS:-$HOME/local_models}"

DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ROOTFS_CACHE="${DIR}/cache-rootfs"

mkdir -p "${HF_CACHE}" "${ROOTFS_CACHE}/triton" "${ROOTFS_CACHE}/inductor" "${ROOTFS_CACHE}/b12x"
mkdir -p "${DIR}/container-tmp"

# ============================================================
# Model: GLM-5.3-Flash-tr3-4bpw local snapshot
# (brandonmusic/GLM-5.3-Flash-tr3-4bpw, uniform K4, unsliced layout)
# ============================================================
MODELNAME="GLM-5.3-Flash"
MODEL_ROOT=/workspace/local_models
MODEL="${LOCAL_MODELS}"/GLM-5.3-Flash-tr3-4bpw
MODEL_CONTAINER="${MODEL_ROOT}/${MODEL##*/}"

# ============================================================
# Settings
# ============================================================
TP_SIZE=2
DCP_SIZE=2                     # the GPUs split the saved conversation (~2x KV space)
GPU_UTIL=0.97                  # ~3% headroom: serving-time Triton JIT OOMs at 0.986 (r5 boot 5)
CONTEXT_SIZE=-1                      # auto: the model cap (1,048,576) clamped by the KV pool
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=1024    # prompt tokens per step. Smaller = more KV space
KV_CACHE_DTYPE=fp8_ds_mla
BLOCK_SIZE=256
CP_KV_INTERLEAVE=4             # both split modes use the same value (4)
MTP_TOKENS=3                   # the MTP draft part ships inside the model file
REASONING_EFFORT=high          # low or high. The template default is max and it talks too much.
HEALTH_START_PERIOD=600

# Expert parallelism is GONE from the karmic b12x runtime surface (the new
# plan_execution contract has no expert maps). Uniform-K4 runs TP-sharded.
EP_FLAG=()

# Ready-made graph sizes. Spec decoding on: (K+1) x seqs. Off: 4 x seqs
if (( MTP_TOKENS > 0 ))
then
    GRAPH_CAP=$(( MAX_NUM_SEQS * (MTP_TOKENS + 1) ))
else
    GRAPH_CAP=$(( MAX_NUM_SEQS * 4 ))
fi
(( GRAPH_CAP < 6 )) && GRAPH_CAP=6

# The prepared sizes must cover every batch width the server can reach
CAPTURE_SIZES=()
s=1
while (( s <= GRAPH_CAP ))
do
    CAPTURE_SIZES+=("$s")
    s=$(( s < 4 ? s*2 : s+4 ))
done
IFS=,
SIZES_CSV="${CAPTURE_SIZES[*]}"
unset IFS

# ============================================================
# Build the server options
# ============================================================
VLLM_ENV=()
VLLM_BACKEND=()
VLLM_SPEC=()
VLLM_EXTRA=()

# The GPUs cannot talk to each other directly. Direct links are off
VLLM_ENV+=(
    -e VLLM_ENGINE_READY_TIMEOUT_S=${HEALTH_START_PERIOD}
    -e VLLM_ENGINE_ITERATION_TIMEOUT_S=120
    -e OMP_NUM_THREADS=4
    -e HF_HUB_OFFLINE=1
    -e SAFETENSORS_FAST_GPU=1
    -e VLLM_USE_V2_MODEL_RUNNER=1
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0
    -e NCCL_P2P_DISABLE=1
    -e VLLM_EXL3_PREFILL_BLOCK_M=32
    -e VLLM_EXL3_PREFILL_TRELLIS=1
    # The uniform-K4 compile factory primes its real-launch variants with
    # live CUDA tensors. That only works in-process: pool workers hide
    # CUDA (r5 boots 9+10). 0 = compile_in_process, the mode the
    # full-rotation prewarm is designed for.
    -e B12X_COMPILE_WORKERS=0
    -e VLLM_GLM53_MTP_DRAFT_HEAD=bf16
    -e CUBLAS_WORKSPACE_CONFIG=:4096:1
)

# Use the b12x fast code for attention and for the expert layers
VLLM_BACKEND=(
    --attention-backend B12X
    --moe-backend b12x
    --linear-backend b12x
)

# Speculative decoding. The draft part is inside the model file
if [[ "${MTP_TOKENS}" != "0" ]]
then
    VLLM_SPEC=(
        --speculative-config \
        "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"moe_backend\":\"b12x\",\"attention_backend\":\"B12X\"}"
    )
fi

VLLM_EXTRA+=(
    --disable-custom-all-reduce
    --prefill-compute-share 0.4
    --prefill-schedule-interval 1
    --mm-encoder-attn-backend TORCH_SDPA
    --limit-mm-per-prompt '{"video":0}'
    --no-enable-flashinfer-autotune
)

# ============================================================
# Startup summary
# ============================================================
{
  printf 'launch %s as %s\n' "${MODEL_CONTAINER}" "${MODELNAME}"
  printf '  image        %s\n' "${IMAGE}"
  printf '  parallel     tp=%s dcp=%s ep=no (uniform K4: TP2 sharded experts)\n' "${TP_SIZE}" "${DCP_SIZE}"
  printf '  quant        exl3 (TR3 uniform K4, routed experts only), kv=%s, block=%s\n' "${KV_CACHE_DTYPE}" "${BLOCK_SIZE}"
  printf '  context      %s tokens, mem-fraction=%s\n' "${CONTEXT_SIZE}" "${GPU_UTIL}"
  printf '  batching     max-seqs=%s, batched-tokens=%s, graph-cap=%s\n' "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" "${GRAPH_CAP}"
  printf '  speculation  mtp (K=%s, probabilistic)\n' "${MTP_TOKENS}"
  printf '  exl3-prefill block size: 32 (kernel claim only, end-to-end slower than 64)\n'
  printf '  reasoning    %s\n' "${REASONING_EFFORT}"
} >&2

# ============================================================
# Container
# ============================================================

podman run --replace --detach --restart=always \
    --entrypoint /bin/bash \
    --health-cmd="curl -f http://localhost:${VLLM_PORT}/health || exit 1" \
    --health-start-period="${HEALTH_START_PERIOD}s" \
    --health-interval=30s \
    --health-on-failure=kill \
    --health-retries=3 \
    --name "${PODNAME}" \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --network=host \
    -v "${LOCAL_MODELS}":/workspace/local_models:ro \
    -v "${ROOTFS_CACHE}":/cache:rw \
    -v "${HF_CACHE}":/root/.cache/huggingface:ro \
    -v "${DIR}/container-tmp":/container-tmp \
    -v "${DIR}/chat_template.multimodal.jinja":/opt/glm53f/chat_template.multimodal.jinja:ro \
    -e TMPDIR=/container-tmp \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
    "${VLLM_ENV[@]}" \
    "${IMAGE}" \
        -lc 'unset MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE CUDAGRAPH_CAPTURE_SIZES PREFILL_SCHEDULE_INTERVAL FAIRNESS_ENGINE PREFILL_COMPUTE_SHARE VLLM_PCIE_ALLREDUCE_BACKEND VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE
exec /opt/venv/bin/vllm serve "$@"' -- \
            "${MODEL_CONTAINER}" \
            `# Networking` \
            --host 0.0.0.0 \
            --port "${VLLM_PORT}" \
            `# Model identity` \
            --served-model-name "${MODELNAME}" \
            `# Quantization` \
            --quantization exl3 \
            --dtype bfloat16 \
            --kv-cache-dtype "${KV_CACHE_DTYPE}" \
            --block-size "${BLOCK_SIZE}" \
            # TODO: explore --mamba-ssm-cache-dtype bfloat16
            # to divide the cache fixed cost by 2 (verify via the
            # rebalance line's max-request cost before adoption)
            --mamba-cache-mode align \
            `# Parallelism` \
            --tensor-parallel-size "${TP_SIZE}" \
            --decode-context-parallel-size "${DCP_SIZE}" \
            --cp-kv-cache-interleave-size "${CP_KV_INTERLEAVE}" \
            --dcp-kv-cache-interleave-size "${CP_KV_INTERLEAVE}" \
            "${EP_FLAG[@]}" \
            `# Resource limits` \
            --gpu-memory-utilization "${GPU_UTIL}" \
            --max-model-len "${CONTEXT_SIZE}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --max-cudagraph-capture-size "${GRAPH_CAP}" \
            --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${SIZES_CSV}]}" \
            `# Tokenizer / tools / reasoning` \
            --reasoning-parser glm45 \
            --tool-call-parser glm47 \
            --enable-auto-tool-choice \
            --chat-template /opt/glm53f/chat_template.multimodal.jinja \
            --default-chat-template-kwargs.reasoning_effort="${REASONING_EFFORT}" \
            `# b12x KDA prefill auto-engages on karmic` \
            `# the old kda_prefill_backend key fails the karmic resolver` \
            `# Serving statistics` \
            --enable-request-id-headers \
            --enable-force-include-usage \
            --enable-per-request-metrics \
            --enable-prompt-tokens-details \
            "${VLLM_BACKEND[@]}" \
            "${VLLM_EXTRA[@]}" \
            "${VLLM_SPEC[@]}" \
            `# SAMPLER` \
            --override-generation-config '{"temperature": 1, "top_p": 0.95}' \
            "$@"

printf 'Started %s on http://127.0.0.1:%s/v1 - podman logs -f %s\n' \
    "${PODNAME}" "${VLLM_PORT}" "${PODNAME}"
