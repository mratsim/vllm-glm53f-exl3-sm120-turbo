#!/bin/bash
set -euo pipefail

# ============================================================
# GLM-5.3-Flash 3.5bpw (satgeze). DFlash2 draft, GPUs split the conversation. Wider batching: 4096 tokens
# per step
# ============================================================

# ============================================================
# Image
# ============================================================
IMAGE="localhost/vllm-glm53f-exl3-sm120-turbo:r6"
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
# Model: GLM-5.3-Flash-EXL3-TR3-3.5bpw local snapshot
# (satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw, mixed K3/K4 per-expert rates)
# ============================================================
MODELNAME="GLM-5.3-Flash"
MODEL_ROOT=/workspace/local_models
MODEL="${LOCAL_MODELS}"/GLM-5.3-Flash-EXL3-TR3-3.5bpw
MODEL_CONTAINER="${MODEL_ROOT}/${MODEL##*/}"

# ============================================================
# Settings
# ============================================================
TP_SIZE=2
DCP_SIZE=2                     # the GPUs split the saved conversation (~2x KV space)
GPU_UTIL=0.986                 # weights take about 74 GiB per GPU. Rest is KV cache
CONTEXT_SIZE=-1                      # auto: the model cap (1,048,576) clamped by the KV pool
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=4096    # prompt tokens per step. Wider = faster prefill, bigger scratch
                               # Bigger = faster long-prompt reading
KV_CACHE_DTYPE=fp8_ds_mla
BLOCK_SIZE=256
CP_KV_INTERLEAVE=4             # both split modes use the same value (4)
# DFlash2 draft, 7 tokens guessed per step. The -mtp3 files run MTP instead
DFLASH_TOKENS=7               # how many tokens ahead the draft guesses each step
# The draft memory cannot be split, so both GPUs keep a full
# copy. The page settings below keep the memory sizes lined up (boots #7b to #7d)
DFLASH_MODEL="${LOCAL_MODELS}"/GLM-5.3-Flash-DFlash2-MXFP8
                              # Small draft model beside the main model.
                              # Wrong guesses are thrown away, output stays exact
REASONING_EFFORT=high          # low or high. The template default is max and it talks too much.

# No expert parallelism. Only the uncut experts of the 4bpw file support it
EP_FLAG=()

# CUDA graph sizes stay on vllm auto-derivation (spec-decode tiers included)
SPEC_K=${DFLASH_TOKENS}      # DFlash draft tokens per decode step
# Graph sizes are token-keyed: base [1,2,4] for piecewise plus q_len x each
# reachable request count, so every decode width has an exact graph
Q=1
if (( SPEC_K > 0 ))
then
    Q=$(( SPEC_K + 1 ))
fi
CAPTURE_SIZES=(1 2 4)
r=1
while (( r <= MAX_NUM_SEQS ))
do
    CAPTURE_SIZES+=("$(( r * Q ))")
    r=$(( r + 1 ))
done
CAPTURE_SIZES=($(printf '%s\n' "${CAPTURE_SIZES[@]}" | sort -n | uniq))
GRAPH_CAP=${CAPTURE_SIZES[-1]}
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
    # Engine and runtime
    -e VLLM_ENGINE_READY_TIMEOUT_S=300
    -e VLLM_ENGINE_ITERATION_TIMEOUT_S=120
    -e OMP_NUM_THREADS=2
    -e HF_HUB_OFFLINE=1
    -e SAFETENSORS_FAST_GPU=1
    -e VLLM_USE_V2_MODEL_RUNNER=1
    # Topology and collectives
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0
    -e NCCL_P2P_DISABLE=1
    # EXL3 settings (also set inside the image)
    -e VLLM_EXL3_PREFILL_BLOCK_M=64
    -e VLLM_EXL3_PREFILL_TRELLIS=1
    # LM head and MTP draft
    -e VLLM_GLM53_MTP_DRAFT_HEAD=bf16
    # Allocator
    -e CUBLAS_WORKSPACE_CONFIG=:4096:1
)

# Use the b12x fast code for attention and for the expert layers
VLLM_BACKEND=(
    --attention-backend B12X
    --moe-backend b12x
    --linear-backend b12x
)

# Speculative decoding. b12x picks the right kernel for the draft size automatically FLASH_ATTN runs
# the draft. The only draft code that supports the split The
# draft memory stays BF16. Adds about 0.6 GiB per GPU
DFLASH_MOUNT=(-v "${DFLASH_MODEL}":/draft:ro)
# The memory bookkeeping needs pages lined up to 8448-byte
# steps. The draft breaks that, so pages are split to line
# up by design. The split width is 4608.
# Needs mamba-cache-mode align below
VLLM_ENV+=(-e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=4608)
VLLM_SPEC=(
    --speculative-config \
    "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":${DFLASH_TOKENS},\"draft_tensor_parallel_size\":${TP_SIZE},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"attention_backend\":\"FLASH_ATTN\",\"kv_cache_dtype\":\"auto\"}"
)

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
  printf '  parallel     tp=%s dcp=%s ep=no (mixed rates: TP2 sharded experts)\n' "${TP_SIZE}" "${DCP_SIZE}"
  printf '  quant        exl3 (TR3 mixed K3/K4 per-expert, routed experts only), kv=%s, block=%s\n' "${KV_CACHE_DTYPE}" "${BLOCK_SIZE}"
  printf '  context      %s tokens, mem-fraction=%s\n' "${CONTEXT_SIZE}" "${GPU_UTIL}"
  printf '  batching     max-seqs=%s, batched-tokens=%s\n' "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}"
  printf '  speculation  dflash2 (K=%s)\n' "${SPEC_K}"
  printf '  reasoning    %s\n' "${REASONING_EFFORT}"
} >&2

# ============================================================
# Container
# ============================================================

podman run --replace --detach --restart=always \
    --entrypoint /bin/bash \
    --health-cmd="curl -f http://localhost:${VLLM_PORT}/health || exit 1" \
    --health-start-period=300s \
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
    "${DFLASH_MOUNT[@]}" \
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
            `# TODO: explore --mamba-ssm-cache-dtype bfloat16 to divide the cache fixed cost by 2` \
            --mamba-ssm-cache-dtype bfloat16 \
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
