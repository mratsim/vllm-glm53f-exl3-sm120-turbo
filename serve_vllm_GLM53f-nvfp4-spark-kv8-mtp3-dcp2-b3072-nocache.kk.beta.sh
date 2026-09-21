#!/bin/bash
# GLM-5.3-Flash NVFP4 Spark (local-inference-lab). 8-bit KV cache, MTP-3, GPUs split (DCP=2). LMCache off.
set -euo pipefail

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
# Model
# ============================================================
MODELNAME="GLM-5.3-Flash"
MODEL="${LOCAL_MODELS}"/GLM-5.3-Flash-NVFP4-Spark

# ============================================================
# Settings (lil preset values)
# ============================================================
TP_SIZE=2
DCP_SIZE=2
GPU_UTIL=0.986
CONTEXT_SIZE=-1                      # auto: the model cap (1,048,576) clamped by the KV pool
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=3072
KV_CACHE_MEMORY_BYTES=4190109696   # 3996 MiB per GPU, fixed to avoid mis-sizing due to cuda graphs / preflight
KV_CACHE_DTYPE=fp8
BLOCK_SIZE=256
CP_KV_INTERLEAVE=4
MTP_TOKENS=3
REASONING_EFFORT=high

# LMCache prefix cache (lil preset values): a CPU-only sidecar
# owns the L1 RAM tier and vLLM gathers and scatters through the MP connector
# in its workers. Retention equals the chunk so recurrent checkpoints stay
# inside one object.
LMCACHE_ENABLED=0
LMCACHE_TRANSFER_MODE=engine_driven
LMCACHE_L1_GB=64
LMCACHE_L1_INIT_GB=2
LMCACHE_L2_ENABLED=0
LMCACHE_CHUNK_SIZE=4096
LMCACHE_INSTANCE_ID=glm53-spark-tp2-kv8
LMCACHE_MP_PORT=5555
LMCACHE_HTTP_PORT=8085
LMCACHE_PROM_PORT=9095

VLLM_CACHE_ENV=()
VLLM_CACHE_ARGS=()
LMCACHE_SERVER=""
if [[ "${LMCACHE_ENABLED}" == "1" ]]
then
    MAX_NUM_BATCHED_TOKENS=${LMCACHE_CHUNK_SIZE}
    SPLIT_TARGET_BLOCK_SIZE=auto
    VLLM_CACHE_ENV=(
        -e LMCACHE_KV_CACHE_DTYPE=fp8_ds_mla
        -e LMCACHE_VLLM_KV_CACHE_DTYPE=fp8
        -e LMCACHE_TRANSFER_MODE=${LMCACHE_TRANSFER_MODE}
        -e LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE}
    )
    VLLM_CACHE_ARGS=(
        --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_buffer_size\":268435456,\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"127.0.0.1\",\"lmcache.mp.port\":${LMCACHE_MP_PORT},\"lmcache.mp.mp_transfer_mode\":\"${LMCACHE_TRANSFER_MODE}\"}}"
        --prefix-cache-retention-interval "${LMCACHE_CHUNK_SIZE}"
        --max-num-scheduled-tokens "${LMCACHE_CHUNK_SIZE}"
    )
    LMCACHE_SERVER="env CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY /opt/venv/bin/lmcache server --instance-id ${LMCACHE_INSTANCE_ID} --host 127.0.0.1 --port ${LMCACHE_MP_PORT} --chunk-size ${LMCACHE_CHUNK_SIZE} --max-workers 8 --max-gpu-workers 2 --max-cpu-workers 16 --hash-algorithm blake3 --supported-transfer-mode ${LMCACHE_TRANSFER_MODE} --separate-object-groups --l1-size-gb ${LMCACHE_L1_GB} --l1-init-size-gb ${LMCACHE_L1_INIT_GB} --eviction-policy LRU --http-host 127.0.0.1 --http-port ${LMCACHE_HTTP_PORT} --prometheus-port ${LMCACHE_PROM_PORT} --no-l1-use-lazy --shm-name lmcache-${LMCACHE_INSTANCE_ID}-${LMCACHE_MP_PORT}"
fi

# CUDA graph sizes stay on vllm auto-derivation (spec-decode tiers included)
# Graph sizes are token-keyed: base [1,2,4] for piecewise plus q_len x each
# reachable request count, so every decode width has an exact graph
Q=1
if (( MTP_TOKENS > 0 ))
then
    Q=$(( MTP_TOKENS + 1 ))
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
    -e TRANSFORMERS_OFFLINE=1
    -e SAFETENSORS_FAST_GPU=1
    # Topology and collectives
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0
    -e NCCL_P2P_DISABLE=1
    -e NCCL_SOCKET_IFNAME=lo
    -e GLOO_SOCKET_IFNAME=lo
    -e NCCL_MIN_NCHANNELS=2
    -e NCCL_MAX_NCHANNELS=2
    -e NCCL_BUFFSIZE=1048576
    -e NCCL_NET_PLUGIN=none
    -e NCCL_TUNER_PLUGIN=none
    -e CUTE_DSL_ARCH=sm_120a
    # Allocator
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:12
    -e CUBLAS_WORKSPACE_CONFIG=:4096:1
    # LM head and MTP draft
    -e VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4
    -e VLLM_MXFP8_LM_HEAD=1
    -e VLLM_MTP_NVFP4_LM_HEAD=1
    -e VLLM_LM_HEAD_A16=0
    # Kernel selection
    -e VLLM_B12X_MOE_FP4_FORCE_A16=0
    -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
    # b12x
    -e VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=65536
    # GLM-5.3 cache policy
    -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=${SPLIT_TARGET_BLOCK_SIZE:-2048}
    -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto
    # b12x
    -e B12X_MHC_PDL=1
    # GLM-5.3 cache policy
    -e VLLM_GLM53_L2_PREFETCH=1
    -e VLLM_GLM53_L2_PREFETCH_PERSIST_MB=0
    -e VLLM_GLM53_DFLASH_ATTN=1
    -e VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1
    -e VLLM_GLM53_KDA_GATE_SIDE_STREAM=1
    -e VLLM_USE_FLASHINFER_SAMPLER=1
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
    --max-parallel-prefills 1
    --no-enable-flashinfer-autotune
)

# ============================================================
# Startup summary
# ============================================================
{
  printf 'launch %s as %s\n' "${MODEL}" "${MODELNAME}"
  printf '  image        %s\n' "${IMAGE}"
  printf '  parallel     tp=%s dcp=%s\n' "${TP_SIZE}" "${DCP_SIZE}"
  printf '  quant        nvfp4 (modelopt_mixed), kv=%s, block=%s\n' "${KV_CACHE_DTYPE}" "${BLOCK_SIZE}"
  printf '  context      %s, kv-pool=%s bytes/gpu, mem-fraction=%s\n' "${CONTEXT_SIZE}" "${KV_CACHE_MEMORY_BYTES}" "${GPU_UTIL}"
  printf '  batching     max-seqs=%s, batched-tokens=%s\n' "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}"
  printf '  speculation  mtp (K=%s, probabilistic)\n' "${MTP_TOKENS}"
  printf '  cache        lmcache %s, mode=%s, l1=%s GiB, l2=%s\n' \
    "$([[ "${LMCACHE_ENABLED}" == 1 ]] && echo on || echo off)" \
    "${LMCACHE_TRANSFER_MODE}" "${LMCACHE_L1_GB}" "${LMCACHE_L2_ENABLED}"
} >&2

# The wrapper unsets host-side env overrides, then brings the cache
# sidecar up first: engine-driven needs its SHM arena before vLLM
# profiles. It health-checks the sidecar before vLLM starts.
WRAP='unset MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE CUDAGRAPH_CAPTURE_SIZES PREFILL_SCHEDULE_INTERVAL FAIRNESS_ENGINE PREFILL_COMPUTE_SHARE VLLM_PCIE_ALLREDUCE_BACKEND VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE MAX_MODEL_LEN'
if [[ -n "${LMCACHE_SERVER}" ]]
then
    WRAP+=$'\n'"${LMCACHE_SERVER} & LM_PID=\$!"
    WRAP+=$'\n''t=0'
    WRAP+=$'\n''until curl -fs --max-time 2 http://127.0.0.1:'"${LMCACHE_HTTP_PORT}"'/healthcheck >/dev/null'
    WRAP+=$'\n''do'
    WRAP+=$'\n''kill -0 $LM_PID 2>/dev/null || exit 1'
    WRAP+=$'\n''t=$((t+1))'
    WRAP+=$'\n''[ $t -ge 120 ] && exit 1'
    WRAP+=$'\n''sleep 1'
    WRAP+=$'\n''done'
fi
WRAP+=$'\n''exec /opt/venv/bin/vllm serve "$@"'

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
    -e TMPDIR=/container-tmp \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
    "${VLLM_ENV[@]}" \
    "${VLLM_CACHE_ENV[@]}" \
    "${IMAGE}" \
        -lc "${WRAP}" -- \
            "${MODEL}" \
            `# Networking` \
            --host 0.0.0.0 \
            --port "${VLLM_PORT}" \
            `# Model identity` \
            --served-model-name "${MODELNAME}" \
            `# Quantization` \
            `# nvfp4: modelopt_mixed, no --trust-remote-code` \
            --quantization modelopt_mixed \
            --load-format safetensors \
            --dtype bfloat16 \
            --kv-cache-dtype "${KV_CACHE_DTYPE}" \
            --block-size "${BLOCK_SIZE}" \
            `# TODO: explore --mamba-ssm-cache-dtype bfloat16 to divide the cache fixed cost by 2` \
            --mamba-ssm-cache-dtype bfloat16 \
            --mamba-cache-mode align \
            --recurrent-checkpoint-policy request_boundaries \
            `# Parallelism` \
            --tensor-parallel-size "${TP_SIZE}" \
            --decode-context-parallel-size "${DCP_SIZE}" \
            --cp-kv-cache-interleave-size "${CP_KV_INTERLEAVE}" \
            --dcp-kv-cache-interleave-size "${CP_KV_INTERLEAVE}" \
            `# Resource limits` \
            --max-model-len "${CONTEXT_SIZE}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
            --gpu-memory-utilization "${GPU_UTIL}" \
            --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${SIZES_CSV}]}" \
            `# Tokenizer / tools / reasoning` \
            --reasoning-parser glm45 \
            --tool-call-parser glm47 \
            --enable-auto-tool-choice \
            --chat-template /opt/glm53f/chat_template.multimodal.jinja \
            --default-chat-template-kwargs.reasoning_effort="${REASONING_EFFORT}" \
            --default-chat-template-kwargs.clear_thinking=false \
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
            "${VLLM_CACHE_ARGS[@]}" \
            "${VLLM_OFFLOAD_ARGS[@]}" \
            `# SAMPLER` \
            --override-generation-config '{"temperature": 1, "top_p": 0.95}' \
            "$@"

printf 'Started %s on http://127.0.0.1:%s/v1 - podman logs -f %s\n' \
    "${PODNAME}" "${VLLM_PORT}" "${PODNAME}"
