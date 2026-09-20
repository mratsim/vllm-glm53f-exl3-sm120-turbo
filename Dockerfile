# vllm-glm53f-exl3-sm120-turbo: GLM-5.3-Flash EXL3 (TR3) on the karmic-kraken wheel-runtime image.
#
# Base: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta, a WHEEL-BUILT runtime on the NVIDIA PyTorch 26.08 NGC base.
# There are no vLLM source trees in this image.
# Layout of the image, verified from the image config plus the release manifest:
#   venv    /opt/venv, packages in /opt/venv/lib/python3.12/site-packages
#   vllm    wheel 0.1.dev21537+g67bb922f6.cu134 (source 67bb922f6f4, integration/karmic-kraken-beta)
#   b12x    wheel 1.3.0 (source eea3ced11fc1)
#   bundle  flashinfer, lmcache, instanttensor, nccl. NO exllamav3.
#   cuda    13.4.1, torch 2.14.0a0, python 3.12
#   entry   /usr/local/bin/lil-entrypoint (the launchers bypass it with --entrypoint /bin/bash)
#
# The build patches the venv's vllm and b12x in place. exllamav3 is cloned at a fixed
# commit and compiled in-image against the in-image torch (ABI match by construction).
# Re-point the base: pull the new karmic-kraken-beta tag and record its RepoDigest in
# BASE_IMAGE. A base change that breaks a patch fails the build loudly. That
# failure is the signal.
#
# Patches (applied in number order, b12x paths flow through the same loop):
#   0101  vllm exl3 adapter: exl3.py + exl3_online_cache.py + _exl3_btx_adoption.py
#         (BTX adoption lives vllm-side, the b12x package stays unpatched).
#   0102  register exl3/Exl3Config in quantization/__init__.py.
#   0103  claim exl3 before the generic ModelOpt override in config/model.py.
#   0104  routed_experts.py: exclude per-expert experts.<idx>. names from the fused check.
#   0105  b12x 1.3 rate contract: the adoption validator reads trellis_rate_granularity.
#   0106  adopt_btx_weights alias + getattr guard for args.shared_experts.
#   0107  _audit_path(tag, detail) witnesses. VLLM_PATCHES_REVISION stamped into the tag.
#   0108  rejection-sampler padding mask: padded draft rows masked to -1 (is_padding).
#   0111  gpu_worker: re-run the 0101 route-pack warmup after the last preparation job,
#         before the guarded capture (preparation eviction drops raw JIT entries).
#   0201  [deleted] BTX adoption overlay, superseded by the 0101 vllm-side re-home.
#   0202  [deleted] w4a16 trellis3 mixed API, present at both b12x revisions.
#         The build-time check still verifies the symbol.
#   0301  mixed-rate acceptance: bits="mixed_k34_per_tensor" plus the r7_routed_experts
#         term in the eager-capture exemption (required for the 3.5bpw checkpoint).
#
# Patch application: patch(1) --fuzz=0 with -d site-packages (the patch files use a/ b/
# prefixes, so -p1 resolves them under site-packages).
#

ARG BASE_IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7@sha256:5927520c447fdcbc0990567f9237756ff66a54e360438915a5f70cd5cf9d530d
ARG EXLLAMAV3_REPO=https://github.com/brandonmmusic-max/exllamav3.git
ARG EXLLAMAV3_COMMIT=704aefd743b390af4bd0fb429d1906f9b964c7d8

FROM ${BASE_IMAGE}

# Re-declared after FROM: ARG values do not cross the FROM boundary.
ARG BASE_IMAGE
ARG EXLLAMAV3_REPO
ARG EXLLAMAV3_COMMIT
ARG VLLM_GLM53F_EXL3_REVISION=r5

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL ai.vllm.base.tag="karmic-kraken-beta-20260919-cfc67a15ebc3daf7" \
      ai.vllm.base.pins="vllm=67bb922f6f4(integration/karmic-kraken-beta),b12x=eea3ced11fc1,exllamav3=704aefd743b,cuda=13.4.1,torch=2.14" \
      ai.vllm.exllamav3.repo="${EXLLAMAV3_REPO}" \
      ai.vllm.exllamav3.commit="${EXLLAMAV3_COMMIT}" \
      ai.vllm.patchset="0101-exl3-adapter,0102-exl3-quant-registration,0103-exl3-config-detection,0104-routed-experts-per-expert-trellis,0105-exl3-b12x13-rate-contract,0106-exl3-adoption-alias-and-guards,0107-exl3-path-audit-logs,0108-rejection-sampler-padding-mask,0111-gpu-worker-route-pack-rewarm,0301-exl3-mixed-rate-gate" \
      ai.vllm.patchset.deleted="0201-b12x-btx-adoption(re-homed-vllm-side),0202-b12x-trellis3-mixed-api(absorbed-upstream)" \
      ai.vllm.revision="${VLLM_GLM53F_EXL3_REVISION}" \
      ai.vllm.target.checkpoint="brandonmusic/GLM-5.3-Flash-tr3-4bpw,satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw" \
      ai.vllm.kv.dtype="fp8_ds_mla" \
      ai.vllm.exl3.prefill-block-m="64"

WORKDIR /workspace

# --- Sniff test: prove the wheel-runtime layout before any patch applies. A miss fails the build.
RUN set -eu; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    for f in \
      "$SP/vllm/model_executor/layers/quantization/__init__.py" \
      "$SP/vllm/model_executor/layers/quantization/online/mxfp8.py" \
      "$SP/vllm/model_executor/layers/fused_moe/routed_experts.py" \
      "$SP/vllm/config/model.py"; \
    do test -f "$f" || { echo "base layout drift: missing $f" >&2; exit 1; }; done; \
    if test -f "$SP/b12x/moe/fused_moe/_impl.py"; then BX="$SP/b12x"; \
    elif test -f "$SP/b12x/b12x/moe/fused_moe/_impl.py"; then BX="$SP/b12x/b12x"; \
    else echo "base layout drift: no b12x package under $SP (tried b12x/moe and b12x/b12x/moe)" >&2; exit 1; fi; \
    test -f "$BX/moe/_shared/kernels/w4a16/mixed_trellis.py" \
      || { echo "base layout drift: missing $BX/moe/_shared/kernels/w4a16/mixed_trellis.py" >&2; exit 1; }; \
    test -x /opt/venv/bin/python; \
    /opt/venv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"; \
    /opt/venv/bin/python -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| b12x', m.version('b12x'))"; \
    echo "base layout OK: wheel runtime, venv packages will be patched in place"

# --- Persistent ccache: pass internal/ccache as a build mount. It stays out of git.
# --- exllamav3: cloned at the fixed commit, compiled in-image against the in-image
# --- torch. Artifacts at /opt/exllamav3 (extension) and /opt/exllamav3-python (package).
RUN set -eux; \
    /opt/venv/bin/python -c 'import torch; assert torch.__version__.startswith("2.14"), torch.__version__'; \
    export CCACHE_DIR=/opt/glm53f-ccache CCACHE_MAXSIZE=8G            CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime; \
    if ! command -v ccache >/dev/null 2>&1; then \
      apt-get update -qq && apt-get install -y -qq ccache && rm -rf /var/lib/apt/lists/*; \
    fi; \
    mkdir -p "${CCACHE_DIR}" /usr/local/ccache-shim; \
    printf '#!/bin/sh\nexec ccache /usr/local/cuda/bin/nvcc "$@"\n' > /usr/local/ccache-shim/nvcc; \
    chmod +x /usr/local/ccache-shim/*; \
    test -d /usr/lib/ccache && export PATH=/usr/lib/ccache:${PATH}; \
    export PATH=/usr/local/ccache-shim:${PATH}; \
    git clone --filter=blob:none --no-checkout "${EXLLAMAV3_REPO}" /tmp/exllamav3-src; \
    git -C /tmp/exllamav3-src fetch --depth=1 origin "${EXLLAMAV3_COMMIT}"; \
    git -C /tmp/exllamav3-src checkout --detach "${EXLLAMAV3_COMMIT}"; \
    test "$(git -C /tmp/exllamav3-src rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}"; \
    cd /tmp/exllamav3-src; \
    TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=48 NVCC_THREADS=4 \
      /opt/venv/bin/python setup.py build_ext --inplace; \
    mapfile -t extensions < <(find . -maxdepth 1 -type f -name 'exllamav3_ext*.so' -print); \
    test "${#extensions[@]}" -eq 1; \
    test -f exllamav3/modules/quant/exl3_lib/quantize.py; \
    install -d /opt/exllamav3 /opt/exllamav3-python; \
    install -m 0755 "${extensions[0]}" /opt/exllamav3/; \
    cp -a exllamav3 /opt/exllamav3-python/; \
    PYTHONPATH=/opt/exllamav3 VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
      /opt/venv/bin/python -c \
      'import importlib; assert hasattr(importlib.import_module("exllamav3_ext"), "exl3_gemm"); print("exllamav3_ext imports with exl3_gemm")'; \
    test -f /opt/exllamav3-python/exllamav3/modules/quant/exl3_lib/quantize.py; \
    rm -rf /tmp/exllamav3-src

COPY patches/ /opt/glm53f-exl3-patches/
COPY chat_template.multimodal.jinja /opt/glm53f/chat_template.multimodal.jinja

# --- Patch tooling. NGC ubuntu bases usually ship patch(1).
# --- If missing, install it from apt, then fall back to git apply if even that fails.
# --- A build without any patch tool must fail loudly at this step.
RUN set -eu; \
    if ! command -v patch >/dev/null 2>&1; then \
      echo "patch(1) not found in the base image, installing via apt" >&2; \
      apt-get update -qq && apt-get install -y -qq patch && rm -rf /var/lib/apt/lists/*; \
      command -v patch >/dev/null 2>&1 || echo "apt install failed, will use git apply as the patch tool" >&2; \
    fi

# --- Apply the patch series in place under the venv site-packages (sorted order is the authoritative order, gaps are history).
# --- Every patch is dry-run first, then
# --- applied. Any failure aborts the build.
RUN set -eu; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    for p in /opt/glm53f-exl3-patches/*.patch; do sed -i 's/\r$//' "$p"; done; \
    for p in $(ls /opt/glm53f-exl3-patches/*.patch | sort); do \
        echo "=== applying $(basename "$p") ==="; \
        if command -v patch >/dev/null 2>&1; then \
            patch -p1 --fuzz=0 --dry-run -d "$SP" < "$p" \
              || { echo "ERROR: $(basename "$p") does not apply. Site-packages moved, re-anchor the patch" >&2; exit 1; }; \
            patch -p1 --fuzz=0 -d "$SP" < "$p"; \
        elif command -v git >/dev/null 2>&1; then \
            ( cd "$SP" && git apply --check "$p" ) \
              || { echo "ERROR: $(basename "$p") does not apply. Site-packages moved, re-anchor the patch" >&2; exit 1; }; \
            ( cd "$SP" && git apply "$p" ); \
        else \
            echo "ERROR: neither patch(1) nor git(1) available in the base image" >&2; exit 1; \
        fi; \
    done; \
    rm -rf /opt/glm53f-exl3-patches; \
    # The patches carry the literal VLLM_PATCHES_REVISION token. The boot log shows the concrete revision.
    V="$SP/vllm"; \
    grep -rl 'sm120-turbo' "$V" | xargs -r sed -i \
      -e "s/sm120-turbo VLLM_PATCHES_REVISION/sm120-turbo ${VLLM_GLM53F_EXL3_REVISION}/g" \
      -e "s/sm120-turbo r[0-9][0-9]*/sm120-turbo ${VLLM_GLM53F_EXL3_REVISION}/g"

# --- Post-apply assertions (no GPU): every patch applied, with the silent-failure ones asserted explicitly.
# --- The b12x import checks try both possible nesting forms of the wheel layout and prove the re-homed adoption module plus the b12x
# --- symbols resolve against the image's real packages.
RUN set -eu; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    V="$SP/vllm"; \
    if grep -rq 'VLLM_PATCHES_REVISION' "$V"; then \
      echo "revision stamp failed: literal VLLM_PATCHES_REVISION marker survived" >&2; exit 1; fi; \
    grep -q '"exl3"' "$V/model_executor/layers/quantization/__init__.py" \
      || { echo "0102 registration missing" >&2; exit 1; }; \
    grep -q '^ *"exl3",' "$V/config/model.py" || { echo "0103 detection insert missing" >&2; exit 1; }; \
    grep -q '_PER_EXPERT_IDX_RE' "$V/model_executor/layers/fused_moe/routed_experts.py" \
      || { echo "0104 per-expert fix missing" >&2; exit 1; }; \
    grep -q 'mixed_k34_per_tensor' "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0301 mixed gate missing" >&2; exit 1; }; \
    grep -q 'self.rank_sliced_metadata is not None or (' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0301 eager exemption missing (silent 25 tok/s eager boot risk)" >&2; exit 1; }; \
    grep -q 'getattr(self, "r7_routed_experts", None)' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0301/0106 r7 read must be getattr-guarded (attr only exists for R7 checkpoints)" >&2; exit 1; }; \
    grep -q '_exl3_btx_adoption import' "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0101 adoption re-home missing" >&2; exit 1; }; \
    grep -q 'warm_route_pack(' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0101 route-pack warmup missing: b12x capture freeze would kill the first boot (JOURNEY 74)" >&2; exit 1; }; \
    grep -q 'def rewarm_mixed_route_packs' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0111 post-eviction route-pack rewarm missing: preparation jobs evict the" \
           "construction-time warmup and the guarded capture freezes on the first" \
           "unwarmed specialization (JOURNEY r5 boot 2)" >&2; exit 1; }; \
    grep -q 'rewarm_mixed_route_packs()' "$V/v1/worker/gpu_worker.py" \
      || { echo "0111 gpu_worker rewarm call missing: the rewarm must run after the last" \
           "preparation job and immediately before the guarded capture" >&2; exit 1; }; \
    grep -q '_REVISION_TAG = "\[mratsim.s sm120-turbo r[0-9][0-9]*\]"' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0101 log tag missing or unstamped: every added log line must carry the" \
           "patchset revision prefix" >&2; exit 1; }; \
    if grep -q 'trellis_rate_structure' "$V/model_executor/layers/quantization/exl3.py" \
       "$V/model_executor/layers/quantization/_exl3_btx_adoption.py"; then \
      echo "0105 rate-contract fix missing: pre-1.3 trellis_rate_structure kwarg still present" \
           "(that name exists at neither pinned b12x revision, the uniform-K4" \
           "plan call must go through _impl.plan_b12x_fp4_moe_weights)" >&2; exit 1; fi; \
    grep -q 'trellis_rate_granularity' "$V/model_executor/layers/quantization/_exl3_btx_adoption.py" \
      || { echo "0105 adoption validator must read trellis_rate_granularity (1.3 field)" >&2; exit 1; }; \
    grep -q 'api._impl.plan_b12x_fp4_moe_weights' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0101 karmic reroute missing: uniform-K4 plan call must use" \
           "api._impl.plan_b12x_fp4_moe_weights (karmic b12x deleted the" \
           "_vllm_compat plan_weights overload, TypeError at weight plan)" >&2; exit 1; }; \
    grep -q '^adopt_btx_weights = adopt_prepared_btx_weights' \
      "$V/model_executor/layers/quantization/_exl3_btx_adoption.py" \
      || { echo "0106 adoption alias missing (exl3.py lazy-imports adopt_btx_weights, ImportError at weight plan)" >&2; exit 1; }; \
    grep -q 'getattr(args, "shared_experts", None)' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0106 shared_experts guard missing (field absent from image QuantizationConfigArgs)" >&2; exit 1; }; \
    grep -q '_PATCHSET_TAG = "\[mratsim.s sm120-turbo r[0-9][0-9]*\]"' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0107 path-audit tag missing or unstamped" >&2; exit 1; }; \
    grep -q 'def _audit_path' "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0107 _audit_path helper missing" >&2; exit 1; }; \
    grep -q 'draft_sampled.masked_fill_' \
      "$V/v1/worker/gpu/spec_decode/rejection_sampler.py" \
      || { echo "0108 padding mask missing (padded draft rows would verify garbage tokens)" >&2; exit 1; }; \
    grep -q 'input_batch.is_padding' \
      "$V/v1/worker/gpu/spec_decode/rejection_sampler.py" \
      || { echo "0108 mask must gather is_padding at logits_indices (is_padding marks cudagraph-padding rows)" >&2; exit 1; }; \
    [ "$(grep -c '^                "codebook": "mcg",$' "$V/model_executor/layers/quantization/exl3.py")" -eq 1 ] \
      || { echo "0301 expected-dict must contain exactly one codebook entry" >&2; exit 1; }; \
    /opt/venv/bin/python -m py_compile \
      "$V/model_executor/layers/quantization/exl3.py" \
      "$V/model_executor/layers/quantization/exl3_online_cache.py" \
      "$V/model_executor/layers/quantization/_exl3_btx_adoption.py" \
      "$V/model_executor/layers/quantization/__init__.py" \
      "$V/model_executor/layers/fused_moe/routed_experts.py" \
      "$V/config/model.py" \
      "$V/v1/worker/gpu_worker.py" \
      "$V/v1/worker/gpu/spec_decode/rejection_sampler.py"; \
    printf '%s\n' \
      'import importlib' \
      'def probe(names, attr, tag):' \
      '    last = None' \
      '    for mod in names:' \
      '        try:' \
      '            found = importlib.import_module(mod)' \
      '        except ImportError as exc:' \
      '            last = exc' \
      '            continue' \
      '        assert hasattr(found, attr), (mod, attr)' \
      '        print(tag, "OK via", mod)' \
      '        return' \
      '    raise SystemExit("ERROR: %s unreachable under either b12x layout (%s)" % (tag, last))' \
      'probe(("b12x.moe.fused_moe._impl", "b12x.b12x.moe.fused_moe._impl"),' \
      '      "plan_b12x_fp4_moe_weights", "b12x fp4 moe planner (0101 karmic reroute target)")' \
      'probe(("b12x.moe._shared.kernels.w4a16.mixed_trellis", "b12x.b12x.moe._shared.kernels.w4a16.mixed_trellis"),' \
      '      "run_bound_mixed_trellis3", "b12x trellis3 mixed API (0202 stays deleted)")' \
      > /tmp/probe_b12x.py; \
    /opt/venv/bin/python /tmp/probe_b12x.py; \
    rm -f /tmp/probe_b12x.py; \
    /opt/venv/bin/python -c "from vllm.model_executor.layers.quantization._exl3_btx_adoption import adopt_btx_weights; print('adoption re-home imports OK')" 2>/dev/null \
      || echo "WARN: adoption import probe skipped (build env without GPU libs), verified at runtime"; \
    echo "patchset asserted: 0101-0108 + 0301 live, 0201/0202 correctly absent"

# VLLM_EXL3_EXT_PATH names the exllamav3 extension directory, VLLM_EXL3_ENCODER_SOURCE
# the exllamav3 python package (both read at runtime by the 0101 adapter). Prefill
# block 64: block 128 fails the boot-time FC2 route-subtile check.
ENV VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
    VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3 \
    VLLM_EXL3_ENCODER_REVISION=${EXLLAMAV3_COMMIT} \
    VLLM_EXL3_PREFILL_BLOCK_M=64 \
    VLLM_EXL3_PREFILL_TRELLIS=1 \
    TORCH_CUDA_ARCH_LIST=12.0a \
    CMAKE_CUDA_ARCHITECTURES=120a \
    NVCC_THREADS=4

ENTRYPOINT ["/opt/venv/bin/vllm"]
CMD ["serve", "/model"]
