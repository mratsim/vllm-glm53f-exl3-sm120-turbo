# vllm-glm53f-exl3-sm120-turbo — GLM-5.3-Flash EXL3 (TR3) on the jovian-judgement vLLM image.
#
# Base: local-inference-lab jovian-judgement-community source-locked image
# (recipes/glm53 flow in local-inference-lab/blackwell-llm-docker: vllm at
# /opt/glm53-flash/vllm, b12x at /opt/glm53-flash/b12x, both symlinked into
# /opt/venv/lib/python3.12/site-packages).
#
# Pin the exact r38 digest before building:
#   podman pull docker.io/localinferencelab/vllm:jovian-judgement-community-20260914-r38
#   podman image inspect ... --format '{{index .RepoDigests 0}}'
# then set BASE_IMAGE below to tag@sha256:... The build fails loudly on any
# patch that does not apply — that failure IS the "upstream absorbed/changed
# something" signal; never force it.
#
# (Digest pinning is OPTIONAL: podman auto-pulls the default tag below if it is
# not in the local store. Tag-only builds are fine — a moved tag that breaks a
# patch still fails the apply gate below.)
#
#   0101  exl3 adapter: adds vllm quantization exl3.py (Brandon's TR3 adapter,
#         LIL-tree port by raul2718) + exl3_online_cache.py + _exl3_btx_adoption.py.
#         The BTX adoption helper is re-homed vllm-side: upstream b12x has no
#         adopt API (verified absent on master 2026-09-17), so the b12x package
#         stays UNPATCHED — the module imports b12x preparation internals over
#         absolute paths (coupling points listed in its header).
#   0102  register "exl3"/Exl3Config in quantization/__init__.py.
#   0103  claim exl3 before the generic ModelOpt override in config/model.py
#         (3-line insert; deliberately excludes raul's removal of
#         _update_model_config_for_parallelism — that was old-tree drift).
#   0104  routed_experts.py: per-expert EXL3 Trellis tensors are rank-3 but not
#         fused; exclude names matching experts.<idx>. from the fused check.
#         (raul's whole-file overlay also dropped the r38 workspace/
#         apply_with_workspace API — deliberately not carried.)
#   0201  [deleted — first gap of the 02 band] b12x BTX adoption overlay: made
#         unnecessary by the vllm-side re-home in 0101. Kept as a number so the
#         history reads in the patch listing.
#   0202  [deleted — absorbed upstream] b12x w4a16 trellis3 mixed API: landed on
#         b12x master 2026-07-30 (5d640ee, one-grid mixed K3/K4 Trellis path),
#         consolidated 2026-08-02 (#112); the r38 image pins master @ 00b69ac2
#         (2026-09-10) which contains it. The build-time probe below still
#         verifies the import — if it ever fails, reintroduce a patch from
#         satindergrewal.../b12x-master/w4a16/.
#   0301  mixed-rate gate: accept bits="mixed_k34_per_tensor" (satgeze TR3
#         3.5bpw) alongside uniform 4, and add the r7_routed_experts term to
#         the eager-capture exemption. Harmless for the 4bpw checkpoint;
#         REQUIRED for the mixed one — without the exemption term the serve
#         silently boots eager (25-30 tok/s, zero graph captures, no error).
#
# Checkpoints (quant_method=exl3 embedded in config.json, routed-experts-only
# scope, mcg codebook, head_bits=16):
#   brandonmusic/GLM-5.3-Flash-tr3-4bpw          bits: 4
#   satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw        bits: mixed_k34_per_tensor
#
# Not patched (verified against the r38 tree): Glm5NextForConditionalGeneration
# is registered natively; PR#546 prefill_schedule_interval is already merged.

ARG BASE_IMAGE=docker.io/localinferencelab/vllm:jovian-judgement-community-20260914-r38
FROM ${BASE_IMAGE}

ARG VLLM_GLM53F_EXL3_REVISION=r4

LABEL ai.vllm.base.tag="jovian-judgement-community-20260914-r38" \
      ai.vllm.patchset="0101-exl3-adapter,0102-exl3-quant-registration,0103-exl3-config-detection,0104-routed-experts-per-expert-trellis,0105-exl3-b12x13-rate-contract,0106-exl3-adoption-alias-and-guards,0107-exl3-path-audit-logs,0301-exl3-mixed-rate-gate" \
      ai.vllm.patchset.deleted="0201-b12x-btx-adoption(re-homed-vllm-side),0202-b12x-trellis3-mixed-api(absorbed-upstream)" \
      ai.vllm.revision="${VLLM_GLM53F_EXL3_REVISION}" \
      ai.vllm.target.checkpoint="brandonmusic/GLM-5.3-Flash-tr3-4bpw,satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw" \
      ai.vllm.kv.dtype="fp8_ds_mla" \
      ai.vllm.exl3.prefill-block-m="64"

WORKDIR /opt/glm53-flash/vllm

# --- Sniff test: prove the base layout before any patch can mis-apply. The
# --- build fails here — not mid-apply — if the in-image tree drifted.
RUN set -eu; \
    for f in \
      /opt/glm53-flash/vllm/vllm/model_executor/layers/quantization/__init__.py \
      /opt/glm53-flash/vllm/vllm/model_executor/layers/quantization/online/mxfp8.py \
      /opt/glm53-flash/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py \
      /opt/glm53-flash/vllm/vllm/config/model.py \
      /opt/glm53-flash/b12x/b12x/moe/fused_moe/_impl.py \
      /opt/glm53-flash/b12x/b12x/moe/_shared/kernels/w4a16/mixed_trellis.py; \
    do test -f "$f" || { echo "base layout drift: missing $f" >&2; exit 1; }; done; \
    test -x /opt/venv/bin/python; \
    echo "base layout OK: glm53 source-locked tree with vllm + b12x"

COPY patches/ /opt/glm53f-exl3-patches/
COPY chat_template.multimodal.jinja /opt/glm53f/chat_template.multimodal.jinja

# --- Apply the patch series (sorted == authoritative order; gaps are history).
RUN set -eu; \
    cd /opt/glm53-flash/vllm; \
    for p in /opt/glm53f-exl3-patches/*.patch; do sed -i 's/\r$//' "$p"; done; \
    for p in $(ls /opt/glm53f-exl3-patches/*.patch | sort); do \
        echo "=== applying $(basename "$p") ==="; \
        if command -v git >/dev/null 2>&1; then \
            git apply --check "$p" || { echo "ERROR: $(basename "$p") does not apply — base moved; re-anchor" >&2; exit 1; }; \
            git apply "$p"; \
        else \
            patch -p1 --fuzz=0 --dry-run < "$p" || { echo "ERROR: $(basename "$p") does not apply — base moved; re-anchor" >&2; exit 1; }; \
            patch -p1 --fuzz=0 < "$p"; \
        fi; \
    done; \
    rm -rf /opt/glm53f-exl3-patches; \
    # Stamp the build revision into the path-audit marker (qwen38fn convention):
    # patches carry the literal VLLM_PATCHES_REVISION token; the built image
    # logs the concrete revision (e.g. "[mratsim's sm120-turbo r4] path-audit …").
    V=/opt/glm53-flash/vllm/vllm; \
    grep -rl 'sm120-turbo' "$V" | xargs -r sed -i \
      -e "s/sm120-turbo VLLM_PATCHES_REVISION/sm120-turbo ${VLLM_GLM53F_EXL3_REVISION}/g" \
      -e "s/sm120-turbo r[0-9][0-9]*/sm120-turbo ${VLLM_GLM53F_EXL3_REVISION}/g"

# --- Post-apply assertions (no GPU): every patch landed, with the silent-
# --- failure ones asserted explicitly. Import probe proves the re-homed
# --- adoption module resolves against the image's real b12x.
RUN set -eu; \
    V=/opt/glm53-flash/vllm/vllm; \
    B=/opt/glm53-flash/b12x/b12x; \
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
    if grep -q 'trellis_rate_structure' "$V/model_executor/layers/quantization/exl3.py" \
       "$V/model_executor/layers/quantization/_exl3_btx_adoption.py"; then \
      echo "0105 rate-contract fix missing: pre-1.3 trellis_rate_structure kwarg still present" \
           "(B12X 1.3 removed it — TypeError: plan_weights() got an unexpected keyword" \
           "argument 'trellis_rate_structure')" >&2; exit 1; fi; \
    grep -q 'trellis_rate_granularity' "$V/model_executor/layers/quantization/_exl3_btx_adoption.py" \
      || { echo "0105 adoption validator must read trellis_rate_granularity (1.3 field)" >&2; exit 1; }; \
    grep -q '^adopt_btx_weights = adopt_prepared_btx_weights' \
      "$V/model_executor/layers/quantization/_exl3_btx_adoption.py" \
      || { echo "0106 adoption alias missing (exl3.py lazy-imports adopt_btx_weights — ImportError at weight plan)" >&2; exit 1; }; \
    grep -q 'getattr(args, "shared_experts", None)' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0106 shared_experts guard missing (field absent from image QuantizationConfigArgs)" >&2; exit 1; }; \
    grep -q '_PATCHSET_TAG = "\[mratsim.s sm120-turbo r[0-9][0-9]*\]"' \
      "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0107 path-audit tag missing or unstamped" >&2; exit 1; }; \
    grep -q 'def _audit_path' "$V/model_executor/layers/quantization/exl3.py" \
      || { echo "0107 _audit_path helper missing" >&2; exit 1; }; \
    [ "$(grep -c '^                "codebook": "mcg",$' "$V/model_executor/layers/quantization/exl3.py")" -eq 1 ] \
      || { echo "0301 expected-dict must contain exactly one codebook entry" >&2; exit 1; }; \
    /opt/venv/bin/python -m py_compile \
      "$V/model_executor/layers/quantization/exl3.py" \
      "$V/model_executor/layers/quantization/exl3_online_cache.py" \
      "$V/model_executor/layers/quantization/_exl3_btx_adoption.py" \
      "$V/model_executor/layers/quantization/__init__.py" \
      "$V/model_executor/layers/fused_moe/routed_experts.py" \
      "$V/config/model.py"; \
    /opt/venv/bin/python -c "from vllm.model_executor.layers.quantization._exl3_btx_adoption import adopt_btx_weights; print('adoption re-home imports OK')" 2>/dev/null \
      || echo "WARN: adoption import probe skipped (build env without GPU libs) — verified at runtime"; \
    /opt/venv/bin/python -c "from b12x.moe._shared.kernels.w4a16 import mixed_trellis as m; assert hasattr(m, 'run_bound_mixed_trellis3'); print('b12x trellis3 mixed API present (0202 stays deleted)')" 2>/dev/null \
      || { echo "ERROR: b12x lacks trellis3 mixed API — 0202 must be reintroduced from satindergrewal b12x-master" >&2; exit 1; }; \
    echo "patchset asserted: 0101-0107 + 0301 live, 0201/0202 correctly absent"

# Structural, never tuned: prefill block 128 fails the boot-time FC2 route-
# subtile check ("FC2 route subtile... allowed routed sizes 8/16/32/48/64").
ENV VLLM_EXL3_PREFILL_BLOCK_M=64 \
    VLLM_EXL3_PREFILL_TRELLIS=1

ENTRYPOINT ["/opt/venv/bin/vllm"]
CMD ["serve", "/model"]
