#!/bin/bash
# Probe an arbitrary jovian-community image and report which patches in
# patches/ are still needed, which got absorbed upstream, and whether any
# deleted patch number must come back. Run before building after a base bump:
#
#   ./tools/check-optional.sh docker.io/localinferencelab/vllm:jovian-judgement-community-20260914-r39
set -euo pipefail

IMAGE="${1:?usage: check-optional.sh IMAGE}"
PY=${PY:-python3}

probe() {  # probe <name> <image-cmd...>
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  [absorbed] ${name}: present upstream — patch may be dropped"
    else
        echo "  [needed  ] ${name}: absent upstream — patch required"
    fi
}

echo "Probing ${IMAGE}"
podman run --rm --entrypoint /bin/bash "${IMAGE}" -lc "
V=/opt/glm53-flash/vllm/vllm
B=/opt/glm53-flash/b12x/b12x

echo 'layout:'
for f in \$V/model_executor/layers/quantization/__init__.py \
         \$V/model_executor/layers/fused_moe/routed_experts.py \
         \$V/config/model.py \
         \$B/moe/fused_moe/_impl.py \
         \$B/moe/_shared/kernels/w4a16/mixed_trellis.py; do
  test -f \$f && echo \"  ok      \$f\" || echo \"  MISSING \$f\"
done

echo 'patch relevance:'
${PY} - <<'PYEOF'
import pathlib
v = pathlib.Path('/opt/glm53-flash/vllm/vllm')
b = pathlib.Path('/opt/glm53-flash/b12x/b12x')

qinit = (v/'model_executor/layers/quantization/__init__.py').read_text()
if '\"exl3\"' in qinit:
    print('  [absorbed] 0102 registration: quantization registry already lists exl3')
else:
    print('  [needed  ] 0102 registration')

model = (v/'config/model.py').read_text()
if '\"exl3\",' in model and 'modelopt_mixed' in model:
    print('  [absorbed] 0103 config detection: exl3 claimed pre-ModelOpt')
else:
    print('  [needed  ] 0103 config detection')

routed = (v/'model_executor/layers/fused_moe/routed_experts.py').read_text()
if '_PER_EXPERT_IDX_RE' in routed:
    print('  [absorbed] 0104 per-expert trellis fix')
else:
    print('  [needed  ] 0104 per-expert trellis fix')

exl3 = v/'model_executor/layers/quantization/exl3.py'
if exl3.exists():
    print('  [absorbed] 0101 exl3.py: file exists upstream — verify it carries r7/mixed support')
else:
    print('  [needed  ] 0101 exl3 adapter (new files)')

# 0202 history check: if trellis3 mixed API ever disappears, it must come back
mt = b/'moe/_shared/kernels/w4a16/mixed_trellis.py'
if mt.exists() and 'run_bound_mixed_trellis3' in mt.read_text():
    print('  [absorbed] 0202 b12x trellis3 mixed API (correctly deleted)')
else:
    print('  [REGRESS ] 0202 must be reintroduced from satindergrewal b12x-master/w4a16')

# if b12x ever ships an adoption API natively, 0101's re-home becomes optional
api = b/'moe/fused_moe/api.py'
if 'adopt_btx_weights' in api.read_text():
    print('  [absorbed] 0101 adoption helper: b12x ships adopt_btx_weights natively')
else:
    print('  [needed  ] 0101 adoption re-home (b12x still has no adopt API)')
PYEOF
"
