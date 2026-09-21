#!/bin/bash
set -euo pipefail

# Build the GLM-5.3-Flash EXL3 serving image (r6, karmic-kraken base)
# Usage: ./build-image.sh [tag]     (default tag: r6)
# The ccache store lives in internal/ (gitignored) so rebuilds reuse
# compiled objects across patch edits and base bumps.

TAG="${1:-r6}"
IMAGE="localhost/vllm-glm53f-exl3-sm120-turbo:${TAG}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${DIR}/internal/ccache"

cd "${DIR}"
podman build --format docker \
    -v "${DIR}/internal/ccache:/opt/glm53f-ccache" \
    -f Dockerfile \
    -t "${IMAGE}" \
    .

printf 'Built %s\n' "${IMAGE}"
