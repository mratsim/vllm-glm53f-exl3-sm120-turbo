#!/usr/bin/env python3
"""Measure the vision-tower weight bytes in a GLM-5.3-Flash EXL3 checkpoint.

Reads only the safetensors headers (no tensor data is loaded), so it runs in
seconds regardless of checkpoint size. Reports total tower bytes and the
per-rank share under TP2 (the ViT is TP-split by default:
mm_encoder_tp_mode="weights").

Usage:
  ./measure-vision-tower.py <checkpoint-dir>

The checkpoint dir must contain model.safetensors.index.json. No paths are
assumed or embedded here.
"""
import collections
import json
import struct
import sys
from pathlib import Path

# Bytes per element for the dtypes that appear in GLM-5.3 checkpoints.
DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "U8": 1, "I8": 1, "I32": 4, "I64": 8, "BOOL": 1,
    "U16": 2, "I16": 2, "U32": 4, "U64": 8,
}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: measure-vision-tower.py <checkpoint-dir>")
    ckpt = Path(sys.argv[1])
    index_path = ckpt / "model.safetensors.index.json"
    if not index_path.is_file():
        sys.exit(f"no index at {index_path} — pass a checkpoint dir as argv[1]")

    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    vision = {
        k: f for k, f in weight_map.items()
        if "vision" in k.lower() or "visual" in k.lower()
    }
    if not vision:
        sys.exit("no vision/visual tensors found — is this the right checkpoint?")

    by_file = collections.defaultdict(list)
    for key, fname in vision.items():
        by_file[fname].append(key)

    total = 0
    per_dtype = collections.Counter()
    unmatched = []
    for fname, keys in by_file.items():
        with open(ckpt / fname, "rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(header_len))
        for key in keys:
            entry = header.get(key)
            if entry is None:
                unmatched.append(key)
                continue
            numel = 1
            for dim in entry["shape"]:
                numel *= dim
            nbytes = numel * DTYPE_BYTES.get(entry["dtype"], 2)
            total += nbytes
            per_dtype[entry["dtype"]] += nbytes

    gib = total / 2**30
    print(f"checkpoint:        {ckpt}")
    print(f"vision tensors:    {len(vision)}")
    print(f"vision tower:      {gib:.2f} GiB total (bf16-equivalent reads)")
    for dtype, nbytes in per_dtype.most_common():
        print(f"  {dtype:<10} {nbytes / 2**30:7.2f} GiB")
    print(f"per-rank under TP2 (weights mode): {gib / 2:.2f} GiB")
    print(f"per-rank if mm_encoder_tp_mode=data: {gib:.2f} GiB")
    if unmatched:
        print(f"WARNING: {len(unmatched)} keys not found in shard headers "
              f"(stale index?): e.g. {unmatched[:3]}")


if __name__ == "__main__":
    main()
