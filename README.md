# vllm-glm53f-exl3-sm120-turbo

This project runs the GLM-5.3-Flash language model on a machine with two
big NVIDIA GPUs. It is a small pile of scripts and patches on top of a
ready-made server image. Pick a script and run it. The model answers
chat requests.

## How to use it

1. Build the image once:
   ```bash
   podman build -t localhost/vllm-glm53f-exl3-sm120-turbo:r4 .
   ```
2. Run one of the serve scripts (see the table below). Each script starts
   the server and waits for it to become healthy.
3. Talk to the server on the port the script prints.

The image banner self-reports `0.26.1rc0+glm53.r38.vllm66c29357`. Patches
are numbered flat files. Gaps in the numbers are deleted history (the
story lives in `internal/JOURNEY.md` on the build machine).

## Which script serves what

Eight scripts. The name tells you the setup: checkpoint width, KV
cache type (kv8), speculation mode, KV split (dcp). The step size
(b2048 or b4096, tokens per prefill step) is in every name. The 4bpw
names end with the prefill block (pb32 or pb64).

| Script | Checkpoint | Speculation | KV split | Step size | Prefill block |
|---|---|---|---|---|---|
| `serve_vllm_GLM53f-exl3-4bpw-kv8-mtp3-dcp2-b2048-pb64.kk.beta.sh` | brandonmusic, uniform K4 | MTP-3 | DCP=2 | 2048 | 64 |
| `serve_vllm_GLM53f-exl3-4bpw-kv8-mtp3-dcp2-b2048-pb32.kk.beta.sh` | same, prefill block 32 | MTP-3 | DCP=2 | 2048 | 32 |
| `serve_vllm_GLM53f-exl3-35bpw-kv8-dflash-dcp1-b2048.kk.beta.sh` | satgeze, mixed K3/K4 | DFlash2-7 | DCP=1 | 2048 | 64 |
| `serve_vllm_GLM53f-exl3-35bpw-kv8-dflash-dcp2-b2048.kk.beta.sh` | satgeze, mixed K3/K4 | DFlash2-7 | DCP=2 | 2048 | 64 |
| `serve_vllm_GLM53f-exl3-35bpw-kv8-dflash-dcp1-b4096.kk.beta.sh` | satgeze, mixed K3/K4 | DFlash2-7 | DCP=1 | 4096 | 64 |
| `serve_vllm_GLM53f-exl3-35bpw-kv8-dflash-dcp2-b4096.kk.beta.sh` | satgeze, mixed K3/K4 | DFlash2-7 | DCP=2 | 4096 | 64 |
| `serve_vllm_GLM53f-exl3-35bpw-kv8-mtp3-dcp1-b2048.kk.beta.sh` | satgeze, mixed K3/K4 | MTP-3 | DCP=1 | 2048 | 64 |
| `serve_vllm_GLM53f-exl3-35bpw-kv8-mtp3-dcp2-b2048.kk.beta.sh` | satgeze, mixed K3/K4 | MTP-3 | DCP=2 | 2048 | 64 |

All eight run the b12x backends. They serve as `GLM-5.3-Flash` with
auto context (`max-model-len -1`: the model cap of 1,048,576 clamped
by the KV pool) and KV cache dtype `fp8_ds_mla`. The b2048 dflash
pair and the two MTP scripts form a 2x2 grid: two speculation modes
times two split sizes. Any two cells differ in exactly one thing so
comparisons are clean. The b4096 dflash pair is the wider-step
variant of the grid.

### The KV pool, context size, concurrency

The scripts reserve the KV pool (`kv-cache-memory-bytes`, 3.9 GiB
per GPU on the NVFP4 preset) and leave `max-model-len` at `-1`
(auto). Two numbers come out of that. Both move with the settings:

- **Per-request cost** = `A * length + F`. `A` is the per-token KV
  cost (about 3,366 bytes per token per GPU at fp8 with DCP2 after
  the cache-group rebalance). `F` is the fixed per-request state
  (GDN/mamba checkpoint bundles, about 0.9 GiB at the 4-sequence
  worst case). The pool holds `pool / cost` concurrent requests.
- **The reported "GPU KV cache size"** is `pool / (A + F/cap)`, where
  cap is `max-model-len`. The fixed per-request state is amortized
  over the cap, so a bigger cap reports a bigger number for the same
  pool: a 327,680 cap reported 661,913 tokens, the 1,048,576 cap
  reports 953,418. Concurrency at any given length stays the same
  (about 2 requests of 327,680 either way). What the cap really sets
  is the longest single request the server admits.

Example mixes on the 3.9 GiB pool (per GPU, engine reservation
arithmetic):

| Mix | Cost | Fits |
|---|---|---|
| 1 x 950,272 tokens | 3.88 GiB | yes (1.00x) |
| 2 x 327,680 tokens | 3.85 GiB | yes (2.02x) |
| 3 x 327,680 tokens | 5.78 GiB | no |
| 4 x 163,840 tokens | 5.65 GiB | no |
| 4 x 131,072 tokens | 5.24 GiB | no |
| 4 x 65,536 tokens | 4.42 GiB | no (close) |
| 3 x 65,536 tokens | 3.31 GiB | yes |
| 4 x 32,768 tokens | 3.98 GiB | no (hair over) |
| 3 x 32,768 tokens | 3.00 GiB | yes |

Concurrency is expensive per request: every extra concurrent
request adds its own fixed state bundle on top of its token cost. The `F` above is
the worst case (full 4-sequence bundle), so lighter mixes come out
slightly better than this arithmetic.

> [!TODO]
> Explore `--mamba-ssm-cache-dtype bfloat16`. The SSM state runs fp32
> by default (auto resolves to float32 for KDA) and the flag halves
> those bytes. The raw states are a small slice of `F`, so measure
> the effect on the rebalance line's max-request cost before
> adoption.

Two words explained:

- **Speculation**: a small extra model guesses the next few tokens. The
  big model checks the guesses. Wrong guesses are thrown away, so the
  output stays exactly the same, just faster.
- **DCP**: the stored conversation memory (KV cache) is cut in half and
  each GPU keeps one half. Same memory, twice the conversation room. It
  costs some speed per step (see the numbers below).

## Measured results

> [!WARNING]
> **Every number in this section was measured on the previous base image**
> `jovian-judgement-community-20260914-r38`
> (vLLM built from `lil-vllm` `dev/jovian-judgement` at commit
> [`66c29357`](https://github.com/local-inference-lab/lil-vllm), the image
> banner commit, with the checkout tip at `5bca5a58d9`). The stack has since
> moved to the **`karmic-kraken-beta` base** (vLLM wheel built from source
> commit `67bb922f6f4` on `integration/karmic-kraken-beta`, b12x at
> `eea3ced11fc1`). Those numbers have **not** been re-measured on the
> karmic base. Treat them as indicative of the r38 rig behavior until
> the karmic re-benchmark numbers exist.

All numbers measured on this rig: two RTX PRO 6000 Blackwell cards on
the NVIDIA open kernel driver, PCIe gen5 x8/x8, power-limited to
360 W per GPU.

**No PCIe P2P**: the open driver does not support
card-to-card transfers over PCIe
([open-gpu-kernel-modules #1215](https://github.com/NVIDIA/open-gpu-kernel-modules/discussions/1215)),
so card-to-card traffic crosses the CPU over host NCCL (the launchers
set `NCCL_P2P_DISABLE=1` with `VLLM_ENABLE_PCIE_ALLREDUCE=0` for this).

Benchmarked with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench).

External prefill claims run higher power envelopes (cstechdev 400 W
and Raul2718 500 W).

| Checkpoint | Speculation | KV split | Step size | Prefill block M | KV pool | KV tokens | Chats @ 327,680 | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| satgeze 3.5bpw | MTP-3 | DCP=2 | 2048 | 64 | 11.07 GiB | 2,460,812 | 7.51x | 3,300 | 138 |
| satgeze 3.5bpw | MTP-3 | DCP=1 | 2048 | 64 | 7.6 GiB | 990,572 | 3.02x | 4,300 | 170 |
| satgeze 3.5bpw | DFlash2-7 | DCP=2 | 2048 | 64 | 14.05 GiB | 1,583,786 | 4.83x | 3,500 | 135 |
| satgeze 3.5bpw | DFlash2-7 | DCP=1 | 2048 | 64 | 13.89 GiB | 1,015,165 | 3.10x | 4,400 | 150 |
| satgeze 3.5bpw | DFlash2-7 | DCP=1 | 4096 | 64 | 8.87 GiB | 639,453 | 1.95x | 4,850 | 145 |
| satgeze 3.5bpw | DFlash2-7 | DCP=2 | 4096 | 64 | 10.22 GiB | 1,134,653 | 3.46x | 3,600 | 145 |
| satgeze 3.5bpw | MTP-3 | DCP=2 | 2048 | 64 | 11.11 GiB | 3,096,774 | 3.10x (at 1M len) | - | - |
| brandonmusic 4bpw | MTP-3 | DCP=2 | 2048 | 64 | 2.69 GiB | 597,534 | 1.82x | - | - |
| brandonmusic 4bpw | MTP-3 | DCP=2 | 2048 | 32 | 6.27 GiB | 1,394,246 | 4.25x | 3,500 | 130 |

The two 4096-step rows ran at a lower memory fraction: 0.95 for the
DCP=1 row, 0.96 for the DCP=2 row (not 0.986). Their KV pools are
smaller than the same geometry at 2048. Prefill gains: 4096 steps
gave +10% at DCP=1 and nothing at DCP=2.

What the numbers say so far:

- Goal: about 5,000 prefill and 170 decode tok/s.
- Splitting the KV cache (DCP=2) costs speed single-stream: decode
  -10% for dflash, -18.8% for MTP, prefill -20.5% for dflash, -23.3%
  for MTP (vs DCP=1). In return the KV pool holds 1.56x more tokens
  for dflash, 2.48x for MTP. DCP=1 wins alone, DCP=2 wins with many
  users.
- How much DCP really multiplies your tokens: 1.71x for MTP, 1.55x for
  dflash (its draft memory cannot be split). Both stay below 2: the
  running-summary memory cannot be split.
- Raising MAX_NUM_BATCHED_TOKENS from 2048 to 4096 gave +10%
  prefill (4,850 vs 4,400 tok/s) at the same decode. On DCP=2 it
  gave nothing (3,600 vs 3,500). The cost is memory: the prefill
  scratch buffers grew from 478 to 862 MiB, so the KV pool shrinks.

## Patches

| # | Name | What it does |
|---|---|---|
| 0101 | exl3-adapter | New files: `exl3.py` (the TR3 adapter, LIL-tree port via raul2718), `exl3_online_cache.py`, `_exl3_btx_adoption.py`. The adoption helper lives vllm-side: upstream b12x has no adopt API (verified 2026-09-17), so b12x stays unpatched. |
| 0102 | exl3-quant-registration | Registers the "exl3" quantization method. |
| 0103 | exl3-config-detection | Claims exl3 configs before ModelOpt (insert only). |
| 0104 | routed-experts-per-expert-trellis | Per-expert Trellis rank-3 non-fused fix in `routed_experts.py`. |
| 0105 | exl3-b12x13-rate-contract | Omits the removed pre-1.3 `trellis_rate_structure` kwarg. |
| 0106 | exl3-adoption-alias-and-guards | `adopt_btx_weights` alias plus getattr guards. |
| 0107 | exl3-path-audit-logs | Log witnesses at every quantization decision point, tagged with the revision. |
| 0301 | exl3-mixed-rate-check | Accepts `mixed_k34_per_tensor` bits and a guarded R7 term in the eager-capture exemption. |

## Gotchas

- `VLLM_EXL3_PREFILL_BLOCK_M=64` is required for the mixed 3.5bpw
  checkpoint (128 fails a startup check there, allowed sizes are
  8/16/32/48/64) and is built into the image. The uniform-K4 4bpw
  checkpoint has no such limit. Block 32 beat block 64 in an isolated
  kernel microbench (288 experts in one process). Each expert got about
  28 rows on average, so 64 padded half of every step. The serving
  config runs expert parallelism with 144 experts per GPU, which
  doubles the rows per expert and inverts that result: the block-32
  serving run measured 3,500 prefill / 130 decode and did not confirm
  the win. Block 64 stays the default. The `-pb32` script remains as an
  experiment. 128 does not work on this image.
- KV cache dtype is `fp8_ds_mla`. Never `nvfp4_ds_mla` here: it needs a
  calibration scales file that only exists in the v84 verdictai image.
- vLLM reads the `quantization_config` embedded in `config.json`.
  Editing a standalone `quantization_config.json` does nothing.
- Never pass `--generation-config` with a path. It treats the value as
  an HF repo id and fails on local paths. Sampler defaults go through
  `--override-generation-config`. Reasoning effort goes through
  `--default-chat-template-kwargs.reasoning_effort` (low or high, the template default max talks too much).
- Expert parallelism needs the unsliced-K4 expert layout, so only the
  4bpw launcher passes `--enable-expert-parallel`.
- Keep cp/dcp KV interleave sizes equal (4).
- If a serve looks slow: grep the boot log for `enforce_eager` and count
  `Capturing CUDA graph` lines before touching anything else.

### Split cache pages (dflash only, set in the launcher)

The model keeps its conversation memory in small notebook pages. A
checker inside the model (the pooled indexer) walks these pages in
fixed steps of 8448 bytes. Every stack of pages must fit those steps
exactly, like books lined up on a shelf.

The dflash draft keeps its own second stack of notebook pages. The
server stores both stacks on one shelf with one shared page size. The
two sizes do not line up. The server refuses to start and prints
"GLM MLA parent-page stride must be an exact number of C4 pages"
(boots #7b and #7c).

The fix: cut the big model's pages so both stacks line up with the
checker's step. The launcher sets `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE
=4608`. The math, with 561 bytes of page payload:

```
4608 pages x 561 bytes = 8448 x 306 = a whole number of C4 index pages
```

Why not use much smaller pages? One piece of state (the KDA running
summary) is written once per page no matter how small the page is
(2,342,912 bytes each time). Small pages mean writing it far more
often:

- At page size 512 the cost was 9x per token and the memory pool fell
  to 797,900 tokens (page-splitting experiment).
- At page size 4608 the pool holds 1,583,786 tokens (measured serving config).

MTP does not need the split: its draft is one shared head
with almost no KV of its own.

The draft attention must advertise `supports_dcp_replicated`, a
property only `FLASH_ATTN` has at this base commit (`TRITON_ATTN`
fails), while prefix caching stays on.

## When the base image updates

```bash
podman pull <new-tag>
./tools/check-optional.sh <new-tag>   # which patches did the new image absorb?
# then update BASE_IMAGE in the Dockerfile and rebuild.
# a failing git apply --check means re-anchor the patches. Never force it.
```

The tag follows `<stack-name>:<revision>`. On every revision bump
keep these three in sync:
- the Dockerfile `ARG`
- the podman tag
- the script `IMAGE` lines

The Dockerfile stamps the revision into the 0107 log marker and asserts
the token is fully consumed.

## Credits

This work stands on the shoulders of others:
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
- local-inference-lab ([GitHub](https://github.com/local-inference-lab),
  [Hugging Face](https://huggingface.co/local-inference-lab)),
  particularly [lukealonso](https://github.com/lukealonso) and
  [voipmonitor](https://github.com/voipmonitor), for the source-locked
  [vLLM fork](https://github.com/local-inference-lab/vllm),
  [b12x](https://github.com/local-inference-lab/b12x) + a wealth of recipes, including the [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) benchmark tool.
- [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  for the reference EXL3/TR3 checkpoint and quantization recipes from which everything derived.
- satgeze/satindergrewal:
  [satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bp](https://huggingface.co/satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw)
  for the mixed K3/K4 3.5bpw checkpoint,
  [satindergrewal/GLM-5.3-Flash-EXL3-3.5bpw-Mixed-SM120-TP2](https://github.com/satindergrewal/GLM-5.3-Flash-EXL3-3.5bpw-Mixed-SM120-TP2)
  for the reference recipe that checkpoint is served from.
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  for the DFlash2 draft model, with
  [local-inference-lab's MXFP8 conversion](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).
- [Raul2718/glm53-flash-exl3-lil](https://github.com/Raul2718/glm53-flash-exl3-lil)
  for the LIL-tree port + patches + measurements at 500 W.
- [legend/glm-5.3-flash-exl3-4bpw](https://github.com/legend/glm-5.3-flash-exl3-4bpw)
  for reference recipes and measurements
- cstechdev for docker builds and measurements
- [vLLM](https://github.com/vllm-project/vllm) for the serving engine.
