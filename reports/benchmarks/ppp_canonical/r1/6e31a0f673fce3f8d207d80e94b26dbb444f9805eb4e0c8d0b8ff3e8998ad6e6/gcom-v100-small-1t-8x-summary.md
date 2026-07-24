# V100 2500 Small-Rank GCoM Run Summary

## Run

- Run ID: `v100-small-gcom-mainbar-1t-8x-20260723T143752`
- Benchmark: `ppp_canonical`
- Rank: `small`
- Case-set digest:
  `6e31a0f673fce3f8d207d80e94b26dbb444f9805eb4e0c8d0b8ff3e8998ad6e6`
- Classification digest:
  `d4cac9b03a0cc0fb6195c12695cce22ae88ae9ff7184d8ec650b7cb18dd950cd`
- Execution shape: 8 disjoint shards, each running GCoM with `omp_threads=1`
- GCoM root:
  `/data00/home/chun.liu/wk/modern-gpu-simulator-micro-2025/simulator-remodeled`
- Trace compile arch: `AMORA_GCOM_NVCC_ARCH=sm_70`

## Outcome

All 834 small-rank cases were attempted and recorded.

| status | count |
|---|---:|
| `simulated` | 98 |
| `failed` | 640 |
| `missing_stat` | 96 |
| total | 834 |

## Per-Shard Counts

| shard | cases | simulated | failed | missing_stat |
|---|---:|---:|---:|---:|
| `shard-00-of-08` | 105 | 12 | 80 | 13 |
| `shard-01-of-08` | 105 | 10 | 86 | 9 |
| `shard-02-of-08` | 104 | 14 | 78 | 12 |
| `shard-03-of-08` | 104 | 13 | 75 | 16 |
| `shard-04-of-08` | 104 | 13 | 79 | 12 |
| `shard-05-of-08` | 104 | 9 | 80 | 15 |
| `shard-06-of-08` | 104 | 15 | 81 | 8 |
| `shard-07-of-08` | 104 | 12 | 81 | 11 |

## Failure Modes

| mode | count |
|---|---:|
| `simulation exceeded 300s timeout` | 637 |
| `simulator emitted no gpu_sim_cycle` | 96 |
| tracing timeout | 3 |

Tracing timeouts:

- `rmsnorm_gemm_fp16`:
  `ppp_canonical:r1:sm_70_v100:rmsnorm_gemm_fp16:r1:K256_M12288_N256`
- `flashmla_dense_decode`:
  `ppp_canonical:r1:sm_70_v100:flashmla_dense_decode:r1:B1_H1_S4096`
- `flash_attention_fwd`:
  `ppp_canonical:r1:sm_70_v100:flash_attention_fwd:r1:B2_D96_H1_S384`

## Kernel Breakdown

### Simulated

| kernel | count |
|---|---:|
| `gelu_gemm_fp16` | 31 |
| `embedding` | 28 |
| `rmsnorm` | 16 |
| `flashmla_dense_decode` | 12 |
| `rmsnorm_gemm_fp16` | 9 |
| `flash_attention_fwd` | 2 |

### Failed

| kernel | count |
|---|---:|
| `embedding` | 131 |
| `rmsnorm_gemm_fp16` | 130 |
| `gelu_gemm_fp16` | 108 |
| `rmsnorm` | 85 |
| `flashmla_dense_decode` | 83 |
| `flash_attention_fwd` | 56 |
| `aligned_gemm_fp16` | 47 |

### Missing `gpu_sim_cycle`

| kernel | count |
|---|---:|
| `aligned_gemm_fp16` | 96 |

## Notes

- The run used eight independent single-threaded GCoM shard workers instead of
  one OpenMP-heavy process.
- No NCU detail collection was run as part of this sharded GCoM-only pass.
- All shard status files ended in `state=complete`.
- No `ncu`, `amora benchmarks detail`, `run_v100_gcom_small_shard`, or
  `accel-sim.out` process remained after completion.
