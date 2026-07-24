# V100 36-Probe + `v100_2500` Benchmark Report

- hardware: `Tesla V100-SXM2-32GB` (`volta`, `sm_70`)
- driver: `535.261.03`
- benchmark generator: `ppp_canonical` r1
- benchmark case-set digest: `6e31a0f673fce3f8d207d80e94b26dbb444f9805eb4e0c8d0b8ff3e8998ad6e6`
- classification digest: `20812205100a965414c2a35b193d10cda3a03d6d72c4fbd007010847b453099e`
- GCoM V100 status: unavailable; no Volta simulator SKU/profile is configured

## Validation anchors

- 36-probe run: 36 probes rendered to `reports/nvidia/volta/probes-v100-32g.md`
- `v100_2500` materialization: 2500 cases, `megamoe_fp8` excluded
- benchmark classification attempt: 16 attempted of 2500 materialized cases
- classification statuses: `failed`=16
- rank overlay: unavailable because no attempted case produced NCU counters

## V100 36-Probe Summary

| fit_status | count |
|---|---:|
| `behavioral_only` | 8 |
| `bounded` | 7 |
| `conditionally_identified` | 3 |
| `direct` | 5 |
| `underconstrained` | 3 |
| `uniquely_identified` | 7 |
| `unsupported` | 3 |

### Key Scalar Probes

| probe | measurement | fit_status |
|---|---:|---|
| `topology.persistent_cta` | 13 blocks | `uniquely_identified` |
| `arithmetic_latency.dependent_chain` | 4.376 cycles_per_op | `direct` |
| `arithmetic_throughput.independent_chains` | 2.1097 cycles_per_op | `uniquely_identified` |
| `shared_memory.pointer_chase` | 26.9988 cycles | `direct` |
| `shared_memory.bank_stride` | 32 banks | `uniquely_identified` |
| `l1_cache.pointer_chase` | 59.0574 cycles | `direct` |
| `scheduler_policy.ready_warps` | 12 warps | `conditionally_identified` |
| `register_file.register_latency` | 1.4382 cycles | `conditionally_identified` |
| `synchronization.barrier_latency` | 57.001 cycles | `uniquely_identified` |
| `l2_cache.pointer_chase` | 246.2571 cycles | `bounded` |
| `global_memory.row_policy_sweep` | 1.5327962187130735 ratio | `bounded` |
| `tensor_core.mma_latency` | 64.3555 cycles_per_op | `uniquely_identified` |
| `tensor_core.mma_throughput` | 0.009 mma/cycle | `uniquely_identified` |
| `interconnect.injection_rate` | 869.47 GB/s | `bounded` |

## `v100_2500` Materialization

| kernel | cases |
|---|---:|
| `aligned_gemm_fp16` | 313 |
| `embedding` | 313 |
| `flash_attention_fwd` | 313 |
| `flashmla_dense_decode` | 313 |
| `gelu` | 312 |
| `gelu_gemm_fp16` | 312 |
| `rmsnorm` | 312 |
| `rmsnorm_gemm_fp16` | 312 |

`megamoe_fp8` is intentionally absent because V100 has no FP8 tensor-core support.

## Benchmark Classification Attempt

| status | count |
|---|---:|
| `failed` | 16 |

### Failure Reasons

| reason | count |
|---|---:|
| `ERR_NVGPUCTRPERM: user lacks permission to access NVIDIA GPU Performance Counters` | 16 |

The metric catalog is available on this host and resolves `inst_executed`, `elapsed_cycles`, and `duration_ns`. Actual NCU collection fails with `ERR_NVGPUCTRPERM`, so no small/medium/large rank overlay can be frozen yet.

## Next Steps

1. Enable NVIDIA performance counters for the VM/user, then rerun full `v100_2500` classification.
2. Once all 2500 cases classify, freeze the V100 rank overlay and run small-rank detailed NCU evidence.
3. Keep GCoM comparison unavailable for V100 until a Volta simulator profile exists.
