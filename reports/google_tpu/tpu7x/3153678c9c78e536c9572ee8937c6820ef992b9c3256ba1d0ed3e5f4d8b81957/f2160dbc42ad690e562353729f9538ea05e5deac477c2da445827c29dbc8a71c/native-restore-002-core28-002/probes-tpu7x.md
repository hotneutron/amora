# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-restore-002-core28-002
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 32
- `fit_status`: `direct`=32
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "1358bbee677ecfeceb93e0ce440171c2a3ace658f4e569970f12cccf6130cca8", "compile_ns": 305925843, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "0a8e80417464c2a8e2758625b5bbc9fb6efa52c0f4c8acc0eeb0a09ac471eccf"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "1f14e58b839ec9b3676e9d40ea965da96bc46f0df583aa2e8511086133eeb1ee"}}, "dynamic_argument_count": 1, "first_execution_ns": 961396, "input_digest": "211297e7891d256480ff2cd59c8c340e13e7dbcad64a6e9c72c4dfde27367223", "matrix": false, "mode": "jax_jit", "staging_ns": 5499254}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A05/axis1/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "1358bbee677ecfeceb93e0ce440171c2a3ace658f4e569970f12cccf6130cca8",
  "compile_ns": 305925843,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "0a8e80417464c2a8e2758625b5bbc9fb6efa52c0f4c8acc0eeb0a09ac471eccf"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1f14e58b839ec9b3676e9d40ea965da96bc46f0df583aa2e8511086133eeb1ee"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 961396,
  "input_digest": "211297e7891d256480ff2cd59c8c340e13e7dbcad64a6e9c72c4dfde27367223",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 5499254
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "4b81b2f47ef111873f4fd448462fb92353664d09c7e79893bb7c772bf91c990c", "compile_ns": 29896271, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "dee714bc6d1041ef96f98a6423d125ddb16c9448ef28965b8e925692d40b3bae"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "d21c2a3849f3f488d1bc1747aa44822bc1ddcaf89458e071a08dc98fb01f9153"}}, "dynamic_argument_count": 1, "first_execution_ns": 765553, "input_digest": "450667752b4d880d9609c62a142484b5c10a7f713e3f4ef3c061c526605de6ab", "matrix": false, "mode": "jax_jit", "staging_ns": 701959}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A05/axis1/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "4b81b2f47ef111873f4fd448462fb92353664d09c7e79893bb7c772bf91c990c",
  "compile_ns": 29896271,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "dee714bc6d1041ef96f98a6423d125ddb16c9448ef28965b8e925692d40b3bae"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "d21c2a3849f3f488d1bc1747aa44822bc1ddcaf89458e071a08dc98fb01f9153"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 765553,
  "input_digest": "450667752b4d880d9609c62a142484b5c10a7f713e3f4ef3c061c526605de6ab",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 701959
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "e63636c8b257bb4e932413397a11f4a8f0ecbec2e66f241b8709a3d19d4dd6fe", "compile_ns": 28038860, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "4c0a88bf5000ec0796557cea71767ac1660ab938e139f33d0fa36634c72e2a41"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "0da1b4aacd748491562cc87c5e6c80a842c4e847297ecc8c30b49a1e49bc24fa"}}, "dynamic_argument_count": 1, "first_execution_ns": 595159, "input_digest": "dac044e9b157a863448e04e5dba433f400758105d31d26883223e105d4e2174a", "matrix": false, "mode": "jax_jit", "staging_ns": 552880}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A05/axis1/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "e63636c8b257bb4e932413397a11f4a8f0ecbec2e66f241b8709a3d19d4dd6fe",
  "compile_ns": 28038860,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4c0a88bf5000ec0796557cea71767ac1660ab938e139f33d0fa36634c72e2a41"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0da1b4aacd748491562cc87c5e6c80a842c4e847297ecc8c30b49a1e49bc24fa"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 595159,
  "input_digest": "dac044e9b157a863448e04e5dba433f400758105d31d26883223e105d4e2174a",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 552880
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "70f92fbabe4ab11d1d500c9c87324b92cfa6850283d12700f47a788992426f4a", "compile_ns": 28101916, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "686e7b56b9ad945bd3c6c3b3ebb440114ce74c4d210abd47575edddea65feabc"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "10faeea45a99f3466ea2c4891d3ccb6fe4b9513d419f85ab6e7c78c20e0fbdfd"}}, "dynamic_argument_count": 1, "first_execution_ns": 697491, "input_digest": "bdb2290b7902456de43d949de25bf9ce128816c1e92fbb4ff6d0254b8437e0a9", "matrix": false, "mode": "jax_jit", "staging_ns": 599413}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A05/axis1/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "70f92fbabe4ab11d1d500c9c87324b92cfa6850283d12700f47a788992426f4a",
  "compile_ns": 28101916,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "686e7b56b9ad945bd3c6c3b3ebb440114ce74c4d210abd47575edddea65feabc"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "10faeea45a99f3466ea2c4891d3ccb6fe4b9513d419f85ab6e7c78c20e0fbdfd"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 697491,
  "input_digest": "bdb2290b7902456de43d949de25bf9ce128816c1e92fbb4ff6d0254b8437e0a9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 599413
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "b0dfb9513a426fe9404655ce9f91f36b99311748ebfe19b9f775fc385ea79c65", "compile_ns": 26692538, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "54d0cd27f43bc4438586888232d756451fd1ac2543257067348d9e173f70c213"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "fd2b438b0d30d3f82ff94ea5d621d7d3a4f45993e94242a81f83f2e3e8e1c169"}}, "dynamic_argument_count": 1, "first_execution_ns": 627910, "input_digest": "c9dbbbc4b74ca88a2ea40d40ed2fce2a31765f24f14000c6b6c42ab0bafbfa8e", "matrix": false, "mode": "jax_jit", "staging_ns": 517963}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis0/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "b0dfb9513a426fe9404655ce9f91f36b99311748ebfe19b9f775fc385ea79c65",
  "compile_ns": 26692538,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "54d0cd27f43bc4438586888232d756451fd1ac2543257067348d9e173f70c213"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "fd2b438b0d30d3f82ff94ea5d621d7d3a4f45993e94242a81f83f2e3e8e1c169"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 627910,
  "input_digest": "c9dbbbc4b74ca88a2ea40d40ed2fce2a31765f24f14000c6b6c42ab0bafbfa8e",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 517963
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "0d339995c6c469af35cf713e476a50647ccb9cf0015d18a2bd948b84d75a3549", "compile_ns": 27057369, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "d06ade1be62f0aba3d11547f5d46433e5cfbcb4d01f3c1f843a2ee83256050f4"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "efe17c52575d3656d43b73c975a6141cc60e414f02dedbc98e9d141cce2d9b07"}}, "dynamic_argument_count": 1, "first_execution_ns": 629090, "input_digest": "b3cd81cb1c511df02a4eb072a3d9ec69b4d57f03c60f5f141f6b0a0c84c5ba27", "matrix": false, "mode": "jax_jit", "staging_ns": 497131}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis0/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "0d339995c6c469af35cf713e476a50647ccb9cf0015d18a2bd948b84d75a3549",
  "compile_ns": 27057369,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "d06ade1be62f0aba3d11547f5d46433e5cfbcb4d01f3c1f843a2ee83256050f4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "efe17c52575d3656d43b73c975a6141cc60e414f02dedbc98e9d141cce2d9b07"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 629090,
  "input_digest": "b3cd81cb1c511df02a4eb072a3d9ec69b4d57f03c60f5f141f6b0a0c84c5ba27",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 497131
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "f22996fb4f4e850448837966b72430dd9057953539c236de7a8fb68078dfd8bb", "compile_ns": 26884861, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "81649f08a0db3fb4430905903f7b4866510e191a274cef416e4809d30ad2436a"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "4a1ce0c7e953e8885230e3cbb056f3fbbdf1de0e7508d0f80cf7e75e4fffb61e"}}, "dynamic_argument_count": 1, "first_execution_ns": 653454, "input_digest": "9d12d9dcfa2157dd5641dc4a39482fa68ea7f489d1b1d2b593b0556bfc0a9bb3", "matrix": false, "mode": "jax_jit", "staging_ns": 495690}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis0/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "f22996fb4f4e850448837966b72430dd9057953539c236de7a8fb68078dfd8bb",
  "compile_ns": 26884861,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "81649f08a0db3fb4430905903f7b4866510e191a274cef416e4809d30ad2436a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4a1ce0c7e953e8885230e3cbb056f3fbbdf1de0e7508d0f80cf7e75e4fffb61e"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 653454,
  "input_digest": "9d12d9dcfa2157dd5641dc4a39482fa68ea7f489d1b1d2b593b0556bfc0a9bb3",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 495690
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "fa90b5e6d5d5ff7b7110f242257edb92d651aea95f7efea9aee11cdb1b5b1843", "compile_ns": 27031787, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "9d26e4e04355ac2533f60615c56dc0fe2006c30b09490bb51c56c742d55855e0"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "27b1f59533fbc028be6cf252f40e1080bed47703e1dadb1caad1f2f5822f6ba5"}}, "dynamic_argument_count": 1, "first_execution_ns": 696261, "input_digest": "82af0a42d1fb16b5671bc4123aba94f0bdec8ed40854cbce07ec60d18b8aa593", "matrix": false, "mode": "jax_jit", "staging_ns": 518486}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis0/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "fa90b5e6d5d5ff7b7110f242257edb92d651aea95f7efea9aee11cdb1b5b1843",
  "compile_ns": 27031787,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9d26e4e04355ac2533f60615c56dc0fe2006c30b09490bb51c56c742d55855e0"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "27b1f59533fbc028be6cf252f40e1080bed47703e1dadb1caad1f2f5822f6ba5"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 696261,
  "input_digest": "82af0a42d1fb16b5671bc4123aba94f0bdec8ed40854cbce07ec60d18b8aa593",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 518486
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "28c43d0abea4ea4ee6876e17ed8d6d27c6ad43145c69f4639e1c5363a27660f0", "compile_ns": 26282308, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0008/optimized_hlo.txt", "sha256": "3949b61663c88f1056dae4000bfc4655ca650078195102cf9cb61d4f76a192eb"}, "stablehlo": {"path": "compiler/case-0008/stablehlo.txt", "sha256": "a38e2ce199577b986866e204d712f420b37df485fc7de2a2fd49e54d5999b44e"}}, "dynamic_argument_count": 1, "first_execution_ns": 656058, "input_digest": "1cefeb285136a0f72d682f4c83712d63f26e37a7aa6eff4dfe5f00c5e3204bae", "matrix": false, "mode": "jax_jit", "staging_ns": 490530}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis1/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "28c43d0abea4ea4ee6876e17ed8d6d27c6ad43145c69f4639e1c5363a27660f0",
  "compile_ns": 26282308,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "3949b61663c88f1056dae4000bfc4655ca650078195102cf9cb61d4f76a192eb"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a38e2ce199577b986866e204d712f420b37df485fc7de2a2fd49e54d5999b44e"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 656058,
  "input_digest": "1cefeb285136a0f72d682f4c83712d63f26e37a7aa6eff4dfe5f00c5e3204bae",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 490530
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "50616c4667f965447b60d6796e663ae2575884075413f00036fed43ba997cddb", "compile_ns": 26302023, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "34cd7a42ed1f7cccb38d5489ac91da4bc81c64a61d922a96dc1357d310667c05"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "048fdf2c74f6878891e581a6063a1dd320051b6dc70e659d19e9fc6d1c3b9eb0"}}, "dynamic_argument_count": 1, "first_execution_ns": 596632, "input_digest": "dba8133090e4fb0ee1680de8cb545f796d160c83ae2d96723dff985d91f62004", "matrix": false, "mode": "jax_jit", "staging_ns": 495246}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis1/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "50616c4667f965447b60d6796e663ae2575884075413f00036fed43ba997cddb",
  "compile_ns": 26302023,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "34cd7a42ed1f7cccb38d5489ac91da4bc81c64a61d922a96dc1357d310667c05"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "048fdf2c74f6878891e581a6063a1dd320051b6dc70e659d19e9fc6d1c3b9eb0"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 596632,
  "input_digest": "dba8133090e4fb0ee1680de8cb545f796d160c83ae2d96723dff985d91f62004",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 495246
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "54823240ab2806c917b8bb6c79a9afafcb6bd8588167bd19a7fa869cb17c26a5", "compile_ns": 26405431, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "7a3a75f079ab1909aad843f7cb31aa65bedd65e848dfc3c2ae48d226379656f8"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "49cfd5f0e71538f9f59f7d751b50651996a96995bfa5e296dee12ae6b12ef5a0"}}, "dynamic_argument_count": 1, "first_execution_ns": 628957, "input_digest": "f05e99eea0eb7f820e657afd80b1836ce731e1f85b86e60e34924476990470fe", "matrix": false, "mode": "jax_jit", "staging_ns": 529318}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis1/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "54823240ab2806c917b8bb6c79a9afafcb6bd8588167bd19a7fa869cb17c26a5",
  "compile_ns": 26405431,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7a3a75f079ab1909aad843f7cb31aa65bedd65e848dfc3c2ae48d226379656f8"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "49cfd5f0e71538f9f59f7d751b50651996a96995bfa5e296dee12ae6b12ef5a0"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 628957,
  "input_digest": "f05e99eea0eb7f820e657afd80b1836ce731e1f85b86e60e34924476990470fe",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 529318
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "b65565b4a74824fc13d7344f22e7888f64ddcf4f91321e260600f078e7dfdac1", "compile_ns": 30716248, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "5b744c4f7634f454c25f07489f0ae2e9d93dfbefad6617e926ac4e64544f3e23"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "3bf1f26b3d17b923cde417b84e35b495ddffe434182231cb68edf78077e89cf0"}}, "dynamic_argument_count": 1, "first_execution_ns": 661900, "input_digest": "a8fdeddf3b6e8d7f7facda985e01b723a5b09a2989c0d2d4ae22cab304a74463", "matrix": false, "mode": "jax_jit", "staging_ns": 530485}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A06/axis1/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "b65565b4a74824fc13d7344f22e7888f64ddcf4f91321e260600f078e7dfdac1",
  "compile_ns": 30716248,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5b744c4f7634f454c25f07489f0ae2e9d93dfbefad6617e926ac4e64544f3e23"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3bf1f26b3d17b923cde417b84e35b495ddffe434182231cb68edf78077e89cf0"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 661900,
  "input_digest": "a8fdeddf3b6e8d7f7facda985e01b723a5b09a2989c0d2d4ae22cab304a74463",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 530485
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "64fce2ea8dae033f22c6206b390f7ad53b92fd80c9f76921b3d43ea85658ef20", "compile_ns": 25219844, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0012/optimized_hlo.txt", "sha256": "78b65a7368456fa2a248fa2a31a0ae18fe64c4cea5139d80c1df844b7b819bf5"}, "stablehlo": {"path": "compiler/case-0012/stablehlo.txt", "sha256": "4c6ff27a381821301aa631b635dcd0cf19436481ae302ecffaa054180c504e51"}}, "dynamic_argument_count": 1, "first_execution_ns": 623420, "input_digest": "49243f33aef57a2998830bd738d09792abfa281870b6d2de56d479897e208ec3", "matrix": false, "mode": "jax_jit", "staging_ns": 483845}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        8
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A07/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        8
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "64fce2ea8dae033f22c6206b390f7ad53b92fd80c9f76921b3d43ea85658ef20",
  "compile_ns": 25219844,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "78b65a7368456fa2a248fa2a31a0ae18fe64c4cea5139d80c1df844b7b819bf5"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4c6ff27a381821301aa631b635dcd0cf19436481ae302ecffaa054180c504e51"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 623420,
  "input_digest": "49243f33aef57a2998830bd738d09792abfa281870b6d2de56d479897e208ec3",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 483845
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "1a7ebb540a8c78efba128e09a084930d81d7ffbfaf5a948eb411d3546d39c0bd", "compile_ns": 24827348, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0013/optimized_hlo.txt", "sha256": "6444c1619afa06933f9c82e83a87d7026ff317ea84e719198609a73a7c3497e4"}, "stablehlo": {"path": "compiler/case-0013/stablehlo.txt", "sha256": "e393e3558b7b4f10970ffeeeede4b6e081ab5e7d1244d842ad7dfaa0aa0fc070"}}, "dynamic_argument_count": 1, "first_execution_ns": 599420, "input_digest": "5d220f1be60a2c6f34dc09b0f10fd05f48ee1795184c6f4c866336c78837acb7", "matrix": false, "mode": "jax_jit", "staging_ns": 513605}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A07/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "1a7ebb540a8c78efba128e09a084930d81d7ffbfaf5a948eb411d3546d39c0bd",
  "compile_ns": 24827348,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6444c1619afa06933f9c82e83a87d7026ff317ea84e719198609a73a7c3497e4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "e393e3558b7b4f10970ffeeeede4b6e081ab5e7d1244d842ad7dfaa0aa0fc070"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 599420,
  "input_digest": "5d220f1be60a2c6f34dc09b0f10fd05f48ee1795184c6f4c866336c78837acb7",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 513605
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "ab9580ff9e680cc5046e1aa36e421ba7d1297db2834dbc5d9dca0633ec0e6a4d", "compile_ns": 24935700, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0014/optimized_hlo.txt", "sha256": "6b058a1cea85a00037f7aa57e3ee4845fa7eb9684ac65fc4732b0db663e9fc3e"}, "stablehlo": {"path": "compiler/case-0014/stablehlo.txt", "sha256": "6a8f389136786a7c0ac2e415b7e5068567da4af08432f1a90e678cd2a2742e7c"}}, "dynamic_argument_count": 1, "first_execution_ns": 628174, "input_digest": "5aa3806640151a5558c8a71581f2ace30e007e0cb195a822a5a16e1a62750374", "matrix": false, "mode": "jax_jit", "staging_ns": 473381}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        17,
        7
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A07/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        17,
        7
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "ab9580ff9e680cc5046e1aa36e421ba7d1297db2834dbc5d9dca0633ec0e6a4d",
  "compile_ns": 24935700,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6b058a1cea85a00037f7aa57e3ee4845fa7eb9684ac65fc4732b0db663e9fc3e"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6a8f389136786a7c0ac2e415b7e5068567da4af08432f1a90e678cd2a2742e7c"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 628174,
  "input_digest": "5aa3806640151a5558c8a71581f2ace30e007e0cb195a822a5a16e1a62750374",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 473381
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "b95d737897591cff4a017133176532ee584c1d2f53f9510628829072767430d7", "compile_ns": 25094413, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0015/optimized_hlo.txt", "sha256": "211eebb94b079c7122ae91e8260bb9f74bc1710bbc8cef8272f7775b7c879bf9"}, "stablehlo": {"path": "compiler/case-0015/stablehlo.txt", "sha256": "200942deb0dec505fbb13a1f16302118ce7f22b767820de42da030006822136a"}}, "dynamic_argument_count": 1, "first_execution_ns": 602838, "input_digest": "0821d7748b02b0f0e95a60b13e470d994afb3949a58d2a58b208ac939c5c6c49", "matrix": false, "mode": "jax_jit", "staging_ns": 555617}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        64,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A07/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        64,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "b95d737897591cff4a017133176532ee584c1d2f53f9510628829072767430d7",
  "compile_ns": 25094413,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "211eebb94b079c7122ae91e8260bb9f74bc1710bbc8cef8272f7775b7c879bf9"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "200942deb0dec505fbb13a1f16302118ce7f22b767820de42da030006822136a"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 602838,
  "input_digest": "0821d7748b02b0f0e95a60b13e470d994afb3949a58d2a58b208ac939c5c6c49",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 555617
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "037f40333a2594715d08daa28aaf8de68049691c608e064877e51727b2269d24", "compile_ns": 46464667, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0016/optimized_hlo.txt", "sha256": "b0c38cf5e4609afdeae60210002ea1873b491874b794fd7eaaa3da7eb0c3e328"}, "stablehlo": {"path": "compiler/case-0016/stablehlo.txt", "sha256": "7f9e7cd44f5dbf6107513339948471312990e135cf8c6728032195ede92204b4"}}, "dynamic_argument_count": 2, "first_execution_ns": 731580, "input_digest": "1dfdb057c80d48686ced1e5beede2b7792165ca6cbf75962e0ea85323fd25dc9", "matrix": false, "mode": "jax_jit", "staging_ns": 932824}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A08/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "037f40333a2594715d08daa28aaf8de68049691c608e064877e51727b2269d24",
  "compile_ns": 46464667,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "b0c38cf5e4609afdeae60210002ea1873b491874b794fd7eaaa3da7eb0c3e328"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7f9e7cd44f5dbf6107513339948471312990e135cf8c6728032195ede92204b4"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 731580,
  "input_digest": "1dfdb057c80d48686ced1e5beede2b7792165ca6cbf75962e0ea85323fd25dc9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 932824
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "a331efc463b3f0e6e048a4a3ff753f1d7bd9e31fa47df11a34abb20cff583677", "compile_ns": 43787818, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0017/optimized_hlo.txt", "sha256": "555205464b01a7f1400596ebac6af95e8a2abbbee839bfbadbb1739c490b6f2a"}, "stablehlo": {"path": "compiler/case-0017/stablehlo.txt", "sha256": "87b6da13d2e5868e633c511d98fb3542d3fd3443d40c9446b84e09d160ceb79b"}}, "dynamic_argument_count": 2, "first_execution_ns": 850914, "input_digest": "145baf05c373803dc472cc8699bcaa1e01f0d8e8f3cb9bda8573ef129acb7f46", "matrix": false, "mode": "jax_jit", "staging_ns": 738373}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A08/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "a331efc463b3f0e6e048a4a3ff753f1d7bd9e31fa47df11a34abb20cff583677",
  "compile_ns": 43787818,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "555205464b01a7f1400596ebac6af95e8a2abbbee839bfbadbb1739c490b6f2a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "87b6da13d2e5868e633c511d98fb3542d3fd3443d40c9446b84e09d160ceb79b"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 850914,
  "input_digest": "145baf05c373803dc472cc8699bcaa1e01f0d8e8f3cb9bda8573ef129acb7f46",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 738373
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "8d898dc2139557c3ae275d552941c225294829ebf3dfc867f486fb2ed231b23b", "compile_ns": 45599806, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0018/optimized_hlo.txt", "sha256": "7bce487720fa8ba685b2132cffac2a6eabc52af031b2a6ec71923f707ebb846b"}, "stablehlo": {"path": "compiler/case-0018/stablehlo.txt", "sha256": "a966b7d2e3bc30cfcca1dce1bbe6cd11c33be8dc9635d7cc9fb244e1c9fe0ba0"}}, "dynamic_argument_count": 2, "first_execution_ns": 891052, "input_digest": "d49f817ff7a5ef0eafaf9f180417439a7af55dd06579f53843ed3b3719d88adf", "matrix": false, "mode": "jax_jit", "staging_ns": 713229}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A08/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "8d898dc2139557c3ae275d552941c225294829ebf3dfc867f486fb2ed231b23b",
  "compile_ns": 45599806,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7bce487720fa8ba685b2132cffac2a6eabc52af031b2a6ec71923f707ebb846b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a966b7d2e3bc30cfcca1dce1bbe6cd11c33be8dc9635d7cc9fb244e1c9fe0ba0"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 891052,
  "input_digest": "d49f817ff7a5ef0eafaf9f180417439a7af55dd06579f53843ed3b3719d88adf",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 713229
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "9624aa13e0c8c9b7c4e6d45617bac5f9ab4007245b1dda00f12c862416114b49", "compile_ns": 45293508, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0019/optimized_hlo.txt", "sha256": "a249ae5f13377cafc295f867cfc0b247def43ab0a789063016647ba7dd5a9a7c"}, "stablehlo": {"path": "compiler/case-0019/stablehlo.txt", "sha256": "5df91a59bc9a9d125ccb2b842b60df2801f7fb1faa00d740e2e6655415d6b803"}}, "dynamic_argument_count": 2, "first_execution_ns": 923862, "input_digest": "867588c8523c41b4e80e6ed4f9f7ec4ff9e47017eecc3fa04a8446af6c92a168", "matrix": false, "mode": "jax_jit", "staging_ns": 765311}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A08/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "9624aa13e0c8c9b7c4e6d45617bac5f9ab4007245b1dda00f12c862416114b49",
  "compile_ns": 45293508,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a249ae5f13377cafc295f867cfc0b247def43ab0a789063016647ba7dd5a9a7c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "5df91a59bc9a9d125ccb2b842b60df2801f7fb1faa00d740e2e6655415d6b803"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 923862,
  "input_digest": "867588c8523c41b4e80e6ed4f9f7ec4ff9e47017eecc3fa04a8446af6c92a168",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 765311
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "092e7ffebc7aae866e3e8387477ce78f968874c5bf097384a53f8c0e73338242", "compile_ns": 35441122, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0020/optimized_hlo.txt", "sha256": "5d88ac303bbf2c93081b598712349172efea623e068f9475c237c68a69e41350"}, "stablehlo": {"path": "compiler/case-0020/stablehlo.txt", "sha256": "5fcd34662f24dfd29adca903e7a65345599abe0c42c9136d3d3c65b207e9a9a4"}}, "dynamic_argument_count": 1, "first_execution_ns": 769530, "input_digest": "6401f42c6029ab27f83f2ee2533600134770e46e8da543fd4fcacd01ece5cf47", "matrix": false, "mode": "jax_jit", "staging_ns": 730714}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.4901161193847656e-08,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C01/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.4901161193847656e-08,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "092e7ffebc7aae866e3e8387477ce78f968874c5bf097384a53f8c0e73338242",
  "compile_ns": 35441122,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5d88ac303bbf2c93081b598712349172efea623e068f9475c237c68a69e41350"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "5fcd34662f24dfd29adca903e7a65345599abe0c42c9136d3d3c65b207e9a9a4"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 769530,
  "input_digest": "6401f42c6029ab27f83f2ee2533600134770e46e8da543fd4fcacd01ece5cf47",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 730714
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "6b205ceaa2f56968974443351ac2264b70687a1c917a5691cdcfdb162d779d84", "compile_ns": 34885261, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0021/optimized_hlo.txt", "sha256": "89cc5a2e8474ba12cad7d805a3702f180494aa04710ae0feb75f65a0fa8ed307"}, "stablehlo": {"path": "compiler/case-0021/stablehlo.txt", "sha256": "14cd18f6e0765365132281db016d14cc46aa0e50c98a633fad946c3a0fc1a22c"}}, "dynamic_argument_count": 1, "first_execution_ns": 640403, "input_digest": "c4edba837467ea2c8f7e697db1ebc76def8a2b9ab9f038b2ae4a72f6bc9a78b9", "matrix": false, "mode": "jax_jit", "staging_ns": 584151}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 7.450580596923828e-09,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C01/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 7.450580596923828e-09,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "6b205ceaa2f56968974443351ac2264b70687a1c917a5691cdcfdb162d779d84",
  "compile_ns": 34885261,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "89cc5a2e8474ba12cad7d805a3702f180494aa04710ae0feb75f65a0fa8ed307"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "14cd18f6e0765365132281db016d14cc46aa0e50c98a633fad946c3a0fc1a22c"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 640403,
  "input_digest": "c4edba837467ea2c8f7e697db1ebc76def8a2b9ab9f038b2ae4a72f6bc9a78b9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 584151
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "0f970d74db2c9193542a652063b02cc2f264d4852df7f03a00acd69d6af737be", "compile_ns": 40985873, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0022/optimized_hlo.txt", "sha256": "f49c77f31c09ef5fa2f250cc7c846cd977e2d4b7c1e23654bc3674261e5de2bd"}, "stablehlo": {"path": "compiler/case-0022/stablehlo.txt", "sha256": "d598e8be31961e5425bc4bd3fb36c66c5464808119be814b268b131b1d42bb0e"}}, "dynamic_argument_count": 1, "first_execution_ns": 650340, "input_digest": "c5dce31911b004a5bc748050f4530b50779406662f76da0cf8812f8a7a551d49", "matrix": false, "mode": "jax_jit", "staging_ns": 547302}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.4901161193847656e-08,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C01/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.4901161193847656e-08,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "0f970d74db2c9193542a652063b02cc2f264d4852df7f03a00acd69d6af737be",
  "compile_ns": 40985873,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "f49c77f31c09ef5fa2f250cc7c846cd977e2d4b7c1e23654bc3674261e5de2bd"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "d598e8be31961e5425bc4bd3fb36c66c5464808119be814b268b131b1d42bb0e"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 650340,
  "input_digest": "c5dce31911b004a5bc748050f4530b50779406662f76da0cf8812f8a7a551d49",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 547302
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "2ab37965cf06fe01c67b013a09d9c96ac83104f8e003986f62db11851761de78", "compile_ns": 35946205, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0023/optimized_hlo.txt", "sha256": "253df7ce56c23a34994f5bf91aec24e98f4c4331e9dd08735be5840546a0173d"}, "stablehlo": {"path": "compiler/case-0023/stablehlo.txt", "sha256": "22d01880ff3a029cf2994888f56300936fcaaa184de16abfca7191a17817c014"}}, "dynamic_argument_count": 1, "first_execution_ns": 753547, "input_digest": "8c4480cf782076a3858cf40744fadbb5be86e9702d0fca56781df7c921494bbd", "matrix": false, "mode": "jax_jit", "staging_ns": 645318}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 7.450580596923828e-09,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C01/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 7.450580596923828e-09,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "2ab37965cf06fe01c67b013a09d9c96ac83104f8e003986f62db11851761de78",
  "compile_ns": 35946205,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "253df7ce56c23a34994f5bf91aec24e98f4c4331e9dd08735be5840546a0173d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "22d01880ff3a029cf2994888f56300936fcaaa184de16abfca7191a17817c014"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 753547,
  "input_digest": "8c4480cf782076a3858cf40744fadbb5be86e9702d0fca56781df7c921494bbd",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 645318
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "be44e898b5219873eb4338d25b83a6db4ca7d40cbdc89104aa64d3dc4c4a8e79", "compile_ns": 38383659, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0024/optimized_hlo.txt", "sha256": "a1201d559b5cb2a3fc94ffae49a2c2e90d916c54df4a7b0fdf86797a784d4fe9"}, "stablehlo": {"path": "compiler/case-0024/stablehlo.txt", "sha256": "202f02d6b5f5b3358c0a8b1f69f68a6eff0676b36909d40931a38117f9d0002d"}}, "dynamic_argument_count": 4, "first_execution_ns": 856416, "input_digest": "6596f8c73450b9a62556fae738a81dec35a176858a531c01c94ea31bf946a8d7", "matrix": false, "mode": "jax_jit", "staging_ns": 1082541}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C02/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "be44e898b5219873eb4338d25b83a6db4ca7d40cbdc89104aa64d3dc4c4a8e79",
  "compile_ns": 38383659,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a1201d559b5cb2a3fc94ffae49a2c2e90d916c54df4a7b0fdf86797a784d4fe9"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "202f02d6b5f5b3358c0a8b1f69f68a6eff0676b36909d40931a38117f9d0002d"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 856416,
  "input_digest": "6596f8c73450b9a62556fae738a81dec35a176858a531c01c94ea31bf946a8d7",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1082541
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "292a217329a3897af73a49a49ee02b83ba882e3ac49a9158a40f11d0e73233d9", "compile_ns": 38356921, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0025/optimized_hlo.txt", "sha256": "5d0c6deb7943bb1154411c45b8217235bfdbef06e9122dbeacdbd6df1bbb8432"}, "stablehlo": {"path": "compiler/case-0025/stablehlo.txt", "sha256": "36a2a32d32f83cefd8cd715a3af7d9c73c5958a1ed0f7fda857ead4542cee282"}}, "dynamic_argument_count": 4, "first_execution_ns": 819856, "input_digest": "1bf362f9128f82b26c318a558845ad125720eeb4c01026048d9571094623a957", "matrix": false, "mode": "jax_jit", "staging_ns": 1004544}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C02/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "292a217329a3897af73a49a49ee02b83ba882e3ac49a9158a40f11d0e73233d9",
  "compile_ns": 38356921,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5d0c6deb7943bb1154411c45b8217235bfdbef06e9122dbeacdbd6df1bbb8432"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "36a2a32d32f83cefd8cd715a3af7d9c73c5958a1ed0f7fda857ead4542cee282"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 819856,
  "input_digest": "1bf362f9128f82b26c318a558845ad125720eeb4c01026048d9571094623a957",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1004544
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "fb3bd8fee9fe5bc7c87a124d2430e6b4d91fcee8d064f5f98c2d0e858bda74e6", "compile_ns": 38245057, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0026/optimized_hlo.txt", "sha256": "21b94b7281010719c52d01b0260fda72aa84cd697876178bc945de0dca8e234b"}, "stablehlo": {"path": "compiler/case-0026/stablehlo.txt", "sha256": "dfad7c09a0befbd1c9510821a39ccaee529af2dd6313b480096b4ed9522c8f50"}}, "dynamic_argument_count": 4, "first_execution_ns": 837487, "input_digest": "b2940337f434a122765a8a8c3ca319a3d5dabcb3aa4c60cb4cfcb089b8abdf69", "matrix": false, "mode": "jax_jit", "staging_ns": 991143}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.1920928955078125e-07,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C02/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.1920928955078125e-07,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "fb3bd8fee9fe5bc7c87a124d2430e6b4d91fcee8d064f5f98c2d0e858bda74e6",
  "compile_ns": 38245057,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "21b94b7281010719c52d01b0260fda72aa84cd697876178bc945de0dca8e234b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "dfad7c09a0befbd1c9510821a39ccaee529af2dd6313b480096b4ed9522c8f50"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 837487,
  "input_digest": "b2940337f434a122765a8a8c3ca319a3d5dabcb3aa4c60cb4cfcb089b8abdf69",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 991143
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "803830d38bbb9ba3bf5d14eedf5147323daeae01a5f07fa8d76a2e255d64437d", "compile_ns": 38320750, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0027/optimized_hlo.txt", "sha256": "da0850c11f3746a948ac95c78e5cd1c0b40bffabcd9ec0e426c71745efc29131"}, "stablehlo": {"path": "compiler/case-0027/stablehlo.txt", "sha256": "c265671413835587789e42af5a713965070ed046d9117cc7b8f8e081dfacd6a5"}}, "dynamic_argument_count": 4, "first_execution_ns": 695056, "input_digest": "1b8dfd6d39cdf2004e8ce1096ed7978bdc33a3f4ca112c29ebd47c948e8631b2", "matrix": false, "mode": "jax_jit", "staging_ns": 999756}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C02/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "803830d38bbb9ba3bf5d14eedf5147323daeae01a5f07fa8d76a2e255d64437d",
  "compile_ns": 38320750,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "da0850c11f3746a948ac95c78e5cd1c0b40bffabcd9ec0e426c71745efc29131"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c265671413835587789e42af5a713965070ed046d9117cc7b8f8e081dfacd6a5"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 695056,
  "input_digest": "1b8dfd6d39cdf2004e8ce1096ed7978bdc33a3f4ca112c29ebd47c948e8631b2",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 999756
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "da81e9d08fbec0aae59d0f689b0b83d0be0004c622e131928052337327cc09b8", "compile_ns": 44033997, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0028/optimized_hlo.txt", "sha256": "886b5ca108309b4262891bfef5750ab37e3466f02e2cd9dffa7028c9c724b90b"}, "stablehlo": {"path": "compiler/case-0028/stablehlo.txt", "sha256": "45e0bfa183143a98367a02c55c5665affc1430d98bd92da7a38b539556abaa0c"}}, "dynamic_argument_count": 5, "first_execution_ns": 881999, "input_digest": "d4916c6e99dcdb8a1f7d601a001a1cebe4f713b6923162b166d93fe6ef75bc5e", "matrix": false, "mode": "jax_jit", "staging_ns": 1159846}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C03/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "da81e9d08fbec0aae59d0f689b0b83d0be0004c622e131928052337327cc09b8",
  "compile_ns": 44033997,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "886b5ca108309b4262891bfef5750ab37e3466f02e2cd9dffa7028c9c724b90b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "45e0bfa183143a98367a02c55c5665affc1430d98bd92da7a38b539556abaa0c"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 881999,
  "input_digest": "d4916c6e99dcdb8a1f7d601a001a1cebe4f713b6923162b166d93fe6ef75bc5e",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1159846
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "2668c24e2f33dd066a7f12771067e6654020e21aa71f85fa32b98a0e2ce63bd2", "compile_ns": 42862990, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0029/optimized_hlo.txt", "sha256": "88c0d7b0a355e9b474d595c55c826982411c94c3ea43cca7cd242ff2fdbd6e7c"}, "stablehlo": {"path": "compiler/case-0029/stablehlo.txt", "sha256": "03abbffbb78bf0310fc1edd2de04c299e60eeab9c2c5bf083989fabc4c715317"}}, "dynamic_argument_count": 5, "first_execution_ns": 1064694, "input_digest": "37caf868440a9cfe78d5989339f7226269ec904fb09560a615a726f0361cf0b1", "matrix": false, "mode": "jax_jit", "staging_ns": 1164129}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C03/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "2668c24e2f33dd066a7f12771067e6654020e21aa71f85fa32b98a0e2ce63bd2",
  "compile_ns": 42862990,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "88c0d7b0a355e9b474d595c55c826982411c94c3ea43cca7cd242ff2fdbd6e7c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "03abbffbb78bf0310fc1edd2de04c299e60eeab9c2c5bf083989fabc4c715317"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 1064694,
  "input_digest": "37caf868440a9cfe78d5989339f7226269ec904fb09560a615a726f0361cf0b1",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1164129
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "b5c13c4b6a73732e21e3a3c49bdce4d724cd02b13ef4f48469eb6ecd11f72bb7", "compile_ns": 42787361, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0030/optimized_hlo.txt", "sha256": "a6487c5358aed0bc1c91bba1d7d7adc6878ea996004cb5915584828874245d63"}, "stablehlo": {"path": "compiler/case-0030/stablehlo.txt", "sha256": "acac4f8ba97cfc5cb47c64c379c7146415dcdf0de0467d17f635ccca2337c99d"}}, "dynamic_argument_count": 5, "first_execution_ns": 997320, "input_digest": "73e03b5f589d4571d17f48575c555315de19f923295919b1b53ddb34badfa6ee", "matrix": false, "mode": "jax_jit", "staging_ns": 1085287}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C03/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        7,
        17
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "b5c13c4b6a73732e21e3a3c49bdce4d724cd02b13ef4f48469eb6ecd11f72bb7",
  "compile_ns": 42787361,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a6487c5358aed0bc1c91bba1d7d7adc6878ea996004cb5915584828874245d63"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "acac4f8ba97cfc5cb47c64c379c7146415dcdf0de0467d17f635ccca2337c99d"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 997320,
  "input_digest": "73e03b5f589d4571d17f48575c555315de19f923295919b1b53ddb34badfa6ee",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1085287
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## portable_core

| field | value |
| --- | --- |
| launch | `jax_jit`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `portable_core` |

- launch.extras: `{"case_sha256": "a5fc8022f3c1a4d57f888d3830c64ea11f4776c788df3f120cbfaa1088452c69", "compile_ns": 42762710, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0031/optimized_hlo.txt", "sha256": "e3cc90f18ef9f31ec893dbe77a01fd54a4cece16c9001a09291d61c79a71d9d1"}, "stablehlo": {"path": "compiler/case-0031/stablehlo.txt", "sha256": "f20e9fa31b0c869f707b7917b1507b475fab4ac39d0c31e75593cf4ab3e060a7"}}, "dynamic_argument_count": 5, "first_execution_ns": 762754, "input_digest": "ac488aadeb456dca7c1136fcc617840547607c13af97a07a00b00ecb3182718f", "matrix": false, "mode": "jax_jit", "staging_ns": 1188270}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/C03/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "a5fc8022f3c1a4d57f888d3830c64ea11f4776c788df3f120cbfaa1088452c69",
  "compile_ns": 42762710,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e3cc90f18ef9f31ec893dbe77a01fd54a4cece16c9001a09291d61c79a71d9d1"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "f20e9fa31b0c869f707b7917b1507b475fab4ac39d0c31e75593cf4ab3e060a7"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 762754,
  "input_digest": "ac488aadeb456dca7c1136fcc617840547607c13af97a07a00b00ecb3182718f",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1188270
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 64,
  "stop": 96,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `3eed9ce9f4df473e0589a94bc9045411e629ce5a14b0072c82ee1cfe01347d50`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-002/out/tpu/portable-core-qualification-v1/a76c99586f3858aa4b3f25ea1f3f363c281460711d3b519b1261e011a74c98ae/native-restore-002-core28-002/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
