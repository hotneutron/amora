# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-restore-002-core28-000
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

- launch.extras: `{"case_sha256": "da4b6a54e558a671385792c299468ebf1ea7559ada862d2233400fc950b0d23f", "compile_ns": 29545467, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "be6e120a72f5aeacf3fa3a883283861c01e9e0ebfe7f8a3a1ea25379058b172d"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "b7f4136857e802b84a1c7a0d4003d62095ce5f087f961d7d53fbb8b03f36daca"}}, "dynamic_argument_count": 2, "first_execution_ns": 710743, "input_digest": "e5986bdf4e49ede61599e27269c6dca1b629f7e471515c49b596e23dee5e019b", "matrix": true, "mode": "jax_jit", "staging_ns": 350229958}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A01/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
        16
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "da4b6a54e558a671385792c299468ebf1ea7559ada862d2233400fc950b0d23f",
  "compile_ns": 29545467,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "be6e120a72f5aeacf3fa3a883283861c01e9e0ebfe7f8a3a1ea25379058b172d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b7f4136857e802b84a1c7a0d4003d62095ce5f087f961d7d53fbb8b03f36daca"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 710743,
  "input_digest": "e5986bdf4e49ede61599e27269c6dca1b629f7e471515c49b596e23dee5e019b",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 350229958
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "75ef619ec359532de44b54289838c0b55d361c2505abfdd2f79e1f7c37b94dff", "compile_ns": 28383558, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "b3a390087c665375780dd3025e83d71a21a15eb134b6e9da03bac70305355c63"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "d95f9dea6edb95124ef332729a40ae9af553f66b5e4c33cd4d729ddfa2f2d1cd"}}, "dynamic_argument_count": 2, "first_execution_ns": 648530, "input_digest": "7d618c24b98a9b0eb8b45564fc879dbfbccf6e2f4fbcb19cf60d0c20e4d288be", "matrix": true, "mode": "jax_jit", "staging_ns": 51457611}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A01/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        16,
        32
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "75ef619ec359532de44b54289838c0b55d361c2505abfdd2f79e1f7c37b94dff",
  "compile_ns": 28383558,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "b3a390087c665375780dd3025e83d71a21a15eb134b6e9da03bac70305355c63"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "d95f9dea6edb95124ef332729a40ae9af553f66b5e4c33cd4d729ddfa2f2d1cd"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 648530,
  "input_digest": "7d618c24b98a9b0eb8b45564fc879dbfbccf6e2f4fbcb19cf60d0c20e4d288be",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51457611
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "011dd0ca13a9c1b22bc668875d9b8caf27d8cd4a77a2fd057f2f61de097d7d59", "compile_ns": 29350378, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "dc5d990dcceac4e1eec413a2311c8de274575386cd6a656cf00d834b432083b5"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "6cd5bc39965c8b3547cdb4562fd6b8b955fe352ee03d4ca19ef3ab54a82d6180"}}, "dynamic_argument_count": 2, "first_execution_ns": 642837, "input_digest": "9de5b2f90e1fb393d23df08dd865f8098480fcaabc39d87d554e6d3e13a17235", "matrix": true, "mode": "jax_jit", "staging_ns": 51014463}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 0.0005,
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
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A01/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
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
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "011dd0ca13a9c1b22bc668875d9b8caf27d8cd4a77a2fd057f2f61de097d7d59",
  "compile_ns": 29350378,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "dc5d990dcceac4e1eec413a2311c8de274575386cd6a656cf00d834b432083b5"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6cd5bc39965c8b3547cdb4562fd6b8b955fe352ee03d4ca19ef3ab54a82d6180"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 642837,
  "input_digest": "9de5b2f90e1fb393d23df08dd865f8098480fcaabc39d87d554e6d3e13a17235",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51014463
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "fa1563edabf5e74a9a929f5a2fc9e981d9b364a4f8b4d2f0916c646c7c780169", "compile_ns": 32311231, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "596b26679662ee54c0ac8ae22ba44e12c521dbb9d18bf4eb65574d9eff43e323"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "0355524b9de88e5c55b29f6a7e3bdbca661928a34236d4530fbd9ccd1d71cdbe"}}, "dynamic_argument_count": 2, "first_execution_ns": 624384, "input_digest": "189bd5296c3e972a3d1f55715eb0adc1ea57dfb0b4370073ad3948192359e9ee", "matrix": true, "mode": "jax_jit", "staging_ns": 51596257}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.1920928955078125e-07,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | smoke/A01/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.1920928955078125e-07,
      "passed": true,
      "shape": [
        32,
        64
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "fa1563edabf5e74a9a929f5a2fc9e981d9b364a4f8b4d2f0916c646c7c780169",
  "compile_ns": 32311231,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "596b26679662ee54c0ac8ae22ba44e12c521dbb9d18bf4eb65574d9eff43e323"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0355524b9de88e5c55b29f6a7e3bdbca661928a34236d4530fbd9ccd1d71cdbe"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 624384,
  "input_digest": "189bd5296c3e972a3d1f55715eb0adc1ea57dfb0b4370073ad3948192359e9ee",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51596257
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "7988043e5ad2407465f8988b9e9db40856ea140910d5e8387d50963562167e22", "compile_ns": 29247379, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "6d9b03571101c0808e3db6913285d2b63af7048b2becedda2bab3955648c39e5"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "17598e5dafc978be8e83f477c5babe7fece62962a72f49cd399468815ef3e6df"}}, "dynamic_argument_count": 3, "first_execution_ns": 813286, "input_digest": "83346db655e109e54a8199a8ae20065370c4bc8f3c286b3aefdb6eb225922d6a", "matrix": false, "mode": "jax_jit", "staging_ns": 863736}`
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
| `case_id` | smoke/A02/affine/dev_small |
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
  "case_sha256": "7988043e5ad2407465f8988b9e9db40856ea140910d5e8387d50963562167e22",
  "compile_ns": 29247379,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6d9b03571101c0808e3db6913285d2b63af7048b2becedda2bab3955648c39e5"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "17598e5dafc978be8e83f477c5babe7fece62962a72f49cd399468815ef3e6df"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 813286,
  "input_digest": "83346db655e109e54a8199a8ae20065370c4bc8f3c286b3aefdb6eb225922d6a",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 863736
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "ed9a4b27810b381300ac9b8fdcc4cfb706e3a42233ba3e48abda7bd270c73f18", "compile_ns": 28979203, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "8d5d2cbf37f7318dd1a7b282c1686b26ddc1c5e789f8225c370ead559572b191"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "adda9476f26e72f237a31bab41cc20bd7038287912573a6373b7d81c59fa985c"}}, "dynamic_argument_count": 3, "first_execution_ns": 772420, "input_digest": "c89ba03d58526fbe413d7d033babf092e164443bb07488c86410cea4b43f38ee", "matrix": false, "mode": "jax_jit", "staging_ns": 826745}`
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
| `case_id` | smoke/A02/affine/dev_medium |
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
  "case_sha256": "ed9a4b27810b381300ac9b8fdcc4cfb706e3a42233ba3e48abda7bd270c73f18",
  "compile_ns": 28979203,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8d5d2cbf37f7318dd1a7b282c1686b26ddc1c5e789f8225c370ead559572b191"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "adda9476f26e72f237a31bab41cc20bd7038287912573a6373b7d81c59fa985c"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 772420,
  "input_digest": "c89ba03d58526fbe413d7d033babf092e164443bb07488c86410cea4b43f38ee",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 826745
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "1bcf45b553d878693b3ffab9767111c5248ce0483249954a8488c786e6bc2681", "compile_ns": 28901667, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "1002753567a98acc5996326ff123fb0c88fb824664aaab8c81f829a02ad1f516"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "6faefa27fc9fe823c84da69981274f88820c2c4cc7b4d65c6cba0bb464a1ec21"}}, "dynamic_argument_count": 3, "first_execution_ns": 864234, "input_digest": "70ff878ca614346b6d7ae40c85bd3ea9d6b826c3c158c3dbe17a8e4ef3b5f381", "matrix": false, "mode": "jax_jit", "staging_ns": 779336}`
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
| `case_id` | smoke/A02/affine/tail |
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
  "case_sha256": "1bcf45b553d878693b3ffab9767111c5248ce0483249954a8488c786e6bc2681",
  "compile_ns": 28901667,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "1002753567a98acc5996326ff123fb0c88fb824664aaab8c81f829a02ad1f516"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6faefa27fc9fe823c84da69981274f88820c2c4cc7b4d65c6cba0bb464a1ec21"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 864234,
  "input_digest": "70ff878ca614346b6d7ae40c85bd3ea9d6b826c3c158c3dbe17a8e4ef3b5f381",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 779336
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "c22ff0e5c1ac0c332e5afe3420b6456a7748844856cb2d5932037fa7dd4ef67a", "compile_ns": 28831390, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "755c0d8edd4aaedaa5780ec6ef06c24ca68a5b8936e127e463a94a4bc9ff0766"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "071e481fd75010449a4033d37b001a557f04adc82f2a34b5089302c16d06983b"}}, "dynamic_argument_count": 3, "first_execution_ns": 789539, "input_digest": "9d77234502648b2440474e63ece18d1ce06eb80a8623fa4305a4afdcbad9f7e6", "matrix": false, "mode": "jax_jit", "staging_ns": 813771}`
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
| `case_id` | smoke/A02/affine/scale |
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
  "case_sha256": "c22ff0e5c1ac0c332e5afe3420b6456a7748844856cb2d5932037fa7dd4ef67a",
  "compile_ns": 28831390,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "755c0d8edd4aaedaa5780ec6ef06c24ca68a5b8936e127e463a94a4bc9ff0766"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "071e481fd75010449a4033d37b001a557f04adc82f2a34b5089302c16d06983b"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 789539,
  "input_digest": "9d77234502648b2440474e63ece18d1ce06eb80a8623fa4305a4afdcbad9f7e6",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 813771
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "370f005d84067321dff823b6e9922a3dbc6d97996ee0cb764bd389f878a40f3d", "compile_ns": 25671108, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0008/optimized_hlo.txt", "sha256": "9e257fadc89696a1baf5dc9e311db24518719c94bba5f7a6fb2593080f72d47f"}, "stablehlo": {"path": "compiler/case-0008/stablehlo.txt", "sha256": "438807524454c0226c0e5f2f5dd7bd6bbfb25d0ddbe5bebf8fcb964aeaa4dcba"}}, "dynamic_argument_count": 2, "first_execution_ns": 765630, "input_digest": "45e892e79ea80fb0d47fb845387bfa15237b8fd31d99a82c2893b7b6f8c60f85", "matrix": false, "mode": "jax_jit", "staging_ns": 673575}`
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
| `case_id` | smoke/A02/add/dev_small |
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
  "case_sha256": "370f005d84067321dff823b6e9922a3dbc6d97996ee0cb764bd389f878a40f3d",
  "compile_ns": 25671108,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9e257fadc89696a1baf5dc9e311db24518719c94bba5f7a6fb2593080f72d47f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "438807524454c0226c0e5f2f5dd7bd6bbfb25d0ddbe5bebf8fcb964aeaa4dcba"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 765630,
  "input_digest": "45e892e79ea80fb0d47fb845387bfa15237b8fd31d99a82c2893b7b6f8c60f85",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 673575
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "894aea4471b815153373960de10a820125bbb5ec5660e610a948b3953d526813", "compile_ns": 25558611, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "e253a937c225614a5fcc3c9468c93cac0060446bfbda34a8769d355c9bf630dc"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "9e9eb4671a5c47dbb00ff26e9b9a40741f41c7d528f7b2d1d92404a984ca07f5"}}, "dynamic_argument_count": 2, "first_execution_ns": 676535, "input_digest": "b151053b7c676d4316b1fc50a003bd700ec96c01c7d4a7544b5475c3c3260a20", "matrix": false, "mode": "jax_jit", "staging_ns": 683010}`
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
| `case_id` | smoke/A02/add/dev_medium |
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
  "case_sha256": "894aea4471b815153373960de10a820125bbb5ec5660e610a948b3953d526813",
  "compile_ns": 25558611,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e253a937c225614a5fcc3c9468c93cac0060446bfbda34a8769d355c9bf630dc"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "9e9eb4671a5c47dbb00ff26e9b9a40741f41c7d528f7b2d1d92404a984ca07f5"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 676535,
  "input_digest": "b151053b7c676d4316b1fc50a003bd700ec96c01c7d4a7544b5475c3c3260a20",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 683010
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "a9b9fc6ba228ab8114b2e6e01884a70604907bf8802331f8309544b7bd43c1d9", "compile_ns": 25652253, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "e1b9f6c7f3f9b628c3830733a542fc3e39aae24acee2a671eeb99d0f8cf7366d"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "ba6ba2e35adf79fba3646b565409f1121d7fff652777808e91243bf944188fc0"}}, "dynamic_argument_count": 2, "first_execution_ns": 661778, "input_digest": "394791c21df8b5977f1dfab48eae155f3fb75ca1da9755672e61ae2759498731", "matrix": false, "mode": "jax_jit", "staging_ns": 663551}`
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
| `case_id` | smoke/A02/add/tail |
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
  "case_sha256": "a9b9fc6ba228ab8114b2e6e01884a70604907bf8802331f8309544b7bd43c1d9",
  "compile_ns": 25652253,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e1b9f6c7f3f9b628c3830733a542fc3e39aae24acee2a671eeb99d0f8cf7366d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ba6ba2e35adf79fba3646b565409f1121d7fff652777808e91243bf944188fc0"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 661778,
  "input_digest": "394791c21df8b5977f1dfab48eae155f3fb75ca1da9755672e61ae2759498731",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 663551
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "040362ac77c6cdb4cd1b62841eeb425108f5555bd40a4f2a1920bb0ab9c78d2d", "compile_ns": 25775055, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "0e8aa5e5fefc875b4d1894a90e4f40692513637600028eb6e0fedcc343a8341a"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "33507034b785f5b47c08fdfc77c76c2f0e736e97c42f967f22aee7b88505bf02"}}, "dynamic_argument_count": 2, "first_execution_ns": 790817, "input_digest": "4a86334f50c4e206426a7c9f04bb8ad9f29e99d1071b98e3dc2420c1be8de8d9", "matrix": false, "mode": "jax_jit", "staging_ns": 733804}`
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
| `case_id` | smoke/A02/add/scale |
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
  "case_sha256": "040362ac77c6cdb4cd1b62841eeb425108f5555bd40a4f2a1920bb0ab9c78d2d",
  "compile_ns": 25775055,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "0e8aa5e5fefc875b4d1894a90e4f40692513637600028eb6e0fedcc343a8341a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "33507034b785f5b47c08fdfc77c76c2f0e736e97c42f967f22aee7b88505bf02"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 790817,
  "input_digest": "4a86334f50c4e206426a7c9f04bb8ad9f29e99d1071b98e3dc2420c1be8de8d9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 733804
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "3e4f8e727f10c2e56b6df9c92d6a131a7a97091f345b4ecac1a794a07d056f98", "compile_ns": 25645834, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0012/optimized_hlo.txt", "sha256": "a0165aa27e47a9eed88827f115026c3fb1d46f60a00dd8640de7bbfaa5bab50a"}, "stablehlo": {"path": "compiler/case-0012/stablehlo.txt", "sha256": "274eeed2ca2053dfdc93b5ce3abcca2165bff31357c5ada7c7cfe5d30b66cac3"}}, "dynamic_argument_count": 2, "first_execution_ns": 782665, "input_digest": "ae4781d69d87bc0ec9d6f87deb08f81739fb51cd93c2249960dc8c33bd04bf95", "matrix": false, "mode": "jax_jit", "staging_ns": 717185}`
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
| `case_id` | smoke/A02/sub/dev_small |
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
  "case_sha256": "3e4f8e727f10c2e56b6df9c92d6a131a7a97091f345b4ecac1a794a07d056f98",
  "compile_ns": 25645834,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a0165aa27e47a9eed88827f115026c3fb1d46f60a00dd8640de7bbfaa5bab50a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "274eeed2ca2053dfdc93b5ce3abcca2165bff31357c5ada7c7cfe5d30b66cac3"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 782665,
  "input_digest": "ae4781d69d87bc0ec9d6f87deb08f81739fb51cd93c2249960dc8c33bd04bf95",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 717185
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "8a4361d7b02ab5315e6a38040f91328974f441f046b36fe3b5fa76e749d170f5", "compile_ns": 25529667, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0013/optimized_hlo.txt", "sha256": "0314a7ca516853f7e107a7918ab948272ac5619a67a1549e4be97d784a478fa3"}, "stablehlo": {"path": "compiler/case-0013/stablehlo.txt", "sha256": "8b4779a8fde0ece593ffb8cc23127f757dc73358532d08324f077e1fd8d59a07"}}, "dynamic_argument_count": 2, "first_execution_ns": 720257, "input_digest": "7cda78ce9a0ec2eb6c9303b1b43126dcacf517f1c3cde138a36993af1d5ba093", "matrix": false, "mode": "jax_jit", "staging_ns": 693391}`
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
| `case_id` | smoke/A02/sub/dev_medium |
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
  "case_sha256": "8a4361d7b02ab5315e6a38040f91328974f441f046b36fe3b5fa76e749d170f5",
  "compile_ns": 25529667,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "0314a7ca516853f7e107a7918ab948272ac5619a67a1549e4be97d784a478fa3"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "8b4779a8fde0ece593ffb8cc23127f757dc73358532d08324f077e1fd8d59a07"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 720257,
  "input_digest": "7cda78ce9a0ec2eb6c9303b1b43126dcacf517f1c3cde138a36993af1d5ba093",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 693391
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "c662e4a4bede2917bc721ec08d9d1f6c6157e8865d2af5134c25b3c4f17f68be", "compile_ns": 29857893, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0014/optimized_hlo.txt", "sha256": "0130755db56eed1871df35a10b28e5758bc54ee15c6cbce7f0f4a5c8b30df750"}, "stablehlo": {"path": "compiler/case-0014/stablehlo.txt", "sha256": "dbe5bdf74dc784f1c8b17b5a2657dce4268dc8b1d1a7396cd04a9eac959cf9bf"}}, "dynamic_argument_count": 2, "first_execution_ns": 816415, "input_digest": "594802c834b2339b999029354b170579c6db48e9de2f5179222ec9445204c50a", "matrix": false, "mode": "jax_jit", "staging_ns": 670676}`
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
| `case_id` | smoke/A02/sub/tail |
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
  "case_sha256": "c662e4a4bede2917bc721ec08d9d1f6c6157e8865d2af5134c25b3c4f17f68be",
  "compile_ns": 29857893,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "0130755db56eed1871df35a10b28e5758bc54ee15c6cbce7f0f4a5c8b30df750"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "dbe5bdf74dc784f1c8b17b5a2657dce4268dc8b1d1a7396cd04a9eac959cf9bf"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 816415,
  "input_digest": "594802c834b2339b999029354b170579c6db48e9de2f5179222ec9445204c50a",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 670676
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "6953bd2926f13694afd8f5329f640aa48f6e4a6ea6fd52606b8e085165f0af17", "compile_ns": 25965323, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0015/optimized_hlo.txt", "sha256": "92aca53be53599816e4b598ee5a4631c3bd91c04cb4380448cdb4c40c76b7b31"}, "stablehlo": {"path": "compiler/case-0015/stablehlo.txt", "sha256": "adcc60fc40533653fd6726fbec44758128d8a6e72718fb26ad3da95d37e40768"}}, "dynamic_argument_count": 2, "first_execution_ns": 663701, "input_digest": "248f2892b5023068a0b55932144d323b2fdddfc2be503200bf12c98cfc2e4dc2", "matrix": false, "mode": "jax_jit", "staging_ns": 713560}`
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
| `case_id` | smoke/A02/sub/scale |
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
  "case_sha256": "6953bd2926f13694afd8f5329f640aa48f6e4a6ea6fd52606b8e085165f0af17",
  "compile_ns": 25965323,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "92aca53be53599816e4b598ee5a4631c3bd91c04cb4380448cdb4c40c76b7b31"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "adcc60fc40533653fd6726fbec44758128d8a6e72718fb26ad3da95d37e40768"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 663701,
  "input_digest": "248f2892b5023068a0b55932144d323b2fdddfc2be503200bf12c98cfc2e4dc2",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 713560
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "46b8c67c4a396d878d70cc3c8512060da8f6a9e8a6d069995c8809647410595b", "compile_ns": 26519786, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0016/optimized_hlo.txt", "sha256": "ebf26f9e9539189cd70cd41c7949718d8751f425786eef0f7b6a2e76f5f46fe4"}, "stablehlo": {"path": "compiler/case-0016/stablehlo.txt", "sha256": "0027fd7ad9725353480a2fd1af4af3e41bdcf50cc0f30e0bf5b1f766d8a452ed"}}, "dynamic_argument_count": 2, "first_execution_ns": 780471, "input_digest": "2fc6c3f585d3d02cb8fd43580a185c77847b7f5f460a29c39bbc6cca116d9abc", "matrix": false, "mode": "jax_jit", "staging_ns": 676489}`
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
| `case_id` | smoke/A02/mul/dev_small |
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
  "case_sha256": "46b8c67c4a396d878d70cc3c8512060da8f6a9e8a6d069995c8809647410595b",
  "compile_ns": 26519786,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ebf26f9e9539189cd70cd41c7949718d8751f425786eef0f7b6a2e76f5f46fe4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0027fd7ad9725353480a2fd1af4af3e41bdcf50cc0f30e0bf5b1f766d8a452ed"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 780471,
  "input_digest": "2fc6c3f585d3d02cb8fd43580a185c77847b7f5f460a29c39bbc6cca116d9abc",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 676489
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "239b7cc2d81142fec86de3801049cdc6a8f5c335eb7d8742ea00ef05a2e10879", "compile_ns": 25588164, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0017/optimized_hlo.txt", "sha256": "07917523b07c2eb94a4324833fc1a6e2280500dc07e10bf8092baa78b5f4e9ae"}, "stablehlo": {"path": "compiler/case-0017/stablehlo.txt", "sha256": "276c35c5b2bb7357878224f6ce7338510c7956e0c6fb551944eeda9550652394"}}, "dynamic_argument_count": 2, "first_execution_ns": 897165, "input_digest": "fd8d1e3256f935dcb2ad5870d94b48d7e68ae1bb6a3fc9d7c45ae7d485226d05", "matrix": false, "mode": "jax_jit", "staging_ns": 684928}`
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
| `case_id` | smoke/A02/mul/dev_medium |
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
  "case_sha256": "239b7cc2d81142fec86de3801049cdc6a8f5c335eb7d8742ea00ef05a2e10879",
  "compile_ns": 25588164,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "07917523b07c2eb94a4324833fc1a6e2280500dc07e10bf8092baa78b5f4e9ae"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "276c35c5b2bb7357878224f6ce7338510c7956e0c6fb551944eeda9550652394"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 897165,
  "input_digest": "fd8d1e3256f935dcb2ad5870d94b48d7e68ae1bb6a3fc9d7c45ae7d485226d05",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 684928
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "1f9f777a76cc1b92d7c2734179960b3b87edd43a2433c246113a1c8eac309902", "compile_ns": 25622173, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0018/optimized_hlo.txt", "sha256": "905c144d3e8be5499be56c968ce7586e3a6b918abcc3433fbaf880bd366db2d8"}, "stablehlo": {"path": "compiler/case-0018/stablehlo.txt", "sha256": "902344b74ae959f87ed88b63e105819247c600112895103335a7db2dcf9cc66d"}}, "dynamic_argument_count": 2, "first_execution_ns": 809464, "input_digest": "05882b9aed31cd866b98be36c6ed0a5f2d26c6e4754cda6b671f1be23af7c34b", "matrix": false, "mode": "jax_jit", "staging_ns": 670024}`
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
| `case_id` | smoke/A02/mul/tail |
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
  "case_sha256": "1f9f777a76cc1b92d7c2734179960b3b87edd43a2433c246113a1c8eac309902",
  "compile_ns": 25622173,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "905c144d3e8be5499be56c968ce7586e3a6b918abcc3433fbaf880bd366db2d8"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "902344b74ae959f87ed88b63e105819247c600112895103335a7db2dcf9cc66d"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 809464,
  "input_digest": "05882b9aed31cd866b98be36c6ed0a5f2d26c6e4754cda6b671f1be23af7c34b",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 670024
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "e18a9ab1532e071c13c74992a81a25aabefa63313878dca22d78928bfd2294f5", "compile_ns": 25827906, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0019/optimized_hlo.txt", "sha256": "20de627c7624f4152835b9a3cd62206ddb502535cbebf84ce8262b597db69f6b"}, "stablehlo": {"path": "compiler/case-0019/stablehlo.txt", "sha256": "822e290a04b738e89d75134ab3d3ff9dae89d78661188792b4a346316979ce8f"}}, "dynamic_argument_count": 2, "first_execution_ns": 791859, "input_digest": "e6bc5153662685145338c471df4dac11f65330bfb460d90ee64595b468b841d9", "matrix": false, "mode": "jax_jit", "staging_ns": 766499}`
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
| `case_id` | smoke/A02/mul/scale |
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
  "case_sha256": "e18a9ab1532e071c13c74992a81a25aabefa63313878dca22d78928bfd2294f5",
  "compile_ns": 25827906,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "20de627c7624f4152835b9a3cd62206ddb502535cbebf84ce8262b597db69f6b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "822e290a04b738e89d75134ab3d3ff9dae89d78661188792b4a346316979ce8f"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 791859,
  "input_digest": "e6bc5153662685145338c471df4dac11f65330bfb460d90ee64595b468b841d9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 766499
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "1a4b4f8ffc0aa1054e6be184ecd49f9fd07d57cefed2fec2f96a313aaceaef2c", "compile_ns": 25664453, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0020/optimized_hlo.txt", "sha256": "5ace1788b80acad33f55c596759c0ac7ef1c3d3045c33188090303fa9d8461e5"}, "stablehlo": {"path": "compiler/case-0020/stablehlo.txt", "sha256": "7526090189817bc0389221f986fa52a6ccdbb20facc16eb57f65feb0df6b8b63"}}, "dynamic_argument_count": 2, "first_execution_ns": 781830, "input_digest": "1c32723ddbb19e359c3f44b9273360575729ad6702bfe2b3889bb3b17e47dd0e", "matrix": false, "mode": "jax_jit", "staging_ns": 817859}`
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
| `case_id` | smoke/A02/div/dev_small |
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
  "case_sha256": "1a4b4f8ffc0aa1054e6be184ecd49f9fd07d57cefed2fec2f96a313aaceaef2c",
  "compile_ns": 25664453,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5ace1788b80acad33f55c596759c0ac7ef1c3d3045c33188090303fa9d8461e5"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7526090189817bc0389221f986fa52a6ccdbb20facc16eb57f65feb0df6b8b63"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 781830,
  "input_digest": "1c32723ddbb19e359c3f44b9273360575729ad6702bfe2b3889bb3b17e47dd0e",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 817859
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "883ece128043a9953856054dc6adbde4e75e673a216fb2945e1abcde04210c92", "compile_ns": 25782089, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0021/optimized_hlo.txt", "sha256": "6f563695101c87f10e04239f5e1ab8eb7e3b069cd090a29f507693348c709092"}, "stablehlo": {"path": "compiler/case-0021/stablehlo.txt", "sha256": "1fcc5ec33133b7cd9bc99c4a394fa42aa12739233a156ff05a1739717c7258d7"}}, "dynamic_argument_count": 2, "first_execution_ns": 737810, "input_digest": "12fa3c942ef6a6e8777426946f84a7d21802b0456e3861ae70cea6a93a76d685", "matrix": false, "mode": "jax_jit", "staging_ns": 708400}`
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
| `case_id` | smoke/A02/div/dev_medium |
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
  "case_sha256": "883ece128043a9953856054dc6adbde4e75e673a216fb2945e1abcde04210c92",
  "compile_ns": 25782089,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6f563695101c87f10e04239f5e1ab8eb7e3b069cd090a29f507693348c709092"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1fcc5ec33133b7cd9bc99c4a394fa42aa12739233a156ff05a1739717c7258d7"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 737810,
  "input_digest": "12fa3c942ef6a6e8777426946f84a7d21802b0456e3861ae70cea6a93a76d685",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 708400
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "ee2d2cadae5e04270ba7429042ba7288eb1e972cd8eec73085afad53e4f500ec", "compile_ns": 25767591, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0022/optimized_hlo.txt", "sha256": "842d5a3e5bd5f92845b2c8bf86f50c555da05e5ad2046156dc27cbd6105db287"}, "stablehlo": {"path": "compiler/case-0022/stablehlo.txt", "sha256": "ed4f32b1b8f1b9327d62cfc73c26c262ac1070d303b7a6451e515fd2f7b7ace1"}}, "dynamic_argument_count": 2, "first_execution_ns": 766875, "input_digest": "428fb9a045238218829d01f47a12983def2a744e23c084cd4d2733c21ec37186", "matrix": false, "mode": "jax_jit", "staging_ns": 678650}`
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
| `case_id` | smoke/A02/div/tail |
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
  "case_sha256": "ee2d2cadae5e04270ba7429042ba7288eb1e972cd8eec73085afad53e4f500ec",
  "compile_ns": 25767591,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "842d5a3e5bd5f92845b2c8bf86f50c555da05e5ad2046156dc27cbd6105db287"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ed4f32b1b8f1b9327d62cfc73c26c262ac1070d303b7a6451e515fd2f7b7ace1"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 766875,
  "input_digest": "428fb9a045238218829d01f47a12983def2a744e23c084cd4d2733c21ec37186",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 678650
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "9375af15c9ff3ac2fd44c5440001f244b81e60761ffa700a745146cf23c19405", "compile_ns": 26215303, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0023/optimized_hlo.txt", "sha256": "4d284111c29920f622c9a5441fc8b8d0ec69a48a8cf2fa72d0d5d17511c78f19"}, "stablehlo": {"path": "compiler/case-0023/stablehlo.txt", "sha256": "911c0d9c0108f7b475f1508a628ee20b69f1b5918247012df9a71081379e26a0"}}, "dynamic_argument_count": 2, "first_execution_ns": 697840, "input_digest": "7fb8f3e6a8e26d5d3d776b7b9ca5fb0c3c5e95acd7d1b229d8eac474c07da1bc", "matrix": false, "mode": "jax_jit", "staging_ns": 747853}`
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
| `case_id` | smoke/A02/div/scale |
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
  "case_sha256": "9375af15c9ff3ac2fd44c5440001f244b81e60761ffa700a745146cf23c19405",
  "compile_ns": 26215303,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4d284111c29920f622c9a5441fc8b8d0ec69a48a8cf2fa72d0d5d17511c78f19"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "911c0d9c0108f7b475f1508a628ee20b69f1b5918247012df9a71081379e26a0"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 697840,
  "input_digest": "7fb8f3e6a8e26d5d3d776b7b9ca5fb0c3c5e95acd7d1b229d8eac474c07da1bc",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 747853
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "d0cd2ff51838f34d92a48988844185b84c7db5a298017632a75d3e8ee63f11bc", "compile_ns": 25629557, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0024/optimized_hlo.txt", "sha256": "947cc09c49cbccde653083fee12cb895dff717f055a23d12cd3a688b38ad46fa"}, "stablehlo": {"path": "compiler/case-0024/stablehlo.txt", "sha256": "dddfec6b9b0604a8da07c0dcf6409e72d9fe2eb2d0cf993b3c4baf2d9d45300c"}}, "dynamic_argument_count": 1, "first_execution_ns": 753722, "input_digest": "23b6d5037699ca169b45c0f1671dbb57eff3fc60b5529b71a646fc20cdddc5c9", "matrix": false, "mode": "jax_jit", "staging_ns": 503040}`
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
| `case_id` | smoke/A02/square/dev_small |
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
  "case_sha256": "d0cd2ff51838f34d92a48988844185b84c7db5a298017632a75d3e8ee63f11bc",
  "compile_ns": 25629557,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "947cc09c49cbccde653083fee12cb895dff717f055a23d12cd3a688b38ad46fa"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "dddfec6b9b0604a8da07c0dcf6409e72d9fe2eb2d0cf993b3c4baf2d9d45300c"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 753722,
  "input_digest": "23b6d5037699ca169b45c0f1671dbb57eff3fc60b5529b71a646fc20cdddc5c9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 503040
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "caa0ad9fc627adb4b0550e924417b29ede2994041471ce7c049cf92806982c95", "compile_ns": 29181582, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0025/optimized_hlo.txt", "sha256": "f21f83bd4c31c0fe757476f96d154668337595eaf35c96a8cc98304305b85c78"}, "stablehlo": {"path": "compiler/case-0025/stablehlo.txt", "sha256": "7bca1242aca3722a36bd8d2e61303c94e47600a1c43f7e7a0eade49fdded6398"}}, "dynamic_argument_count": 1, "first_execution_ns": 682430, "input_digest": "430bc069b45fe881ee3e47b2b20eb4db2e9c2365326775c5357069579f28e73e", "matrix": false, "mode": "jax_jit", "staging_ns": 535086}`
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
| `case_id` | smoke/A02/square/dev_medium |
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
  "case_sha256": "caa0ad9fc627adb4b0550e924417b29ede2994041471ce7c049cf92806982c95",
  "compile_ns": 29181582,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "f21f83bd4c31c0fe757476f96d154668337595eaf35c96a8cc98304305b85c78"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7bca1242aca3722a36bd8d2e61303c94e47600a1c43f7e7a0eade49fdded6398"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 682430,
  "input_digest": "430bc069b45fe881ee3e47b2b20eb4db2e9c2365326775c5357069579f28e73e",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 535086
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "4c79344ba3f11992e325db4c32db59d369238851ec168061f1fbf226aecf717c", "compile_ns": 25614717, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0026/optimized_hlo.txt", "sha256": "60fa5ac219a92f012300ab7ac02847cf09e78054de6dd94982a6324c3fcb2cbd"}, "stablehlo": {"path": "compiler/case-0026/stablehlo.txt", "sha256": "4f61c8bf6e1396565c2e60350214671debcdea207898dfde4b9036340d8d86ed"}}, "dynamic_argument_count": 1, "first_execution_ns": 671355, "input_digest": "110120c8a215a0d3b0564f5578e0a5651bb5f0e306ca92087f3d524ea99ad4e3", "matrix": false, "mode": "jax_jit", "staging_ns": 493579}`
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
| `case_id` | smoke/A02/square/tail |
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
  "case_sha256": "4c79344ba3f11992e325db4c32db59d369238851ec168061f1fbf226aecf717c",
  "compile_ns": 25614717,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "60fa5ac219a92f012300ab7ac02847cf09e78054de6dd94982a6324c3fcb2cbd"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4f61c8bf6e1396565c2e60350214671debcdea207898dfde4b9036340d8d86ed"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 671355,
  "input_digest": "110120c8a215a0d3b0564f5578e0a5651bb5f0e306ca92087f3d524ea99ad4e3",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 493579
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "c682099970f16a7cd6267766c11316866adc61b8a105082325937387cb99259d", "compile_ns": 25655582, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0027/optimized_hlo.txt", "sha256": "a6ec97d58f330aa8d6eb7314f813de707f93d2a088908e6c38bac89b5e5718e7"}, "stablehlo": {"path": "compiler/case-0027/stablehlo.txt", "sha256": "ad667c834e25aca0617f345d8b745e2806d6a08afbc93824faed4909e3558509"}}, "dynamic_argument_count": 1, "first_execution_ns": 691741, "input_digest": "9380cab1a888483ea9addbbcea4acc8b8d01c1a7a2be59c50accdc36bf406f07", "matrix": false, "mode": "jax_jit", "staging_ns": 542294}`
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
| `case_id` | smoke/A02/square/scale |
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
  "case_sha256": "c682099970f16a7cd6267766c11316866adc61b8a105082325937387cb99259d",
  "compile_ns": 25655582,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a6ec97d58f330aa8d6eb7314f813de707f93d2a088908e6c38bac89b5e5718e7"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ad667c834e25aca0617f345d8b745e2806d6a08afbc93824faed4909e3558509"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 691741,
  "input_digest": "9380cab1a888483ea9addbbcea4acc8b8d01c1a7a2be59c50accdc36bf406f07",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 542294
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "608589a18f89d831b6327298439bd4c8b89489bd7133bf3fe40e116f5dcc12e3", "compile_ns": 25093019, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0028/optimized_hlo.txt", "sha256": "e59e23e1b58dbe26d4ed310fc2bf9d1d7348a46f179d4752f6eb3a37212676ab"}, "stablehlo": {"path": "compiler/case-0028/stablehlo.txt", "sha256": "77773aeb7bc3838588fe9ce99e16cfbd8750766089c7f745dadba6e3145ba8d8"}}, "dynamic_argument_count": 1, "first_execution_ns": 600687, "input_digest": "cf0933f4e848a33e4f3427786fa3a7a1aeb26ddf8a9ff98f5d66ea0e9bc70362", "matrix": false, "mode": "jax_jit", "staging_ns": 540696}`
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
      "dtype": "bfloat16",
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
| `case_id` | smoke/A02/to_bf16/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "bfloat16",
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
  "case_sha256": "608589a18f89d831b6327298439bd4c8b89489bd7133bf3fe40e116f5dcc12e3",
  "compile_ns": 25093019,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e59e23e1b58dbe26d4ed310fc2bf9d1d7348a46f179d4752f6eb3a37212676ab"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "77773aeb7bc3838588fe9ce99e16cfbd8750766089c7f745dadba6e3145ba8d8"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 600687,
  "input_digest": "cf0933f4e848a33e4f3427786fa3a7a1aeb26ddf8a9ff98f5d66ea0e9bc70362",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 540696
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "1e4335151ffacd1066199f11a1e6e260bff0e4aca0add41b517214562c73d768", "compile_ns": 24923655, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0029/optimized_hlo.txt", "sha256": "3966d6cbbc7f4fd1ee006df72d34920c4aa86df56be693fb53d24f9e0b829c3e"}, "stablehlo": {"path": "compiler/case-0029/stablehlo.txt", "sha256": "bb6a4d75ae8c78e025cec0c41a1dc1a7352f22fe379350116d65eda8fb6ebec6"}}, "dynamic_argument_count": 1, "first_execution_ns": 508402, "input_digest": "e0372c40556d8478b31fb27c19d0f45f5a104ed12c1d710c6ccdf0c5ef62eb15", "matrix": false, "mode": "jax_jit", "staging_ns": 579895}`
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
      "dtype": "bfloat16",
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
| `case_id` | smoke/A02/to_bf16/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "bfloat16",
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
  "case_sha256": "1e4335151ffacd1066199f11a1e6e260bff0e4aca0add41b517214562c73d768",
  "compile_ns": 24923655,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "3966d6cbbc7f4fd1ee006df72d34920c4aa86df56be693fb53d24f9e0b829c3e"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "bb6a4d75ae8c78e025cec0c41a1dc1a7352f22fe379350116d65eda8fb6ebec6"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 508402,
  "input_digest": "e0372c40556d8478b31fb27c19d0f45f5a104ed12c1d710c6ccdf0c5ef62eb15",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 579895
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "9be8fbcedaf7fbe64d274d137c63a386e033f805ea71b9b1e6ba0924f84426d4", "compile_ns": 25010865, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0030/optimized_hlo.txt", "sha256": "7194598049707a529b684ad42e88d16a81dbae12cd93d82b793661ac07aeb608"}, "stablehlo": {"path": "compiler/case-0030/stablehlo.txt", "sha256": "c79022c5c407cecebe8e5cafdad70706205bc212f90aee812de415c2fb2ce990"}}, "dynamic_argument_count": 1, "first_execution_ns": 610794, "input_digest": "aa9db48a726a1fea6b9fcf6c06b2830be7bc4079b98ca79000b252260f32158e", "matrix": false, "mode": "jax_jit", "staging_ns": 537270}`
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
      "dtype": "bfloat16",
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
| `case_id` | smoke/A02/to_bf16/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "bfloat16",
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
  "case_sha256": "9be8fbcedaf7fbe64d274d137c63a386e033f805ea71b9b1e6ba0924f84426d4",
  "compile_ns": 25010865,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7194598049707a529b684ad42e88d16a81dbae12cd93d82b793661ac07aeb608"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c79022c5c407cecebe8e5cafdad70706205bc212f90aee812de415c2fb2ce990"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 610794,
  "input_digest": "aa9db48a726a1fea6b9fcf6c06b2830be7bc4079b98ca79000b252260f32158e",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 537270
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
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

- launch.extras: `{"case_sha256": "a935361310e0bc12dedf9c557cab29b0273e26650dcdd0d49639a9fa52f70032", "compile_ns": 24984189, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0031/optimized_hlo.txt", "sha256": "bf157975418014e24db645f5aa14dcb88e55129e4fcb6d105abfcb3f665c6fd6"}, "stablehlo": {"path": "compiler/case-0031/stablehlo.txt", "sha256": "3ed7623f85f5a7184fe8756203d443ffba2a7cedb1748f66e7a4ce2b7625d362"}}, "dynamic_argument_count": 1, "first_execution_ns": 525998, "input_digest": "e628f0e3b7ae471a92600f74fcacb8263428a6b1e86e79084e7f661a7e09f6cd", "matrix": false, "mode": "jax_jit", "staging_ns": 607627}`
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
      "dtype": "bfloat16",
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
| `case_id` | smoke/A02/to_bf16/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "bfloat16",
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
  "case_sha256": "a935361310e0bc12dedf9c557cab29b0273e26650dcdd0d49639a9fa52f70032",
  "compile_ns": 24984189,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "bf157975418014e24db645f5aa14dcb88e55129e4fcb6d105abfcb3f665c6fd6"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3ed7623f85f5a7184fe8756203d443ffba2a7cedb1748f66e7a4ce2b7625d362"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 525998,
  "input_digest": "e628f0e3b7ae471a92600f74fcacb8263428a6b1e86e79084e7f661a7e09f6cd",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 607627
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 0,
  "stop": 32,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `10a7279a430ba7b3cc04c9ddfa169741128021fa3670fbfc5b6610de90e2cd51`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-002/out/tpu/portable-core-qualification-v1/22f1a0757f76c4af7dbe5e9b04900e73c29715d4c498c0cf0306e71355d6cf93/native-restore-002-core28-000/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
