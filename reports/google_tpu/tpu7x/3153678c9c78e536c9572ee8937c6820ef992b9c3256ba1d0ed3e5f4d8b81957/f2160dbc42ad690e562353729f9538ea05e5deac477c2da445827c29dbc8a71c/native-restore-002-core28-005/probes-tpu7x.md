# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-restore-002-core28-005
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 24
- `fit_status`: `direct`=24
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core) · [portable_core](#portable_core)

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

- launch.extras: `{"case_sha256": "c9a3538150a3bd26e19cbd407305e19efa1c2b768957b1379d25bd9de62fa7be", "compile_ns": 74527770, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "16cb61d9c99bde79b6798943caed3040b84f582a980b4e2b1c9245fe7d76085f"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "20a278740eeb68612a945fbb57547c29dfb9435a1f8cbc073fe3ad6244814839"}}, "dynamic_argument_count": 5, "first_execution_ns": 840015, "input_digest": "033023b496be1dbdf335d78d9414845750d781c743c56d11582e4de69ef04fad", "matrix": true, "mode": "jax_jit", "staging_ns": 309071097}`
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
      "max_abs_error": 0.0,
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
| `case_id` | smoke/N23/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
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
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "c9a3538150a3bd26e19cbd407305e19efa1c2b768957b1379d25bd9de62fa7be",
  "compile_ns": 74527770,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "16cb61d9c99bde79b6798943caed3040b84f582a980b4e2b1c9245fe7d76085f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "20a278740eeb68612a945fbb57547c29dfb9435a1f8cbc073fe3ad6244814839"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 840015,
  "input_digest": "033023b496be1dbdf335d78d9414845750d781c743c56d11582e4de69ef04fad",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 309071097
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "3a1d305edec5ed4a29326cf6b4081c3c12950f750b4e58dd3b65da97095c8579", "compile_ns": 69851435, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "40975022c2556b44262f20e95b5480c7617aafeee1fbcc25d11b9149c775b966"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "f1b4bc9cbfcb2e2d9d80bf0b868d5bc567ff17f4e1bf02faa34291848961fdf0"}}, "dynamic_argument_count": 5, "first_execution_ns": 779795, "input_digest": "9f5205852fc19aa9867016e8e0e76e7805848fa676a596722aca8ad437bc0366", "matrix": true, "mode": "jax_jit", "staging_ns": 27616854}`
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
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
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
| `case_id` | smoke/N23/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        16,
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
  "case_sha256": "3a1d305edec5ed4a29326cf6b4081c3c12950f750b4e58dd3b65da97095c8579",
  "compile_ns": 69851435,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "40975022c2556b44262f20e95b5480c7617aafeee1fbcc25d11b9149c775b966"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "f1b4bc9cbfcb2e2d9d80bf0b868d5bc567ff17f4e1bf02faa34291848961fdf0"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 779795,
  "input_digest": "9f5205852fc19aa9867016e8e0e76e7805848fa676a596722aca8ad437bc0366",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 27616854
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "d66c737bf15ebae984efceef5153520eb00b24eba95d6b175dc6975e4208afa0", "compile_ns": 68030393, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "81409620ba0109c912177b994e80c7a6a0fb3a379aa1cc28e476ee4bb5fe9568"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "b7070ecce9bcfd288b2e43ed349b0a344f75408091f7505db4a52c0da7237ef0"}}, "dynamic_argument_count": 5, "first_execution_ns": 807556, "input_digest": "b3febee324bdadaeb8b061a3aa79a22715ec0e435053dc24dd701219345306fc", "matrix": true, "mode": "jax_jit", "staging_ns": 27133871}`
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
      "max_abs_error": 3.725290298461914e-09,
      "passed": true,
      "shape": [
        7,
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
| `case_id` | smoke/N23/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 3.725290298461914e-09,
      "passed": true,
      "shape": [
        7,
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
  "case_sha256": "d66c737bf15ebae984efceef5153520eb00b24eba95d6b175dc6975e4208afa0",
  "compile_ns": 68030393,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "81409620ba0109c912177b994e80c7a6a0fb3a379aa1cc28e476ee4bb5fe9568"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b7070ecce9bcfd288b2e43ed349b0a344f75408091f7505db4a52c0da7237ef0"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 807556,
  "input_digest": "b3febee324bdadaeb8b061a3aa79a22715ec0e435053dc24dd701219345306fc",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 27133871
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "0c6796dfbb8b2695fd431596a3bfb55b26858dbdd8efc62c11601681fff1c113", "compile_ns": 73077996, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "d63bb74c6b82668e0aebe9a0ba7cb04c02f0cef6fe83e2bd52958f321afa700f"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "eb5c87e5d7563ac7748a0fef8e890ca79603efe359d8a9e2d6c9951542dc1fb5"}}, "dynamic_argument_count": 5, "first_execution_ns": 864123, "input_digest": "0a1c7ac8bb4aaf12e252c011ba48289baf90f95f9ac52621c3ded949354cc2f4", "matrix": true, "mode": "jax_jit", "staging_ns": 26709677}`
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
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        32,
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
| `case_id` | smoke/N23/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        32,
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
  "case_sha256": "0c6796dfbb8b2695fd431596a3bfb55b26858dbdd8efc62c11601681fff1c113",
  "compile_ns": 73077996,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "d63bb74c6b82668e0aebe9a0ba7cb04c02f0cef6fe83e2bd52958f321afa700f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "eb5c87e5d7563ac7748a0fef8e890ca79603efe359d8a9e2d6c9951542dc1fb5"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 864123,
  "input_digest": "0a1c7ac8bb4aaf12e252c011ba48289baf90f95f9ac52621c3ded949354cc2f4",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 26709677
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "10233e886e8e99b8bae8c7308e549ae3bcf839d93babe8959a2a97c3b0ffe9c6", "compile_ns": 42385999, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "9518082f13ec4dc2c04146882bba4aff6b4f28ab05b4b6c9d17e42dd8fbd224b"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "9aece0b4fe08d931358f33eac3865cf798c9d5a62b1b6595a9c7039e72716a4e"}}, "dynamic_argument_count": 9, "first_execution_ns": 1045367, "input_digest": "5b7e52d099b3bd553b6336e99cbe84d26c5246c3d84b88a37eee680e92793f66", "matrix": true, "mode": "jax_jit", "staging_ns": 130738219}`
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
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        1,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        2,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        4,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        1,
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
| `case_id` | smoke/N24/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        1,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        2,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        4,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        1,
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
  "case_sha256": "10233e886e8e99b8bae8c7308e549ae3bcf839d93babe8959a2a97c3b0ffe9c6",
  "compile_ns": 42385999,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9518082f13ec4dc2c04146882bba4aff6b4f28ab05b4b6c9d17e42dd8fbd224b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "9aece0b4fe08d931358f33eac3865cf798c9d5a62b1b6595a9c7039e72716a4e"
    }
  },
  "dynamic_argument_count": 9,
  "first_execution_ns": 1045367,
  "input_digest": "5b7e52d099b3bd553b6336e99cbe84d26c5246c3d84b88a37eee680e92793f66",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 130738219
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "dda8c81103aa47d668260f9986e41abb0847af47e7adcdb12d436f0ab2b18043", "compile_ns": 39913464, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "cfa81b533ba04d7abb770b5633b3e1deac61e9a0bdfde2b3b5aefe584337df31"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "a46caa53d421953b23910b7af30ca3a10282274d0d4a58d94dbaf4c9137bd86c"}}, "dynamic_argument_count": 9, "first_execution_ns": 888155, "input_digest": "2dfd9650941c78293b60219a3eb3275167f7fe29ead2e80961b9bd4486b23a9e", "matrix": true, "mode": "jax_jit", "staging_ns": 130251733}`
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
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        2,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        4,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        2,
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
| `case_id` | smoke/N24/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        2,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        4,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
        16
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        2,
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
  "case_sha256": "dda8c81103aa47d668260f9986e41abb0847af47e7adcdb12d436f0ab2b18043",
  "compile_ns": 39913464,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "cfa81b533ba04d7abb770b5633b3e1deac61e9a0bdfde2b3b5aefe584337df31"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a46caa53d421953b23910b7af30ca3a10282274d0d4a58d94dbaf4c9137bd86c"
    }
  },
  "dynamic_argument_count": 9,
  "first_execution_ns": 888155,
  "input_digest": "2dfd9650941c78293b60219a3eb3275167f7fe29ead2e80961b9bd4486b23a9e",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 130251733
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "b04bbb53b00fd38ce3d8b8082feff7bae8cf1dbb59820a0396cdb3ab95dc0ab4", "compile_ns": 41644155, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "604db57db695b913902e2fa06ee92acfb1dbba4a50986f8e62353222e1399636"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "3c09f46bbcf4d40ce9c4655678b7e28b5de14f55db64dc97bcbe25f38da06f06"}}, "dynamic_argument_count": 9, "first_execution_ns": 894946, "input_digest": "20612165fbed9ae1c411f417caef36c5589dfcf21d71e6a73de5921effcaf51e", "matrix": true, "mode": "jax_jit", "staging_ns": 129625531}`
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
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        1,
        17
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        1,
        17
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        3,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        2,
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
| `case_id` | smoke/N24/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        1,
        17
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        1,
        17
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        3,
        8
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        2,
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
  "case_sha256": "b04bbb53b00fd38ce3d8b8082feff7bae8cf1dbb59820a0396cdb3ab95dc0ab4",
  "compile_ns": 41644155,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "604db57db695b913902e2fa06ee92acfb1dbba4a50986f8e62353222e1399636"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3c09f46bbcf4d40ce9c4655678b7e28b5de14f55db64dc97bcbe25f38da06f06"
    }
  },
  "dynamic_argument_count": 9,
  "first_execution_ns": 894946,
  "input_digest": "20612165fbed9ae1c411f417caef36c5589dfcf21d71e6a73de5921effcaf51e",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 129625531
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "30c03c0c642f16ddfe353beab8efa9f627d1b994011b7ba84a605f9925976190", "compile_ns": 41326968, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "122903059e098b4a64e71e566d451a1611ec4c3b57dfb6050794d127b5a18cbc"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "6163f9202ae8906a9eccb2e7d503e8d6da50900c30b6d515aec6324ddaf59c8e"}}, "dynamic_argument_count": 9, "first_execution_ns": 872327, "input_digest": "fbadbcf4b8aeedd7d608116a76b407ce211159e86b6529d0d4e92b511027a576", "matrix": true, "mode": "jax_jit", "staging_ns": 155993049}`
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
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        4,
        64
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
        64
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        16,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        4,
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
| `case_id` | smoke/N24/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        0,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        4,
        64
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
        64
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        16,
        32
      ]
    },
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        4,
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
  "case_sha256": "30c03c0c642f16ddfe353beab8efa9f627d1b994011b7ba84a605f9925976190",
  "compile_ns": 41326968,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "122903059e098b4a64e71e566d451a1611ec4c3b57dfb6050794d127b5a18cbc"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6163f9202ae8906a9eccb2e7d503e8d6da50900c30b6d515aec6324ddaf59c8e"
    }
  },
  "dynamic_argument_count": 9,
  "first_execution_ns": 872327,
  "input_digest": "fbadbcf4b8aeedd7d608116a76b407ce211159e86b6529d0d4e92b511027a576",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 155993049
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "515a586cbd78a8fbb4247400b7d474c5cfdb5aeb29693e62d6829773abfe6825", "compile_ns": 36927484, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0008/optimized_hlo.txt", "sha256": "73635e07ff638a988f5f8928130a19698d5b9e5df4081aa364c909acb2109d69"}, "stablehlo": {"path": "compiler/case-0008/stablehlo.txt", "sha256": "e62024e3fb247b8291a4af48544965ac86343ec236875f291439fc3f539003a3"}}, "dynamic_argument_count": 1, "first_execution_ns": 1080036, "input_digest": "ddaa1431a6cca93ada830fa5ca80bcd428614ec3de8f6faaf1c9c01f57d41c71", "matrix": false, "mode": "jax_jit", "staging_ns": 565193}`
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
      "max_abs_error": 9.5367431640625e-07,
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
| `case_id` | smoke/N25/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 9.5367431640625e-07,
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
  "case_sha256": "515a586cbd78a8fbb4247400b7d474c5cfdb5aeb29693e62d6829773abfe6825",
  "compile_ns": 36927484,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "73635e07ff638a988f5f8928130a19698d5b9e5df4081aa364c909acb2109d69"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "e62024e3fb247b8291a4af48544965ac86343ec236875f291439fc3f539003a3"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1080036,
  "input_digest": "ddaa1431a6cca93ada830fa5ca80bcd428614ec3de8f6faaf1c9c01f57d41c71",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 565193
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "34739e67575da2d1b7a8e0e9566d666f32d1787ae67966459e5139e158be1f08", "compile_ns": 42689876, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "c36cb8df37fd71f9d0ea1d2986ca016ef76b79e1c1f8fdda506ae5baee53d968"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "ec9101eccb1d44e569718804389372a68d340b151eb904f9209f23d4c2466661"}}, "dynamic_argument_count": 1, "first_execution_ns": 801919, "input_digest": "0487eb08b2673a48bbf4a066d66d586dd28aabbfe0b9084673192b83fc3cb570", "matrix": false, "mode": "jax_jit", "staging_ns": 535462}`
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
      "max_abs_error": 1.430511474609375e-06,
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
| `case_id` | smoke/N25/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.430511474609375e-06,
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
  "case_sha256": "34739e67575da2d1b7a8e0e9566d666f32d1787ae67966459e5139e158be1f08",
  "compile_ns": 42689876,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "c36cb8df37fd71f9d0ea1d2986ca016ef76b79e1c1f8fdda506ae5baee53d968"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ec9101eccb1d44e569718804389372a68d340b151eb904f9209f23d4c2466661"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 801919,
  "input_digest": "0487eb08b2673a48bbf4a066d66d586dd28aabbfe0b9084673192b83fc3cb570",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 535462
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "e01b84f787432f9d27dd561cb2f865efe78cc7f8ee3dccb671d53880cf527737", "compile_ns": 33555689, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "f547e1cb64c3de147081fbec64a98e9502977a3918177f8cb1ccbb67c1f92eab"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "066329e3d89f451bc5fd174ee45c951a47a6f6fbf0210efbe75f945344d0bb3f"}}, "dynamic_argument_count": 1, "first_execution_ns": 702869, "input_digest": "a8339f5b5dd4858acf7f93e21ab74bf6f0e0969b5faf6d1dc7ba60465da2e1ca", "matrix": false, "mode": "jax_jit", "staging_ns": 536718}`
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
      "max_abs_error": 4.76837158203125e-07,
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
| `case_id` | smoke/N25/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 4.76837158203125e-07,
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
  "case_sha256": "e01b84f787432f9d27dd561cb2f865efe78cc7f8ee3dccb671d53880cf527737",
  "compile_ns": 33555689,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "f547e1cb64c3de147081fbec64a98e9502977a3918177f8cb1ccbb67c1f92eab"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "066329e3d89f451bc5fd174ee45c951a47a6f6fbf0210efbe75f945344d0bb3f"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 702869,
  "input_digest": "a8339f5b5dd4858acf7f93e21ab74bf6f0e0969b5faf6d1dc7ba60465da2e1ca",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 536718
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "3281e37b0f8bc0f4afefd350688f86a8ca202b22a129bdf4267bebf7573d74bd", "compile_ns": 81637228, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "8b0c50020f90d54b3aff3cbbb41a51957bfb7afce30ff19ca1f25e59b0202ba5"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "7f8ac5f5e7dd19a22bebf319b8681f48814bc15de3449b464d04088dc23f3b60"}}, "dynamic_argument_count": 1, "first_execution_ns": 672333, "input_digest": "f49b373a1afbe1e6ed9f22d2c508584aa6947a327743b3b84f204ac3ca9364ef", "matrix": false, "mode": "jax_jit", "staging_ns": 579878}`
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
      "max_abs_error": 2.86102294921875e-06,
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
| `case_id` | smoke/N25/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.86102294921875e-06,
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
  "case_sha256": "3281e37b0f8bc0f4afefd350688f86a8ca202b22a129bdf4267bebf7573d74bd",
  "compile_ns": 81637228,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8b0c50020f90d54b3aff3cbbb41a51957bfb7afce30ff19ca1f25e59b0202ba5"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7f8ac5f5e7dd19a22bebf319b8681f48814bc15de3449b464d04088dc23f3b60"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 672333,
  "input_digest": "f49b373a1afbe1e6ed9f22d2c508584aa6947a327743b3b84f204ac3ca9364ef",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 579878
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "3125648e37c7055fd52853eecacd1243b41839579be95a4cb674ce554369fcc1", "compile_ns": 40059816, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0012/optimized_hlo.txt", "sha256": "f2e3940aae961f1eaca5f74f161d5a29f046f65c8fade4ff497d0193c54bb6e0"}, "stablehlo": {"path": "compiler/case-0012/stablehlo.txt", "sha256": "fb0964a9271f07cffeada04d37c62c4a9e977254fc4c8653358dad099d24d950"}}, "dynamic_argument_count": 2, "first_execution_ns": 890632, "input_digest": "efa66c061bfb196e4c7d1e792fbd1df0898f76d64021b44bd62537efd1cb3d37", "matrix": false, "mode": "jax_jit", "staging_ns": 805174}`
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
        5,
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
| `case_id` | smoke/N26/base/dev_small |
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
        5,
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
  "case_sha256": "3125648e37c7055fd52853eecacd1243b41839579be95a4cb674ce554369fcc1",
  "compile_ns": 40059816,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "f2e3940aae961f1eaca5f74f161d5a29f046f65c8fade4ff497d0193c54bb6e0"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "fb0964a9271f07cffeada04d37c62c4a9e977254fc4c8653358dad099d24d950"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 890632,
  "input_digest": "efa66c061bfb196e4c7d1e792fbd1df0898f76d64021b44bd62537efd1cb3d37",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 805174
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "3ba5aaead4a5f65c5ded954c0120bc00117da76296724352bfa3ebf181e3e54c", "compile_ns": 41116209, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0013/optimized_hlo.txt", "sha256": "df42bdb473fab60d2b4825dcbe1ac045e0b156ade6ba8dc8ed0f0650e9e80348"}, "stablehlo": {"path": "compiler/case-0013/stablehlo.txt", "sha256": "31824e54a11355ce43dcad7c7401852fc46c90e6cd765fb20d6fd04e44d23be1"}}, "dynamic_argument_count": 2, "first_execution_ns": 845312, "input_digest": "b8bf1577fad64af0f9e5983b0019f4f1684735005f9a855fe7fed30a21e81061", "matrix": false, "mode": "jax_jit", "staging_ns": 748949}`
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
        5,
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
| `case_id` | smoke/N26/base/dev_medium |
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
        5,
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
  "case_sha256": "3ba5aaead4a5f65c5ded954c0120bc00117da76296724352bfa3ebf181e3e54c",
  "compile_ns": 41116209,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "df42bdb473fab60d2b4825dcbe1ac045e0b156ade6ba8dc8ed0f0650e9e80348"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "31824e54a11355ce43dcad7c7401852fc46c90e6cd765fb20d6fd04e44d23be1"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 845312,
  "input_digest": "b8bf1577fad64af0f9e5983b0019f4f1684735005f9a855fe7fed30a21e81061",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 748949
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "cf3304746dda2c267b2ad383bd188b6c977f2a57c1b3140b67979ff4024c336f", "compile_ns": 40749272, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0014/optimized_hlo.txt", "sha256": "44762a9aa334a822e7c81383a6ca3371b47fad11c78d2a8291066b44a09289aa"}, "stablehlo": {"path": "compiler/case-0014/stablehlo.txt", "sha256": "2f7a8fec74086349e59389279602462e4f7e7342697560565008efd7f3f25447"}}, "dynamic_argument_count": 2, "first_execution_ns": 913457, "input_digest": "0607dfd5e500a0f8ef9e8b3070c6ef88c5cce53ba0c52bb22e5456a63d64b7ca", "matrix": false, "mode": "jax_jit", "staging_ns": 742381}`
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
        5,
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
| `case_id` | smoke/N26/base/tail |
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
        5,
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
  "case_sha256": "cf3304746dda2c267b2ad383bd188b6c977f2a57c1b3140b67979ff4024c336f",
  "compile_ns": 40749272,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "44762a9aa334a822e7c81383a6ca3371b47fad11c78d2a8291066b44a09289aa"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "2f7a8fec74086349e59389279602462e4f7e7342697560565008efd7f3f25447"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 913457,
  "input_digest": "0607dfd5e500a0f8ef9e8b3070c6ef88c5cce53ba0c52bb22e5456a63d64b7ca",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 742381
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "b3612aa1f351f4547813bd289c038b26bf8acf8a2427daa78a319840675b8feb", "compile_ns": 40603052, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0015/optimized_hlo.txt", "sha256": "50e013321476063617d13f6c2eb9d7cdefd7b6c44c2ffc2d95c4b14b9f276c8f"}, "stablehlo": {"path": "compiler/case-0015/stablehlo.txt", "sha256": "6fc9f80332c648344b1eba4f8a3c6df6e25dfeb405dc66eeec0365334cf74a93"}}, "dynamic_argument_count": 2, "first_execution_ns": 906818, "input_digest": "b35dd9aa7e3e431280eafaf01febcb69405dddccf310c0e0e1d3cdf50191f151", "matrix": false, "mode": "jax_jit", "staging_ns": 784016}`
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
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        5,
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
| `case_id` | smoke/N26/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        5,
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
  "case_sha256": "b3612aa1f351f4547813bd289c038b26bf8acf8a2427daa78a319840675b8feb",
  "compile_ns": 40603052,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "50e013321476063617d13f6c2eb9d7cdefd7b6c44c2ffc2d95c4b14b9f276c8f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6fc9f80332c648344b1eba4f8a3c6df6e25dfeb405dc66eeec0365334cf74a93"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 906818,
  "input_digest": "b35dd9aa7e3e431280eafaf01febcb69405dddccf310c0e0e1d3cdf50191f151",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 784016
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "f4fe3f0cf2b043a8e7f4a13652843d7a846e6369afa7897269ef265b35f91a41", "compile_ns": 327281575, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0016/optimized_hlo.txt", "sha256": "4f87a192ce53f320ff3c9a4b6a4aa821d01db27d3f9cb6a0fd3447f90a2cc82f"}, "stablehlo": {"path": "compiler/case-0016/stablehlo.txt", "sha256": "0a3b97b8871c2195783d9ab514a27e90ad5d26f57a84e9cdd0d88b882d5fca03"}}, "dynamic_argument_count": 2, "first_execution_ns": 1255773, "input_digest": "f4830dc9823c508d69e856f2148af239fed752202b02361d1e2b03a98b1174b7", "matrix": false, "mode": "jax_jit", "staging_ns": 822849}`
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
        2,
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
| `case_id` | smoke/N27/base/dev_small |
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
        2,
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
  "case_sha256": "f4fe3f0cf2b043a8e7f4a13652843d7a846e6369afa7897269ef265b35f91a41",
  "compile_ns": 327281575,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4f87a192ce53f320ff3c9a4b6a4aa821d01db27d3f9cb6a0fd3447f90a2cc82f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0a3b97b8871c2195783d9ab514a27e90ad5d26f57a84e9cdd0d88b882d5fca03"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1255773,
  "input_digest": "f4830dc9823c508d69e856f2148af239fed752202b02361d1e2b03a98b1174b7",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 822849
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "ea8f433175912ef4d914de0e537687ad5287bbb35f45f5dde02de8e1f3ddca40", "compile_ns": 295667406, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0017/optimized_hlo.txt", "sha256": "65442db8fdb722285ef9690702394ccc80efc03a7dcf30a6153d7c19506cbc8a"}, "stablehlo": {"path": "compiler/case-0017/stablehlo.txt", "sha256": "605e6a65c10c1782938c7718e035b6ab5c83bcf0e5253a232b0c8f76b9577de0"}}, "dynamic_argument_count": 2, "first_execution_ns": 1011882, "input_digest": "15b0c1b6b79ba080a2e5282d1ccb31be6bfabca38308946edd090a51c4bf2ee1", "matrix": false, "mode": "jax_jit", "staging_ns": 819752}`
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
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        4,
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
| `case_id` | smoke/N27/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        4,
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
  "case_sha256": "ea8f433175912ef4d914de0e537687ad5287bbb35f45f5dde02de8e1f3ddca40",
  "compile_ns": 295667406,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "65442db8fdb722285ef9690702394ccc80efc03a7dcf30a6153d7c19506cbc8a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "605e6a65c10c1782938c7718e035b6ab5c83bcf0e5253a232b0c8f76b9577de0"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1011882,
  "input_digest": "15b0c1b6b79ba080a2e5282d1ccb31be6bfabca38308946edd090a51c4bf2ee1",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 819752
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "424812cb4fbbcab39123440c70b0c60dc38af59945c4de55e248c2d1a8f16496", "compile_ns": 279066370, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0018/optimized_hlo.txt", "sha256": "5fe70c10db32fa9fc1fd97f85a78b6781efbdecb204ba0634a7adcde91d84058"}, "stablehlo": {"path": "compiler/case-0018/stablehlo.txt", "sha256": "1ca732d8d4f7bc64a59aaaf352e4b4e924ff85dd634de902a6863fe00754fb86"}}, "dynamic_argument_count": 2, "first_execution_ns": 1289018, "input_digest": "ab57d66c8f0960c6d8f1db03956fdb4e27b79c292998a5851b63ac8bd9f84f3d", "matrix": false, "mode": "jax_jit", "staging_ns": 740942}`
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
        2,
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
| `case_id` | smoke/N27/base/tail |
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
        2,
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
  "case_sha256": "424812cb4fbbcab39123440c70b0c60dc38af59945c4de55e248c2d1a8f16496",
  "compile_ns": 279066370,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5fe70c10db32fa9fc1fd97f85a78b6781efbdecb204ba0634a7adcde91d84058"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1ca732d8d4f7bc64a59aaaf352e4b4e924ff85dd634de902a6863fe00754fb86"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1289018,
  "input_digest": "ab57d66c8f0960c6d8f1db03956fdb4e27b79c292998a5851b63ac8bd9f84f3d",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 740942
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "1cb0edad50cf81738a5e4033dfc48341a5e1abd8ff03a8ad13d6284b66e05740", "compile_ns": 292749113, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0019/optimized_hlo.txt", "sha256": "a15f878951f8602b2d07349965201e07870cdb34109bcac0b8055f8589faca18"}, "stablehlo": {"path": "compiler/case-0019/stablehlo.txt", "sha256": "fc997d6ba57d4845c579f49d69256a04e0e35dbed39130918c15c4aedeba04d1"}}, "dynamic_argument_count": 2, "first_execution_ns": 1199836, "input_digest": "b44095b0b5bb3554f10f0851f2e7e2f8e1260e678609c4aed2ab7e21d59a168f", "matrix": false, "mode": "jax_jit", "staging_ns": 825700}`
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
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        8,
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
| `case_id` | smoke/N27/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        8,
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
  "case_sha256": "1cb0edad50cf81738a5e4033dfc48341a5e1abd8ff03a8ad13d6284b66e05740",
  "compile_ns": 292749113,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a15f878951f8602b2d07349965201e07870cdb34109bcac0b8055f8589faca18"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "fc997d6ba57d4845c579f49d69256a04e0e35dbed39130918c15c4aedeba04d1"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1199836,
  "input_digest": "b44095b0b5bb3554f10f0851f2e7e2f8e1260e678609c4aed2ab7e21d59a168f",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 825700
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "d73bb8a70c210a2164f237d795dfd9b051aee28f51f0011112e295d0be4a6eae", "compile_ns": 70230143, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0020/optimized_hlo.txt", "sha256": "4fbcf6ddad0d64440e91e95bd7ac4fc911303ce8e018ba54f11250262878a1bb"}, "stablehlo": {"path": "compiler/case-0020/stablehlo.txt", "sha256": "a0a5f0458359be3ab8d25d50c131bac684e9a0436db7d2b702bae320e31121ac"}}, "dynamic_argument_count": 2, "first_execution_ns": 1231461, "input_digest": "5a02ad4efaf57c8ee590aff22848c89df26cbdacd090f4615e0b7c8e4dc4d302", "matrix": false, "mode": "jax_jit", "staging_ns": 1036917}`
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
        8,
        16,
        4
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
| `case_id` | smoke/N28/base/dev_small |
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
        8,
        16,
        4
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
  "case_sha256": "d73bb8a70c210a2164f237d795dfd9b051aee28f51f0011112e295d0be4a6eae",
  "compile_ns": 70230143,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4fbcf6ddad0d64440e91e95bd7ac4fc911303ce8e018ba54f11250262878a1bb"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a0a5f0458359be3ab8d25d50c131bac684e9a0436db7d2b702bae320e31121ac"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1231461,
  "input_digest": "5a02ad4efaf57c8ee590aff22848c89df26cbdacd090f4615e0b7c8e4dc4d302",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1036917
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "c3396648b19e7ef88ee89080a9119cc87357f1383be2c0dbfb4c6557bb69ecef", "compile_ns": 78492812, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0021/optimized_hlo.txt", "sha256": "c1ae26303fe2316e274937e919f289045c409a6d6eb58a7dd31875fb7d41fd42"}, "stablehlo": {"path": "compiler/case-0021/stablehlo.txt", "sha256": "22f832e2edb19e534d1a99904cc3f5bc94d1d0d27794a77db4604b8ec4f273c6"}}, "dynamic_argument_count": 2, "first_execution_ns": 1281743, "input_digest": "e81004129a0172d16068f400b2266ebe967172b394722fe83b6b7140f47d7101", "matrix": false, "mode": "jax_jit", "staging_ns": 1000766}`
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
        16,
        32,
        4
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
| `case_id` | smoke/N28/base/dev_medium |
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
        16,
        32,
        4
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
  "case_sha256": "c3396648b19e7ef88ee89080a9119cc87357f1383be2c0dbfb4c6557bb69ecef",
  "compile_ns": 78492812,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "c1ae26303fe2316e274937e919f289045c409a6d6eb58a7dd31875fb7d41fd42"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "22f832e2edb19e534d1a99904cc3f5bc94d1d0d27794a77db4604b8ec4f273c6"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1281743,
  "input_digest": "e81004129a0172d16068f400b2266ebe967172b394722fe83b6b7140f47d7101",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1000766
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "683a3ab9424ddec73fbc5955fae8e35cdf4d3236972f9b639ee8d5b05f0ef3f1", "compile_ns": 66512575, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0022/optimized_hlo.txt", "sha256": "57962d22e41b4b5294afec2ad0c718287b4d8a89f06e28580854259fc29ad677"}, "stablehlo": {"path": "compiler/case-0022/stablehlo.txt", "sha256": "1b380c373d4df6b08ea1aa4461ccd5fd24bfcf7af03773228b00623683184dda"}}, "dynamic_argument_count": 2, "first_execution_ns": 1315018, "input_digest": "49e2b4f1e8e145b31ec92ec59fe3292424b84d5855739691ecf2eeb433879363", "matrix": false, "mode": "jax_jit", "staging_ns": 898692}`
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
      "max_abs_error": 8.940696716308594e-08,
      "passed": true,
      "shape": [
        7,
        17,
        4
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
| `case_id` | smoke/N28/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 8.940696716308594e-08,
      "passed": true,
      "shape": [
        7,
        17,
        4
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
  "case_sha256": "683a3ab9424ddec73fbc5955fae8e35cdf4d3236972f9b639ee8d5b05f0ef3f1",
  "compile_ns": 66512575,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "57962d22e41b4b5294afec2ad0c718287b4d8a89f06e28580854259fc29ad677"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1b380c373d4df6b08ea1aa4461ccd5fd24bfcf7af03773228b00623683184dda"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1315018,
  "input_digest": "49e2b4f1e8e145b31ec92ec59fe3292424b84d5855739691ecf2eeb433879363",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 898692
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
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

- launch.extras: `{"case_sha256": "ac998dbfd10d038860ed52537e64e94d1c79ea959e9b647b2e39382110c598fa", "compile_ns": 108268394, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0023/optimized_hlo.txt", "sha256": "9e753495e70e57c759747fc11d4c4b69c2564b682d48201d7883ae967aa9dfb7"}, "stablehlo": {"path": "compiler/case-0023/stablehlo.txt", "sha256": "967bb86111768ba1b5076fba5b140b8d869c2a2a198777cdd935d1c469bd95e5"}}, "dynamic_argument_count": 2, "first_execution_ns": 1242766, "input_digest": "6d2420b1ef405178f1bbb4a5c3d3faa8f21982ae929275252181036b64044cca", "matrix": false, "mode": "jax_jit", "staging_ns": 1256995}`
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
        64,
        4
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
| `case_id` | smoke/N28/base/scale |
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
        64,
        4
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
  "case_sha256": "ac998dbfd10d038860ed52537e64e94d1c79ea959e9b647b2e39382110c598fa",
  "compile_ns": 108268394,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9e753495e70e57c759747fc11d4c4b69c2564b682d48201d7883ae967aa9dfb7"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "967bb86111768ba1b5076fba5b140b8d869c2a2a198777cdd935d1c469bd95e5"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 1242766,
  "input_digest": "6d2420b1ef405178f1bbb4a5c3d3faa8f21982ae929275252181036b64044cca",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1256995
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 160,
  "stop": 184,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `e9c905b1b3e893aa1c4d3f85daebb9753d8210c9103c25628657d9fc1900b21b`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-002/out/tpu/portable-core-qualification-v1/1a136003c1c9cd7c17c1f51a2a5a02c2e3508f0762e54ccdf3efa4c7fb4116f5/native-restore-002-core28-005/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
