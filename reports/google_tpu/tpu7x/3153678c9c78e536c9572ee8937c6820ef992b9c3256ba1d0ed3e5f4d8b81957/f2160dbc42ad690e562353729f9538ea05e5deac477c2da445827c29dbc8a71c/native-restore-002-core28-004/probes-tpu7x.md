# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-restore-002-core28-004
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

- launch.extras: `{"case_sha256": "3529332186306f964197cc0f339e4b86e23cc9d8420b333602c001b71af51d78", "compile_ns": 35938659, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "f32301e64f380d58d6f6b14ca2eb60569f7717f543063ce2953d835b37d8effb"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "c68aabc958f59f78125bb17235685bf043996a0e51d5ed6125481f8e992cc86c"}}, "dynamic_argument_count": 3, "first_execution_ns": 3805096, "input_digest": "3e28654d984818d10d61cc4ea587746b22154259e302a283e0c68318dbbaa8d7", "matrix": true, "mode": "jax_jit", "staging_ns": 339863484}`
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
| `case_id` | smoke/D03/MP/dev_small |
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
  "case_sha256": "3529332186306f964197cc0f339e4b86e23cc9d8420b333602c001b71af51d78",
  "compile_ns": 35938659,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "f32301e64f380d58d6f6b14ca2eb60569f7717f543063ce2953d835b37d8effb"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c68aabc958f59f78125bb17235685bf043996a0e51d5ed6125481f8e992cc86c"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 3805096,
  "input_digest": "3e28654d984818d10d61cc4ea587746b22154259e302a283e0c68318dbbaa8d7",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 339863484
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "84517be7dbe61289dfb0aa422cb99c5a216cd7cada0c38e9a82ef5f868a95870", "compile_ns": 34216816, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "5e09a34ddeb072d81e48644fdacc6bf7a2e616a32b2592e8ea71dba565757a4c"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "9afb09e8dea8d61d14638b9d848e7448afc716e3906f2c41efb3765586774224"}}, "dynamic_argument_count": 3, "first_execution_ns": 650493, "input_digest": "d3f45623b1a4bbc515f4e64149be5ede1d1e7004679220e070dec453a953ebd4", "matrix": true, "mode": "jax_jit", "staging_ns": 51332333}`
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
| `case_id` | smoke/D03/MP/dev_medium |
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
  "case_sha256": "84517be7dbe61289dfb0aa422cb99c5a216cd7cada0c38e9a82ef5f868a95870",
  "compile_ns": 34216816,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5e09a34ddeb072d81e48644fdacc6bf7a2e616a32b2592e8ea71dba565757a4c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "9afb09e8dea8d61d14638b9d848e7448afc716e3906f2c41efb3765586774224"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 650493,
  "input_digest": "d3f45623b1a4bbc515f4e64149be5ede1d1e7004679220e070dec453a953ebd4",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51332333
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "8268b5934a90f1c4ce3300fb156d7b490043a0781bcb1fb7a89f557099fc4952", "compile_ns": 34299326, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "7602b4f0311147b269e0bca785b5019864b989b7964876ee7438888fef1d719c"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "3e05f5a3ed3bcc154635b89496b43ffd87d196381fb6f241fe117b3469c3ba37"}}, "dynamic_argument_count": 3, "first_execution_ns": 630578, "input_digest": "78810d73b8940f8b0349026384a19234e59bf0978f74763bd41e0e017a25278f", "matrix": true, "mode": "jax_jit", "staging_ns": 51234295}`
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
| `case_id` | smoke/D03/MP/tail |
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
  "case_sha256": "8268b5934a90f1c4ce3300fb156d7b490043a0781bcb1fb7a89f557099fc4952",
  "compile_ns": 34299326,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7602b4f0311147b269e0bca785b5019864b989b7964876ee7438888fef1d719c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3e05f5a3ed3bcc154635b89496b43ffd87d196381fb6f241fe117b3469c3ba37"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 630578,
  "input_digest": "78810d73b8940f8b0349026384a19234e59bf0978f74763bd41e0e017a25278f",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51234295
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "397bdccd891f26ef508dfccad2d36484b172f5cdceb62f4459b96f19cf5628ae", "compile_ns": 38697556, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "ac4fb779e3dd9a11efca264b847e7b5448b15539d2a190663080f257f4bcdb2d"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "1601fc737aa9c720933bcdc9eae9e00c8281443393499f5fb99f22f9712df4d9"}}, "dynamic_argument_count": 3, "first_execution_ns": 593390, "input_digest": "629e23e288bcb1d0bb23fdfe546577b2e9eafa9fcdc5e0aba48217878c4b3fe5", "matrix": true, "mode": "jax_jit", "staging_ns": 51601169}`
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
| `case_id` | smoke/D03/MP/scale |
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
  "case_sha256": "397bdccd891f26ef508dfccad2d36484b172f5cdceb62f4459b96f19cf5628ae",
  "compile_ns": 38697556,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ac4fb779e3dd9a11efca264b847e7b5448b15539d2a190663080f257f4bcdb2d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1601fc737aa9c720933bcdc9eae9e00c8281443393499f5fb99f22f9712df4d9"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 593390,
  "input_digest": "629e23e288bcb1d0bb23fdfe546577b2e9eafa9fcdc5e0aba48217878c4b3fe5",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51601169
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "5abfabe47c8c97defdea6453d3f94a07d80e0346efc5df1dd16832cf8ea4831b", "compile_ns": 32075160, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "df3534955c6db054550aaa2aba15d0e9e9877f1d616b7ea3cb48395564cdf687"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "6bcac81934cb95f43fc91b2e2bc34798185deb8b8571c635a32de60359ea26d8"}}, "dynamic_argument_count": 4, "first_execution_ns": 627501, "input_digest": "3ad9506df750514363aea11f5dd746aea81fa145d07bc8d62e178b38d8faa27d", "matrix": true, "mode": "jax_jit", "staging_ns": 1842432}`
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
| `case_id` | smoke/D03/MM/dev_small |
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
  "case_sha256": "5abfabe47c8c97defdea6453d3f94a07d80e0346efc5df1dd16832cf8ea4831b",
  "compile_ns": 32075160,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "df3534955c6db054550aaa2aba15d0e9e9877f1d616b7ea3cb48395564cdf687"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6bcac81934cb95f43fc91b2e2bc34798185deb8b8571c635a32de60359ea26d8"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 627501,
  "input_digest": "3ad9506df750514363aea11f5dd746aea81fa145d07bc8d62e178b38d8faa27d",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1842432
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "2e3704f3b95451e0229f795ca47193a9ead669a960b0ba89793427ad2b9d75f2", "compile_ns": 32182972, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "eb846d24c96d7d0e9752b97fbbcdb764fc4f4f620b4f7163e52d9a5823d4a8de"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "4a8a488abc1f315f35f56310b004d9a4682d5873ba75db6c7adbb81bdd6aa6cc"}}, "dynamic_argument_count": 4, "first_execution_ns": 670374, "input_digest": "44912745eea939bc2757bd9bfc70316283348555ad0df1beb14be6072784d61e", "matrix": true, "mode": "jax_jit", "staging_ns": 1892515}`
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
| `case_id` | smoke/D03/MM/dev_medium |
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
  "case_sha256": "2e3704f3b95451e0229f795ca47193a9ead669a960b0ba89793427ad2b9d75f2",
  "compile_ns": 32182972,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "eb846d24c96d7d0e9752b97fbbcdb764fc4f4f620b4f7163e52d9a5823d4a8de"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4a8a488abc1f315f35f56310b004d9a4682d5873ba75db6c7adbb81bdd6aa6cc"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 670374,
  "input_digest": "44912745eea939bc2757bd9bfc70316283348555ad0df1beb14be6072784d61e",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1892515
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "7f35d07a61aaac6db7cbb4fa0149f28989b2923680efc99d759910090d6e7ead", "compile_ns": 31812779, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "c27994186a8bf63d372ab6afa4f527c086fa53997bba60df3fb8f34eed8cbae8"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "b0db5ae48e7ed36dd1f73cc071f03643fe8e1893c948063e26d1c165b5dc7ddb"}}, "dynamic_argument_count": 4, "first_execution_ns": 660006, "input_digest": "c7df04c99ff031d50ccf46e47e9a0e5606b517362ac0128f17a6a302eaf4f361", "matrix": true, "mode": "jax_jit", "staging_ns": 1778378}`
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
| `case_id` | smoke/D03/MM/tail |
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
  "case_sha256": "7f35d07a61aaac6db7cbb4fa0149f28989b2923680efc99d759910090d6e7ead",
  "compile_ns": 31812779,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "c27994186a8bf63d372ab6afa4f527c086fa53997bba60df3fb8f34eed8cbae8"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b0db5ae48e7ed36dd1f73cc071f03643fe8e1893c948063e26d1c165b5dc7ddb"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 660006,
  "input_digest": "c7df04c99ff031d50ccf46e47e9a0e5606b517362ac0128f17a6a302eaf4f361",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1778378
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "a27fd5ac77e2257c940f87741bfa6a7570cf277d1cce0bfe8501bb091ee00da2", "compile_ns": 31603314, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "da53de4d0eb1bedb5bdcd734133752fa218f6efc21a639cc833cdb673bddeac7"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "df765dbd753f3270d4bc3a48a9ff948affe8fa11f4402a95aae28403b48fee7c"}}, "dynamic_argument_count": 4, "first_execution_ns": 635322, "input_digest": "484d9840f52ae97cfb8e5639e258fffccbf3cbe0ea5912aa12a1b62eb340ea1c", "matrix": true, "mode": "jax_jit", "staging_ns": 2113168}`
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
| `case_id` | smoke/D03/MM/scale |
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
  "case_sha256": "a27fd5ac77e2257c940f87741bfa6a7570cf277d1cce0bfe8501bb091ee00da2",
  "compile_ns": 31603314,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "da53de4d0eb1bedb5bdcd734133752fa218f6efc21a639cc833cdb673bddeac7"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "df765dbd753f3270d4bc3a48a9ff948affe8fa11f4402a95aae28403b48fee7c"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 635322,
  "input_digest": "484d9840f52ae97cfb8e5639e258fffccbf3cbe0ea5912aa12a1b62eb340ea1c",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 2113168
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "59bcc8f107b66af106dcfe9b1141a991b4a37b89503cb36c3e63b41e836da879", "compile_ns": 31982010, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0008/optimized_hlo.txt", "sha256": "9e3420c62d44439c1173b63a6f7c60a41c488c56e6bba83ecacc9a8ed5feebdf"}, "stablehlo": {"path": "compiler/case-0008/stablehlo.txt", "sha256": "566cde1f41b2fb3996a9ff3179fd8823c17370265f19677484315b80c242f3f1"}}, "dynamic_argument_count": 2, "first_execution_ns": 771480, "input_digest": "5bfedbcb6c8781b65fe883a118a7fd0f0c068382e2ab7012ddbce256b1e14c81", "matrix": false, "mode": "jax_jit", "staging_ns": 783808}`
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
| `case_id` | smoke/D03/RR/dev_small |
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
  "case_sha256": "59bcc8f107b66af106dcfe9b1141a991b4a37b89503cb36c3e63b41e836da879",
  "compile_ns": 31982010,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9e3420c62d44439c1173b63a6f7c60a41c488c56e6bba83ecacc9a8ed5feebdf"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "566cde1f41b2fb3996a9ff3179fd8823c17370265f19677484315b80c242f3f1"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 771480,
  "input_digest": "5bfedbcb6c8781b65fe883a118a7fd0f0c068382e2ab7012ddbce256b1e14c81",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 783808
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "628fddf68081345a4547dab51dfeff138291b0072d8ea9e35e7cc70909baa721", "compile_ns": 31267592, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "27bf938f3ec2b3395d42454030a96ac1002dcf236fbc1c93eed4ffdc9ab93c74"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "a31a075fe6343b3b66c605040f504a17cf49753e6e6f1aec0c5be67a76a5e306"}}, "dynamic_argument_count": 2, "first_execution_ns": 733345, "input_digest": "063c326852e5de727177823337b271f1e9b55b0715a7e9657896dbe7d9a619e2", "matrix": false, "mode": "jax_jit", "staging_ns": 724379}`
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
| `case_id` | smoke/D03/RR/dev_medium |
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
  "case_sha256": "628fddf68081345a4547dab51dfeff138291b0072d8ea9e35e7cc70909baa721",
  "compile_ns": 31267592,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "27bf938f3ec2b3395d42454030a96ac1002dcf236fbc1c93eed4ffdc9ab93c74"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a31a075fe6343b3b66c605040f504a17cf49753e6e6f1aec0c5be67a76a5e306"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 733345,
  "input_digest": "063c326852e5de727177823337b271f1e9b55b0715a7e9657896dbe7d9a619e2",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 724379
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "11ce7f6a4938a30ea9070f4303326c329ea4964f52a0fa3a658d4f41cbf5da65", "compile_ns": 32458228, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "385d95cdc7ee036a648e8570b0e71b0d5acb680b9a81107b81c5c28c71c69cd1"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "a6e670ac538d0a3acf232b03ca40fd5c2afba9d164efe722f6ff492fc3ab0a8d"}}, "dynamic_argument_count": 2, "first_execution_ns": 721930, "input_digest": "c2c26e3878bade557b637ab0cbd657f6675c6518cabfbd744d89bdb2b7de2d86", "matrix": false, "mode": "jax_jit", "staging_ns": 707140}`
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
| `case_id` | smoke/D03/RR/tail |
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
  "case_sha256": "11ce7f6a4938a30ea9070f4303326c329ea4964f52a0fa3a658d4f41cbf5da65",
  "compile_ns": 32458228,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "385d95cdc7ee036a648e8570b0e71b0d5acb680b9a81107b81c5c28c71c69cd1"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a6e670ac538d0a3acf232b03ca40fd5c2afba9d164efe722f6ff492fc3ab0a8d"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 721930,
  "input_digest": "c2c26e3878bade557b637ab0cbd657f6675c6518cabfbd744d89bdb2b7de2d86",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 707140
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "c38a276d05af348ada6e6d3f534032ab8af797c699562a38bcaa96ae52403517", "compile_ns": 31443789, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "65b840ad2e6278b4afb00641e9753f6a713e47037b82f7664a6964cbad8af413"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "84d999b2aaab79b4405dca96fae75a99dc159f158306cac8744791f256607372"}}, "dynamic_argument_count": 2, "first_execution_ns": 802804, "input_digest": "7e403e82d459bd7eab5093d4e8b0d0470286c2e93bd55bea7ea6e64f04c73b38", "matrix": false, "mode": "jax_jit", "staging_ns": 760797}`
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
| `case_id` | smoke/D03/RR/scale |
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
  "case_sha256": "c38a276d05af348ada6e6d3f534032ab8af797c699562a38bcaa96ae52403517",
  "compile_ns": 31443789,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "65b840ad2e6278b4afb00641e9753f6a713e47037b82f7664a6964cbad8af413"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "84d999b2aaab79b4405dca96fae75a99dc159f158306cac8744791f256607372"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 802804,
  "input_digest": "7e403e82d459bd7eab5093d4e8b0d0470286c2e93bd55bea7ea6e64f04c73b38",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 760797
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "b6ce3c693f2c7ac20b4e968161e23fed819d0fe8a53465e76f8103a623cf49e7", "compile_ns": 50004070, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0012/optimized_hlo.txt", "sha256": "3a89c3806b3c7926ad4dd8de8252f5778e954034937f6099acec05cd9a4b2060"}, "stablehlo": {"path": "compiler/case-0012/stablehlo.txt", "sha256": "9a09ea308507ae43f449a03eee517d993d1f8949777ddb83919cd9a8c7084f69"}}, "dynamic_argument_count": 3, "first_execution_ns": 800907, "input_digest": "b02b059ce0ea90a543dd9c9879cd4e343752bb1ee1ba447f3824cffd3dd4efc9", "matrix": false, "mode": "jax_jit", "staging_ns": 1153059}`
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
| `case_id` | smoke/D03/GP/dev_small |
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
  "case_sha256": "b6ce3c693f2c7ac20b4e968161e23fed819d0fe8a53465e76f8103a623cf49e7",
  "compile_ns": 50004070,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "3a89c3806b3c7926ad4dd8de8252f5778e954034937f6099acec05cd9a4b2060"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "9a09ea308507ae43f449a03eee517d993d1f8949777ddb83919cd9a8c7084f69"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 800907,
  "input_digest": "b02b059ce0ea90a543dd9c9879cd4e343752bb1ee1ba447f3824cffd3dd4efc9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1153059
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "b5fef03c5c42575325c3c12a4e7a8afba96b69abf4315722cb0f4ecfe70425fe", "compile_ns": 47607086, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0013/optimized_hlo.txt", "sha256": "458d2376d0c3c27656acb86f8283b70723d59695d2cb3472170bbefc862cc413"}, "stablehlo": {"path": "compiler/case-0013/stablehlo.txt", "sha256": "af7495e0d37f42e9ab40cb13f9a93242d29f86d96a69f4ac4016e1398ebdc484"}}, "dynamic_argument_count": 3, "first_execution_ns": 790406, "input_digest": "59cf9b9b53237ed9773513a0d4e382ca47500775874bb5ee1efb119f07aa0d37", "matrix": false, "mode": "jax_jit", "staging_ns": 935555}`
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
| `case_id` | smoke/D03/GP/dev_medium |
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
  "case_sha256": "b5fef03c5c42575325c3c12a4e7a8afba96b69abf4315722cb0f4ecfe70425fe",
  "compile_ns": 47607086,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "458d2376d0c3c27656acb86f8283b70723d59695d2cb3472170bbefc862cc413"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "af7495e0d37f42e9ab40cb13f9a93242d29f86d96a69f4ac4016e1398ebdc484"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 790406,
  "input_digest": "59cf9b9b53237ed9773513a0d4e382ca47500775874bb5ee1efb119f07aa0d37",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 935555
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "467f0a0d20741502685cfb62e5d9208aa31cf64e37f7558412598dd78b48c869", "compile_ns": 51983672, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0014/optimized_hlo.txt", "sha256": "23d86b586d349717a14fbec318e3528e163e18d6a54840e48e604587f8677869"}, "stablehlo": {"path": "compiler/case-0014/stablehlo.txt", "sha256": "afca148eddc9492642a85b6f02cd4e145e999e27844c24c3b723c2de20ffe0bd"}}, "dynamic_argument_count": 3, "first_execution_ns": 943927, "input_digest": "de2b7c784c5d6c193d2571867553e0de9db9fa4a3c8569d3b4cf251f7e0c3138", "matrix": false, "mode": "jax_jit", "staging_ns": 916233}`
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
| `case_id` | smoke/D03/GP/tail |
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
  "case_sha256": "467f0a0d20741502685cfb62e5d9208aa31cf64e37f7558412598dd78b48c869",
  "compile_ns": 51983672,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "23d86b586d349717a14fbec318e3528e163e18d6a54840e48e604587f8677869"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "afca148eddc9492642a85b6f02cd4e145e999e27844c24c3b723c2de20ffe0bd"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 943927,
  "input_digest": "de2b7c784c5d6c193d2571867553e0de9db9fa4a3c8569d3b4cf251f7e0c3138",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 916233
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "0a3865d34fd1312bf3b4413fc49729e0020c59ebcbb68777c09ad11de4bba0d6", "compile_ns": 47433238, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0015/optimized_hlo.txt", "sha256": "822b3868e1d42e079cc8c0b913983500995f68baa613e8149d2132106f92629a"}, "stablehlo": {"path": "compiler/case-0015/stablehlo.txt", "sha256": "60d7d24f567b9d1a75dfcbaafbf8e5f0b0b90673dade0ec1215663a4ac6f4e59"}}, "dynamic_argument_count": 3, "first_execution_ns": 945233, "input_digest": "183db183b715e93a663add5aef8fbf3998b7fc3e38d541064d2b403cb720c019", "matrix": false, "mode": "jax_jit", "staging_ns": 1001638}`
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
| `case_id` | smoke/D03/GP/scale |
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
  "case_sha256": "0a3865d34fd1312bf3b4413fc49729e0020c59ebcbb68777c09ad11de4bba0d6",
  "compile_ns": 47433238,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "822b3868e1d42e079cc8c0b913983500995f68baa613e8149d2132106f92629a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "60d7d24f567b9d1a75dfcbaafbf8e5f0b0b90673dade0ec1215663a4ac6f4e59"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 945233,
  "input_digest": "183db183b715e93a663add5aef8fbf3998b7fc3e38d541064d2b403cb720c019",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1001638
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "bafa12d86035469cf9a0fe8b3f66fda14ab4d61519efd67e7816d3a7d5042f1b", "compile_ns": 26839257, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0016/optimized_hlo.txt", "sha256": "41e87706c68019f3fb2397f300063df6264a45d32c5c468b2b3e71ec65c9784e"}, "stablehlo": {"path": "compiler/case-0016/stablehlo.txt", "sha256": "3598df66245938a7f56ab6efe45285739171083bb539b7fb66b00d2d6512b3fd"}}, "dynamic_argument_count": 2, "first_execution_ns": 814064, "input_digest": "d287406fe449e7ea4d7a9cc443a38f5375b0259d324b53d6c672516416424135", "matrix": false, "mode": "jax_jit", "staging_ns": 726451}`
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
| `case_id` | smoke/D03/TT/dev_small |
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
  "case_sha256": "bafa12d86035469cf9a0fe8b3f66fda14ab4d61519efd67e7816d3a7d5042f1b",
  "compile_ns": 26839257,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "41e87706c68019f3fb2397f300063df6264a45d32c5c468b2b3e71ec65c9784e"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3598df66245938a7f56ab6efe45285739171083bb539b7fb66b00d2d6512b3fd"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 814064,
  "input_digest": "d287406fe449e7ea4d7a9cc443a38f5375b0259d324b53d6c672516416424135",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 726451
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "69c35fa911a5006d3794efaac2906aa154a392730d3bb05c736f0958d6abf2cb", "compile_ns": 26531933, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0017/optimized_hlo.txt", "sha256": "8256c3e8fa651ba38facff47cdc5d6cfbbc2e7ceb20a404e446c88e7a5df8b57"}, "stablehlo": {"path": "compiler/case-0017/stablehlo.txt", "sha256": "65658644aa634a6b69032652e31e1abd6fc83d790816038b747c82f8154760e6"}}, "dynamic_argument_count": 2, "first_execution_ns": 657814, "input_digest": "bd3dfb38f46e16526051658194fd7dfe133e140d2015ddb8467b4e9aec80cfa9", "matrix": false, "mode": "jax_jit", "staging_ns": 721244}`
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
| `case_id` | smoke/D03/TT/dev_medium |
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
  "case_sha256": "69c35fa911a5006d3794efaac2906aa154a392730d3bb05c736f0958d6abf2cb",
  "compile_ns": 26531933,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8256c3e8fa651ba38facff47cdc5d6cfbbc2e7ceb20a404e446c88e7a5df8b57"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "65658644aa634a6b69032652e31e1abd6fc83d790816038b747c82f8154760e6"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 657814,
  "input_digest": "bd3dfb38f46e16526051658194fd7dfe133e140d2015ddb8467b4e9aec80cfa9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 721244
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "8aa6272dcab1de06d7ca2c903a349183133ffc09048235fdbf71d5cf3159ea84", "compile_ns": 26392027, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0018/optimized_hlo.txt", "sha256": "92463b172a10c0dd5d5f9901c60a94d7983938476140bf15f3b03326be7ab97a"}, "stablehlo": {"path": "compiler/case-0018/stablehlo.txt", "sha256": "0fc67bb989cba297c89b576858e0293dbac0b55bd1cd400e4f1412e1adc38b1d"}}, "dynamic_argument_count": 2, "first_execution_ns": 655456, "input_digest": "7a16858ac10ef7128f652e6f38ecee75aa0c6198c1dbd5775ef8c5b5e5af5445", "matrix": false, "mode": "jax_jit", "staging_ns": 701913}`
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
| `case_id` | smoke/D03/TT/tail |
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
  "case_sha256": "8aa6272dcab1de06d7ca2c903a349183133ffc09048235fdbf71d5cf3159ea84",
  "compile_ns": 26392027,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "92463b172a10c0dd5d5f9901c60a94d7983938476140bf15f3b03326be7ab97a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0fc67bb989cba297c89b576858e0293dbac0b55bd1cd400e4f1412e1adc38b1d"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 655456,
  "input_digest": "7a16858ac10ef7128f652e6f38ecee75aa0c6198c1dbd5775ef8c5b5e5af5445",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 701913
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "b242e601a70a2a6f6fc5810a0e30bfd6e4b6243ea25735648ff0ee7ea02aa0b3", "compile_ns": 26641623, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0019/optimized_hlo.txt", "sha256": "440ceed3738085fce1e575a4a12e04e8b641f369129f0e1b3887f4f0826df361"}, "stablehlo": {"path": "compiler/case-0019/stablehlo.txt", "sha256": "78c98b5e7241ad79757f394521fa8ec289caff1e7bc504a2b65b68a6af7f8f79"}}, "dynamic_argument_count": 2, "first_execution_ns": 636606, "input_digest": "d40b6983296551f260223f58f921a4edb75a021e4176cba40c59a5d85ea7ceed", "matrix": false, "mode": "jax_jit", "staging_ns": 735125}`
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
| `case_id` | smoke/D03/TT/scale |
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
  "case_sha256": "b242e601a70a2a6f6fc5810a0e30bfd6e4b6243ea25735648ff0ee7ea02aa0b3",
  "compile_ns": 26641623,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "440ceed3738085fce1e575a4a12e04e8b641f369129f0e1b3887f4f0826df361"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "78c98b5e7241ad79757f394521fa8ec289caff1e7bc504a2b65b68a6af7f8f79"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 636606,
  "input_digest": "d40b6983296551f260223f58f921a4edb75a021e4176cba40c59a5d85ea7ceed",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 735125
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "a545f6068365966cc52d1885a6bf5c33f264ac2a1ccf8d789fc7a6207cb8b288", "compile_ns": 35625862, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0020/optimized_hlo.txt", "sha256": "8308650892eda3c95a40a85bad531dfa1d04a47b4e0b4de35f1f27d2c39635b2"}, "stablehlo": {"path": "compiler/case-0020/stablehlo.txt", "sha256": "4c2f16b5068a88e3fcbc2674c317ebf46cd9069b9a2f23ae8e28304c0591d3f5"}}, "dynamic_argument_count": 3, "first_execution_ns": 633210, "input_digest": "843f303b3d418ecb1dc4ecbfa25f04449edcf290d882e0065dc9ccaa2c144f95", "matrix": true, "mode": "jax_jit", "staging_ns": 26419463}`
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
| `case_id` | smoke/D04/base/dev_small |
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
  "case_sha256": "a545f6068365966cc52d1885a6bf5c33f264ac2a1ccf8d789fc7a6207cb8b288",
  "compile_ns": 35625862,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8308650892eda3c95a40a85bad531dfa1d04a47b4e0b4de35f1f27d2c39635b2"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4c2f16b5068a88e3fcbc2674c317ebf46cd9069b9a2f23ae8e28304c0591d3f5"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 633210,
  "input_digest": "843f303b3d418ecb1dc4ecbfa25f04449edcf290d882e0065dc9ccaa2c144f95",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 26419463
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "842103549d5e6fd09a2d911c6d22b9007c67755f2e308a240082d9a28d619555", "compile_ns": 35601213, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0021/optimized_hlo.txt", "sha256": "029a60fd3cb68c2957d2a82c198863ee2785ad2ecfc051638fb981866ef140e1"}, "stablehlo": {"path": "compiler/case-0021/stablehlo.txt", "sha256": "6f115adebbf239c937121893d3b6ef4b20f0698135f063ee1d62c000a082bfc4"}}, "dynamic_argument_count": 3, "first_execution_ns": 650990, "input_digest": "95d7e829a7ccebda1314d28df51fa87ed96151934ffb6f0b3b82fe352c50cf51", "matrix": true, "mode": "jax_jit", "staging_ns": 26618544}`
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
      "max_abs_error": 2.2351741790771484e-08,
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
| `case_id` | smoke/D04/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.2351741790771484e-08,
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
  "case_sha256": "842103549d5e6fd09a2d911c6d22b9007c67755f2e308a240082d9a28d619555",
  "compile_ns": 35601213,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "029a60fd3cb68c2957d2a82c198863ee2785ad2ecfc051638fb981866ef140e1"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6f115adebbf239c937121893d3b6ef4b20f0698135f063ee1d62c000a082bfc4"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 650990,
  "input_digest": "95d7e829a7ccebda1314d28df51fa87ed96151934ffb6f0b3b82fe352c50cf51",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 26618544
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "077b71c38425177ec2681b9dd66e098fef6cdbb8665742afe8991f6174feeb3c", "compile_ns": 35509731, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0022/optimized_hlo.txt", "sha256": "c16e9af6e6507685635d3e8c2f4b026b30d05d2f1ac9f185ecef09b0f4c30530"}, "stablehlo": {"path": "compiler/case-0022/stablehlo.txt", "sha256": "1635fdaad4d572d4ec01c59cf394f71d028f2f8be82edcf864523b8a63b1f7f9"}}, "dynamic_argument_count": 3, "first_execution_ns": 643873, "input_digest": "ea0331a90a4e6e6328c6639cda396ebbdf6d7a92513860ad301057c63a760ee8", "matrix": true, "mode": "jax_jit", "staging_ns": 53690748}`
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
| `case_id` | smoke/D04/base/tail |
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
  "case_sha256": "077b71c38425177ec2681b9dd66e098fef6cdbb8665742afe8991f6174feeb3c",
  "compile_ns": 35509731,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "c16e9af6e6507685635d3e8c2f4b026b30d05d2f1ac9f185ecef09b0f4c30530"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1635fdaad4d572d4ec01c59cf394f71d028f2f8be82edcf864523b8a63b1f7f9"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 643873,
  "input_digest": "ea0331a90a4e6e6328c6639cda396ebbdf6d7a92513860ad301057c63a760ee8",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 53690748
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "7bab5cc43f6422cf14d916ebbbb99c6fe2681e5da660a8e7e09250e730958e53", "compile_ns": 36063385, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0023/optimized_hlo.txt", "sha256": "adac1b2751619eb4c14697db9e3da536c5af32c2a3291ed4091a23f22a18e245"}, "stablehlo": {"path": "compiler/case-0023/stablehlo.txt", "sha256": "cb9a2b2dd19cd9035f328bacc3f5670ba478fdf804033254fac57ab6839bf720"}}, "dynamic_argument_count": 3, "first_execution_ns": 617950, "input_digest": "39030aa61aa99ddddf249e07c39bab1022e1ec959dc3eb148c3d89f6ee4abc1c", "matrix": true, "mode": "jax_jit", "staging_ns": 52070057}`
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
| `case_id` | smoke/D04/base/scale |
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
  "case_sha256": "7bab5cc43f6422cf14d916ebbbb99c6fe2681e5da660a8e7e09250e730958e53",
  "compile_ns": 36063385,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "adac1b2751619eb4c14697db9e3da536c5af32c2a3291ed4091a23f22a18e245"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "cb9a2b2dd19cd9035f328bacc3f5670ba478fdf804033254fac57ab6839bf720"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 617950,
  "input_digest": "39030aa61aa99ddddf249e07c39bab1022e1ec959dc3eb148c3d89f6ee4abc1c",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 52070057
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "6e6b1d1d2e9a9a14fc53f76c664924a20aee017300b651221962886ac564de7a", "compile_ns": 41708063, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0024/optimized_hlo.txt", "sha256": "6dae17bbe1ab6baa8d1cb770a98bb65cb59b944db12eca08da7e553a7a61dbbb"}, "stablehlo": {"path": "compiler/case-0024/stablehlo.txt", "sha256": "1ac6ab63d579721719c1c1336f8ece58bee5ed04a89a58f5c9a891194d8127a8"}}, "dynamic_argument_count": 4, "first_execution_ns": 718026, "input_digest": "ce08afb2e468c7d32f4881104140f0e391cc037e71e27bbe91d11f37e7beadc8", "matrix": true, "mode": "jax_jit", "staging_ns": 1717379}`
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
| `case_id` | smoke/N21/base/dev_small |
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
  "case_sha256": "6e6b1d1d2e9a9a14fc53f76c664924a20aee017300b651221962886ac564de7a",
  "compile_ns": 41708063,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6dae17bbe1ab6baa8d1cb770a98bb65cb59b944db12eca08da7e553a7a61dbbb"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1ac6ab63d579721719c1c1336f8ece58bee5ed04a89a58f5c9a891194d8127a8"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 718026,
  "input_digest": "ce08afb2e468c7d32f4881104140f0e391cc037e71e27bbe91d11f37e7beadc8",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1717379
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "48e351496ed703d214dd416847416780a0a9feafd54bb6e4243cb179e0958b74", "compile_ns": 42999064, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0025/optimized_hlo.txt", "sha256": "296d4850f548e941857f2233587af4c963c0e8501643737b5ce2cb17ecdab08d"}, "stablehlo": {"path": "compiler/case-0025/stablehlo.txt", "sha256": "3020c35bdd259e66c34d13c0f0bae18652c4b0e9453105523a188365e14e16ef"}}, "dynamic_argument_count": 4, "first_execution_ns": 672912, "input_digest": "9bfa8231a3960cbc6c2b4ed114414542b58aab6967c33c99801aa0db8180b3ce", "matrix": true, "mode": "jax_jit", "staging_ns": 1728378}`
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
| `case_id` | smoke/N21/base/dev_medium |
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
  "case_sha256": "48e351496ed703d214dd416847416780a0a9feafd54bb6e4243cb179e0958b74",
  "compile_ns": 42999064,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "296d4850f548e941857f2233587af4c963c0e8501643737b5ce2cb17ecdab08d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3020c35bdd259e66c34d13c0f0bae18652c4b0e9453105523a188365e14e16ef"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 672912,
  "input_digest": "9bfa8231a3960cbc6c2b4ed114414542b58aab6967c33c99801aa0db8180b3ce",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1728378
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "71fbb7ba6a757b6b92dd3d265e372c4b4ee1b0216850e4edef312fd18a9bbef7", "compile_ns": 42178271, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0026/optimized_hlo.txt", "sha256": "34175a2ad247be12f7ae7965d8abadc409031527a426ee621f45232c5358f9f8"}, "stablehlo": {"path": "compiler/case-0026/stablehlo.txt", "sha256": "e6ebd2d7a17c0be6f19b4562782c5ff249b1aa98fc0e6fef0f7bd5d5fb7102b5"}}, "dynamic_argument_count": 4, "first_execution_ns": 680466, "input_digest": "31b476b6c15d6a234b366ffc73eb4b71e59aa0edc4ea1347105fb006b6032fb4", "matrix": true, "mode": "jax_jit", "staging_ns": 76453902}`
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
| `case_id` | smoke/N21/base/tail |
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
  "case_sha256": "71fbb7ba6a757b6b92dd3d265e372c4b4ee1b0216850e4edef312fd18a9bbef7",
  "compile_ns": 42178271,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "34175a2ad247be12f7ae7965d8abadc409031527a426ee621f45232c5358f9f8"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "e6ebd2d7a17c0be6f19b4562782c5ff249b1aa98fc0e6fef0f7bd5d5fb7102b5"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 680466,
  "input_digest": "31b476b6c15d6a234b366ffc73eb4b71e59aa0edc4ea1347105fb006b6032fb4",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 76453902
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "02b878f83b77f92470d7f6b0ae22a26a5d341a2ea72d6ebcca72f4859bcc2537", "compile_ns": 42846606, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0027/optimized_hlo.txt", "sha256": "5db41b2f3fca77ebf54bc47e828d975deede52b2480c923812a91fb954fd5595"}, "stablehlo": {"path": "compiler/case-0027/stablehlo.txt", "sha256": "0eb4d1723a320a14740ab00dd09ca1f960d9d960ce5551beab1857536b919cb5"}}, "dynamic_argument_count": 4, "first_execution_ns": 625421, "input_digest": "2543691716d80c2903c50dfc9596bc329d51584dee88e078aeebb21f851aa3a8", "matrix": true, "mode": "jax_jit", "staging_ns": 53826425}`
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
| `case_id` | smoke/N21/base/scale |
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
  "case_sha256": "02b878f83b77f92470d7f6b0ae22a26a5d341a2ea72d6ebcca72f4859bcc2537",
  "compile_ns": 42846606,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5db41b2f3fca77ebf54bc47e828d975deede52b2480c923812a91fb954fd5595"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0eb4d1723a320a14740ab00dd09ca1f960d9d960ce5551beab1857536b919cb5"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 625421,
  "input_digest": "2543691716d80c2903c50dfc9596bc329d51584dee88e078aeebb21f851aa3a8",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 53826425
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "e9345e2471a2042af57c7ba833ddb871a1aedd822ca55c1f887c34703a856079", "compile_ns": 37438366, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0028/optimized_hlo.txt", "sha256": "d6b11da58f37bfc8e8c1efee2097992a16156506700bb036757e734f13d6c8c4"}, "stablehlo": {"path": "compiler/case-0028/stablehlo.txt", "sha256": "fca75d6ebe8f8b4adbe6d6fc3b6c9df0bc528f97d4d01c326b4956b7484a25dc"}}, "dynamic_argument_count": 4, "first_execution_ns": 660852, "input_digest": "f72da849948324f53a309ce233d804c258174391d8c6e6650e706356a06694b3", "matrix": true, "mode": "jax_jit", "staging_ns": 1895708}`
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
| `case_id` | smoke/N22/base/dev_small |
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
  "case_sha256": "e9345e2471a2042af57c7ba833ddb871a1aedd822ca55c1f887c34703a856079",
  "compile_ns": 37438366,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "d6b11da58f37bfc8e8c1efee2097992a16156506700bb036757e734f13d6c8c4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "fca75d6ebe8f8b4adbe6d6fc3b6c9df0bc528f97d4d01c326b4956b7484a25dc"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 660852,
  "input_digest": "f72da849948324f53a309ce233d804c258174391d8c6e6650e706356a06694b3",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1895708
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "367b6b0f8a42ad7cf31fd1fa33c1b855df0e49bcea7ed8beb866a47a4246bfdf", "compile_ns": 37528388, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0029/optimized_hlo.txt", "sha256": "dd47309f63f057c9ab5c5d7032fb0ef4b37d6254f64b7978117cda64588ee6f4"}, "stablehlo": {"path": "compiler/case-0029/stablehlo.txt", "sha256": "f5cf2ca2ebe82c7eacea286d70c2811b3bafed00c265c9fac00bb9c4b9206aed"}}, "dynamic_argument_count": 4, "first_execution_ns": 637827, "input_digest": "ed4ba17c33a6269a342bbd69d8f151117dcc358b44f897542c4868da190cedc0", "matrix": true, "mode": "jax_jit", "staging_ns": 2075246}`
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
      "max_abs_error": 7.450580596923828e-09,
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
| `case_id` | smoke/N22/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
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
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "367b6b0f8a42ad7cf31fd1fa33c1b855df0e49bcea7ed8beb866a47a4246bfdf",
  "compile_ns": 37528388,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "dd47309f63f057c9ab5c5d7032fb0ef4b37d6254f64b7978117cda64588ee6f4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "f5cf2ca2ebe82c7eacea286d70c2811b3bafed00c265c9fac00bb9c4b9206aed"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 637827,
  "input_digest": "ed4ba17c33a6269a342bbd69d8f151117dcc358b44f897542c4868da190cedc0",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 2075246
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "0d1e05bcf354e834016ed9511ad867401a2e6a84e9516d9389d7074ba3080106", "compile_ns": 38740073, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0030/optimized_hlo.txt", "sha256": "a0f4dac4bd72d308bac5d09cac79c7acedf65d2723c0d10f26f23db7696c8bd9"}, "stablehlo": {"path": "compiler/case-0030/stablehlo.txt", "sha256": "4949cc23903fb817a76ecbd27c47cf46e65b30a58d9f7e8af7a71c950c9c79f4"}}, "dynamic_argument_count": 4, "first_execution_ns": 610353, "input_digest": "a4a70e0489476918b2dcd33e4767ebbfe151ba257c9f7daf8f3590b8ed49c895", "matrix": true, "mode": "jax_jit", "staging_ns": 1851443}`
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
| `case_id` | smoke/N22/base/tail |
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
  "case_sha256": "0d1e05bcf354e834016ed9511ad867401a2e6a84e9516d9389d7074ba3080106",
  "compile_ns": 38740073,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a0f4dac4bd72d308bac5d09cac79c7acedf65d2723c0d10f26f23db7696c8bd9"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4949cc23903fb817a76ecbd27c47cf46e65b30a58d9f7e8af7a71c950c9c79f4"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 610353,
  "input_digest": "a4a70e0489476918b2dcd33e4767ebbfe151ba257c9f7daf8f3590b8ed49c895",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1851443
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
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

- launch.extras: `{"case_sha256": "91fbaa659a4bd298824aa58bd55d46f147f276ed02a92aa9abcaa25f1a8616ec", "compile_ns": 37801651, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0031/optimized_hlo.txt", "sha256": "1f8769c3376d054feb67d120a77b66d655e20ee7576cc9d8b954299c39c252ea"}, "stablehlo": {"path": "compiler/case-0031/stablehlo.txt", "sha256": "c6acc92fcba91f997515ac1b612e9fc4a5f58adddc7ee23764bd26003ac85c91"}}, "dynamic_argument_count": 4, "first_execution_ns": 622553, "input_digest": "1afae3d5f19c942bfbd2731e4b5231753c46ae9c9b2c188b263058d890ee7470", "matrix": true, "mode": "jax_jit", "staging_ns": 2816676}`
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
| `case_id` | smoke/N22/base/scale |
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
  "case_sha256": "91fbaa659a4bd298824aa58bd55d46f147f276ed02a92aa9abcaa25f1a8616ec",
  "compile_ns": 37801651,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "1f8769c3376d054feb67d120a77b66d655e20ee7576cc9d8b954299c39c252ea"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c6acc92fcba91f997515ac1b612e9fc4a5f58adddc7ee23764bd26003ac85c91"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 622553,
  "input_digest": "1afae3d5f19c942bfbd2731e4b5231753c46ae9c9b2c188b263058d890ee7470",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 2816676
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 128,
  "stop": 160,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `0076be988e0bdf83d098ef25bcf9ce5fa11a76414ac591d5d939930646bc1b3a`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-002/out/tpu/portable-core-qualification-v1/e5bce37d0a6d7c13a8da32dc39e97617bccb6216c3b78d598ef85f7862a3e63f/native-restore-002-core28-004/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
