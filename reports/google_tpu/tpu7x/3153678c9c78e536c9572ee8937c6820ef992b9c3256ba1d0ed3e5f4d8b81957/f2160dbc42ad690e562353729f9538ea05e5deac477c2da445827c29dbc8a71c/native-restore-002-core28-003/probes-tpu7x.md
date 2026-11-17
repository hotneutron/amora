# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-restore-002-core28-003
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

- launch.extras: `{"case_sha256": "a9b2488a2bbf927859ed68eaaadbec3d55a9b02ec2fffaec41e80eff0e2d0eb3", "compile_ns": 343547725, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "34567046e9fb84853d2c1cf3d10819130eda3a809e132a05b1c1a0119fb277cc"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "16f2d4f3f2846c81a62e096f060a8260d615dfe0e10381db2d9e4c1aa3e3ea33"}}, "dynamic_argument_count": 3, "first_execution_ns": 1059131, "input_digest": "c86dc086596a5c13036b110837983672aab04192340aa29fe73910028480a929", "matrix": false, "mode": "jax_jit", "staging_ns": 6520485}`
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
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        4,
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
| `case_id` | smoke/C04/base/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.9802322387695312e-08,
      "passed": true,
      "shape": [
        4,
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
  "case_sha256": "a9b2488a2bbf927859ed68eaaadbec3d55a9b02ec2fffaec41e80eff0e2d0eb3",
  "compile_ns": 343547725,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "34567046e9fb84853d2c1cf3d10819130eda3a809e132a05b1c1a0119fb277cc"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "16f2d4f3f2846c81a62e096f060a8260d615dfe0e10381db2d9e4c1aa3e3ea33"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 1059131,
  "input_digest": "c86dc086596a5c13036b110837983672aab04192340aa29fe73910028480a929",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 6520485
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "541dcad7a110389ae4feada6b99f29798b69ee53715cbca0e337617f10908e7b", "compile_ns": 68248349, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "e3326b98ed0d8ebac88c0521a3fc3536bcf62fad9ece5e70a81be7e0d51fb856"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "4741857b296efa269523620a5ad5f372aa2f95fb334dd78df69900a73f404e31"}}, "dynamic_argument_count": 3, "first_execution_ns": 972429, "input_digest": "842ad0f31306830d9502b81e496027a5b5240569ed72c3bc768064653018b944", "matrix": false, "mode": "jax_jit", "staging_ns": 1139218}`
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
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
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
| `case_id` | smoke/C04/base/dev_medium |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        8,
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
  "case_sha256": "541dcad7a110389ae4feada6b99f29798b69ee53715cbca0e337617f10908e7b",
  "compile_ns": 68248349,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e3326b98ed0d8ebac88c0521a3fc3536bcf62fad9ece5e70a81be7e0d51fb856"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4741857b296efa269523620a5ad5f372aa2f95fb334dd78df69900a73f404e31"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 972429,
  "input_digest": "842ad0f31306830d9502b81e496027a5b5240569ed72c3bc768064653018b944",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1139218
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "0a5686997105cdfceaecc1a309b7a038cee99c77ad58a87d2a619dca68a69cf9", "compile_ns": 57078026, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "6c6af482992e09ae332afb65fb2aa5ec9e4a74325706bac2608da1fd7eefed0a"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "5cec50d29946870a5c3f491747c26dae035ec6e5b1b7036cc606f76ba2e517c2"}}, "dynamic_argument_count": 3, "first_execution_ns": 879588, "input_digest": "87060d2a4951f740fb760aac05a75741b6a362cc654165f938cee2204fb9e28f", "matrix": false, "mode": "jax_jit", "staging_ns": 982140}`
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
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        3,
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
| `case_id` | smoke/C04/base/tail |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        3,
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
  "case_sha256": "0a5686997105cdfceaecc1a309b7a038cee99c77ad58a87d2a619dca68a69cf9",
  "compile_ns": 57078026,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6c6af482992e09ae332afb65fb2aa5ec9e4a74325706bac2608da1fd7eefed0a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "5cec50d29946870a5c3f491747c26dae035ec6e5b1b7036cc606f76ba2e517c2"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 879588,
  "input_digest": "87060d2a4951f740fb760aac05a75741b6a362cc654165f938cee2204fb9e28f",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 982140
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "88861e97e8bbe1179c2ef579de70dae525d1258676279b200ed5e1631add1ac6", "compile_ns": 81108281, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "a8d3096d8eba47d77e24577d00985120ec8b5e34b93901f28dfc42e971c7963c"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "c062c62af55925772b10caa24120cfe39f4350a9042519010f8743f6ea44333f"}}, "dynamic_argument_count": 3, "first_execution_ns": 1136025, "input_digest": "2b57741164ae7b5221120de9ecabb12ac29b2f65180ae7a95d4621f2996c8365", "matrix": false, "mode": "jax_jit", "staging_ns": 1128907}`
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
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        16,
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
| `case_id` | smoke/C04/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 5.960464477539063e-08,
      "passed": true,
      "shape": [
        16,
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
  "case_sha256": "88861e97e8bbe1179c2ef579de70dae525d1258676279b200ed5e1631add1ac6",
  "compile_ns": 81108281,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a8d3096d8eba47d77e24577d00985120ec8b5e34b93901f28dfc42e971c7963c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c062c62af55925772b10caa24120cfe39f4350a9042519010f8743f6ea44333f"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 1136025,
  "input_digest": "2b57741164ae7b5221120de9ecabb12ac29b2f65180ae7a95d4621f2996c8365",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1128907
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "9e43f1da436b30381ed2dea1e97d34a950719cba1f1128bea3afff5f8149cc55", "compile_ns": 35987461, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "955a9f0e5d783ff94cee03982cf75ed516ad70eeddd5c5ea7e466c28a078b30c"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "c82991000816433de56df979f98c6fdb9db95f60408640821eb7e67e4624d992"}}, "dynamic_argument_count": 3, "first_execution_ns": 687835, "input_digest": "d4e5fa0af757cd96a1371b144d54e51e3e68d10e43a9fa99bd3951b05fb0d4a9", "matrix": true, "mode": "jax_jit", "staging_ns": 53689788}`
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
| `case_id` | smoke/C05/base/dev_small |
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
  "case_sha256": "9e43f1da436b30381ed2dea1e97d34a950719cba1f1128bea3afff5f8149cc55",
  "compile_ns": 35987461,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "955a9f0e5d783ff94cee03982cf75ed516ad70eeddd5c5ea7e466c28a078b30c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c82991000816433de56df979f98c6fdb9db95f60408640821eb7e67e4624d992"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 687835,
  "input_digest": "d4e5fa0af757cd96a1371b144d54e51e3e68d10e43a9fa99bd3951b05fb0d4a9",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 53689788
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "41d54e5f840b00e8f566f6f3bc9aaa2baef756241a8222093558835aacbeacd1", "compile_ns": 36466830, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "45625f0af85fc3388d1ba995c6d74b6cb886a0e425957a76df1aaab718f44581"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "b2f2f2d346f2e0ccb0615dae47a68462facfb4af60a08038abc4ffc27f12a912"}}, "dynamic_argument_count": 3, "first_execution_ns": 610603, "input_digest": "28c7fd5c31cf195a6096cbbe922b68c0af4a5fef3963d13fb55c4ac3668c892e", "matrix": true, "mode": "jax_jit", "staging_ns": 51667213}`
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
| `case_id` | smoke/C05/base/dev_medium |
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
  "case_sha256": "41d54e5f840b00e8f566f6f3bc9aaa2baef756241a8222093558835aacbeacd1",
  "compile_ns": 36466830,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "45625f0af85fc3388d1ba995c6d74b6cb886a0e425957a76df1aaab718f44581"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b2f2f2d346f2e0ccb0615dae47a68462facfb4af60a08038abc4ffc27f12a912"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 610603,
  "input_digest": "28c7fd5c31cf195a6096cbbe922b68c0af4a5fef3963d13fb55c4ac3668c892e",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51667213
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "266ce61130159577901d47e7bb7f6d4dc54e12d2affe5cd8b2a62c8b36017dff", "compile_ns": 34866444, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "77f8c91e95e760fc5ce227aebdb197585af22462c7ea9526d5f05445c0939b1f"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "142f996165ec94c1442df8850b45dbd03790dd71355757eb3c55fd70b133421d"}}, "dynamic_argument_count": 3, "first_execution_ns": 583396, "input_digest": "17411d2918653695e99946086b5de46e5432f199683e06e89ccaf3c77deb7ae8", "matrix": true, "mode": "jax_jit", "staging_ns": 55109890}`
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
| `case_id` | smoke/C05/base/tail |
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
  "case_sha256": "266ce61130159577901d47e7bb7f6d4dc54e12d2affe5cd8b2a62c8b36017dff",
  "compile_ns": 34866444,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "77f8c91e95e760fc5ce227aebdb197585af22462c7ea9526d5f05445c0939b1f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "142f996165ec94c1442df8850b45dbd03790dd71355757eb3c55fd70b133421d"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 583396,
  "input_digest": "17411d2918653695e99946086b5de46e5432f199683e06e89ccaf3c77deb7ae8",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 55109890
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "883357fa7c7c3e597eb20bb2c2ccb92e00f92f9003e04c7055fceb31e8c78b72", "compile_ns": 36110918, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "cdafd639d8da409777e48af7d59f00c2e5e4825a3597667799f7404dd3956cdd"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "0af75aaa6a679c949f7e04a387545c95ef6da0caee8096a36e8ebbb7d3aacdb5"}}, "dynamic_argument_count": 3, "first_execution_ns": 569499, "input_digest": "e922687d4c65b4d971844cd1daca02644c3c7a47d68d7918e6de9d935816b0f6", "matrix": true, "mode": "jax_jit", "staging_ns": 51532880}`
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
      "max_abs_error": 2.384185791015625e-07,
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
| `case_id` | smoke/C05/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
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
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "883357fa7c7c3e597eb20bb2c2ccb92e00f92f9003e04c7055fceb31e8c78b72",
  "compile_ns": 36110918,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "cdafd639d8da409777e48af7d59f00c2e5e4825a3597667799f7404dd3956cdd"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0af75aaa6a679c949f7e04a387545c95ef6da0caee8096a36e8ebbb7d3aacdb5"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 569499,
  "input_digest": "e922687d4c65b4d971844cd1daca02644c3c7a47d68d7918e6de9d935816b0f6",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 51532880
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "b2217a6526f0a5cfad36ade7b43abba347fc7985abe1355acb37004094b637a0", "compile_ns": 30034732, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0008/optimized_hlo.txt", "sha256": "acbe825aaced39671d26f5e59941aed1901060a7a162fa53deef3fa514ebc7b7"}, "stablehlo": {"path": "compiler/case-0008/stablehlo.txt", "sha256": "fea796cfe48422f3a5f1377aec239befbab3f086c6874f031465273fc7a60358"}}, "dynamic_argument_count": 2, "first_execution_ns": 569394, "input_digest": "f057870959cf30f54da59ccf5f0e2d29b73c62df681dc86fe79baf47b8978d98", "matrix": true, "mode": "jax_jit", "staging_ns": 1063638}`
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
| `case_id` | smoke/C06/base/dev_small |
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
  "case_sha256": "b2217a6526f0a5cfad36ade7b43abba347fc7985abe1355acb37004094b637a0",
  "compile_ns": 30034732,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "acbe825aaced39671d26f5e59941aed1901060a7a162fa53deef3fa514ebc7b7"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "fea796cfe48422f3a5f1377aec239befbab3f086c6874f031465273fc7a60358"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 569394,
  "input_digest": "f057870959cf30f54da59ccf5f0e2d29b73c62df681dc86fe79baf47b8978d98",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1063638
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "ecc60c9c3ebb950bfa27d34a4bb82315817546bbeb97117f9678b130d719c31b", "compile_ns": 30361006, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "99beca1e9d88ff07dfb214c5a90652d2ff13bdc9e76562b18ce53f244f337932"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "9356476317fcb7e059e57e801ceb578bf07f6330adc6db5bc3b76fe7acd897df"}}, "dynamic_argument_count": 2, "first_execution_ns": 559777, "input_digest": "aa33d21022a0b2c55ab098216d350f8b03abdf038611c42f83612656b6b72316", "matrix": true, "mode": "jax_jit", "staging_ns": 1078834}`
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
| `case_id` | smoke/C06/base/dev_medium |
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
  "case_sha256": "ecc60c9c3ebb950bfa27d34a4bb82315817546bbeb97117f9678b130d719c31b",
  "compile_ns": 30361006,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "99beca1e9d88ff07dfb214c5a90652d2ff13bdc9e76562b18ce53f244f337932"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "9356476317fcb7e059e57e801ceb578bf07f6330adc6db5bc3b76fe7acd897df"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 559777,
  "input_digest": "aa33d21022a0b2c55ab098216d350f8b03abdf038611c42f83612656b6b72316",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1078834
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "0dc28accccd851772e1c07072c1697a94028fb5a66c56ac2351baad821d54fcb", "compile_ns": 30457922, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "e0713e16de35a053bdac1ad69610fa61ac59d81ccc9246e6b665799d814e4514"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "dd480d8d082572c26dd9248603679435b20bad8fc8a11383cb05828b7c4261f3"}}, "dynamic_argument_count": 2, "first_execution_ns": 553750, "input_digest": "9f91df88a338aa9eebc86e71be5f3681b495eedbcaf0222c743253409ebec5e1", "matrix": true, "mode": "jax_jit", "staging_ns": 1050555}`
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
| `case_id` | smoke/C06/base/tail |
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
  "case_sha256": "0dc28accccd851772e1c07072c1697a94028fb5a66c56ac2351baad821d54fcb",
  "compile_ns": 30457922,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e0713e16de35a053bdac1ad69610fa61ac59d81ccc9246e6b665799d814e4514"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "dd480d8d082572c26dd9248603679435b20bad8fc8a11383cb05828b7c4261f3"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 553750,
  "input_digest": "9f91df88a338aa9eebc86e71be5f3681b495eedbcaf0222c743253409ebec5e1",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1050555
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "1e919101e939d8d5fd6c66ed717924ea692ac7b90ed3b044da529e0a0d69280e", "compile_ns": 30906200, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "89b1458c0f506565e506ae0c7b640cbef5f53d6968a71c7bbe5ca94c79da6bc4"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "020faa6eddcb4487f6ae36b1472610a76c6567a7e253d9dce223564609b876cc"}}, "dynamic_argument_count": 2, "first_execution_ns": 554840, "input_digest": "a3e967f5a970483dcb4d0304e027f43aa49d28947e7e4f05f7b1b117a2b8dbe4", "matrix": true, "mode": "jax_jit", "staging_ns": 1258878}`
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
| `case_id` | smoke/C06/base/scale |
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
  "case_sha256": "1e919101e939d8d5fd6c66ed717924ea692ac7b90ed3b044da529e0a0d69280e",
  "compile_ns": 30906200,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "89b1458c0f506565e506ae0c7b640cbef5f53d6968a71c7bbe5ca94c79da6bc4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "020faa6eddcb4487f6ae36b1472610a76c6567a7e253d9dce223564609b876cc"
    }
  },
  "dynamic_argument_count": 2,
  "first_execution_ns": 554840,
  "input_digest": "a3e967f5a970483dcb4d0304e027f43aa49d28947e7e4f05f7b1b117a2b8dbe4",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1258878
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "6e8e70d2f38ac7c636f2ee86e6c0e543f6448356305b9bd443a238343b7c0677", "compile_ns": 41703671, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0012/optimized_hlo.txt", "sha256": "ce06b0169df52ecfbe7642892d5024fbfb81606b9614b41fde6757ef33083194"}, "stablehlo": {"path": "compiler/case-0012/stablehlo.txt", "sha256": "4ebb9be3164526844703a055c4be820b44470ec5b5c9301b842f2a1adf0e74a1"}}, "dynamic_argument_count": 5, "first_execution_ns": 646590, "input_digest": "383104bb4379d38dd8f72f8da4c107a3cc520fa9ba669540a8e937cae9647c93", "matrix": true, "mode": "jax_jit", "staging_ns": 1510495}`
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
| `case_id` | smoke/C07/base/dev_small |
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
  "case_sha256": "6e8e70d2f38ac7c636f2ee86e6c0e543f6448356305b9bd443a238343b7c0677",
  "compile_ns": 41703671,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ce06b0169df52ecfbe7642892d5024fbfb81606b9614b41fde6757ef33083194"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4ebb9be3164526844703a055c4be820b44470ec5b5c9301b842f2a1adf0e74a1"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 646590,
  "input_digest": "383104bb4379d38dd8f72f8da4c107a3cc520fa9ba669540a8e937cae9647c93",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1510495
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "410d84bf0ea0fcbff10abdb7152c48c53a01f7a9d3a8f4d5c2d1c85d6ffbf8ff", "compile_ns": 41578914, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0013/optimized_hlo.txt", "sha256": "2f9e49cbdc0ebe2200322ed7f1a30771886f6f4b9caeae30d8e12bd5a41b401a"}, "stablehlo": {"path": "compiler/case-0013/stablehlo.txt", "sha256": "a381c6a259495cd8180e9ad3b69c9081ad89a604a782b11931bd2a07f4ac19ba"}}, "dynamic_argument_count": 5, "first_execution_ns": 663062, "input_digest": "c7200d4f156d9a97cdeda94940372ebf27ae8f8e868e1087dd9694808ab1c1b8", "matrix": true, "mode": "jax_jit", "staging_ns": 1517365}`
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
| `case_id` | smoke/C07/base/dev_medium |
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
  "case_sha256": "410d84bf0ea0fcbff10abdb7152c48c53a01f7a9d3a8f4d5c2d1c85d6ffbf8ff",
  "compile_ns": 41578914,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "2f9e49cbdc0ebe2200322ed7f1a30771886f6f4b9caeae30d8e12bd5a41b401a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a381c6a259495cd8180e9ad3b69c9081ad89a604a782b11931bd2a07f4ac19ba"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 663062,
  "input_digest": "c7200d4f156d9a97cdeda94940372ebf27ae8f8e868e1087dd9694808ab1c1b8",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1517365
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "61d22ef5c7e94d7341c6f74e4d1ff7766bc3849e2b1d85000bb6f3a11af7f42b", "compile_ns": 44832268, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0014/optimized_hlo.txt", "sha256": "cab8d25ad90ef2f57f7f45854e27183d13cd41d469bfaa201fe529e41220b22f"}, "stablehlo": {"path": "compiler/case-0014/stablehlo.txt", "sha256": "c929494acaab615fc5ab87feab44956de91c0094b5d96c1972cb53ec228cc35d"}}, "dynamic_argument_count": 5, "first_execution_ns": 656985, "input_digest": "6031d95d1783e4a63433f825113d71817e736d7eb68e61db3d52827fd3eb85b3", "matrix": true, "mode": "jax_jit", "staging_ns": 1411456}`
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
| `case_id` | smoke/C07/base/tail |
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
  "case_sha256": "61d22ef5c7e94d7341c6f74e4d1ff7766bc3849e2b1d85000bb6f3a11af7f42b",
  "compile_ns": 44832268,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "cab8d25ad90ef2f57f7f45854e27183d13cd41d469bfaa201fe529e41220b22f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c929494acaab615fc5ab87feab44956de91c0094b5d96c1972cb53ec228cc35d"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 656985,
  "input_digest": "6031d95d1783e4a63433f825113d71817e736d7eb68e61db3d52827fd3eb85b3",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1411456
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "17ab11c9d5f01e2101a9a5da69dbf603f23be5feb48ab6b9e7dddc7852339c3c", "compile_ns": 42692417, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0015/optimized_hlo.txt", "sha256": "7d96204b160d817f5c98e4de954fcbd2388f0727b99491ebdb0a2259348f378b"}, "stablehlo": {"path": "compiler/case-0015/stablehlo.txt", "sha256": "82958f90058c533c73273a88637a98e63193360c4c6930626e016f2976465f80"}}, "dynamic_argument_count": 5, "first_execution_ns": 748700, "input_digest": "122fb182151d04c2aafcea32cd31da0b5cf40cdfcbb569abf2888c216550d9a1", "matrix": true, "mode": "jax_jit", "staging_ns": 1655985}`
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
      "max_abs_error": 2.384185791015625e-07,
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
| `case_id` | smoke/C07/base/scale |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
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
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "17ab11c9d5f01e2101a9a5da69dbf603f23be5feb48ab6b9e7dddc7852339c3c",
  "compile_ns": 42692417,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7d96204b160d817f5c98e4de954fcbd2388f0727b99491ebdb0a2259348f378b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "82958f90058c533c73273a88637a98e63193360c4c6930626e016f2976465f80"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 748700,
  "input_digest": "122fb182151d04c2aafcea32cd31da0b5cf40cdfcbb569abf2888c216550d9a1",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1655985
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "f6120244142618730e3ead7329b00c8534fa714e8675750c19c151fb4d3c0989", "compile_ns": 47961752, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0016/optimized_hlo.txt", "sha256": "4c1e25e741e5fa86de336236213dd8200eb4166b905215b2272a4ba6c2450ec6"}, "stablehlo": {"path": "compiler/case-0016/stablehlo.txt", "sha256": "06d40db7db1facca4d8fdbae7aa5e5e6007e27e398990e377327265338586aef"}}, "dynamic_argument_count": 3, "first_execution_ns": 623661, "input_digest": "ec962875849aa244ffef9c9719ef541f87958a971e06592f91309cb802b37f8d", "matrix": true, "mode": "jax_jit", "staging_ns": 1232942}`
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
| `case_id` | smoke/C08/base/dev_small |
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
  "case_sha256": "f6120244142618730e3ead7329b00c8534fa714e8675750c19c151fb4d3c0989",
  "compile_ns": 47961752,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4c1e25e741e5fa86de336236213dd8200eb4166b905215b2272a4ba6c2450ec6"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "06d40db7db1facca4d8fdbae7aa5e5e6007e27e398990e377327265338586aef"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 623661,
  "input_digest": "ec962875849aa244ffef9c9719ef541f87958a971e06592f91309cb802b37f8d",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1232942
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "565eacdcb92c24b7aedca51523194308b3b59409a5d3b3e98a2fc114de17b3b2", "compile_ns": 49079415, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0017/optimized_hlo.txt", "sha256": "079e5dcc1f6f8b1150930720084e735254825c4d628a530bc2ceda8fd6137d98"}, "stablehlo": {"path": "compiler/case-0017/stablehlo.txt", "sha256": "551777a9bde6a2e5b13c859c6e894bf6847182416e22a13448c898b858bd17f0"}}, "dynamic_argument_count": 3, "first_execution_ns": 650944, "input_digest": "da5d922b462ca3f69b125fcdf0151afcc2a108bd1039cca584a2689bfc4fec10", "matrix": true, "mode": "jax_jit", "staging_ns": 1257222}`
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
| `case_id` | smoke/C08/base/dev_medium |
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
  "case_sha256": "565eacdcb92c24b7aedca51523194308b3b59409a5d3b3e98a2fc114de17b3b2",
  "compile_ns": 49079415,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "079e5dcc1f6f8b1150930720084e735254825c4d628a530bc2ceda8fd6137d98"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "551777a9bde6a2e5b13c859c6e894bf6847182416e22a13448c898b858bd17f0"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 650944,
  "input_digest": "da5d922b462ca3f69b125fcdf0151afcc2a108bd1039cca584a2689bfc4fec10",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1257222
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "53da4f874ac4ec76890a9184314b68d4290241427fb91d61693307610223828e", "compile_ns": 47188886, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0018/optimized_hlo.txt", "sha256": "a1810e7d3d77c3675083c07af51ff92ed1fb2228352d24bab932aa0cb847c784"}, "stablehlo": {"path": "compiler/case-0018/stablehlo.txt", "sha256": "10c7a4bed955b22474bd233d3064319c92b30f65b414b7a8126d9b1f93eb44eb"}}, "dynamic_argument_count": 3, "first_execution_ns": 645540, "input_digest": "0c2be2051f656ef67b275717b0b95997679127307efc59bf653ab8f80c86b4f3", "matrix": true, "mode": "jax_jit", "staging_ns": 1205321}`
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
| `case_id` | smoke/C08/base/tail |
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
  "case_sha256": "53da4f874ac4ec76890a9184314b68d4290241427fb91d61693307610223828e",
  "compile_ns": 47188886,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a1810e7d3d77c3675083c07af51ff92ed1fb2228352d24bab932aa0cb847c784"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "10c7a4bed955b22474bd233d3064319c92b30f65b414b7a8126d9b1f93eb44eb"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 645540,
  "input_digest": "0c2be2051f656ef67b275717b0b95997679127307efc59bf653ab8f80c86b4f3",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1205321
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "8d61fb78e580a36fe60a49443687d97c1d0946d6d484869c875781a3e0afb3ac", "compile_ns": 48444660, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0019/optimized_hlo.txt", "sha256": "14957cec4fa7d370d6c85bf034ae393d521b82be1fe9af925cc3849866efcc55"}, "stablehlo": {"path": "compiler/case-0019/stablehlo.txt", "sha256": "37a2c704551ae1a2205251b9fbde7f9ded283b7787000cf7908bcd4ce76a566e"}}, "dynamic_argument_count": 3, "first_execution_ns": 711563, "input_digest": "59772c428bc3f303e353412715be9470aa1046e4ebdf89b49e2d25e7bdd31624", "matrix": true, "mode": "jax_jit", "staging_ns": 1389340}`
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
| `case_id` | smoke/C08/base/scale |
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
  "case_sha256": "8d61fb78e580a36fe60a49443687d97c1d0946d6d484869c875781a3e0afb3ac",
  "compile_ns": 48444660,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "14957cec4fa7d370d6c85bf034ae393d521b82be1fe9af925cc3849866efcc55"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "37a2c704551ae1a2205251b9fbde7f9ded283b7787000cf7908bcd4ce76a566e"
    }
  },
  "dynamic_argument_count": 3,
  "first_execution_ns": 711563,
  "input_digest": "59772c428bc3f303e353412715be9470aa1046e4ebdf89b49e2d25e7bdd31624",
  "matrix": true,
  "mode": "jax_jit",
  "staging_ns": 1389340
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "dee160bd90a993f379d0a70d099160e7181c32bd56b50c8087303ffb00b4f5d9", "compile_ns": 34657621, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0020/optimized_hlo.txt", "sha256": "26a2672059226314305451d04cd0edc5d82bbb6df7e21ad4232183a150994ae9"}, "stablehlo": {"path": "compiler/case-0020/stablehlo.txt", "sha256": "fb635e167a604511398a8c307e05d3a15a9af1aee3a11256355b6347f5b16f4f"}}, "dynamic_argument_count": 5, "first_execution_ns": 920832, "input_digest": "8fd577711fc0584aaca1bf543821c35a5a1501d4d2a4dd8718804e3159b14fc0", "matrix": false, "mode": "jax_jit", "staging_ns": 1110086}`
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
| `case_id` | smoke/D01/base/dev_small |
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
  "case_sha256": "dee160bd90a993f379d0a70d099160e7181c32bd56b50c8087303ffb00b4f5d9",
  "compile_ns": 34657621,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "26a2672059226314305451d04cd0edc5d82bbb6df7e21ad4232183a150994ae9"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "fb635e167a604511398a8c307e05d3a15a9af1aee3a11256355b6347f5b16f4f"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 920832,
  "input_digest": "8fd577711fc0584aaca1bf543821c35a5a1501d4d2a4dd8718804e3159b14fc0",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1110086
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "e5387be5f56bd7d0630332418f90ac8b5fafed8f7cfec8f67bd7101aa7adbeef", "compile_ns": 35353888, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0021/optimized_hlo.txt", "sha256": "7efe13a35134a6b5357b797821f34e81c3b6422f669f490553bbd1afd53a8924"}, "stablehlo": {"path": "compiler/case-0021/stablehlo.txt", "sha256": "e19dc15e6ac4f4d68145948652c98f6e80fe910cbf29f299a8ac79fc346a2e46"}}, "dynamic_argument_count": 5, "first_execution_ns": 806193, "input_digest": "32a4bd8f60d050de509c62eae68aafc6cf042ded86f4bf734d900d23361043f4", "matrix": false, "mode": "jax_jit", "staging_ns": 1121378}`
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
| `case_id` | smoke/D01/base/dev_medium |
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
  "case_sha256": "e5387be5f56bd7d0630332418f90ac8b5fafed8f7cfec8f67bd7101aa7adbeef",
  "compile_ns": 35353888,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7efe13a35134a6b5357b797821f34e81c3b6422f669f490553bbd1afd53a8924"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "e19dc15e6ac4f4d68145948652c98f6e80fe910cbf29f299a8ac79fc346a2e46"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 806193,
  "input_digest": "32a4bd8f60d050de509c62eae68aafc6cf042ded86f4bf734d900d23361043f4",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1121378
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "de945958c6d127672069734ec7ecb15b8588496896458baa461efe17a05e5f5d", "compile_ns": 34926262, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0022/optimized_hlo.txt", "sha256": "ce7542030f8b2628c1985159bf4a6f584b1801c2c75c313ae39fc12049c55f80"}, "stablehlo": {"path": "compiler/case-0022/stablehlo.txt", "sha256": "36c8f81eedd6db154b6c9ea0df10eaea23b6629a4412a32c210a15ec6ef1b925"}}, "dynamic_argument_count": 5, "first_execution_ns": 820117, "input_digest": "2275502841c2b506f95309f2bdcd8c03ad06fe4c074c6dc619c8f3090d311d2e", "matrix": false, "mode": "jax_jit", "staging_ns": 1061283}`
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
| `case_id` | smoke/D01/base/tail |
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
  "case_sha256": "de945958c6d127672069734ec7ecb15b8588496896458baa461efe17a05e5f5d",
  "compile_ns": 34926262,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ce7542030f8b2628c1985159bf4a6f584b1801c2c75c313ae39fc12049c55f80"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "36c8f81eedd6db154b6c9ea0df10eaea23b6629a4412a32c210a15ec6ef1b925"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 820117,
  "input_digest": "2275502841c2b506f95309f2bdcd8c03ad06fe4c074c6dc619c8f3090d311d2e",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1061283
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "1409edbf0200f1d1491c498f98b555c85bf259a909e9cbf49032bd15d19de4a3", "compile_ns": 36100486, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0023/optimized_hlo.txt", "sha256": "62fa13bdb6e7985a52bd3e05f97767e103fe0b3bfe1f7ff8345435ef8fc3bcc2"}, "stablehlo": {"path": "compiler/case-0023/stablehlo.txt", "sha256": "0b493bc780b22a96d9261d7563139728dc5a8fbe1235623ed084021577552eaa"}}, "dynamic_argument_count": 5, "first_execution_ns": 835580, "input_digest": "f8f689e912c1f7e1672364e97ba9c7936bd6ab7c44c696bc4c6b99e82b7b8d4c", "matrix": false, "mode": "jax_jit", "staging_ns": 1176220}`
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
| `case_id` | smoke/D01/base/scale |
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
  "case_sha256": "1409edbf0200f1d1491c498f98b555c85bf259a909e9cbf49032bd15d19de4a3",
  "compile_ns": 36100486,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "62fa13bdb6e7985a52bd3e05f97767e103fe0b3bfe1f7ff8345435ef8fc3bcc2"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0b493bc780b22a96d9261d7563139728dc5a8fbe1235623ed084021577552eaa"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 835580,
  "input_digest": "f8f689e912c1f7e1672364e97ba9c7936bd6ab7c44c696bc4c6b99e82b7b8d4c",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1176220
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "5154af9a1bdecc35b905e10b461cb4c099ab09dbe9ea9aaf256794e736d97004", "compile_ns": 33761511, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0024/optimized_hlo.txt", "sha256": "920b95c481a3b340db25ea550d4bfb3452d2f757f8af2f39acce82c22e923551"}, "stablehlo": {"path": "compiler/case-0024/stablehlo.txt", "sha256": "abec4bff11a90fe388192bc8eb00c5ad984dcc49cfaaeb9ffc26223e19f3e5ab"}}, "dynamic_argument_count": 5, "first_execution_ns": 770201, "input_digest": "d9b27289f9fe151ef57297780c73f83ce987833aaa5e4cde9d848135c3b6a493", "matrix": false, "mode": "jax_jit", "staging_ns": 1078494}`
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
| `case_id` | smoke/D02/base/dev_small |
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
  "case_sha256": "5154af9a1bdecc35b905e10b461cb4c099ab09dbe9ea9aaf256794e736d97004",
  "compile_ns": 33761511,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "920b95c481a3b340db25ea550d4bfb3452d2f757f8af2f39acce82c22e923551"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "abec4bff11a90fe388192bc8eb00c5ad984dcc49cfaaeb9ffc26223e19f3e5ab"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 770201,
  "input_digest": "d9b27289f9fe151ef57297780c73f83ce987833aaa5e4cde9d848135c3b6a493",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1078494
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "d4022fbe81b9198102c9f48534ff35cc73f8483d2e90b1aab4c0e9833dfc19f8", "compile_ns": 38207802, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0025/optimized_hlo.txt", "sha256": "352e238bf346174f280c8c285e2b46195a5ab56768e49dab22f31723b5b1ba83"}, "stablehlo": {"path": "compiler/case-0025/stablehlo.txt", "sha256": "41306cc1a06e3e95ed1528c09c0b6c5d4717b6582895f2529d18af1d7104bbef"}}, "dynamic_argument_count": 5, "first_execution_ns": 839799, "input_digest": "8a3fe7471bb3fc7f9481f3d99cd52bb5fc2d89047b1f49fc1f2e3fda0d84bae5", "matrix": false, "mode": "jax_jit", "staging_ns": 1114361}`
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
| `case_id` | smoke/D02/base/dev_medium |
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
  "case_sha256": "d4022fbe81b9198102c9f48534ff35cc73f8483d2e90b1aab4c0e9833dfc19f8",
  "compile_ns": 38207802,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "352e238bf346174f280c8c285e2b46195a5ab56768e49dab22f31723b5b1ba83"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "41306cc1a06e3e95ed1528c09c0b6c5d4717b6582895f2529d18af1d7104bbef"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 839799,
  "input_digest": "8a3fe7471bb3fc7f9481f3d99cd52bb5fc2d89047b1f49fc1f2e3fda0d84bae5",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1114361
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "505e2532d314b0ec94c061ea939c1664fdd102768694e4600a973397666e717f", "compile_ns": 33962905, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0026/optimized_hlo.txt", "sha256": "7e2074b691115304c3b2a16b9e23ecc3b643887eab5952906995b92aa9890e00"}, "stablehlo": {"path": "compiler/case-0026/stablehlo.txt", "sha256": "ea22574d5bbc57cf66441a1f2cd42851266d8845a61459b4369015afdb2f0151"}}, "dynamic_argument_count": 5, "first_execution_ns": 756114, "input_digest": "a6fa3769a27fc3b76babd2a7a37cc7e7565e9c8035476edee7e3adf17e8934c0", "matrix": false, "mode": "jax_jit", "staging_ns": 1064353}`
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
| `case_id` | smoke/D02/base/tail |
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
  "case_sha256": "505e2532d314b0ec94c061ea939c1664fdd102768694e4600a973397666e717f",
  "compile_ns": 33962905,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7e2074b691115304c3b2a16b9e23ecc3b643887eab5952906995b92aa9890e00"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ea22574d5bbc57cf66441a1f2cd42851266d8845a61459b4369015afdb2f0151"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 756114,
  "input_digest": "a6fa3769a27fc3b76babd2a7a37cc7e7565e9c8035476edee7e3adf17e8934c0",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1064353
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "777065f72b3a64f462899f730ebed4845f5bf4a88da01ead12fa6661a48835ed", "compile_ns": 35334791, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0027/optimized_hlo.txt", "sha256": "cc85d00fed402e0aef0f75cd354d1cf42dd82ffbf60d30e0cb9d15fe88eec9d7"}, "stablehlo": {"path": "compiler/case-0027/stablehlo.txt", "sha256": "50d8801bcff14e4b9dda74b52fe6b119d7f9b64cc83c5941d9811eb5581b28fa"}}, "dynamic_argument_count": 5, "first_execution_ns": 744747, "input_digest": "bf6d6b7c4270dd2237e8a1eb27e514b4c6d71aac9f65ca9ca7e9eacc1a723d79", "matrix": false, "mode": "jax_jit", "staging_ns": 1194095}`
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
| `case_id` | smoke/D02/base/scale |
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
  "case_sha256": "777065f72b3a64f462899f730ebed4845f5bf4a88da01ead12fa6661a48835ed",
  "compile_ns": 35334791,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "cc85d00fed402e0aef0f75cd354d1cf42dd82ffbf60d30e0cb9d15fe88eec9d7"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "50d8801bcff14e4b9dda74b52fe6b119d7f9b64cc83c5941d9811eb5581b28fa"
    }
  },
  "dynamic_argument_count": 5,
  "first_execution_ns": 744747,
  "input_digest": "bf6d6b7c4270dd2237e8a1eb27e514b4c6d71aac9f65ca9ca7e9eacc1a723d79",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1194095
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "074130e9a43de09c6b950110f69f5f88cfd327bde89285fa87f69e185288b435", "compile_ns": 31060224, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0028/optimized_hlo.txt", "sha256": "23375f5c3013825d584eeeb0e06c43a0bd449167253615b8b54b1b342158915e"}, "stablehlo": {"path": "compiler/case-0028/stablehlo.txt", "sha256": "2d01efe0f7fa17a8fa22403271407eeea14f0270e338c7e0fd899afd1d877115"}}, "dynamic_argument_count": 4, "first_execution_ns": 736910, "input_digest": "99f3ed5c145f233100a3cbb8f059d363c4e662e4dd6fdc6d912e61204a94490b", "matrix": false, "mode": "jax_jit", "staging_ns": 918436}`
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
      "max_abs_error": 5.960464477539063e-08,
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
| `case_id` | smoke/D03/PP/dev_small |
| `probe_id` | portable_core |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
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
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "case_sha256": "074130e9a43de09c6b950110f69f5f88cfd327bde89285fa87f69e185288b435",
  "compile_ns": 31060224,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "23375f5c3013825d584eeeb0e06c43a0bd449167253615b8b54b1b342158915e"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "2d01efe0f7fa17a8fa22403271407eeea14f0270e338c7e0fd899afd1d877115"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 736910,
  "input_digest": "99f3ed5c145f233100a3cbb8f059d363c4e662e4dd6fdc6d912e61204a94490b",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 918436
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "02671daf6196b882bf61271240a6242bff9cf8bf18f0a204e67bfb4b084c7c4f", "compile_ns": 30686372, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0029/optimized_hlo.txt", "sha256": "565edfa2b683a26d5951de182e282b5a430a1c8f016c0d6ec77e37d9e8431747"}, "stablehlo": {"path": "compiler/case-0029/stablehlo.txt", "sha256": "4d41cf8f7213f00b6e4690aeb3756099704df45f553705b5c4d69b205a8434d1"}}, "dynamic_argument_count": 4, "first_execution_ns": 718300, "input_digest": "65892d50cac8e72e4a6611254170cb1eae6445d92a7201f159daf2eb68f36be9", "matrix": false, "mode": "jax_jit", "staging_ns": 963806}`
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
| `case_id` | smoke/D03/PP/dev_medium |
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
  "case_sha256": "02671daf6196b882bf61271240a6242bff9cf8bf18f0a204e67bfb4b084c7c4f",
  "compile_ns": 30686372,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "565edfa2b683a26d5951de182e282b5a430a1c8f016c0d6ec77e37d9e8431747"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "4d41cf8f7213f00b6e4690aeb3756099704df45f553705b5c4d69b205a8434d1"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 718300,
  "input_digest": "65892d50cac8e72e4a6611254170cb1eae6445d92a7201f159daf2eb68f36be9",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 963806
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "1d50830569a195299071f1feb5cc3b61a750e39486ae099eac8833a22a42b27d", "compile_ns": 30737743, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0030/optimized_hlo.txt", "sha256": "8857a8dd5a26e87fde9287dc3003e4c03f6e1d955c06e895ea1da557ed41f3f2"}, "stablehlo": {"path": "compiler/case-0030/stablehlo.txt", "sha256": "d71469f93231d91f9a59c40a95a8eaad99039255eddf39c5c302597e3861e1ce"}}, "dynamic_argument_count": 4, "first_execution_ns": 706265, "input_digest": "8358e537cf682e43aa22950b56388d9b4a8bca823138522807d0e615115b6d7c", "matrix": false, "mode": "jax_jit", "staging_ns": 946211}`
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
| `case_id` | smoke/D03/PP/tail |
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
  "case_sha256": "1d50830569a195299071f1feb5cc3b61a750e39486ae099eac8833a22a42b27d",
  "compile_ns": 30737743,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8857a8dd5a26e87fde9287dc3003e4c03f6e1d955c06e895ea1da557ed41f3f2"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "d71469f93231d91f9a59c40a95a8eaad99039255eddf39c5c302597e3861e1ce"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 706265,
  "input_digest": "8358e537cf682e43aa22950b56388d9b4a8bca823138522807d0e615115b6d7c",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 946211
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
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

- launch.extras: `{"case_sha256": "ff69708b0f44d10d07b14ed1c2cf4bddb2a73781912ed6703ef03c0db08ce7be", "compile_ns": 31666580, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0031/optimized_hlo.txt", "sha256": "2d2dcaf0ec070e5aedbf2475ec455e8eefef7346e6c19b75dd5398b33470f7d4"}, "stablehlo": {"path": "compiler/case-0031/stablehlo.txt", "sha256": "35a4b5d8ec7fabd5d3b78c7029517ad4e47a4fba88e142f55d019c18cd9e8731"}}, "dynamic_argument_count": 4, "first_execution_ns": 724380, "input_digest": "5d956a0d2b1e1f2dd1137a9a034a76826ec88603171e36b2036a0c72635752da", "matrix": false, "mode": "jax_jit", "staging_ns": 1001653}`
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
| `case_id` | smoke/D03/PP/scale |
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
  "case_sha256": "ff69708b0f44d10d07b14ed1c2cf4bddb2a73781912ed6703ef03c0db08ce7be",
  "compile_ns": 31666580,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "2d2dcaf0ec070e5aedbf2475ec455e8eefef7346e6c19b75dd5398b33470f7d4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "35a4b5d8ec7fabd5d3b78c7029517ad4e47a4fba88e142f55d019c18cd9e8731"
    }
  },
  "dynamic_argument_count": 4,
  "first_execution_ns": 724380,
  "input_digest": "5d956a0d2b1e1f2dd1137a9a034a76826ec88603171e36b2036a0c72635752da",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 1001653
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 96,
  "stop": 128,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `298a1a8f7e2dd145160211817cc0d9731572e206b1b95e28dcdee11bee19a11e`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-002/out/tpu/portable-core-qualification-v1/cd28e27d2ce351ba709a5fcea3c31335de3e931832b3beb93888775c768a2d91/native-restore-002-core28-003/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
