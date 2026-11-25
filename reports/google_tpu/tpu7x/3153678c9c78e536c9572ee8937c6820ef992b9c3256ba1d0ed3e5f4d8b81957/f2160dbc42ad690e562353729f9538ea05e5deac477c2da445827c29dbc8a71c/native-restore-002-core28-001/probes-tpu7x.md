# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-restore-002-core28-001
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

- launch.extras: `{"case_sha256": "be66f5067567c8c539faed688039d9973b44833c90910eb2208105e219f04df0", "compile_ns": 27806492, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "d72fb3114bdb7f571dcee433c6d8cab7319790a0b35c3af7e85da4523df0bb4e"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "2b302c6d75ae71a3f7405cd27c99a7703bc3a6b6408c53c7b9431ad1db1a443e"}}, "dynamic_argument_count": 1, "first_execution_ns": 653316, "input_digest": "97bfd7a0c15b6311962a703b14e9aec56e027521864ae2dfa7ec71a9859a92e8", "matrix": false, "mode": "jax_jit", "staging_ns": 311452434}`
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
| `case_id` | smoke/A02/to_fp32/dev_small |
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
  "case_sha256": "be66f5067567c8c539faed688039d9973b44833c90910eb2208105e219f04df0",
  "compile_ns": 27806492,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "d72fb3114bdb7f571dcee433c6d8cab7319790a0b35c3af7e85da4523df0bb4e"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "2b302c6d75ae71a3f7405cd27c99a7703bc3a6b6408c53c7b9431ad1db1a443e"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 653316,
  "input_digest": "97bfd7a0c15b6311962a703b14e9aec56e027521864ae2dfa7ec71a9859a92e8",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 311452434
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "3be021c853bed57d51cfbb0c6e516c911ceef56948997fd6c1438055cd6571e6", "compile_ns": 26180209, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "f13674a62c93091f02bca70a4bb4ed63b4bd6386b7620f0041f65d5d1b1d7592"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "312b55a35b302b06e6d0f14eb058d42c6072e66d05e8e739bd095806e3b35e1e"}}, "dynamic_argument_count": 1, "first_execution_ns": 577735, "input_digest": "978c014676b547e6769ee799c8767be40d2a843c0708552688b70ddf5d11b417", "matrix": false, "mode": "jax_jit", "staging_ns": 27000712}`
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
| `case_id` | smoke/A02/to_fp32/dev_medium |
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
  "case_sha256": "3be021c853bed57d51cfbb0c6e516c911ceef56948997fd6c1438055cd6571e6",
  "compile_ns": 26180209,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "f13674a62c93091f02bca70a4bb4ed63b4bd6386b7620f0041f65d5d1b1d7592"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "312b55a35b302b06e6d0f14eb058d42c6072e66d05e8e739bd095806e3b35e1e"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 577735,
  "input_digest": "978c014676b547e6769ee799c8767be40d2a843c0708552688b70ddf5d11b417",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 27000712
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "4dc4350dfc2276240fce9b55a60582c52b04bee782d59af0b72d9a158a50aba5", "compile_ns": 25514744, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "a6af1f979eefef0faabd2a923bed9186718f0abc1607e4ee62547773b7578dda"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "344084c81d1bea0bf2cd89879dac070caab46e2444691af7ad6ff2101e9089d2"}}, "dynamic_argument_count": 1, "first_execution_ns": 562252, "input_digest": "80795f414b7f6f2cfc14db3bbd274aee8b69191c80cc7b67ac45be6aea21e0ed", "matrix": false, "mode": "jax_jit", "staging_ns": 26690009}`
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
| `case_id` | smoke/A02/to_fp32/tail |
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
  "case_sha256": "4dc4350dfc2276240fce9b55a60582c52b04bee782d59af0b72d9a158a50aba5",
  "compile_ns": 25514744,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "a6af1f979eefef0faabd2a923bed9186718f0abc1607e4ee62547773b7578dda"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "344084c81d1bea0bf2cd89879dac070caab46e2444691af7ad6ff2101e9089d2"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 562252,
  "input_digest": "80795f414b7f6f2cfc14db3bbd274aee8b69191c80cc7b67ac45be6aea21e0ed",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 26690009
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "29fa522a6fe6a24445dffeda45da64d96b850da3c8a2f49621e35239c2143568", "compile_ns": 25625501, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "d4e063efc24b618220d40ee53a3d9298ce8c4287be76b4457aee5fc335b34310"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "50df2322773229f4ba1299a05636e5c34b628b5f6a596f695aa8ab3a88540880"}}, "dynamic_argument_count": 1, "first_execution_ns": 559314, "input_digest": "01fbdeccb6d12afaec8266727d8a4d0eb44eda14478370285d9eabbf218c025a", "matrix": false, "mode": "jax_jit", "staging_ns": 26539900}`
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
| `case_id` | smoke/A02/to_fp32/scale |
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
  "case_sha256": "29fa522a6fe6a24445dffeda45da64d96b850da3c8a2f49621e35239c2143568",
  "compile_ns": 25625501,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "d4e063efc24b618220d40ee53a3d9298ce8c4287be76b4457aee5fc335b34310"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "50df2322773229f4ba1299a05636e5c34b628b5f6a596f695aa8ab3a88540880"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 559314,
  "input_digest": "01fbdeccb6d12afaec8266727d8a4d0eb44eda14478370285d9eabbf218c025a",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 26539900
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "47397331a408fe82d6a499c08f7cd57d8bba38c0bc49918b3e5309c181613450", "compile_ns": 27928118, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "5aedccd254209f36bf6651648837f32c184b20a1a6b7799f2324078c7b22842e"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "b9a81389e6424b24b64d399277b440f7aedce4e9be64a9cc12624296e887773e"}}, "dynamic_argument_count": 1, "first_execution_ns": 699206, "input_digest": "33c9637b2138d4c931e40d735cc2ee9d1fd9c0eab45866bfbd1b5a39f680fca6", "matrix": false, "mode": "jax_jit", "staging_ns": 570774}`
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
| `case_id` | smoke/A03/relu/dev_small |
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
  "case_sha256": "47397331a408fe82d6a499c08f7cd57d8bba38c0bc49918b3e5309c181613450",
  "compile_ns": 27928118,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "5aedccd254209f36bf6651648837f32c184b20a1a6b7799f2324078c7b22842e"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b9a81389e6424b24b64d399277b440f7aedce4e9be64a9cc12624296e887773e"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 699206,
  "input_digest": "33c9637b2138d4c931e40d735cc2ee9d1fd9c0eab45866bfbd1b5a39f680fca6",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 570774
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "a59388eb9aa22d909e56f7270c09318580792f3a10805cd92ea30953bc7be49a", "compile_ns": 27873943, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "f4a0dde0fb319d4f1a709f863c7cae96fe9b3eea41e86462a371a5288025912c"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "5a7a43883ba6e1f3d3854072f41decd310b980b067e924b301b92029ac328bc2"}}, "dynamic_argument_count": 1, "first_execution_ns": 756446, "input_digest": "575c460846fd774e17e0dcd5e5497e6cde960a1dc7ad10453541a50a7f4883f7", "matrix": false, "mode": "jax_jit", "staging_ns": 509613}`
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
| `case_id` | smoke/A03/relu/dev_medium |
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
  "case_sha256": "a59388eb9aa22d909e56f7270c09318580792f3a10805cd92ea30953bc7be49a",
  "compile_ns": 27873943,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "f4a0dde0fb319d4f1a709f863c7cae96fe9b3eea41e86462a371a5288025912c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "5a7a43883ba6e1f3d3854072f41decd310b980b067e924b301b92029ac328bc2"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 756446,
  "input_digest": "575c460846fd774e17e0dcd5e5497e6cde960a1dc7ad10453541a50a7f4883f7",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 509613
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "a32186bace282cf292c7b86168890b6819c2eadb4405eec32e8bdafee4396a42", "compile_ns": 27201597, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "6406a778a0acf158ff51cacb91250f136b8b76b8091b5661d73ab07c509091ee"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "1e0b364c2da163d61ef6dfecde53b1ae11474a3862f4464fc206ff0997fca111"}}, "dynamic_argument_count": 1, "first_execution_ns": 751975, "input_digest": "7750c2da004a52616cc87f45deb8d7f41d83ff36b0528ceda17ea070438d0c69", "matrix": false, "mode": "jax_jit", "staging_ns": 485809}`
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
| `case_id` | smoke/A03/relu/tail |
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
  "case_sha256": "a32186bace282cf292c7b86168890b6819c2eadb4405eec32e8bdafee4396a42",
  "compile_ns": 27201597,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6406a778a0acf158ff51cacb91250f136b8b76b8091b5661d73ab07c509091ee"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1e0b364c2da163d61ef6dfecde53b1ae11474a3862f4464fc206ff0997fca111"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 751975,
  "input_digest": "7750c2da004a52616cc87f45deb8d7f41d83ff36b0528ceda17ea070438d0c69",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 485809
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "b468bd41f95e7d01f30a5e9130009a3e70d9d5f879a5bfb59d57459343d2249b", "compile_ns": 30690341, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "4b5379072c6b3b0d3a8751ad7f759bb12bd659cf3e37b7349a37394b63e5bef2"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "415aca6fdbb40ab7af6c5d79739c3c46c98e7eb9ae97de774a9fd9b676e4b6de"}}, "dynamic_argument_count": 1, "first_execution_ns": 651157, "input_digest": "027a7b91bbe59d629cd6ba48673d22a1a506a0fbdf3ef677df8b6609ad835562", "matrix": false, "mode": "jax_jit", "staging_ns": 558572}`
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
| `case_id` | smoke/A03/relu/scale |
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
  "case_sha256": "b468bd41f95e7d01f30a5e9130009a3e70d9d5f879a5bfb59d57459343d2249b",
  "compile_ns": 30690341,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4b5379072c6b3b0d3a8751ad7f759bb12bd659cf3e37b7349a37394b63e5bef2"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "415aca6fdbb40ab7af6c5d79739c3c46c98e7eb9ae97de774a9fd9b676e4b6de"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 651157,
  "input_digest": "027a7b91bbe59d629cd6ba48673d22a1a506a0fbdf3ef677df8b6609ad835562",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 558572
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "1b94f5d2e9e43564fde30f60d93ce41260579d222815797e150f34f36c43684a", "compile_ns": 26053263, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0008/optimized_hlo.txt", "sha256": "acaf96fb38c8b9826a4a7cbc47306e4dea9ededdc727ffb28269f00fe13c41b7"}, "stablehlo": {"path": "compiler/case-0008/stablehlo.txt", "sha256": "f9213528612597e68bd808fae4e1eea81cc8cf0e8684d84cc4e0992174ea6ffa"}}, "dynamic_argument_count": 1, "first_execution_ns": 710435, "input_digest": "67cce99456cbd962421ac384bee43ea8b87e7a5d3937a72fac7a46cc58a3026c", "matrix": false, "mode": "jax_jit", "staging_ns": 501408}`
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
| `case_id` | smoke/A03/exp/dev_small |
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
  "case_sha256": "1b94f5d2e9e43564fde30f60d93ce41260579d222815797e150f34f36c43684a",
  "compile_ns": 26053263,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "acaf96fb38c8b9826a4a7cbc47306e4dea9ededdc727ffb28269f00fe13c41b7"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "f9213528612597e68bd808fae4e1eea81cc8cf0e8684d84cc4e0992174ea6ffa"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 710435,
  "input_digest": "67cce99456cbd962421ac384bee43ea8b87e7a5d3937a72fac7a46cc58a3026c",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 501408
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "6eea690f5a0658f9e9f781f839f6f35bfe646d0f020910b8bffe05670ae38c4a", "compile_ns": 26152794, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "9f2dd51c5d10856e6adc679c0fb404c7d542c4302a5e9c7b5528db6ec55a301a"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "50cf5aaaaaa05cb66880f42a09c5a73a701830ede002c142da9cdb846769151e"}}, "dynamic_argument_count": 1, "first_execution_ns": 727364, "input_digest": "726a76fa30b72af83894070f226cc72f2eecce7fe0b3f0dd9d28912da2e5e43c", "matrix": false, "mode": "jax_jit", "staging_ns": 563494}`
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
| `case_id` | smoke/A03/exp/dev_medium |
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
  "case_sha256": "6eea690f5a0658f9e9f781f839f6f35bfe646d0f020910b8bffe05670ae38c4a",
  "compile_ns": 26152794,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9f2dd51c5d10856e6adc679c0fb404c7d542c4302a5e9c7b5528db6ec55a301a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "50cf5aaaaaa05cb66880f42a09c5a73a701830ede002c142da9cdb846769151e"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 727364,
  "input_digest": "726a76fa30b72af83894070f226cc72f2eecce7fe0b3f0dd9d28912da2e5e43c",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 563494
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "790041c2e654135d7931003f2199e142636ac67c75ded53f7a4727eb0844fd59", "compile_ns": 25729856, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "b7ab27c571192ef9e9ed86ac7747c016d0d25864231463af0b89930f4b4447cc"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "ad12e74ad5ef0c402b4d88f2a693d482bcc8e33ba931c0d718d8a208459168f9"}}, "dynamic_argument_count": 1, "first_execution_ns": 698021, "input_digest": "7011506503fd904cf8a54cbfff2146668c03484a740e98701c1d5b63578276e6", "matrix": false, "mode": "jax_jit", "staging_ns": 520535}`
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
| `case_id` | smoke/A03/exp/tail |
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
  "case_sha256": "790041c2e654135d7931003f2199e142636ac67c75ded53f7a4727eb0844fd59",
  "compile_ns": 25729856,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "b7ab27c571192ef9e9ed86ac7747c016d0d25864231463af0b89930f4b4447cc"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ad12e74ad5ef0c402b4d88f2a693d482bcc8e33ba931c0d718d8a208459168f9"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 698021,
  "input_digest": "7011506503fd904cf8a54cbfff2146668c03484a740e98701c1d5b63578276e6",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 520535
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "c157204035f757cb8b3d7c385ebb57faca32cb0d1b27f175649e8722383f7626", "compile_ns": 25965425, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "25b4e5ed4f3145e645d1696f9f5849cff71fcb6c71e0a8ff322a17019f0bedea"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "903376de1f298179c154c585281264b14b615514475c5cdb56ee99b8887357b0"}}, "dynamic_argument_count": 1, "first_execution_ns": 861972, "input_digest": "2ee05dbd0416ef7179b9ca508841c6723456db5273207595ec6a6b800a7f6390", "matrix": false, "mode": "jax_jit", "staging_ns": 536374}`
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
| `case_id` | smoke/A03/exp/scale |
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
  "case_sha256": "c157204035f757cb8b3d7c385ebb57faca32cb0d1b27f175649e8722383f7626",
  "compile_ns": 25965425,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "25b4e5ed4f3145e645d1696f9f5849cff71fcb6c71e0a8ff322a17019f0bedea"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "903376de1f298179c154c585281264b14b615514475c5cdb56ee99b8887357b0"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 861972,
  "input_digest": "2ee05dbd0416ef7179b9ca508841c6723456db5273207595ec6a6b800a7f6390",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 536374
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "34983b88c58deed32ef40f7765f2e705fbc28868e563107d364426001fa8de1d", "compile_ns": 25722705, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0012/optimized_hlo.txt", "sha256": "1466ab89bf34f696fc4b9cb6f003548605cb3c80222eff8bb2aec12a4cb62f54"}, "stablehlo": {"path": "compiler/case-0012/stablehlo.txt", "sha256": "131cc555506930cc53e3a5f85f38f3537fc2275f8984919a209a4211571871fd"}}, "dynamic_argument_count": 1, "first_execution_ns": 842849, "input_digest": "0ba147371dd03fbe9e34e890bb4b28f3ec27f70b8bb14e4210f0aaeb552d1d78", "matrix": false, "mode": "jax_jit", "staging_ns": 672399}`
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
| `case_id` | smoke/A03/rsqrt/dev_small |
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
  "case_sha256": "34983b88c58deed32ef40f7765f2e705fbc28868e563107d364426001fa8de1d",
  "compile_ns": 25722705,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "1466ab89bf34f696fc4b9cb6f003548605cb3c80222eff8bb2aec12a4cb62f54"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "131cc555506930cc53e3a5f85f38f3537fc2275f8984919a209a4211571871fd"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 842849,
  "input_digest": "0ba147371dd03fbe9e34e890bb4b28f3ec27f70b8bb14e4210f0aaeb552d1d78",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 672399
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "a28e4bd2523bcd560c867317c8a29a6fdb5e83210acf3289468db11818324a6f", "compile_ns": 25442779, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0013/optimized_hlo.txt", "sha256": "1822f7fe90db5aaeab61628525481bac926714a30078446c704c16da9c3df951"}, "stablehlo": {"path": "compiler/case-0013/stablehlo.txt", "sha256": "0e3f56d6fdedbeae9804366894261f9d84c5650b4b4bd8878a98537a538976a5"}}, "dynamic_argument_count": 1, "first_execution_ns": 716489, "input_digest": "5cdada56d8c29fd76f74c09d6d1a343cfb6eda7648ad3f53c04500efea584dc5", "matrix": false, "mode": "jax_jit", "staging_ns": 543068}`
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
| `case_id` | smoke/A03/rsqrt/dev_medium |
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
  "case_sha256": "a28e4bd2523bcd560c867317c8a29a6fdb5e83210acf3289468db11818324a6f",
  "compile_ns": 25442779,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "1822f7fe90db5aaeab61628525481bac926714a30078446c704c16da9c3df951"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0e3f56d6fdedbeae9804366894261f9d84c5650b4b4bd8878a98537a538976a5"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 716489,
  "input_digest": "5cdada56d8c29fd76f74c09d6d1a343cfb6eda7648ad3f53c04500efea584dc5",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 543068
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "dd317d7b01daccbe23c7e4f313e8cf21a65ef363841f1339a96e87fa869cb195", "compile_ns": 25672662, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0014/optimized_hlo.txt", "sha256": "e57ddd52afb95fed46d482383b8610262fd081d7663e8870fb071f3d79940e6f"}, "stablehlo": {"path": "compiler/case-0014/stablehlo.txt", "sha256": "c949b39ecd485ca824582c2affc1e355931b56fb013b48991d8066715abd3544"}}, "dynamic_argument_count": 1, "first_execution_ns": 677653, "input_digest": "88086ea1182069a7bb4ab7b2aa805c697847c78ab546eb5dfe661a3757622ab3", "matrix": false, "mode": "jax_jit", "staging_ns": 507442}`
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
| `case_id` | smoke/A03/rsqrt/tail |
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
  "case_sha256": "dd317d7b01daccbe23c7e4f313e8cf21a65ef363841f1339a96e87fa869cb195",
  "compile_ns": 25672662,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e57ddd52afb95fed46d482383b8610262fd081d7663e8870fb071f3d79940e6f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c949b39ecd485ca824582c2affc1e355931b56fb013b48991d8066715abd3544"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 677653,
  "input_digest": "88086ea1182069a7bb4ab7b2aa805c697847c78ab546eb5dfe661a3757622ab3",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 507442
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "925ee557bd5749dcfb092696ad1bcc02423c0384a11c2d6cf6d50abd60030a31", "compile_ns": 28049843, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0015/optimized_hlo.txt", "sha256": "301a0f663b50ecf4fc3223780ac19b6aa8f2e0df748cabafbcc31ed74cd14235"}, "stablehlo": {"path": "compiler/case-0015/stablehlo.txt", "sha256": "8039f70382c8f744049454c0e4e4e0d21fbdbb31ea1add0d8411af2a225c68c1"}}, "dynamic_argument_count": 1, "first_execution_ns": 2919606, "input_digest": "20166ba624b2ee41a5c48e43d656e0ee259211a9f23440687b6b01584b6030b8", "matrix": false, "mode": "jax_jit", "staging_ns": 551916}`
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
| `case_id` | smoke/A03/rsqrt/scale |
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
  "case_sha256": "925ee557bd5749dcfb092696ad1bcc02423c0384a11c2d6cf6d50abd60030a31",
  "compile_ns": 28049843,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "301a0f663b50ecf4fc3223780ac19b6aa8f2e0df748cabafbcc31ed74cd14235"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "8039f70382c8f744049454c0e4e4e0d21fbdbb31ea1add0d8411af2a225c68c1"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 2919606,
  "input_digest": "20166ba624b2ee41a5c48e43d656e0ee259211a9f23440687b6b01584b6030b8",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 551916
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "e3e682a011c73319dc3041e7a978bef006fa08914793364e1e6b6695c986f462", "compile_ns": 29355377, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0016/optimized_hlo.txt", "sha256": "600958387b23e677c6bb64e6434df51609ab8cd51ff3c153bee1d9e392d04684"}, "stablehlo": {"path": "compiler/case-0016/stablehlo.txt", "sha256": "c58beb8e1bf0ac6b450195150ee53aeccc158053d521c325501d82ccb3939ba4"}}, "dynamic_argument_count": 1, "first_execution_ns": 839047, "input_digest": "5f6e83bfa1b37cf00ee5c15e9c215d94e450ffe003d9c4ffbdcb9d0b0d6f1f1f", "matrix": false, "mode": "jax_jit", "staging_ns": 599868}`
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
| `case_id` | smoke/A03/silu/dev_small |
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
  "case_sha256": "e3e682a011c73319dc3041e7a978bef006fa08914793364e1e6b6695c986f462",
  "compile_ns": 29355377,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "600958387b23e677c6bb64e6434df51609ab8cd51ff3c153bee1d9e392d04684"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "c58beb8e1bf0ac6b450195150ee53aeccc158053d521c325501d82ccb3939ba4"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 839047,
  "input_digest": "5f6e83bfa1b37cf00ee5c15e9c215d94e450ffe003d9c4ffbdcb9d0b0d6f1f1f",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 599868
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "a25dc28ab8ccf9a7979878a09c51bce40672efdb71a13cc044db416ae792dbc1", "compile_ns": 28832772, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0017/optimized_hlo.txt", "sha256": "407ef32e5a66199cadd9eb425ed507760c67b54271bc91eaaa1c3b06c225e4e3"}, "stablehlo": {"path": "compiler/case-0017/stablehlo.txt", "sha256": "332c0baf30bf90ec923cf71a754ec6227b37f708b6eaa579d307490fb98bc655"}}, "dynamic_argument_count": 1, "first_execution_ns": 782724, "input_digest": "90ae245c10fe03791953cb3deb14da8dd80473b9a986ad19b27c882df5ff681f", "matrix": false, "mode": "jax_jit", "staging_ns": 538937}`
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
| `case_id` | smoke/A03/silu/dev_medium |
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
  "case_sha256": "a25dc28ab8ccf9a7979878a09c51bce40672efdb71a13cc044db416ae792dbc1",
  "compile_ns": 28832772,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "407ef32e5a66199cadd9eb425ed507760c67b54271bc91eaaa1c3b06c225e4e3"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "332c0baf30bf90ec923cf71a754ec6227b37f708b6eaa579d307490fb98bc655"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 782724,
  "input_digest": "90ae245c10fe03791953cb3deb14da8dd80473b9a986ad19b27c882df5ff681f",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 538937
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "b59161fc1d2ed7c58a6d0cc57e456667cb524399132ada7cce52ca9b7c4cc421", "compile_ns": 32936176, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0018/optimized_hlo.txt", "sha256": "1e02d3336f23710e959a77da8e15701cbe281be723ce70a3301b4e5e3a3169d6"}, "stablehlo": {"path": "compiler/case-0018/stablehlo.txt", "sha256": "a3d6a1d52d6ac556cfca616ed7e426fa1681591d8702df5cdd7d1d78fdd156f6"}}, "dynamic_argument_count": 1, "first_execution_ns": 664582, "input_digest": "9a7b138b1771d102181615d0ef4bbc8eb5c501db59510e99a06b4b9ef0221eaa", "matrix": false, "mode": "jax_jit", "staging_ns": 526440}`
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
| `case_id` | smoke/A03/silu/tail |
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
  "case_sha256": "b59161fc1d2ed7c58a6d0cc57e456667cb524399132ada7cce52ca9b7c4cc421",
  "compile_ns": 32936176,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "1e02d3336f23710e959a77da8e15701cbe281be723ce70a3301b4e5e3a3169d6"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "a3d6a1d52d6ac556cfca616ed7e426fa1681591d8702df5cdd7d1d78fdd156f6"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 664582,
  "input_digest": "9a7b138b1771d102181615d0ef4bbc8eb5c501db59510e99a06b4b9ef0221eaa",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 526440
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "403a93e9409b64e1990107a777542a0b880e8e0559d49a5cb2a98a86f5b30039", "compile_ns": 29107612, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0019/optimized_hlo.txt", "sha256": "0d669f04ddb7fbc6afcd816058c0fd9fb988c2b2e40fe9f0b47d027c0a3d07e9"}, "stablehlo": {"path": "compiler/case-0019/stablehlo.txt", "sha256": "bae5e87ec8d359ab1f24e987e7eff8b4bf9908a4a510845c3e2c2d51dca1f8e5"}}, "dynamic_argument_count": 1, "first_execution_ns": 666452, "input_digest": "8ceddf990b16b699d2b8b232fec163b430b7139b7e25bcc4b76b551066ef68df", "matrix": false, "mode": "jax_jit", "staging_ns": 562063}`
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
| `case_id` | smoke/A03/silu/scale |
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
  "case_sha256": "403a93e9409b64e1990107a777542a0b880e8e0559d49a5cb2a98a86f5b30039",
  "compile_ns": 29107612,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "0d669f04ddb7fbc6afcd816058c0fd9fb988c2b2e40fe9f0b47d027c0a3d07e9"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "bae5e87ec8d359ab1f24e987e7eff8b4bf9908a4a510845c3e2c2d51dca1f8e5"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 666452,
  "input_digest": "8ceddf990b16b699d2b8b232fec163b430b7139b7e25bcc4b76b551066ef68df",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 562063
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "58a329330aed8b410d850ec9b65e62355069846724912a4d93d7d6d0a1abad07", "compile_ns": 29074390, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0020/optimized_hlo.txt", "sha256": "ea24b9de22e7879254116199451e68b73804b823a160c29253c3862189570a6d"}, "stablehlo": {"path": "compiler/case-0020/stablehlo.txt", "sha256": "ca728c5ec935b7f374ed2fbd88130230b692e5b109eba96485389caf65669f0b"}}, "dynamic_argument_count": 1, "first_execution_ns": 708671, "input_digest": "e26884f36aa572ede8b56e610450d3bb875cca2512681ae36885cb071f6883f3", "matrix": false, "mode": "jax_jit", "staging_ns": 566977}`
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
| `case_id` | smoke/A04/axis0/dev_small |
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
  "case_sha256": "58a329330aed8b410d850ec9b65e62355069846724912a4d93d7d6d0a1abad07",
  "compile_ns": 29074390,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ea24b9de22e7879254116199451e68b73804b823a160c29253c3862189570a6d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ca728c5ec935b7f374ed2fbd88130230b692e5b109eba96485389caf65669f0b"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 708671,
  "input_digest": "e26884f36aa572ede8b56e610450d3bb875cca2512681ae36885cb071f6883f3",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 566977
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "9f4330320e1579ae5903167bfb03e91008483fba6480aa64628327fc7499dfa8", "compile_ns": 28364390, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0021/optimized_hlo.txt", "sha256": "da5ab72947cf8c56fbb4c4759b5ac3f21ed77ac5b672179ea997dd86e5cee219"}, "stablehlo": {"path": "compiler/case-0021/stablehlo.txt", "sha256": "ff8e76b0751d96b4d7fb8a81c58f236caba9e871d3333f0e9dd6cf9377d08e3b"}}, "dynamic_argument_count": 1, "first_execution_ns": 792887, "input_digest": "4caf547f76457c1c17b53b0dd971c54e3578cb7b7e89c55bfc37d974dc56ea83", "matrix": false, "mode": "jax_jit", "staging_ns": 534898}`
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
| `case_id` | smoke/A04/axis0/dev_medium |
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
  "case_sha256": "9f4330320e1579ae5903167bfb03e91008483fba6480aa64628327fc7499dfa8",
  "compile_ns": 28364390,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "da5ab72947cf8c56fbb4c4759b5ac3f21ed77ac5b672179ea997dd86e5cee219"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ff8e76b0751d96b4d7fb8a81c58f236caba9e871d3333f0e9dd6cf9377d08e3b"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 792887,
  "input_digest": "4caf547f76457c1c17b53b0dd971c54e3578cb7b7e89c55bfc37d974dc56ea83",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 534898
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "285badd941538d52a681c99b6cca7ff6b3c0dc3405b0de61eab1e4ad093e3971", "compile_ns": 28451383, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0022/optimized_hlo.txt", "sha256": "9f15c7f25b1f4d19cc69f25f3cd59e4fb84d1a1ab789c398d1ccaf6fe4715276"}, "stablehlo": {"path": "compiler/case-0022/stablehlo.txt", "sha256": "ce89c5f296cbc12fddbbaf9dc93867a9713d5f4e0f310c909fa8afaf29b6d777"}}, "dynamic_argument_count": 1, "first_execution_ns": 861016, "input_digest": "abfdc728c4946f51826a63cbc7a0ffcd603b3c35a7ec28d831656d6fbc2ef0b2", "matrix": false, "mode": "jax_jit", "staging_ns": 525192}`
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
| `case_id` | smoke/A04/axis0/tail |
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
  "case_sha256": "285badd941538d52a681c99b6cca7ff6b3c0dc3405b0de61eab1e4ad093e3971",
  "compile_ns": 28451383,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9f15c7f25b1f4d19cc69f25f3cd59e4fb84d1a1ab789c398d1ccaf6fe4715276"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ce89c5f296cbc12fddbbaf9dc93867a9713d5f4e0f310c909fa8afaf29b6d777"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 861016,
  "input_digest": "abfdc728c4946f51826a63cbc7a0ffcd603b3c35a7ec28d831656d6fbc2ef0b2",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 525192
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "2847da91a59bfa3e99c5a8e8ed8c9cdacb098717c0e61802cfebe08b845bff69", "compile_ns": 29360632, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0023/optimized_hlo.txt", "sha256": "fd04ff5bbb591dbaec9e2baf9cb0fe7577bbf6c76ffdabdc3fb1325b02766e10"}, "stablehlo": {"path": "compiler/case-0023/stablehlo.txt", "sha256": "ba636c766a0b8d146745a8f52352e7910c6f74a28765ee0b8fa2110bf2ef7e42"}}, "dynamic_argument_count": 1, "first_execution_ns": 747636, "input_digest": "9872bd079fe87eb14ba9bf9a6d9ebc69273cbd2361c76d4dc4795b9fb915f0ba", "matrix": false, "mode": "jax_jit", "staging_ns": 555555}`
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
| `case_id` | smoke/A04/axis0/scale |
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
  "case_sha256": "2847da91a59bfa3e99c5a8e8ed8c9cdacb098717c0e61802cfebe08b845bff69",
  "compile_ns": 29360632,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "fd04ff5bbb591dbaec9e2baf9cb0fe7577bbf6c76ffdabdc3fb1325b02766e10"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "ba636c766a0b8d146745a8f52352e7910c6f74a28765ee0b8fa2110bf2ef7e42"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 747636,
  "input_digest": "9872bd079fe87eb14ba9bf9a6d9ebc69273cbd2361c76d4dc4795b9fb915f0ba",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 555555
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "f59da98a6768ce282ade7b9f6674e48f97421582334e1dbf70b53192357aa7d1", "compile_ns": 28684444, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0024/optimized_hlo.txt", "sha256": "7a1dbd23b571be1ce325499c2862035df0a9e5200641c8f22eee969bdf06f926"}, "stablehlo": {"path": "compiler/case-0024/stablehlo.txt", "sha256": "3182db0867f7a4bd9738ab440765baed314f2002583578acae1544e08e45e6be"}}, "dynamic_argument_count": 1, "first_execution_ns": 737107, "input_digest": "59e028f23af9d3af49c5dd79f5ea903dd2347a05b8ea62bfe611ed5432fd60c4", "matrix": false, "mode": "jax_jit", "staging_ns": 520639}`
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
| `case_id` | smoke/A04/axis1/dev_small |
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
  "case_sha256": "f59da98a6768ce282ade7b9f6674e48f97421582334e1dbf70b53192357aa7d1",
  "compile_ns": 28684444,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "7a1dbd23b571be1ce325499c2862035df0a9e5200641c8f22eee969bdf06f926"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3182db0867f7a4bd9738ab440765baed314f2002583578acae1544e08e45e6be"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 737107,
  "input_digest": "59e028f23af9d3af49c5dd79f5ea903dd2347a05b8ea62bfe611ed5432fd60c4",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 520639
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "9a9d0690611666b69a19a75e05dc4b71bae899b5c345dbb5fe2bcfdc6b5babec", "compile_ns": 28273878, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0025/optimized_hlo.txt", "sha256": "4aa2a869298ed833930d55fc654f6ff0899474657ae942ef4740de214a5723ed"}, "stablehlo": {"path": "compiler/case-0025/stablehlo.txt", "sha256": "b984e618ef71866f7e996f3e90b47302a6c0887d241ac4b139c3c2d25f2eb142"}}, "dynamic_argument_count": 1, "first_execution_ns": 883729, "input_digest": "b8f8c26c8e34f7f0097151e61ebd539ae6b7b55da766382d06564e695f823efa", "matrix": false, "mode": "jax_jit", "staging_ns": 536233}`
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
| `case_id` | smoke/A04/axis1/dev_medium |
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
  "case_sha256": "9a9d0690611666b69a19a75e05dc4b71bae899b5c345dbb5fe2bcfdc6b5babec",
  "compile_ns": 28273878,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4aa2a869298ed833930d55fc654f6ff0899474657ae942ef4740de214a5723ed"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b984e618ef71866f7e996f3e90b47302a6c0887d241ac4b139c3c2d25f2eb142"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 883729,
  "input_digest": "b8f8c26c8e34f7f0097151e61ebd539ae6b7b55da766382d06564e695f823efa",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 536233
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "dab86fa540215a2b4edbffad5eb961469a59783d73d10ddbde486a4b68ed5e6a", "compile_ns": 28534892, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0026/optimized_hlo.txt", "sha256": "1205c949a83bdda463de4598e0d65c19856633dc917f87bd8603f587a29d5078"}, "stablehlo": {"path": "compiler/case-0026/stablehlo.txt", "sha256": "3363c85b796ecbb0f2c252114b0c53c9e56ed4e3d96c2d4dd11f146e1ad77196"}}, "dynamic_argument_count": 1, "first_execution_ns": 759697, "input_digest": "a7a66d459e9f0950d757150cd742222e20d592665ca5b4f8cdb03573f85d589d", "matrix": false, "mode": "jax_jit", "staging_ns": 512742}`
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
| `case_id` | smoke/A04/axis1/tail |
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
  "case_sha256": "dab86fa540215a2b4edbffad5eb961469a59783d73d10ddbde486a4b68ed5e6a",
  "compile_ns": 28534892,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "1205c949a83bdda463de4598e0d65c19856633dc917f87bd8603f587a29d5078"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3363c85b796ecbb0f2c252114b0c53c9e56ed4e3d96c2d4dd11f146e1ad77196"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 759697,
  "input_digest": "a7a66d459e9f0950d757150cd742222e20d592665ca5b4f8cdb03573f85d589d",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 512742
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "cf513c358a57d1a9517babd03332f52081b988d80cb4df3ccadd7bda8c13a7a9", "compile_ns": 28521939, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0027/optimized_hlo.txt", "sha256": "c90a6bdcdfdcd421e252fbbd534bffe15ec3157a26e48d40cd8a95d421de322d"}, "stablehlo": {"path": "compiler/case-0027/stablehlo.txt", "sha256": "d3d2740b35cbce3eb3b63ff59e36ef3dad8272bb9bd61a2c5302719c4856bd9b"}}, "dynamic_argument_count": 1, "first_execution_ns": 797115, "input_digest": "1df0fe2659128b152699b2fe7130dab95ce8cf148e9992f0757a5468fa9e7a67", "matrix": false, "mode": "jax_jit", "staging_ns": 540759}`
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
| `case_id` | smoke/A04/axis1/scale |
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
  "case_sha256": "cf513c358a57d1a9517babd03332f52081b988d80cb4df3ccadd7bda8c13a7a9",
  "compile_ns": 28521939,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "c90a6bdcdfdcd421e252fbbd534bffe15ec3157a26e48d40cd8a95d421de322d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "d3d2740b35cbce3eb3b63ff59e36ef3dad8272bb9bd61a2c5302719c4856bd9b"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 797115,
  "input_digest": "1df0fe2659128b152699b2fe7130dab95ce8cf148e9992f0757a5468fa9e7a67",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 540759
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "6d3c3467b096afac26feec68753605100358f1c7eac9c15db48779981e3076d5", "compile_ns": 28537890, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0028/optimized_hlo.txt", "sha256": "6b7d015ea22f034d331aaa2779ea4ee63bf0a1b69387b0a6bf92cac9d32b14ca"}, "stablehlo": {"path": "compiler/case-0028/stablehlo.txt", "sha256": "18bbec56f333b3fc9fcd643c64455ccf75beaf7cba0c6ac5a02ce1308b82b583"}}, "dynamic_argument_count": 1, "first_execution_ns": 684770, "input_digest": "0ca7ae3df92ea22634a208847d725b1ce3c75e7e0f1c9ace994e2647915981e0", "matrix": false, "mode": "jax_jit", "staging_ns": 507026}`
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
| `case_id` | smoke/A05/axis0/dev_small |
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
  "case_sha256": "6d3c3467b096afac26feec68753605100358f1c7eac9c15db48779981e3076d5",
  "compile_ns": 28537890,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6b7d015ea22f034d331aaa2779ea4ee63bf0a1b69387b0a6bf92cac9d32b14ca"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "18bbec56f333b3fc9fcd643c64455ccf75beaf7cba0c6ac5a02ce1308b82b583"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 684770,
  "input_digest": "0ca7ae3df92ea22634a208847d725b1ce3c75e7e0f1c9ace994e2647915981e0",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 507026
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "92413cab2a7f46fe98018f8568bc614b587e8805d950bddd18bc84e83330c65b", "compile_ns": 32329102, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0029/optimized_hlo.txt", "sha256": "018deb9ffaf33543b38b572670f3554e51f38a1b7081686ec383779b1b031912"}, "stablehlo": {"path": "compiler/case-0029/stablehlo.txt", "sha256": "7968d4fe121cfd1588a1d751f2903d9d96ff8c321465df05d31a297b6c358d3a"}}, "dynamic_argument_count": 1, "first_execution_ns": 801324, "input_digest": "415d97e898bac8166a3694381fefbd51671db99fa65e3ef3127699ae6a10255c", "matrix": false, "mode": "jax_jit", "staging_ns": 546123}`
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
| `case_id` | smoke/A05/axis0/dev_medium |
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
  "case_sha256": "92413cab2a7f46fe98018f8568bc614b587e8805d950bddd18bc84e83330c65b",
  "compile_ns": 32329102,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "018deb9ffaf33543b38b572670f3554e51f38a1b7081686ec383779b1b031912"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7968d4fe121cfd1588a1d751f2903d9d96ff8c321465df05d31a297b6c358d3a"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 801324,
  "input_digest": "415d97e898bac8166a3694381fefbd51671db99fa65e3ef3127699ae6a10255c",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 546123
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "c6b06f228302cfd56df7c2122e05907ee5ddd1dbe524f0cd0efa3ed96ee6c372", "compile_ns": 28728974, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0030/optimized_hlo.txt", "sha256": "e3049e88048d7deee48768a9052b5bbe4791ed37c82334e45cb8856261d1597a"}, "stablehlo": {"path": "compiler/case-0030/stablehlo.txt", "sha256": "8fe5b7c239b1241f6c2a3a78068147ef53d02999844060872ffd792b6f9c42e2"}}, "dynamic_argument_count": 1, "first_execution_ns": 722547, "input_digest": "80213a9335312652e35081fb882bda7a61bc3d451e21135b2dcc896f7c07f5d4", "matrix": false, "mode": "jax_jit", "staging_ns": 521183}`
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
| `case_id` | smoke/A05/axis0/tail |
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
  "case_sha256": "c6b06f228302cfd56df7c2122e05907ee5ddd1dbe524f0cd0efa3ed96ee6c372",
  "compile_ns": 28728974,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "e3049e88048d7deee48768a9052b5bbe4791ed37c82334e45cb8856261d1597a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "8fe5b7c239b1241f6c2a3a78068147ef53d02999844060872ffd792b6f9c42e2"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 722547,
  "input_digest": "80213a9335312652e35081fb882bda7a61bc3d451e21135b2dcc896f7c07f5d4",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 521183
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
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

- launch.extras: `{"case_sha256": "83ccff343e81fffb56685e75177f1e4499dcfe22a3963ef8edba020f5814b75e", "compile_ns": 28368545, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0031/optimized_hlo.txt", "sha256": "6cdc7b02464af94dbd48b2a19cfd9da3850b38855d7149e618640c2bea3d97e9"}, "stablehlo": {"path": "compiler/case-0031/stablehlo.txt", "sha256": "e13f0dff75a4a2a55e2319d73d12392a4b426fbfcb30001cca782b755b7fd4ba"}}, "dynamic_argument_count": 1, "first_execution_ns": 739050, "input_digest": "d72a040a10f1b69aaaf683c0e75444ccb4248d29f40eddfbf4a62e75b0d59ba3", "matrix": false, "mode": "jax_jit", "staging_ns": 545383}`
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
| `case_id` | smoke/A05/axis0/scale |
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
  "case_sha256": "83ccff343e81fffb56685e75177f1e4499dcfe22a3963ef8edba020f5814b75e",
  "compile_ns": 28368545,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6cdc7b02464af94dbd48b2a19cfd9da3850b38855d7149e618640c2bea3d97e9"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "e13f0dff75a4a2a55e2319d73d12392a4b426fbfcb30001cca782b755b7fd4ba"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 739050,
  "input_digest": "d72a040a10f1b69aaaf683c0e75444ccb4248d29f40eddfbf4a62e75b0d59ba3",
  "matrix": false,
  "mode": "jax_jit",
  "staging_ns": 545383
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "kind": "portable_core",
  "measure": false,
  "profile": "smoke",
  "start": 32,
  "stop": 64,
  "tier": 28
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `e8842bfe0749f1d1cb3a5536e5b70cb60724beaf62873bc4f4baacfaf9032999`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-002/out/tpu/portable-core-qualification-v1/55a254e8b83e1dfdc08645f615d1ad0b6ab9186ac19861361a9ed59792d0b369/native-restore-002-core28-001/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
