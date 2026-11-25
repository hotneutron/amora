# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-restore-002-collectives
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 18
- `fit_status`: `direct`=18
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 2, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 88527.9 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 328347719, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "577ee802ba78835a43a601d700081757cb54f2b39e8a93a0dc76a758fd436f6d"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "371e923860d105be702c3f5cead0efbebde75658890249b6234985ae04ea94a3"}}, "dynamic_argument_count": 1, "first_execution_ns": 1327339, "input_digest": "04012f09d325596f241a3e36349cdbecf60fa6a4854794fe93abf5810d561b73", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 2, "primitive": "psum", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 1024, "shape": [128], "staging_ns": 5512376}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/0 |
| `effective_bytes_per_second` | 1.1567e+07 |
| `probe_id` | interconnect.collectives |
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
        2,
        128
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
  "compile_ns": 328347719,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "577ee802ba78835a43a601d700081757cb54f2b39e8a93a0dc76a758fd436f6d"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "371e923860d105be702c3f5cead0efbebde75658890249b6234985ae04ea94a3"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1327339,
  "input_digest": "04012f09d325596f241a3e36349cdbecf60fa6a4854794fe93abf5810d561b73",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 2,
  "primitive": "psum",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 1024,
  "shape": [
    128
  ],
  "staging_ns": 5512376
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 2,
  "primitive": "psum",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 88527.875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 27416.0,
      "max": 1612769.0,
      "median": 1416446.0,
      "min": 1358692.0,
      "relative_mad": 0.019355485489739813
    },
    {
      "count": 30,
      "mad": 14615.5,
      "max": 1511975.0,
      "median": 1427425.0,
      "min": 1355924.0,
      "relative_mad": 0.010239066851148046
    },
    {
      "count": 30,
      "mad": 21580.5,
      "max": 1443039.0,
      "median": 1408393.5,
      "min": 1357788.0,
      "relative_mad": 0.015322777334601445
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 8052.5,
    "max": 1427425.0,
    "median": 1416446.0,
    "min": 1408393.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 2, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 86492.4 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 48551170, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "326314ffdff730608aab86f2acf1f6bdfeec06fcff69cc73ea1572cb6b4a73e4"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "9e4034dc8244c8f7a9930dee2f01322f7073ff126d15c134a1234a1479e7227c"}}, "dynamic_argument_count": 1, "first_execution_ns": 1100456, "input_digest": "04012f09d325596f241a3e36349cdbecf60fa6a4854794fe93abf5810d561b73", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 2, "primitive": "all_gather", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 1024, "shape": [128], "staging_ns": 803939}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/1 |
| `effective_bytes_per_second` | 1.18392e+07 |
| `probe_id` | interconnect.collectives |
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
        2,
        128
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
  "compile_ns": 48551170,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "326314ffdff730608aab86f2acf1f6bdfeec06fcff69cc73ea1572cb6b4a73e4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "9e4034dc8244c8f7a9930dee2f01322f7073ff126d15c134a1234a1479e7227c"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1100456,
  "input_digest": "04012f09d325596f241a3e36349cdbecf60fa6a4854794fe93abf5810d561b73",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 2,
  "primitive": "all_gather",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 1024,
  "shape": [
    128
  ],
  "staging_ns": 803939
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 2,
  "primitive": "all_gather",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 86492.40625,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 21747.5,
      "max": 1547721.0,
      "median": 1383878.5,
      "min": 1311585.0,
      "relative_mad": 0.015714891155545807
    },
    {
      "count": 30,
      "mad": 22075.5,
      "max": 1449218.0,
      "median": 1387341.0,
      "min": 1337849.0,
      "relative_mad": 0.015912093710198142
    },
    {
      "count": 30,
      "mad": 14987.5,
      "max": 1412170.0,
      "median": 1370573.5,
      "min": 1339742.0,
      "relative_mad": 0.010935203402079494
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 3462.5,
    "max": 1387341.0,
    "median": 1383878.5,
    "min": 1370573.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 2, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 86487.8 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 48272053, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "4af8f481f963cffd4e9f267a78a220bdff6eabeee5871042134c61bc29e559a9"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "9a3378e7189dad20038bad971e77e3e9c18fb7ef089a235f20b28703d86f0340"}}, "dynamic_argument_count": 1, "first_execution_ns": 954368, "input_digest": "04012f09d325596f241a3e36349cdbecf60fa6a4854794fe93abf5810d561b73", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 2, "primitive": "ppermute", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 1024, "routes": [[0, 1], [1, 0]], "shape": [128], "staging_ns": 695484}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/2 |
| `effective_bytes_per_second` | 1.18398e+07 |
| `probe_id` | interconnect.collectives |
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
        2,
        128
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
  "compile_ns": 48272053,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "4af8f481f963cffd4e9f267a78a220bdff6eabeee5871042134c61bc29e559a9"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "9a3378e7189dad20038bad971e77e3e9c18fb7ef089a235f20b28703d86f0340"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 954368,
  "input_digest": "04012f09d325596f241a3e36349cdbecf60fa6a4854794fe93abf5810d561b73",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 2,
  "primitive": "ppermute",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 1024,
  "routes": [
    [
      0,
      1
    ],
    [
      1,
      0
    ]
  ],
  "shape": [
    128
  ],
  "staging_ns": 695484
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 2,
  "primitive": "ppermute",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 86487.78125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 19173.0,
      "max": 1439177.0,
      "median": 1383804.5,
      "min": 1294925.0,
      "relative_mad": 0.013855280857953562
    },
    {
      "count": 30,
      "mad": 25589.5,
      "max": 2402342.0,
      "median": 1388035.0,
      "min": 1331809.0,
      "relative_mad": 0.018435774314048275
    },
    {
      "count": 30,
      "mad": 17668.5,
      "max": 1448962.0,
      "median": 1378279.0,
      "min": 1331077.0,
      "relative_mad": 0.012819247772040349
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 4230.5,
    "max": 1388035.0,
    "median": 1383804.5,
    "min": 1378279.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 2, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 85646.1 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 58724744, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "887ebd2719e6fdaf28101192fc550f15325b427afdab5b85154a334cecb108b2"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "b04ffea95546c3616cfc7217c5a0b854a6ebb607d688010923af83a7de9db7e4"}}, "dynamic_argument_count": 1, "first_execution_ns": 1025885, "input_digest": "cbe447f1028b687c648fe65f18adbfd22e9d3d56a61aa83a0b30bf848960a42c", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 2, "primitive": "psum", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 524288, "shape": [65536], "staging_ns": 2342763}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/3 |
| `effective_bytes_per_second` | 6.12156e+09 |
| `probe_id` | interconnect.collectives |
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
        2,
        65536
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
  "compile_ns": 58724744,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "887ebd2719e6fdaf28101192fc550f15325b427afdab5b85154a334cecb108b2"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b04ffea95546c3616cfc7217c5a0b854a6ebb607d688010923af83a7de9db7e4"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1025885,
  "input_digest": "cbe447f1028b687c648fe65f18adbfd22e9d3d56a61aa83a0b30bf848960a42c",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 2,
  "primitive": "psum",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 524288,
  "shape": [
    65536
  ],
  "staging_ns": 2342763
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 2,
  "primitive": "psum",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 85646.125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 18403.5,
      "max": 1406158.0,
      "median": 1371186.5,
      "min": 1315961.0,
      "relative_mad": 0.013421587800054915
    },
    {
      "count": 30,
      "mad": 16665.0,
      "max": 1397127.0,
      "median": 1345393.0,
      "min": 1303733.0,
      "relative_mad": 0.012386715257177643
    },
    {
      "count": 30,
      "mad": 18326.0,
      "max": 1439995.0,
      "median": 1370338.0,
      "min": 1305282.0,
      "relative_mad": 0.013373342927073466
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 848.5,
    "max": 1371186.5,
    "median": 1370338.0,
    "min": 1345393.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 2, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 86055.8 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 88563325, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "1dd0c647d02759161b90f506bddb6cefc689f9a6577e357ae18712e12f6ffca4"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "313c4e53c997617ab7d4deaec9e8688f006da6e1f1eba2284b6c661fa7ae36e3"}}, "dynamic_argument_count": 1, "first_execution_ns": 1000426, "input_digest": "cbe447f1028b687c648fe65f18adbfd22e9d3d56a61aa83a0b30bf848960a42c", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 2, "primitive": "all_gather", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 524288, "shape": [65536], "staging_ns": 2119406}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/4 |
| `effective_bytes_per_second` | 6.09242e+09 |
| `probe_id` | interconnect.collectives |
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
        2,
        65536
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
  "compile_ns": 88563325,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "1dd0c647d02759161b90f506bddb6cefc689f9a6577e357ae18712e12f6ffca4"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "313c4e53c997617ab7d4deaec9e8688f006da6e1f1eba2284b6c661fa7ae36e3"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1000426,
  "input_digest": "cbe447f1028b687c648fe65f18adbfd22e9d3d56a61aa83a0b30bf848960a42c",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 2,
  "primitive": "all_gather",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 524288,
  "shape": [
    65536
  ],
  "staging_ns": 2119406
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 2,
  "primitive": "all_gather",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 86055.8125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 22001.5,
      "max": 1449753.0,
      "median": 1376893.0,
      "min": 1323892.0,
      "relative_mad": 0.015979092057262258
    },
    {
      "count": 30,
      "mad": 16885.5,
      "max": 1420043.0,
      "median": 1375198.0,
      "min": 1299147.0,
      "relative_mad": 0.012278595518608956
    },
    {
      "count": 30,
      "mad": 17625.0,
      "max": 1444428.0,
      "median": 1379226.0,
      "min": 1333530.0,
      "relative_mad": 0.012778906430128202
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 1695.0,
    "max": 1379226.0,
    "median": 1376893.0,
    "min": 1375198.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 2, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 85021.7 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 47393728, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "800372f0700cae2e723a248e2472ed7086a178e3dd29c1372a78e48b4e64f0b5"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "abda444e994a4cbd0c39fcf1371407d9a341d6b2291999725d8305c2122a0be7"}}, "dynamic_argument_count": 1, "first_execution_ns": 1051882, "input_digest": "cbe447f1028b687c648fe65f18adbfd22e9d3d56a61aa83a0b30bf848960a42c", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 2, "primitive": "ppermute", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 524288, "routes": [[0, 1], [1, 0]], "shape": [65536], "staging_ns": 2152729}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/5 |
| `effective_bytes_per_second` | 6.16652e+09 |
| `probe_id` | interconnect.collectives |
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
        2,
        65536
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
  "compile_ns": 47393728,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "800372f0700cae2e723a248e2472ed7086a178e3dd29c1372a78e48b4e64f0b5"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "abda444e994a4cbd0c39fcf1371407d9a341d6b2291999725d8305c2122a0be7"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1051882,
  "input_digest": "cbe447f1028b687c648fe65f18adbfd22e9d3d56a61aa83a0b30bf848960a42c",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 2,
  "primitive": "ppermute",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 524288,
  "routes": [
    [
      0,
      1
    ],
    [
      1,
      0
    ]
  ],
  "shape": [
    65536
  ],
  "staging_ns": 2152729
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 2,
  "primitive": "ppermute",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 85021.6875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 20041.0,
      "max": 1442191.0,
      "median": 1366489.0,
      "min": 1303218.0,
      "relative_mad": 0.0146660529283441
    },
    {
      "count": 30,
      "mad": 18652.5,
      "max": 1414231.0,
      "median": 1360347.0,
      "min": 1312367.0,
      "relative_mad": 0.01371157506136302
    },
    {
      "count": 30,
      "mad": 17269.0,
      "max": 1433301.0,
      "median": 1357026.5,
      "min": 1320594.0,
      "relative_mad": 0.012725617370036621
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 3320.5,
    "max": 1366489.0,
    "median": 1360347.0,
    "min": 1357026.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 4, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 137463 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 44944261, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "b6ba64b62af5e8687fb253a40f727b609254952513541e8e20f8b5987ec107e2"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "db7af58e6ff6d6c791562689e38eb3cd0dae134031aa189d7364451dcddc5c7c"}}, "dynamic_argument_count": 1, "first_execution_ns": 1053528, "input_digest": "e60e5245b92eec993db5f53da7b2585f01bc20a2f01e2fe3294adfa09e7a1bef", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 4, "primitive": "psum", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 2048, "shape": [128], "staging_ns": 990568}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/6 |
| `effective_bytes_per_second` | 1.48985e+07 |
| `probe_id` | interconnect.collectives |
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
        4,
        128
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
  "compile_ns": 44944261,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "b6ba64b62af5e8687fb253a40f727b609254952513541e8e20f8b5987ec107e2"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "db7af58e6ff6d6c791562689e38eb3cd0dae134031aa189d7364451dcddc5c7c"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1053528,
  "input_digest": "e60e5245b92eec993db5f53da7b2585f01bc20a2f01e2fe3294adfa09e7a1bef",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 4,
  "primitive": "psum",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 2048,
  "shape": [
    128
  ],
  "staging_ns": 990568
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 4,
  "primitive": "psum",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 137463.0625,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 23208.0,
      "max": 2281136.0,
      "median": 2190909.5,
      "min": 2090869.0,
      "relative_mad": 0.010592861092619298
    },
    {
      "count": 30,
      "mad": 19482.5,
      "max": 2323375.0,
      "median": 2214816.0,
      "min": 2115953.0,
      "relative_mad": 0.008796441781168278
    },
    {
      "count": 30,
      "mad": 31306.0,
      "max": 2301779.0,
      "median": 2199409.0,
      "min": 2086356.0,
      "relative_mad": 0.014233823722645492
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 8499.5,
    "max": 2214816.0,
    "median": 2199409.0,
    "min": 2190909.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 4, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 138342 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 43277723, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "cf1f1432c5c3b922f9a3f1b7278110bb03bdd9fd2343331c388a4f3d0c841411"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "0e7413fe6defbe5fffdd71b32b488bdad97f6f17c850101922572ac0ff16adf3"}}, "dynamic_argument_count": 1, "first_execution_ns": 1164671, "input_digest": "e60e5245b92eec993db5f53da7b2585f01bc20a2f01e2fe3294adfa09e7a1bef", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 4, "primitive": "all_gather", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 2048, "shape": [128], "staging_ns": 747519}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/7 |
| `effective_bytes_per_second` | 1.48039e+07 |
| `probe_id` | interconnect.collectives |
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
        4,
        128
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
  "compile_ns": 43277723,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "cf1f1432c5c3b922f9a3f1b7278110bb03bdd9fd2343331c388a4f3d0c841411"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "0e7413fe6defbe5fffdd71b32b488bdad97f6f17c850101922572ac0ff16adf3"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1164671,
  "input_digest": "e60e5245b92eec993db5f53da7b2585f01bc20a2f01e2fe3294adfa09e7a1bef",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 4,
  "primitive": "all_gather",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 2048,
  "shape": [
    128
  ],
  "staging_ns": 747519
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 4,
  "primitive": "all_gather",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 138342.09375,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 45475.0,
      "max": 3969685.0,
      "median": 2238820.5,
      "min": 2084236.0,
      "relative_mad": 0.020312034841560547
    },
    {
      "count": 30,
      "mad": 28173.0,
      "max": 2307448.0,
      "median": 2213473.5,
      "min": 2105250.0,
      "relative_mad": 0.01272795902006507
    },
    {
      "count": 30,
      "mad": 26746.0,
      "max": 2324995.0,
      "median": 2210020.5,
      "min": 2115853.0,
      "relative_mad": 0.012102150183674767
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 3453.0,
    "max": 2238820.5,
    "median": 2213473.5,
    "min": 2210020.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 4, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 137711 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 45029882, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0008/optimized_hlo.txt", "sha256": "2e23cb6eb09b02190ff27a33764fbe49545126d5664864cf231465a87c540c87"}, "stablehlo": {"path": "compiler/case-0008/stablehlo.txt", "sha256": "6b347f24f496cf71d6c8e172a2db7346171a7d77acc38d2148b8e191e5fa838d"}}, "dynamic_argument_count": 1, "first_execution_ns": 1203387, "input_digest": "e60e5245b92eec993db5f53da7b2585f01bc20a2f01e2fe3294adfa09e7a1bef", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 4, "primitive": "ppermute", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 2048, "routes": [[0, 1], [1, 2], [2, 3], [3, 0]], "shape": [128], "staging_ns": 790090}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/8 |
| `effective_bytes_per_second` | 1.48717e+07 |
| `probe_id` | interconnect.collectives |
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
        4,
        128
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
  "compile_ns": 45029882,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "2e23cb6eb09b02190ff27a33764fbe49545126d5664864cf231465a87c540c87"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6b347f24f496cf71d6c8e172a2db7346171a7d77acc38d2148b8e191e5fa838d"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1203387,
  "input_digest": "e60e5245b92eec993db5f53da7b2585f01bc20a2f01e2fe3294adfa09e7a1bef",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 4,
  "primitive": "ppermute",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 2048,
  "routes": [
    [
      0,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      3
    ],
    [
      3,
      0
    ]
  ],
  "shape": [
    128
  ],
  "staging_ns": 790090
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 4,
  "primitive": "ppermute",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 137711.4375,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 41533.5,
      "max": 2319459.0,
      "median": 2207529.5,
      "min": 2012760.0,
      "relative_mad": 0.01881447110899311
    },
    {
      "count": 30,
      "mad": 28615.0,
      "max": 2274107.0,
      "median": 2180459.0,
      "min": 2124046.0,
      "relative_mad": 0.01312338365454246
    },
    {
      "count": 30,
      "mad": 24774.5,
      "max": 2286800.0,
      "median": 2203383.0,
      "min": 2140750.0,
      "relative_mad": 0.011243846394385361
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 4146.5,
    "max": 2207529.5,
    "median": 2203383.0,
    "min": 2180459.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 4, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 138589 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 68663341, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "ba7a5e4036f71b286c927f0c6a4c4160f572e8cea3b22dce5a7b71a68318fe32"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "793669c71026c0c2eb923a32a78b494907282cf19ab8814e2d21814ea5c95e29"}}, "dynamic_argument_count": 1, "first_execution_ns": 1117911, "input_digest": "289820c9229c27f0663ddc32393c99fc89a60a775b73ce162e338ac911b058b4", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 4, "primitive": "psum", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 1048576, "shape": [65536], "staging_ns": 3450328}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/9 |
| `effective_bytes_per_second` | 7.56607e+09 |
| `probe_id` | interconnect.collectives |
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
        4,
        65536
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
  "compile_ns": 68663341,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ba7a5e4036f71b286c927f0c6a4c4160f572e8cea3b22dce5a7b71a68318fe32"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "793669c71026c0c2eb923a32a78b494907282cf19ab8814e2d21814ea5c95e29"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1117911,
  "input_digest": "289820c9229c27f0663ddc32393c99fc89a60a775b73ce162e338ac911b058b4",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 4,
  "primitive": "psum",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 1048576,
  "shape": [
    65536
  ],
  "staging_ns": 3450328
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 4,
  "primitive": "psum",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 138589.3125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 34722.5,
      "max": 2307204.0,
      "median": 2217429.0,
      "min": 1999437.0,
      "relative_mad": 0.015658900465358755
    },
    {
      "count": 30,
      "mad": 30636.5,
      "max": 2298877.0,
      "median": 2221595.0,
      "min": 2124521.0,
      "relative_mad": 0.01379031731706274
    },
    {
      "count": 30,
      "mad": 29363.0,
      "max": 4200156.0,
      "median": 2214961.5,
      "min": 2132932.0,
      "relative_mad": 0.013256663829145563
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 2467.5,
    "max": 2221595.0,
    "median": 2217429.0,
    "min": 2214961.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 4, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 138663 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 128449411, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "8d6c38db0adf6145e00f9a0d37830691db4be2141829b60638f53124bb694805"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "6a4bb4ad1f2d683854e775e6473b85e3757cadc0ea4f25609858e6590383d62b"}}, "dynamic_argument_count": 1, "first_execution_ns": 1358001, "input_digest": "289820c9229c27f0663ddc32393c99fc89a60a775b73ce162e338ac911b058b4", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 4, "primitive": "all_gather", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 1048576, "shape": [65536], "staging_ns": 3851296}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/10 |
| `effective_bytes_per_second` | 7.56204e+09 |
| `probe_id` | interconnect.collectives |
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
        4,
        65536
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
  "compile_ns": 128449411,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8d6c38db0adf6145e00f9a0d37830691db4be2141829b60638f53124bb694805"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6a4bb4ad1f2d683854e775e6473b85e3757cadc0ea4f25609858e6590383d62b"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1358001,
  "input_digest": "289820c9229c27f0663ddc32393c99fc89a60a775b73ce162e338ac911b058b4",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 4,
  "primitive": "all_gather",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 1048576,
  "shape": [
    65536
  ],
  "staging_ns": 3851296
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 4,
  "primitive": "all_gather",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 138663.1875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 22925.5,
      "max": 2340368.0,
      "median": 2218611.0,
      "min": 2132214.0,
      "relative_mad": 0.010333267075661304
    },
    {
      "count": 30,
      "mad": 31189.0,
      "max": 2344823.0,
      "median": 2227284.0,
      "min": 2142646.0,
      "relative_mad": 0.0140031536166919
    },
    {
      "count": 30,
      "mad": 28969.0,
      "max": 2283586.0,
      "median": 2218453.5,
      "min": 2113427.0,
      "relative_mad": 0.01305819572057742
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 157.5,
    "max": 2227284.0,
    "median": 2218611.0,
    "min": 2218453.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 4, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 137823 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 52226420, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "8c73026a3027607e099826191278e77782fa5a56244c7ae21f657e99e884c4c0"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "6d320a1b52a0611d5d31d0aa9411dbe79c746a162f931bf180a88fc1aac5fa4f"}}, "dynamic_argument_count": 1, "first_execution_ns": 1237825, "input_digest": "289820c9229c27f0663ddc32393c99fc89a60a775b73ce162e338ac911b058b4", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 4, "primitive": "ppermute", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 1048576, "routes": [[0, 1], [1, 2], [2, 3], [3, 0]], "shape": [65536], "staging_ns": 3310639}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/11 |
| `effective_bytes_per_second` | 7.60811e+09 |
| `probe_id` | interconnect.collectives |
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
        4,
        65536
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
  "compile_ns": 52226420,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8c73026a3027607e099826191278e77782fa5a56244c7ae21f657e99e884c4c0"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6d320a1b52a0611d5d31d0aa9411dbe79c746a162f931bf180a88fc1aac5fa4f"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1237825,
  "input_digest": "289820c9229c27f0663ddc32393c99fc89a60a775b73ce162e338ac911b058b4",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 4,
  "primitive": "ppermute",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 1048576,
  "routes": [
    [
      0,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      3
    ],
    [
      3,
      0
    ]
  ],
  "shape": [
    65536
  ],
  "staging_ns": 3310639
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 4,
  "primitive": "ppermute",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 137823.4375,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 27880.5,
      "max": 2252162.0,
      "median": 2202993.0,
      "min": 2122840.0,
      "relative_mad": 0.01265573699053969
    },
    {
      "count": 30,
      "mad": 30539.0,
      "max": 2308630.0,
      "median": 2207861.5,
      "min": 2113920.0,
      "relative_mad": 0.013831936468841004
    },
    {
      "count": 30,
      "mad": 20556.0,
      "max": 2272707.0,
      "median": 2205175.0,
      "min": 2150146.0,
      "relative_mad": 0.009321709161404424
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 2182.0,
    "max": 2207861.5,
    "median": 2205175.0,
    "min": 2202993.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 224780 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 48161051, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0012/optimized_hlo.txt", "sha256": "80a27b79dfebdf7b70084d0fc55c88262514f0a81ea8f70378190f6eb3497749"}, "stablehlo": {"path": "compiler/case-0012/stablehlo.txt", "sha256": "1cfe0acb37125b1eb475cd2a7df2cdbf1cab70e279770fbe60a4f4b3bcc6e376"}}, "dynamic_argument_count": 1, "first_execution_ns": 1492310, "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 8, "primitive": "psum", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 4096, "shape": [128], "staging_ns": 1173394}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/12 |
| `effective_bytes_per_second` | 1.82223e+07 |
| `probe_id` | interconnect.collectives |
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
        128
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
  "compile_ns": 48161051,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "80a27b79dfebdf7b70084d0fc55c88262514f0a81ea8f70378190f6eb3497749"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1cfe0acb37125b1eb475cd2a7df2cdbf1cab70e279770fbe60a4f4b3bcc6e376"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1492310,
  "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 8,
  "primitive": "psum",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 4096,
  "shape": [
    128
  ],
  "staging_ns": 1173394
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "primitive": "psum",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 224780.0625,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 40775.5,
      "max": 3771683.0,
      "median": 3603824.0,
      "min": 3379987.0,
      "relative_mad": 0.011314509254613988
    },
    {
      "count": 30,
      "mad": 25864.0,
      "max": 3722331.0,
      "median": 3596481.0,
      "min": 3435453.0,
      "relative_mad": 0.007191474110387348
    },
    {
      "count": 30,
      "mad": 57289.0,
      "max": 3748992.0,
      "median": 3543207.0,
      "min": 3442159.0,
      "relative_mad": 0.01616868560036148
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 7343.0,
    "max": 3603824.0,
    "median": 3596481.0,
    "min": 3543207.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 219242 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 362608650, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0013/optimized_hlo.txt", "sha256": "8c88c4c355034a8c5947695f5d82620b44b962a6c8aff01cd432f3a3cf0a250c"}, "stablehlo": {"path": "compiler/case-0013/stablehlo.txt", "sha256": "7fa03db27220f4d32a9ad1f778983c079789b45f5b5b015d7ff6bb93ee8c4248"}}, "dynamic_argument_count": 1, "first_execution_ns": 1146660, "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 8, "primitive": "all_gather", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 4096, "shape": [128], "staging_ns": 840909}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/13 |
| `effective_bytes_per_second` | 1.86825e+07 |
| `probe_id` | interconnect.collectives |
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
        128
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
  "compile_ns": 362608650,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8c88c4c355034a8c5947695f5d82620b44b962a6c8aff01cd432f3a3cf0a250c"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7fa03db27220f4d32a9ad1f778983c079789b45f5b5b015d7ff6bb93ee8c4248"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1146660,
  "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 8,
  "primitive": "all_gather",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 4096,
  "shape": [
    128
  ],
  "staging_ns": 840909
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "primitive": "all_gather",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 219242.21875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 63672.5,
      "max": 3648493.0,
      "median": 3462160.5,
      "min": 3281358.0,
      "relative_mad": 0.018390972919944064
    },
    {
      "count": 30,
      "mad": 63307.0,
      "max": 3683985.0,
      "median": 3511640.0,
      "min": 3341961.0,
      "relative_mad": 0.01802775910970373
    },
    {
      "count": 30,
      "mad": 58348.5,
      "max": 3975197.0,
      "median": 3507875.5,
      "min": 3314710.0,
      "relative_mad": 0.01663357208658061
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 3764.5,
    "max": 3511640.0,
    "median": 3507875.5,
    "min": 3462160.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 225846 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 44712720, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0014/optimized_hlo.txt", "sha256": "9162987a5e4e053f3f54e3d5dfe8667163317863b3e8271546911fee84619a26"}, "stablehlo": {"path": "compiler/case-0014/stablehlo.txt", "sha256": "46524df99c2ecb0375c9b47af93f54ace87574618ba763c71c22733ef08325e6"}}, "dynamic_argument_count": 1, "first_execution_ns": 1210798, "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 8, "primitive": "ppermute", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 4096, "routes": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]], "shape": [128], "staging_ns": 932481}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/14 |
| `effective_bytes_per_second` | 1.81362e+07 |
| `probe_id` | interconnect.collectives |
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
        128
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
  "compile_ns": 44712720,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "9162987a5e4e053f3f54e3d5dfe8667163317863b3e8271546911fee84619a26"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "46524df99c2ecb0375c9b47af93f54ace87574618ba763c71c22733ef08325e6"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1210798,
  "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 8,
  "primitive": "ppermute",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 4096,
  "routes": [
    [
      0,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      3
    ],
    [
      3,
      4
    ],
    [
      4,
      5
    ],
    [
      5,
      6
    ],
    [
      6,
      7
    ],
    [
      7,
      0
    ]
  ],
  "shape": [
    128
  ],
  "staging_ns": 932481
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "primitive": "ppermute",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 225846.25,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 58384.0,
      "max": 4354866.0,
      "median": 3601996.0,
      "min": 3310383.0,
      "relative_mad": 0.016208790903710055
    },
    {
      "count": 30,
      "mad": 47050.5,
      "max": 3792853.0,
      "median": 3613540.0,
      "min": 3465381.0,
      "relative_mad": 0.013020611367246523
    },
    {
      "count": 30,
      "mad": 37675.0,
      "max": 5310540.0,
      "median": 3620296.0,
      "min": 3503410.0,
      "relative_mad": 0.010406607636502651
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 6756.0,
    "max": 3620296.0,
    "median": 3613540.0,
    "min": 3601996.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 205167 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 711574546, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0015/optimized_hlo.txt", "sha256": "cd064c9e89295f64488f60c4fa408f0aab7399742b0980fa931918bc395e2a7b"}, "stablehlo": {"path": "compiler/case-0015/stablehlo.txt", "sha256": "3ccdc5e4aa42ea129b570bae07fa030518d426a48b34c2b3146b2cbd5c5b8894"}}, "dynamic_argument_count": 1, "first_execution_ns": 1709141, "input_digest": "64129a9f1182bbfedcbb298a006c5501d54dc690232aa34534ac91453d89f702", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 8, "primitive": "psum", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 2097152, "shape": [65536], "staging_ns": 5868921}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/15 |
| `effective_bytes_per_second` | 1.02217e+10 |
| `probe_id` | interconnect.collectives |
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
        65536
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
  "compile_ns": 711574546,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "cd064c9e89295f64488f60c4fa408f0aab7399742b0980fa931918bc395e2a7b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3ccdc5e4aa42ea129b570bae07fa030518d426a48b34c2b3146b2cbd5c5b8894"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1709141,
  "input_digest": "64129a9f1182bbfedcbb298a006c5501d54dc690232aa34534ac91453d89f702",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 8,
  "primitive": "psum",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 2097152,
  "shape": [
    65536
  ],
  "staging_ns": 5868921
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "primitive": "psum",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 205166.875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 81651.0,
      "max": 3760012.0,
      "median": 3229308.5,
      "min": 3029272.0,
      "relative_mad": 0.02528436041338262
    },
    {
      "count": 30,
      "mad": 81757.5,
      "max": 4560584.0,
      "median": 3282670.0,
      "min": 3071327.0,
      "relative_mad": 0.02490579315008819
    },
    {
      "count": 30,
      "mad": 110068.5,
      "max": 3981858.0,
      "median": 3318415.0,
      "min": 3116990.0,
      "relative_mad": 0.033168997849877126
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 35745.0,
    "max": 3318415.0,
    "median": 3282670.0,
    "min": 3229308.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 217317 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 544592987, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0016/optimized_hlo.txt", "sha256": "943877633d6ac36d37ea4ef03f258e1a5fdc0d1ce7e2590203517264c161b576"}, "stablehlo": {"path": "compiler/case-0016/stablehlo.txt", "sha256": "d9df88843e47b79fc9b2a9af901f96cfd852f9526771d672211ab06b9ea98860"}}, "dynamic_argument_count": 1, "first_execution_ns": 1479299, "input_digest": "64129a9f1182bbfedcbb298a006c5501d54dc690232aa34534ac91453d89f702", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 8, "primitive": "all_gather", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 2097152, "shape": [65536], "staging_ns": 6032783}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/16 |
| `effective_bytes_per_second` | 9.65019e+09 |
| `probe_id` | interconnect.collectives |
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
        65536
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
  "compile_ns": 544592987,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "943877633d6ac36d37ea4ef03f258e1a5fdc0d1ce7e2590203517264c161b576"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "d9df88843e47b79fc9b2a9af901f96cfd852f9526771d672211ab06b9ea98860"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1479299,
  "input_digest": "64129a9f1182bbfedcbb298a006c5501d54dc690232aa34534ac91453d89f702",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 8,
  "primitive": "all_gather",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 2097152,
  "shape": [
    65536
  ],
  "staging_ns": 6032783
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "primitive": "all_gather",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 217317.125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 42846.5,
      "max": 3985773.0,
      "median": 3478946.5,
      "min": 3304000.0,
      "relative_mad": 0.012315941047095724
    },
    {
      "count": 30,
      "mad": 33881.0,
      "max": 3597897.0,
      "median": 3435720.5,
      "min": 3328203.0,
      "relative_mad": 0.009861395884793307
    },
    {
      "count": 30,
      "mad": 61463.0,
      "max": 5174226.0,
      "median": 3477074.0,
      "min": 3273263.0,
      "relative_mad": 0.017676644212921554
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 1872.5,
    "max": 3478946.5,
    "median": 3477074.0,
    "min": 3435720.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [65536]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 223945 ns |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 48944337, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0017/optimized_hlo.txt", "sha256": "6b1f25965986df24f14db9346a04c36a7979630fe8ed6f1d97c4bcde9973d033"}, "stablehlo": {"path": "compiler/case-0017/stablehlo.txt", "sha256": "89bfa05f5982cbd769e10277733c24d5720d8e646e9b74bb9d6066c99db07003"}}, "dynamic_argument_count": 1, "first_execution_ns": 1125902, "input_digest": "64129a9f1182bbfedcbb298a006c5501d54dc690232aa34534ac91453d89f702", "kind": "collective", "matrix": false, "mode": "shard_map", "participants": 8, "primitive": "ppermute", "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86", "requested_bytes": 2097152, "routes": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]], "shape": [65536], "staging_ns": 6205723}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | interconnect.collectives/17 |
| `effective_bytes_per_second` | 9.36457e+09 |
| `probe_id` | interconnect.collectives |
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
        65536
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
  "compile_ns": 48944337,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6b1f25965986df24f14db9346a04c36a7979630fe8ed6f1d97c4bcde9973d033"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "89bfa05f5982cbd769e10277733c24d5720d8e646e9b74bb9d6066c99db07003"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1125902,
  "input_digest": "64129a9f1182bbfedcbb298a006c5501d54dc690232aa34534ac91453d89f702",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participants": 8,
  "primitive": "ppermute",
  "probe_source_sha256": "0d4bf8170374e240d82ffa995e711caa51b91f47d73160d11e0786d9950a3d86",
  "requested_bytes": 2097152,
  "routes": [
    [
      0,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      3
    ],
    [
      3,
      4
    ],
    [
      4,
      5
    ],
    [
      5,
      6
    ],
    [
      6,
      7
    ],
    [
      7,
      0
    ]
  ],
  "shape": [
    65536
  ],
  "staging_ns": 6205723
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "primitive": "ppermute",
  "probe_id": "interconnect.collectives",
  "shape": [
    65536
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 16,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 223945.3125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 41828.0,
      "max": 3662591.0,
      "median": 3583125.0,
      "min": 3409006.0,
      "relative_mad": 0.011673608930751788
    },
    {
      "count": 30,
      "mad": 40695.0,
      "max": 3781622.0,
      "median": 3562800.0,
      "min": 3421314.0,
      "relative_mad": 0.011422196025597844
    },
    {
      "count": 30,
      "mad": 34028.5,
      "max": 3700153.0,
      "median": 3589918.5,
      "min": 3458031.0,
      "relative_mad": 0.009478905997448132
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 6793.5,
    "max": 3589918.5,
    "median": 3583125.0,
    "min": 3562800.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `cad07e424af93e6963b0981ffe1c6a17ffcca3e2cd0c92d87f349c21feb1500e`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-002/out/tpu/tpu-collective-sweep-v1/9b5a8040d51a329f64ca42978f5708dce9fd7ffd05218fcf49ce5dc917414852/native-restore-002-collectives/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
