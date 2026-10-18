# Support and Roadmap {#linalg_support_roadmap}

For source organization and numerical layering, see the [architecture guide](ARCHITECTURE.md).

## Current support

| Capability | Status |
| --- | --- |
| Host CPU execution | Supported on validated compiler/toolchain combinations |
| Fixed-size dense matrices and vectors | Implemented and tested |
| CUDA device execution | Not supported or validated; under consideration |
| FPGA/HLS synthesis | Not supported or validated; under consideration |
| Sparse matrix storage and algorithms | Not implemented; under consideration |

## Remaining work

- **CUDA:** select toolchains and an API subset; audit device annotations and
  library calls; verify GPU kernels against independent numerical references.
  Host calculations in a CUDA project are distinct from device execution.
- **HLS:** select a tool/device and API subset; audit synthesizable constructs;
  verify simulation/co-simulation and report resources, latency, and timing.
- **Sparse:** choose formats and capacity/pattern semantics; implement validated
  construction, conversion, and products. Assess sparse solves, ordering, fill-in,
  and allocation/constexpr guarantees separately.

These extensions have no delivery commitment. Promote support only after the
API subset, toolchains, correctness checks, and target-specific results are recorded.
