# akreon-linalg

Header-only C++26 fixed-size linear algebra and numerical methods. Production dependencies are limited to the C++ standard library.

[Documentation](https://akreon-llc.github.io/akreon-linalg/) · [Architecture](docs/ARCHITECTURE.md) · [Support and roadmap](docs/ROADMAP.md) · [C++ example](examples/consumer/main.cpp) · [Releases](https://github.com/Akreon-LLC/akreon-linalg/releases)

[![DOI](https://zenodo.org/badge/1423703658.svg)](https://zenodo.org/badge/latestdoi/1423703658)

## Capabilities

- Inline vector and row-major matrix storage with compile-time dimensions; arithmetic, products, blocks, and slices.
- Mathematical spaces, symmetric coordinates, linear/affine maps, inner products, and norms.
- QR, Cholesky, LDLᵀ, eigen decomposition, SVD, linear solves, and matrix functions.
- Constexpr-capable construction and operations where supported.

Storage, mathematical contracts, and numerical algorithms are separate abstractions. Inspect numerical events/status where provided; fixed-size storage does not establish execution-time guarantees.

See the [consumer example](examples/consumer/main.cpp) for usage. CUDA, FPGA/HLS, and sparse algorithms are not currently supported or validated; details are in the [roadmap](docs/ROADMAP.md).

## Build and test

Requirements:

- CMake 3.30 or newer.
- A C++26-capable compiler with the required constexpr math support.

Build from the repository:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The `host-default` and `host-functional-only` CMake presets are also available.

Unit and functional tests default to enabled when Linalg is the top-level CMake project. They can be disabled using:

```sh
-DBUILD_UNIT_TESTS=OFF -DBUILD_FUNCTIONAL_TESTS=OFF
```

### Dependencies

Tests use Catch2 3, resolved from an installed package or fetched at version `v3.8.1`.

For offline tests, provide a local Catch2 checkout:

```sh
-DFETCHCONTENT_SOURCE_DIR_CATCH2=/absolute/path/to/Catch2
```

## Use in your CMake project

The public target is `Akreon::Linalg`, exported through the `AkreonLinalg` package.

### Installed package

Install the library:

```sh
cmake --install build --prefix /absolute/path/akreon-install
```

Consume it in another project:

```cmake
find_package(AkreonLinalg 0.1 CONFIG REQUIRED)

add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE Akreon::Linalg)
```

Configure your consuming project with:

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/absolute/path/akreon-install
```

### CMake FetchContent

Alternatively, consume a pinned release directly:

```cmake
include(FetchContent)

FetchContent_Declare(
    akreon_linalg
    GIT_REPOSITORY https://github.com/Akreon-LLC/akreon-linalg.git
    GIT_TAG v0.1.0
)

FetchContent_MakeAvailable(akreon_linalg)

target_link_libraries(my_application PRIVATE Akreon::Linalg)
```

Public headers use canonical `akreon/...` paths:

```cpp
#include <akreon/linalg/linalg.hpp>
```

The CMake target propagates its include directories and C++ standard requirement.

A separate [consumer example](examples/consumer) demonstrates installed-package integration.

## Documentation and verification

See [architecture](docs/ARCHITECTURE.md), [support and roadmap](docs/ROADMAP.md), [contributing](CONTRIBUTING.md), and [security](SECURITY.md).

Generate API documentation with Doxygen and Graphviz installed:

```sh
doxygen Doxyfile
```

Open `docs/doxygen/html/index.html`. Tests cover implemented cases, not every dimension or conditioning regime.

## Citation and license

For research citation, use [CITATION.cff](CITATION.cff). Licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
