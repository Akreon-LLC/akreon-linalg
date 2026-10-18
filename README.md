# akreon-linalg

Fixed-size vectors and matrices, constexpr scalar primitives, affine maps, spaces, norms, QR, SVD, eigen decomposition, matrix exponentials, Cholesky and LDL factorization.

Header-only C++26 library. Public target: `Akreon::Linalg`; installed package:
`AkreonLinalg`.
Public APIs use `linalg`, with factorization settings in `config`.
Optimization domains and cones belong to akreon-convex.
Public headers live under `include/akreon/linalg/`. Only canonical `akreon/...` include paths are supported.

## Compatibility

For toolchain and target compatibility, storage support, and planned
extensions, see the [support details and roadmap](docs/ROADMAP.md).

## Build and test

Use CMake 3.30+ (for the exported `cxx_std_26` requirement) and a C++26-capable compiler with constexpr math support.
From this checkout:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

`host-default` and `host-functional-only` presets are also available. Tests default on only for
the top-level project. Disable tests using `-DBUILD_UNIT_TESTS=OFF
-DBUILD_FUNCTIONAL_TESTS=OFF`.

## Dependencies

The library has no dependencies beyond the C++ standard library. Unit tests
use installed Catch2 3 or fetch v3.8.1. For offline testing, pass
`-DFETCHCONTENT_SOURCE_DIR_CATCH2=/absolute/path/to/Catch2`.

## Install and consume

The [consumer example](examples/consumer) shows how a separate CMake project
uses the installed library. After building above, install the package, then
configure, build, and test that example:

```sh
cmake --install build --prefix /absolute/path/akreon-install
cmake -S examples/consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH=/absolute/path/akreon-install
cmake --build build/consumer
ctest --test-dir build/consumer --output-on-failure
```

The example's [CMakeLists.txt](examples/consumer/CMakeLists.txt) finds the package
and links its `consumer` executable:

```cmake
find_package(AkreonLinalg 0.1 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Akreon::Linalg)
```

Its [main.cpp](examples/consumer/main.cpp) includes `<akreon/linalg/linalg.hpp>`
and uses a fixed-size vector. Follow the same pattern in your own project,
setting `CMAKE_PREFIX_PATH` to the installation prefix. Include directories and
the C++ standard requirement propagate through `Akreon::Linalg`.

## Documentation

The [architecture guide](docs/ARCHITECTURE.md) is the Doxygen landing page.
Contributor workflow is in [CONTRIBUTING.md](CONTRIBUTING.md); test coverage is
in [tests/README.md](tests/README.md).
API and algorithm documentation is generated from the header comments. To build
both together, install Doxygen and Graphviz, then run from this repository:

```sh
doxygen Doxyfile
```

Open `docs/doxygen/html/index.html`. The same generated HTML directory can be
published to GitHub Pages; no separate copy of the architecture guide is needed.
