# Akreon Linalg

Fixed-size linear algebra and numerical methods in C++26.

## Start here

- [Build, install, and examples](https://github.com/Akreon-LLC/akreon-linalg#readme)
- [Architecture](ARCHITECTURE.md): source organization and design.
- [Support and roadmap](ROADMAP.md): implemented capabilities and limitations.
- \ref linalg "API reference": types and functions.

## Capabilities

| Area | Purpose |
|---|---|
| Containers and operations | Vectors, matrices, products, blocks, and slices. |
| Spaces and maps | Symmetric coordinates, linear/affine transformations, norms, and inner products. |
| Numerical methods | QR, Cholesky, LDLᵀ, eigen decomposition, SVD, linear solves, and matrix functions. |

## C++ integration

```cpp
#include <akreon/linalg/linalg.hpp>
```

Link the `Akreon::Linalg` CMake target. The library is header-only; dimensions and
mathematical contracts are represented in C++ types.
