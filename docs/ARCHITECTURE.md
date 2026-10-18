# Akreon Linalg Architecture

This guide describes how the numerical layers fit together. For target/storage
compatibility and future work, see the [support roadmap](ROADMAP.md).

## Source organization

Paths below are relative to `include/akreon/linalg/`.

| Path | Responsibility |
| --- | --- |
| `scalar/`, `constants.hpp` | Scalar contracts and numerical constants |
| `vector/`, `matrix/` | Fixed-size containers, traits, concepts, and Householder reflectors |
| `linalg_ops.hpp`, `linalg_util.hpp` | Arithmetic, products, slices, blocks, and structural conversions |
| `space/` | Mathematical spaces and element-to-coordinate mappings |
| `affine/` | Affine maps, traits, and transformations |
| `norms/` | Vector and matrix norm contracts and implementations |
| `operator/` | Operator contracts, structured matrices, factorization, and inverse utilities |
| `spectral/` | QR, SVD, eigen decomposition, and matrix exponential |
| `numerical_config.hpp`, `assert.hpp` | Factorization settings, numerical events, and assertions |

## Representation and mathematical meaning

`linalg::Vector` and `linalg::Matrix` own fixed-size storage; dimensions and
scalar types are encoded in their types. Arithmetic and structural operations
are implemented outside the containers so storage does not imply symmetry,
definiteness, or membership in a particular mathematical space.

Traits and concepts express separate contracts:

- Container traits describe element types and dimensions.
- Space traits describe mathematical elements and their finite coordinates.
- Coordinate maps convert between those representations. Symmetric matrix
  coordinates use scaled off-diagonal entries to preserve inner products.
- Operator traits describe domains, codomains, and matrix representations.
- Inner products and norms describe pairings and magnitudes over those types.

An affine map combines a linear action and an offset. Factorizations represent
numerical algorithms and their results; they do not automatically become spaces.
Symmetric matrix spaces and numerical definiteness checks live here;
cone membership and optimization domains belong to akreon-convex.

## Numerical flow

A caller selects a container or structured representation, checks its required
contracts, and passes it to a numerical algorithm. QR and LDL can retain factors
for subsequent solves. SVD and eigen decomposition expose decomposition results;
matrix exponential uses the eigenvector/generalized-eigenvector basis. Individual
headers document these algorithms, their assumptions, and result semantics.

APIs favor constexpr evaluation and dimensions known at compile time. Numerical
success still depends on the input and algorithm: callers must inspect convergence
indicators or numerical events where returned. Compiler constexpr budgets and
factorization settings are distinct controls.

## Dependency boundary

Akreon Convex, Control, and Dynamics consume this library. Linalg does not depend
on them. Generic numerical kernels belong here; optimization modeling, controller
synthesis, and simulation policies belong in their respective libraries.

## API reference

Use Doxygen's namespace, class, and file indexes for signatures, numerical
assumptions, and algorithm flow documented in the headers.
