Follow AGENTS.md and the checked-in clang-format configuration.
Build and run the independent suite as described in README.md.
Keep algorithms constexpr and keep dependency edges within the documented graph.

CI checks out development dependencies explicitly in `.deps`. For private
repositories, configure `AKREON_DEPENDENCY_TOKEN` with read access.
`AKREON_LINALG_REF` and `AKREON_CONVEX_REF` repository variables can pin CI
to reviewed commits; switch these to release tags when published.
The library fallback itself is always pinned to the documented release tag.
