/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "endomorphism.hpp"
#include "akreon/linalg/vector/householder.hpp"

namespace linalg {
/**
 * @brief Reduces A to upper Hessenberg form H = U^* A U.
 *
 * Householder similarities eliminate entries below the first subdiagonal.
 * Packed reflector tails reconstruct the orthogonal or unitary basis U.
 *
 * @tparam Mat Square real or complex matrix type.
 */
template <SquareMatrix Mat>
struct Hessenberg : Endomorphism<Mat, Hessenberg<Mat>> {
  /// @brief alias from base class
  using OperatorBase = Endomorphism<Mat, Hessenberg<Mat>>;
  using typename OperatorBase::T;

  /// @brief helper from base to get num rows
  using OperatorBase::N;

  /**
   * @brief Reduce A by two-sided Householder similarities.
   *
   * @param[in] A Square input matrix to reduce to upper Hessenberg form.
   */
  constexpr explicit Hessenberg(const Mat& A) {
    H_pack = A;
    constexpr_for<0, N - 2>([&]<std::size_t k>(
                                std::integral_constant<std::size_t, k>) {
      const auto& H_block = get_n_rows<k + 1, N>(H_pack);
      const auto& H_col = get_col(H_block, k);
      // Reflect the column tail onto its first subdiagonal entry.
      const auto h = HouseholderVector(H_col);

      const auto& v = as_col_vector(h.v);
      beta(k) = h.beta;

      if (beta(k) == real_type_t<T>(0)) return;

      // Left update: H <- (I - beta*v*v^*) H on the active rows.
      auto H_block_kn = get_n_cols<k, N>(H_block);
      if constexpr (Complex<T>)
        H_block_kn =
            H_block_kn - (beta(k) * v * conjugate_transpose(v)) * H_block_kn;
      else H_block_kn = H_block_kn - (beta(k) * v * transpose(v)) * H_block_kn;

      for (std::size_t i = k + 1, a = 0; i < N; ++i, ++a)
        for (std::size_t j = k, b = 0; j < N; ++j, ++b)
          H_pack(i, j) = H_block_kn(a, b);

      // Right update: H <- H (I - beta*v*v^*) completes the similarity.
      auto H_zn_kn = get_block<0, N, k + 1, N>(H_pack);
      if constexpr (Complex<T>)
        H_zn_kn = H_zn_kn - H_zn_kn * (beta(k) * v * conjugate_transpose(v));
      else H_zn_kn = H_zn_kn - H_zn_kn * (beta(k) * v * transpose(v));

      for (std::size_t i = 0, a = 0; i < N; ++i, ++a)
        for (std::size_t j = k + 1, b = 0; j < N; ++j, ++b)
          H_pack(i, j) = H_zn_kn(a, b);

      // Pack reflector tails below the band; H() restores these zeros.
      for (std::size_t j = k + 2, a = 1; j < N; ++j, ++a)
        H_pack(j, k) = v(a, 0);
    });
  }

  /**
   * @brief Extract the upper Hessenberg matrix.
   *
   * @return Matrix<T, N, N> with zeros below the first subdiagonal.
   */
  constexpr auto H() const {
    Matrix<T, N, N> res{};
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        if (i >= 2 && j <= (i - 2)) continue;
        res(i, j) = H_pack(i, j);
      }
    }
    return res;
  }

  /**
   * @brief Reconstruct the orthogonal or unitary similarity basis.
   *
   * @return Matrix<T, N, N> containing U such that H = U^* A U.
   */
  constexpr auto U() const {
    auto U = make_identity<T, N>();
    constexpr_for<0, N - 2>([&]<std::size_t k>(
                                std::integral_constant<std::size_t, k>) {
      const std::size_t j = N - k - 1;
      Vector<T, j> v;
      v(0) = T(1);
      for (std::size_t i = k + 2, a = 1; i < N; ++i, ++a) v(a) = H_pack(i, k);

      const auto& v_col = as_col_vector(v);

      // Accumulate the same column-space changes used in the reduction.
      auto U_block = get_n_cols<k + 1, N>(U);
      if constexpr (Complex<T>)
        U_block = U_block -
                  beta(k) * (U_block * (v_col * conjugate_transpose(v_col)));
      else U_block = U_block - beta(k) * (U_block * (v_col * transpose(v_col)));

      for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = k + 1, a = 0; c < N; ++c, ++a)
          U(r, c) = U_block(r, a);
    });
    return U;
  }

  /**
   * @brief Read an entry of the Hessenberg matrix.
   *
   * @param[in] i Zero-based row index.
   * @param[in] j Zero-based column index.
   * @return Scalar T; zero below the first subdiagonal.
   */
  constexpr auto operator()(std::size_t i, std::size_t j) const {
    if (i >= 2 && j <= (i - 2)) return T(0);
    return H_pack(i, j);
  }

  /** @brief Vector member variables */
  Vector<T, N> beta; ///< beta vector from householder reflections
private:
  Matrix<T, N, N>
      H_pack; ///< Hessenberg matrix (packed w/ householder below diag)
};

} // namespace linalg
