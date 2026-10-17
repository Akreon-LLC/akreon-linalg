/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "endomorphism.hpp"
#include "operator_traits.hpp"

namespace linalg {

/**
 * @brief Symmetric linear operator CRTP base
 */
template <SquareMatrix Mat, typename Derived>
struct SymmetricOperator : Endomorphism<Mat, SymmetricOperator<Mat, Derived>> {
  /// @brief alias to extract base types
  using OperatorBase = Endomorphism<Mat, SymmetricOperator<Mat, Derived>>;
  using typename OperatorBase::ComplexT;
  using typename OperatorBase::ScalarT;
  using typename OperatorBase::T;

  /// @brief extract N from base
  using OperatorBase::N;

  /// @brief helpers to get DIM from mat
  static constexpr std::size_t DIM = (N * (N + 1)) / 2;

  /**
   * @brief CTOR
   *
   * @param Mat A
   */
  constexpr explicit SymmetricOperator(const Mat& A) : svec(get_svec(A)) {}

  /**
   * @brief get element at (i,j) index
   *
   * @param row index
   * @param col index
   *
   * @return element
   */
  constexpr auto operator()(std::size_t i, std::size_t j) const {
    const auto x = svec(index(i, j));
    if (i == j) return x;
    return x / sqrt2_v<T>;
  }

  /**
   * @brief index from (row, col)
   *
   * @param std::size_t i = row index
   * @param std::size_t j = col index
   *
   * @return index
   */
  constexpr std::size_t index(std::size_t i, std::size_t j) const {
    AK_ASSERT((j < N && i < N) &&
              "SymmetricOperator index expects row/col inside [0, N)");
    if (i >= j) return i * (i + 1) / 2 + j;
    return j * (j + 1) / 2 + i;
  }

  /**
   * @brief is pd check
   *
   * @return T/F
   */
  constexpr bool is_pd() const { return type == SymmetricOperatorType::PD; }

  /**
   * @brief is nd check
   *
   * @return T/F
   */
  constexpr bool is_nd() const { return type == SymmetricOperatorType::ND; }

  /**
   * @brief is psd check
   *
   * @return T/F
   */
  constexpr bool is_psd() const {
    return type == SymmetricOperatorType::PD ||
           type == SymmetricOperatorType::PSD;
  }

  /**
   * @brief is nsd check
   *
   * @return T/F
   */
  constexpr bool is_nsd() const {
    return type == SymmetricOperatorType::ND ||
           type == SymmetricOperatorType::NSD;
  }

  /**
   * @brief is indefinite
   *
   * @return T/F
   */
  constexpr bool is_ind() const { return type == SymmetricOperatorType::IND; }

  /**
   * @brief is valid
   *
   * @return T/F
   */
  constexpr bool is_valid() const { return type != SymmetricOperatorType::INV; }

  /**
   * @brief compute definiteness
   *
   * @param vec (diagonal of elements)
   *
   * @return void
   */
  template <typename DiagT>
  constexpr void compute_definiteness(const Vector<DiagT, N>& diag,
                                      const ScalarT& tol) {
    std::size_t npos = 0, nneg = 0, nzero = 0;

    auto classify_scalar = [&](const auto& d_real) {
      if (d_real > tol) ++npos;
      else if (d_real < -tol) ++nneg;
      else ++nzero;
    };

    if constexpr (is_complex_v<DiagT>) {
      for (const auto& d : diag) classify_scalar(d.real());
    } else {
      for (const auto& d : diag) classify_scalar(d);
    }

    if (nneg == 0 && nzero == 0) type = SymmetricOperatorType::PD;
    else if (nneg == 0 && nzero > 0) type = SymmetricOperatorType::PSD;
    else if (npos == 0 && nzero == 0) type = SymmetricOperatorType::ND;
    else if (npos == 0 && nzero > 0) type = SymmetricOperatorType::NSD;
    else type = SymmetricOperatorType::IND;
  }

  /**
   * @brief self
   *
   * @return self
   */
  constexpr const Derived& self() const { return static_cast<Derived&>(*this); }

  /** @brief member variables */
  SymmetricOperatorType type; //< SymmetricOperatorType value
  Vector<T, DIM> svec;        //< Data stored as svec(A): lower triangle with
                              // sqrt(2)-scaled off-diagonals
};

} // namespace linalg
