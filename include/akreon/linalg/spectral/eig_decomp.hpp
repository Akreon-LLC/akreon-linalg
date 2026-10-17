/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "akreon/linalg/operator/hessenberg.hpp"
#include "qr.hpp"

namespace linalg {

/**
 * @brief Computes eigenvalues and eigenvectors satisfying A V = V diag(lambda).
 *
 * Reduces A to Hessenberg form, applies shifted QR to obtain Schur form,
 * then recovers eigenvectors in the original coordinates. Hermitian matrices
 * use the accumulated orthonormal basis directly.
 *
 * @tparam Mat Square real or complex matrix type.
 */
template <SquareMatrix Mat>
struct EigenDecomposition {
  // @brief alias for underlying type
  using T = matrix_scalar_t<Mat>;

  // @brief aliasing complex and real types for eigen decomposition
  using Real = real_type_t<T>;
  using Complex = std::conditional_t<linalg::Complex<T>, T, std::complex<Real>>;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static_assert(N > 0, "Eigen decomposition requires a nonempty matrix");

  /**
   * @brief Construct the nominal shifted-QR configuration.
   * @return FactorConfig<Real> with 256 steps, relative tol = 8*epsilon,
   * and floor_eps = the smallest positive normal Real value.
   */
  static constexpr config::FactorConfig<Real> default_config() {
    return {.steps = 256,
            .tol = Real(8) * std::numeric_limits<Real>::epsilon(),
            .floor_eps = std::numeric_limits<Real>::min()};
  }

  /**
   * @brief Compute eigenvalues and eigenvectors using the nominal
   * configuration.
   *
   * @param[in] A Square input matrix.
   */
  constexpr explicit EigenDecomposition(const Mat& A) { solve_qr(A); }

  /**
   * @brief Compute eigenvalues and eigenvectors with explicit settings.
   *
   * @param[in] A Square input matrix.
   * @param[in] config_in Iteration budget, relative deflation tolerance, and
   * normalization floor.
   */
  constexpr explicit EigenDecomposition(
      const Mat& A, const config::FactorConfig<Real>& config_in)
      : config(config_in) {
    solve_qr(A);
  }

  /**
   * @brief Select the eigenvalue whose magnitude is the spectral radius.
   *
   * @return Complex eigenvalue with greatest magnitude; zero for a zero
   * spectrum.
   */
  constexpr auto spectral_radius() const {
    Complex rho(0.0);
    for (std::size_t i = 0; i < N; ++i) {
      const auto mag_rho = constexpr_sqrt_safe(rho.real() * rho.real() +
                                               rho.imag() * rho.imag());
      const auto mag_lambda =
          constexpr_sqrt_safe(lambda(i).real() * lambda(i).real() +
                              lambda(i).imag() * lambda(i).imag());
      rho = mag_lambda > mag_rho ? lambda(i) : rho;
    }
    return rho;
  }

  /** @brief eigendecomposition member variables */
  Vector<Complex, N> lambda; ///< eigenvalues
  Matrix<Complex, N, N> V;   ///< eigenvectors
  bool converged{false};     ///< eigenvalue convergence flag

private:
  /**
   * @brief Compute a shift root with relative squared-residual accuracy.
   * @param[in] value Nonnegative real radicand.
   * @return Real approximation to sqrt(value).
   */
  static constexpr Real spectral_sqrt(Real value) {
    return constexpr_sqrt_safe(value, Real(0), Real(1),
                               Real(4) * std::numeric_limits<Real>::epsilon() *
                                   value);
  }

  /**
   * @brief Recover eigenpairs by shifted QR and Schur back substitution.
   *
   * @param[in] A Square input matrix; updates lambda, V, and converged.
   */
  constexpr void solve_qr(const Mat& A) {
    if (!std::isfinite(config.tol) || config.tol <= Real(0) ||
        !std::isfinite(config.floor_eps) || config.floor_eps <= Real(0))
      return;
    for (std::size_t r = 0; r < N; ++r)
      for (std::size_t c = 0; c < N; ++c)
        if (!std::isfinite(constexpr_abs(A(r, c)))) return;

    Real scale = 0;
    for (std::size_t r = 0; r < N; ++r)
      for (std::size_t c = 0; c < N; ++c)
        scale = std::max(scale, constexpr_abs(A(r, c)));
    // Normalize A by its largest entry; restore eigenvalue units at the end.
    auto normalized = A;
    if (scale > Real(0))
      for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c) normalized(r, c) /= scale;
    // Reduce to H = U^* (A/scale) U; orders one and two are already Hessenberg.
    auto [H, U] = [&] {
      if constexpr (N > 2) {
        const auto reduction = Hessenberg(normalized);
        return std::tuple{reduction.H(), reduction.U()};
      } else return std::tuple{normalized, make_identity<T, N>()};
    }();
    const Real threshold = config.tol;
    const bool hermitian = [&] {
      if constexpr (linalg::Real<T>)
        return all_equal(normalized, transpose(normalized), threshold);
      else
        return all_equal(normalized, conjugate_transpose(normalized),
                         threshold);
    }();
    Vector<Complex, N> eigs;

    // Split blocks at negligible subdiagonals. Convergence requires triangular
    // form, or isolated real 2x2 blocks representing conjugate eigenvalues.
    auto deflate = [&] {
      for (std::size_t r = 1; r < N; ++r)
        if (constexpr_abs(H(r, r - 1)) <= threshold) H(r, r - 1) = T(0);
      bool done = true;
      for (std::size_t r = 1; r < N; ++r) {
        if (H(r, r - 1) == T(0)) continue;
        if constexpr (linalg::Complex<T>) done = false;
        else {
          const Real half_difference = (H(r - 1, r - 1) - H(r, r)) / 2;
          const Real discriminant =
              half_difference * half_difference + H(r - 1, r) * H(r, r - 1);
          if (hermitian || discriminant >= 0 ||
              (r > 1 && H(r - 1, r - 2) != T(0)) ||
              (r + 1 < N && H(r + 1, r) != T(0)))
            done = false;
        }
      }
      return done;
    };
    converged = deflate();
    for (std::size_t iteration = 0; !converged && iteration < config.steps;
         ++iteration) {
      // Locate the last unresolved block, skipping deflated scalar roots and
      // isolated real blocks with conjugate eigenvalues.
      std::size_t end = N;
      while (end > 1 && H(end - 1, end - 2) == T(0)) --end;
      if constexpr (linalg::Real<T>) {
        while (!hermitian && end > 2 && H(end - 2, end - 3) == T(0)) {
          const Real d = (H(end - 2, end - 2) - H(end - 1, end - 1)) / 2;
          if (d * d + H(end - 2, end - 1) * H(end - 1, end - 2) >= 0) break;
          end -= 2;
          while (end > 1 && H(end - 1, end - 2) == T(0)) --end;
        }
      }
      // Choose the trailing 2x2 root nearest the last diagonal as the shift.
      T shift = H(end - 1, end - 1);
      if constexpr (linalg::Real<T>) {
        if (end > 1) {
          const Real a = H(end - 2, end - 2), d = H(end - 1, end - 1);
          const Real delta = (a - d) / 2;
          const Real discriminant =
              delta * delta + H(end - 2, end - 1) * H(end - 1, end - 2);
          if (discriminant >= 0) {
            const Real root = spectral_sqrt(discriminant);
            shift = (a + d) / 2 + (delta >= 0 ? -root : root);
          }
        }
      }
      if constexpr (linalg::Complex<T>) {
        if (end > 1) {
          const Complex center =
              (H(end - 2, end - 2) + H(end - 1, end - 1)) / Real(2);
          const Complex delta =
              (H(end - 2, end - 2) - H(end - 1, end - 1)) / Real(2);
          const Complex disc =
              delta * delta + H(end - 2, end - 1) * H(end - 1, end - 2);
          const Real radius = constexpr_abs(disc);
          const Complex root(
              spectral_sqrt(
                  std::max(Real(0), (radius + disc.real()) / Real(2))),
              (disc.imag() < Real(0) ? Real(-1) : Real(1)) *
                  spectral_sqrt(
                      std::max(Real(0), (radius - disc.real()) / Real(2))));
          const Complex plus = center + root, minus = center - root;
          shift = constexpr_abs(plus - shift) < constexpr_abs(minus - shift)
                      ? plus
                      : minus;
        }
      }
      auto shifted = H;
      for (std::size_t j = 0; j < N; ++j) shifted(j, j) -= shift;
      bool double_shift = false;
      T trace = T(0), determinant = T(0);
      if constexpr (linalg::Real<T>) {
        if (end > 1 && !hermitian) {
          const T a = H(end - 2, end - 2), d = H(end - 1, end - 1);
          const T b = H(end - 2, end - 1), c = H(end - 1, end - 2);
          double_shift = (a - d) * (a - d) / 4 + b * c < 0;
          trace = a + d;
          determinant = a * d - b * c;
        }
      }
      if (double_shift) {
        // Use the real quadratic H^2 - trace*H + determinant*I for a pair.
        shifted = H * H - trace * H;
        for (std::size_t j = 0; j < N; ++j) shifted(j, j) += determinant;
      }
      // Update H by QR similarity and accumulate its basis change in U.
      const auto qr = QR(shifted);
      const auto Q = qr.Q();
      if (double_shift) H = transpose(Q) * (H * Q);
      else {
        H = qr.R() * Q;
        for (std::size_t j = 0; j < N; ++j) H(j, j) += shift;
      }
      // Restore the exact Hessenberg zeros after the similarity update.
      for (std::size_t r = 2; r < N; ++r)
        for (std::size_t c = 0; c + 1 < r; ++c) H(r, c) = T(0);
      U = U * Q;
      converged = deflate();
    }
    if (!converged) return;

    // Extract eigenvalues from diagonal entries and real 2x2 Schur blocks.
    for (std::size_t j = 0; j < N; ++j) {
      eigs(j) = Complex(H(j, j));
      if constexpr (linalg::Real<T>) {
        if (j + 1 < N && H(j + 1, j) != T(0)) {
          const Real center = (H(j, j) + H(j + 1, j + 1)) / 2;
          const Real delta = (H(j, j) - H(j + 1, j + 1)) / 2;
          const Real discriminant = delta * delta + H(j, j + 1) * H(j + 1, j);
          const Real imaginary = spectral_sqrt(-discriminant);
          eigs(j) = Complex(center, imaginary);
          eigs(j + 1) = Complex(center, -imaginary);
          ++j;
        }
      }
    }
    // Populate the spectrum used by Schur-block seeding and back substitution.
    lambda = eigs;
    if (hermitian) {
      // Hermitian Schur form is diagonal; its eigenvectors are U.
      V = as_complex_matrix(U);
      // Sort coupled Hermitian eigenpairs; keep diagonal input order.
      bool diagonal_input = true;
      for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c)
          if (r != c && A(r, c) != T(0)) diagonal_input = false;
      if (!diagonal_input) {
        for (std::size_t j = 0; j < N; ++j) {
          std::size_t largest = j;
          for (std::size_t k = j + 1; k < N; ++k)
            if (lambda(k).real() > lambda(largest).real()) largest = k;
          std::swap(lambda(j), lambda(largest));
          for (std::size_t r = 0; r < N; ++r) std::swap(V(r, j), V(r, largest));
        }
      }
      for (std::size_t j = 0; j < N; ++j) lambda(j) *= scale;
      return;
    }

    // Track columns assigned to repeated roots and generalized eigenvectors.
    std::array<bool, N> used{false};

    auto indices_equal = [&](const Vector<Complex, N>& D,
                             Complex val) -> std::array<int, N> {
      std::array<int, N> idx;
      idx.fill(-1);
      for (int j = 0; j < int(N); ++j)
        if (is_same(D(j), val)) idx[j] = j;
      return idx;
    };

    auto take_last_unused = [&](std::array<int, N>& slots) -> int {
      for (int j = 0; j < int(N); ++j) {
        if (!used[j] && slots[j] != -1) {
          used[j] = true;
          return j;
        }
      }
      return -1;
    };

    // Solve in Schur coordinates from the last block upward, then map by U.
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
      if (used[i]) continue;

      // Estimate the Jordan defect from multiplicity and nullity.
      int AM = 0;
      std::for_each(eigs.begin(), eigs.end(), [&](const auto& val) {
        AM += constexpr_abs(eigs(i) - val) < config.tol;
      });
      int GM = 0, defect = 0;
      if (AM > 1) {
        const auto I = make_identity<Complex, N>();
        GM = N - QR(as_complex_matrix(normalized) - eigs(i) * I).rank();
        defect = std::max(0, AM - GM);
      }

      if (0 == defect) {
        Vector<Complex, N> x;

        // Recover ordinary eigenvectors when no Jordan defect is detected.
        if (linalg::Real<T> && eigs(i).imag() != Real(0) && i > 0) {
          // Seed the Schur block and solve (H - lambda_i I)x = 0.
          seed_complex_block(H, i, x);
          back_sub_1x1_block(H, i, x);

          // Map both conjugate eigenvectors to the original coordinates.
          auto v = map_to_A(U, x);
          for (std::size_t j = 0; j < N; ++j) V(j, i) = v(j);

          i--;

          v = map_to_A(U, conjugate(x));
          for (std::size_t j = 0; j < N; ++j) V(j, i) = v(j);

          lambda(i) = eigs(i);

          used[i] = true;
        } else {
          // Seed the Schur block and solve (H - lambda_i I)x = 0.
          seed_real_block(i, x);
          back_sub_1x1_block(H, i, x);

          // Map the Schur-coordinate eigenvector by U.
          auto v = map_to_A(U, x);
          for (std::size_t j = 0; j < N; ++j) V(j, i) = v(j);

          lambda(i) = eigs(i);

          used[i] = true;
        }
      } else {
        // Build generalized eigenvectors for a defective repeated root.
        if (linalg::Real<T> && eigs(i).imag() != Real(0) && i > 0) {
          // Bound the chain length by the defect and available Schur columns.
          const int chain_len = std::min(1 + defect, (i + 1) / 2);
          Vector<Vector<Complex, N>, N> chain;

          // Start the chain with a solution of (H - lambda_i I)x_0 = 0.
          seed_complex_block(H, i, chain(0));
          back_sub_1x1_block(H, i, chain(0));

          // Extend the chain with (H - lambda_i I)x_k = x_{k-1}.
          for (int k = 1; k < chain_len; ++k)
            back_sub_shifted_rhs(H, lambda(i), chain(k - 1), chain(k));

          // Assign mapped chain vectors to matching eigenvalue columns.
          auto slots = indices_equal(eigs, eigs(i));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots);
            if (slot_idx < 0) break;

            auto v = map_to_A(U, chain(k));

            for (std::size_t j = 0; j < N; ++j) V(j, slot_idx) = v(j);

            lambda(slot_idx) = eigs(i);
          }

          // Map the conjugate chain to columns for the conjugate root.
          auto slots_conj = indices_equal(eigs, std::conj(eigs(i)));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots_conj);
            if (slot_idx < 0) break;

            auto vbar = map_to_A(U, conjugate(chain(k)));

            for (std::size_t j = 0; j < N; ++j) V(j, slot_idx) = vbar(j);

            lambda(slot_idx) = std::conj(eigs(i));
          }
        } else {
          // Bound the chain length by the defect and available Schur columns.
          const int chain_len = std::min(1 + defect, 1 + i);
          Vector<Vector<Complex, N>, N> chain;

          // Start the chain with a solution of (H - lambda_i I)x_0 = 0.
          seed_real_block(i, chain(0));
          back_sub_1x1_block(H, i, chain(0));

          // Extend the chain with (H - lambda_i I)x_k = x_{k-1}.
          for (int k = 1; k < chain_len; ++k)
            back_sub_shifted_rhs(H, eigs(i), chain(k - 1), chain(k));

          // Map the chain to the original coordinates.
          auto slots = indices_equal(eigs, eigs(i));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots);
            if (slot_idx < 0) break;

            auto v = map_to_A(U, chain(k));

            for (std::size_t j = 0; j < N; ++j) V(j, slot_idx) = v(j);

            lambda(slot_idx) = eigs(i);
          }
        }
      }
    }
    for (std::size_t j = 0; j < N; ++j) lambda(j) *= scale;
  }

  /**
   * @brief Back-substitute through scalar and real conjugate-pair Schur blocks.
   *
   * @param[in] H Schur matrix in normalized coordinates.
   * @param[in] idx Index of the seeded eigenvalue or end of its conjugate
   * block.
   * @param[in,out] x Seeded Schur-coordinate vector, completed by back
   * substitution.
   */
  constexpr void back_sub_1x1_block(const Mat& H, int idx,
                                    Vector<Complex, N>& x) {
    // Start above the seeded scalar or 2x2 block and solve toward the first
    // row.
    for (int r = ((linalg::Real<T> && lambda(idx).imag() != Real(0)) ? idx - 2
                                                                     : idx - 1);
         r >= 0; --r) {
      Complex rhs(0);
      for (int c = r + 1; c <= idx; ++c) rhs -= Complex(H(r, c)) * x(c);
      if constexpr (linalg::Real<T>) {
        if (r > 0 && H(r, r - 1) != T(0)) {
          // Solve the coupled 2x2 system for the preceding conjugate block.
          Complex upper_rhs(0);
          for (int c = r + 1; c <= idx; ++c)
            upper_rhs -= Complex(H(r - 1, c)) * x(c);
          const Complex a = Complex(H(r - 1, r - 1)) - lambda(idx);
          const Complex b = Complex(H(r - 1, r));
          const Complex c = Complex(H(r, r - 1));
          const Complex d = Complex(H(r, r)) - lambda(idx);
          const Complex det = a * d - b * c;
          if (det != Complex(0)) {
            x(r - 1) = (d * upper_rhs - b * rhs) / det;
            x(r) = (a * rhs - c * upper_rhs) / det;
          }
          --r;
          continue;
        }
      }
      const auto denom = Complex(H(r, r)) - lambda(idx);
      x(r) = denom == Complex(0) ? Complex(0) : rhs / denom;
    }
  }

  /**
   * @brief Solve (H - eig*I)x = rhs by backward substitution.
   *
   * @param[in] H Schur matrix in normalized coordinates.
   * @param[in] eig Eigenvalue defining the shifted system.
   * @param[in] rhs Previous generalized eigenvector in the chain.
   * @param[out] x Computed Schur-coordinate solution.
   */
  constexpr void back_sub_shifted_rhs(const Mat& H, const Complex& eig,
                                      const Vector<Complex, N>& rhs,
                                      Vector<Complex, N>& x) {
    // Solve (H - eig*I)x = rhs from the last row upward.
    for (int r = int(N) - 1; r >= 0; --r) {
      Complex sum(0);
      for (int c = r + 1; c < int(N); ++c) sum += H(r, c) * x(c);
      const auto denom = Complex(H(r, r)) - eig;
      if (denom == Complex(0)) x(r) = Complex(0);
      else x(r) = (rhs(r) - sum) / denom;
    }
  }

  /**
   * @brief Seed a null vector of a real 2x2 conjugate-eigenvalue block.
   *
   * @param[in] H Real Schur matrix.
   * @param[in] idx Last row of the 2x2 block and index of its eigenvalue.
   * @param[in,out] x Vector receiving the two block components.
   */
  constexpr void seed_complex_block(const Mat& H, int idx,
                                    Vector<Complex, N>& x) {
    const auto& a = H(idx - 1, idx - 1);
    const auto& b = H(idx - 1, idx);
    const auto& c = H(idx, idx - 1);
    const auto& d = H(idx, idx);

    // Choose between two null vectors of the 2x2 block using the larger pivot.
    const Complex la = lambda(idx) - Complex(a);
    Complex z0, z1;
    if (L2<Vector<T, 1>>()(Vector<T, 1>{b}) >= L2<Vector<Complex, 1>>()(la)) {
      // Use [b; lambda - a]
      z0 = Complex(b);
      z1 = la;
    } else {
      // Use [lambda - d; c]
      z0 = lambda(idx) - Complex(d);
      z1 = Complex(c);
    }
    Real s = constexpr_sqrt_safe(L2<Vector<Complex, 1>>()(z0) +
                                 L2<Vector<Complex, 1>>()(z1));
    if (s > Real(0)) {
      z0 /= s;
      z1 /= s;
    }

    x(idx - 1) = z0;
    x(idx) = z1;
  }

  /**
   * @brief Initialize a scalar Schur block with a coordinate basis vector.
   *
   * @param[in] idx Coordinate to set to one.
   * @param[out] x Vector set to zero except at idx.
   */
  constexpr void seed_real_block(int idx, Vector<Complex, N>& x) {
    for (std::size_t i = 0; i < N; ++i) x(i) = Complex(0);
    x(idx) = Complex(1);
  }

  /**
   * @brief Map a Schur-coordinate vector into the original basis and normalize
   * it.
   *
   * @param[in] U Accumulated orthogonal or unitary similarity basis.
   * @param[in] x Vector in Schur coordinates.
   * @return Vector<Complex, N> containing U*x divided by its floored norm.
   */
  constexpr auto map_to_A(const Mat& U, const Vector<Complex, N>& x) {
    Vector<Complex, N> v;

    for (std::size_t j = 0; j < N; ++j) {
      Complex sum(0);
      for (std::size_t k = 0; k < N; ++k) sum += Complex(U(j, k)) * x(k);
      v(j) = sum;
    }

    // Normalize v = U*x with a lower bound on its norm.
    const auto norm = std::max(L2<Vector<Complex, N>>()(v), config.floor_eps);
    return T(1) / norm * v;
  }

  /** @brief member vars */
  config::FactorConfig<Real> config = default_config();
};

} // namespace linalg
