/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include "akreon/linalg/linalg.hpp"

using namespace affine;
using namespace linalg;

TEST_CASE("Affine maps compose and invert", "[affine]") {
  constexpr auto I = make_identity<double, 2>();
  constexpr auto zeros = Vectord<2>{};
  constexpr auto F = Matrixd<2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{4.2, -2.0}};
  constexpr auto f_bias = Vectord<2>{1.0, 1.0};
  constexpr auto G = Matrixd<2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{4.2, -2.0}};
  constexpr auto g_bias = Vectord<2>{-1.0, 2.0};

  constexpr auto identity = AffineMap2d<double>(I, zeros);
  constexpr auto f = AffineMap2d<double>(F, f_bias);
  constexpr auto g = AffineMap2d<double>(G, g_bias);
  constexpr auto fg = f.compose(g);
  constexpr auto gf = g.compose(f);

  STATIC_REQUIRE(f.compose(identity) == identity.compose(f));
  STATIC_REQUIRE(g.compose(identity) == identity.compose(g));
  STATIC_REQUIRE(all_equal(fg.linear_part(), F * G));
  STATIC_REQUIRE(all_equal(fg.translation(), F * g_bias + f_bias));
  STATIC_REQUIRE(all_equal(gf.linear_part(), G * F));
  STATIC_REQUIRE(all_equal(gf.translation(), G * f_bias + g_bias));
  STATIC_REQUIRE(f.compose(g).compose(identity) ==
                 f.compose(g.compose(identity)));

  constexpr auto f_inverse = f.inverse();
  constexpr auto f_identity = f.compose(f_inverse);
  STATIC_REQUIRE(all_equal(f_identity.linear_part(), I));
  STATIC_REQUIRE(all_equal(f_identity.translation(), zeros));
}

TEST_CASE("Homogeneous matrices construct affine maps", "[affine]") {
  constexpr auto homogeneous = Matrixd<4, 4>{
      Vectord<4>{1.0, 0.0, 0.0, 2.0}, Vectord<4>{0.0, 1.0, 0.0, -3.0},
      Vectord<4>{0.0, 0.0, 1.0, 4.0}, Vectord<4>{0.0, 0.0, 0.0, 1.0}};
  constexpr auto map = AffineMap3d<double>(homogeneous);

  STATIC_REQUIRE(all_equal(map.linear_part(), make_identity<double, 3>()));
  STATIC_REQUIRE(all_equal(map.translation(), Vectord<3>{2.0, -3.0, 4.0}));
}

TEST_CASE("Affine transform generators", "[affine]") {
  constexpr auto identity2 = make_identity<double, 2>();
  constexpr auto identity3 = make_identity<double, 3>();

  constexpr auto rotation2 = Rotation2d<double>()(0.0);
  constexpr auto rotation3 = Rotation3d<double>();
  STATIC_REQUIRE(all_equal(rotation2, identity2));
  STATIC_REQUIRE(all_equal(rotation3.x(0.0), identity3));
  STATIC_REQUIRE(all_equal(rotation3.y(0.0), identity3));
  STATIC_REQUIRE(all_equal(rotation3.z(0.0), identity3));

  constexpr auto scale = Scale<double, 3>()(2.0, 3.0, -4.0);
  STATIC_REQUIRE(all_equal(scale, Matrixd<3, 3>{Vectord<3>{2.0, 0.0, 0.0},
                                                Vectord<3>{0.0, 3.0, 0.0},
                                                Vectord<3>{0.0, 0.0, -4.0}}));

  constexpr auto shear = Shear<double, 3>()(1, 2, -1.5);
  STATIC_REQUIRE(shear(1, 2) == -1.5);
  STATIC_REQUIRE(shear(0, 0) == 1.0);
  STATIC_REQUIRE(shear(1, 1) == 1.0);
  STATIC_REQUIRE(shear(2, 2) == 1.0);
}
