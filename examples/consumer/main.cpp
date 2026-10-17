#include <akreon/linalg/linalg.hpp>
int main() {
  constexpr linalg::Vector<double, 2> v{1.0, 2.0};
  static_assert(v(0) == 1.0);
  return v(1) == 2.0 ? 0 : 1;
}
