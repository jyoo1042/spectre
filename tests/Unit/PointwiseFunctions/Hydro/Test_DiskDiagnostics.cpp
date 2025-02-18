// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "PointwiseFunctions/Hydro/DiskDiagnostics.hpp"

namespace {
template <typename DataType>
void test_disk_diagnostics(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                       const Scalar<DataType>&)>(
          &hydro::edot<DataType>),
      "Test_DiskDiagnostics", "edot", {{{0.01, 1.0}}}, used_for_size);

  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                       const Scalar<DataType>&)>(
          &hydro::edot<DataType>),
      "Test_DiskDiagnostics", "ldot", {{{0.01, 1.0}}}, used_for_size);
}
}  // namespace
