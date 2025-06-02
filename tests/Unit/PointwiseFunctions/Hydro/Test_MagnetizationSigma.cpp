// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/MagnetizationSigma.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename DataType>
void test_magnetization_sigma(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                       const Scalar<DataType>&)>(
          &hydro::magnetization_sigma<DataType>),
      "MagnetizationSigma", "magnetization_sigma", {{{0.01, 1.0}}},
      used_for_size);
}
}  // namespace

namespace hydro {
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.MagnetizationSigma",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro"};

  test_magnetization_sigma(std::numeric_limits<double>::signaling_NaN());
  test_magnetization_sigma(DataVector(5));

  // Check compute item works correctly in DataBox
  TestHelpers::db::test_compute_tag<
      Tags::MagnetizationSigmaCompute<DataVector>>("MagnetizationSigma");
  const Scalar<DataVector> comoving_magnetic_field_magnitude{
      {{DataVector{5, 0.11}}}};
  const Scalar<DataVector> rest_mass_density{{{DataVector{5, 0.05}}}};

  const auto box = db::create<
      db::AddSimpleTags<Tags::ComovingMagneticFieldMagnitude<DataVector>,
                        Tags::RestMassDensity<DataVector>>,
      db::AddComputeTags<Tags::MagnetizationSigmaCompute<DataVector>>>(
      comoving_magnetic_field_magnitude, rest_mass_density);
  CHECK(db::get<Tags::MagnetizationSigma<DataVector>>(box) ==
        magnetization_sigma(comoving_magnetic_field_magnitude,
                            rest_mass_density));
}
}  // namespace hydro
