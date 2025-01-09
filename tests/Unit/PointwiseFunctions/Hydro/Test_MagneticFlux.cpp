// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MagneticFlux.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace hydro {
namespace {
template <size_t Dim, typename Frame, typename DataType>
void test_magnetic_flux(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataType, Dim, Frame> (*)(
          const tnsr::I<DataType, Dim, Frame>&, const Scalar<DataType>&)>(
          &magnetic_flux<DataType, Dim, Frame>),
      "TestFunctions", "magnetic_flux", {{{-10.0, 10.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.MagneticFlux",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector dv(5);
  test_magnetic_flux<1, Frame::Inertial>(dv);
  test_magnetic_flux<1, Frame::Grid>(dv);
  test_magnetic_flux<2, Frame::Inertial>(dv);
  test_magnetic_flux<2, Frame::Grid>(dv);
  test_magnetic_flux<3, Frame::Inertial>(dv);
  test_magnetic_flux<3, Frame::Grid>(dv);

  test_magnetic_flux<1, Frame::Inertial>(0.0);
  test_magnetic_flux<1, Frame::Grid>(0.0);
  test_magnetic_flux<2, Frame::Inertial>(0.0);
  test_magnetic_flux<2, Frame::Grid>(0.0);
  test_magnetic_flux<3, Frame::Inertial>(0.0);
  test_magnetic_flux<3, Frame::Grid>(0.0);

  // Check compute item works correctly in DataBox
  TestHelpers::db::test_compute_tag<
      Tags::MagneticFluxCompute<DataVector, 2, Frame::Inertial>>(
      "MagneticFlux");
  tnsr::I<DataVector, 3> magnetic_field{
      {{DataVector{5, 0.25}, DataVector{5, 0.1}, DataVector{5, 0.35}}}};
  Scalar<DataVector> sqrt_det_g{{{DataVector{5, 0.25}}}};
  const auto box = db::create<
      db::AddSimpleTags<Tags::MagneticField<DataVector, 3>,
                        ::gr::Tags::SqrtDetSpatialMetric<DataVector>>,
      db::AddComputeTags<
          Tags::MagneticFluxCompute<DataVector, 3, Frame::Inertial>>>(
      magnetic_field, sqrt_det_g);
  CHECK(db::get<Tags::MagneticFlux<DataVector, 3, Frame::Inertial>>(box) ==
        magnetic_flux(magnetic_field, sqrt_det_g));
}
}  // namespace hydro
