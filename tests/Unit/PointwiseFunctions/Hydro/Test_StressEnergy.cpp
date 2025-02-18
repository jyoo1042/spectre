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
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.StressEnergy",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/Python/");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(&energy_density<DataVector>,
                                    "Test_StressEnergy", {"energy_density"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(&momentum_density<DataVector>,
                                    "Test_StressEnergy", {"momentum_density"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(&stress_trace<DataVector>,
                                    "Test_StressEnergy", {"stress_trace"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &stress_energy_tensor<DataVector>, "Test_StressEnergy",
      {"stress_energy_tensor"}, {{{0.0, 1.0}}}, used_for_size);

  // Check compute item works correctly in DataBox
  // For simplicity, just do flat spacetime, spatial_velocity = (0.6, 0.0, 0.0),
  // magnetic_field = (0.5, 0.0, 0.0).
  TestHelpers::db::test_compute_tag<
      Tags::StressEnergyTensorCompute<DataVector>>("StressEnergyTensor");

  const Scalar<DataVector> rest_mass_density{{{DataVector{5, 0.45}}}};
  const Scalar<DataVector> specific_internal_energy{{{DataVector{5, 0.34}}}};
  const Scalar<DataVector> pressure{{{DataVector{5, 0.23}}}};
  const Scalar<DataVector> lorentz_factor{{{DataVector{5, 1.5625}}}};
  const Scalar<DataVector> comoving_magnetic_field_magnitude{
      {{DataVector{5, 0.5660388679}}}};
  const Scalar<DataVector> lapse{{{DataVector{5, 1.}}}};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3>>(rest_mass_density, 0.);
  const auto shift =
      make_with_value<tnsr::I<DataVector, 3>>(rest_mass_density, 0.);
  auto magnetic_field =
      make_with_value<tnsr::I<DataVector, 3>>(rest_mass_density, 0.);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3>>(rest_mass_density, 0.);
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3>>(rest_mass_density, 0.);

  tnsr::AA<DataVector, 3> result{};

  spatial_velocity.get(0) = 0.6;
  magnetic_field.get(0) = 0.5;
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0;
    inverse_spatial_metric.get(i, i) = 1.0;
  }

  stress_energy_tensor(make_not_null(&result), rest_mass_density,
                       specific_internal_energy, pressure, lorentz_factor,
                       comoving_magnetic_field_magnitude, lapse,
                       spatial_velocity, shift, magnetic_field, spatial_metric,
                       inverse_spatial_metric);

  const auto box = db::create<
      db::AddSimpleTags<
          Tags::RestMassDensity<DataVector>,
          Tags::SpecificInternalEnergy<DataVector>, Tags::Pressure<DataVector>,
          Tags::LorentzFactor<DataVector>, ::gr::Tags::Lapse<DataVector>,
          Tags::ComovingMagneticFieldMagnitude<DataVector>,
          Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
          ::gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
          Tags::MagneticField<DataVector, 3, Frame::Inertial>,
          ::gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
          ::gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>,
      db::AddComputeTags<Tags::StressEnergyTensorCompute<DataVector>>>(
      rest_mass_density, specific_internal_energy, pressure, lorentz_factor,
      comoving_magnetic_field_magnitude, lapse, spatial_velocity, shift,
      magnetic_field, spatial_metric, inverse_spatial_metric);
  CHECK(db::get<Tags::StressEnergyTensorCompute<DataVector>>(box) == result);
}
}  // namespace hydro
