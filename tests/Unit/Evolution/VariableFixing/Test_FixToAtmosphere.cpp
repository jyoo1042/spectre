// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_fix_reconstructed_state_to_atmosphere() {
  using Frs = VariableFixing::FixReconstructedStateToAtmosphere;
  for (const auto t :
       {Frs::Always, Frs::AtDgFdInterfaceOnly, Frs::OnFdOnly, Frs::Never}) {
    CHECK(t == TestHelpers::test_creation<Frs>(get_output(t)));
  }
}

template <size_t Dim>
void test_variable_fixer(
    const VariableFixing::FixToAtmosphere<Dim>& variable_fixer,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) {
  // 2.e-12 -> below cutoff, velocity goes to zero
  // 2.e-11 -> above cutoff, no changes
  // 4.e-12 -> density unchanged, velocity restricted
  // 4.e-10 -> b^2/density > 100 for given magnetic field. density increased
  //           pressure and energy increaed accordingly.

  const bool use_magnetization_limiting =
      variable_fixer.magnetization_limiting() != std::nullopt;

  Scalar<DataVector> density{DataVector{2.e-12, 2.e-11, 4.e-12, 4e-10}};
  auto pressure = equation_of_state.pressure_from_density(density);
  auto specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density(density);
  auto temperature = equation_of_state.temperature_from_density(density);
  const Scalar<DataVector> electron_fraction{
      DataVector{get(density).size(), 0.5}};
  auto magnetic_field =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  // comoing magnetic field squared, b^2 = 7.84 e-4 given values of spatial
  // metric and spatial_velocity specified below.
  magnetic_field.get(0) = DataVector{0.0, 0.0, 0.0, 2.e-4};

  Scalar<DataVector> lorentz_factor{DataVector{
      5.0 / 3.0, 7.0710678118654752, 1.8898223650461359, 1.4433756729740643}};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  spatial_velocity.get(0) = DataVector{0.8, 0.7, 0.6, 0.5};
  spatial_velocity.get(1) = DataVector{0.0, 0.0, 0.0, 0.1};
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    spatial_metric.get(i, i) = 2.0;
  }
  Scalar<DataVector> expected_density{
      use_magnetization_limiting ? DataVector{1.e-12, 2.e-11, 4.e-12, 7.84e-10}
                                 : DataVector{1.e-12, 2.e-11, 4.e-12, 4.e-10}};
  auto expected_pressure =
      equation_of_state.pressure_from_density(expected_density);
  auto expected_specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density(expected_density);
  Scalar<DataVector> expected_lorentz_factor{DataVector{
      1.0, 7.0710678118654752, 1.0000000001020408, 1.4433756729740643}};
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  expected_spatial_velocity.get(0)[1] = 0.7;
  // The [2] component of the expected velocity is:
  //     velocity *= max_velocity_magnitude_ * (rho - rho_cut) /
  //                 (trans_rho - rho_cut) / |v^i|
  expected_spatial_velocity.get(0)[2] = 0.6 * (4.e-12 - 3.e-12) /
                                        (1.e-11 - 3.e-12) * 1.e-4 /
                                        sqrt(0.6 * 0.6 * 2.0);
  expected_spatial_velocity.get(0)[3] = 0.5;
  expected_spatial_velocity.get(1)[3] = 0.1;
  // If we apply magnetization limiting, enthalpy of the system is
  // increased when rest mass density and pressure
  // is increased to limit magnetization and inverse plasma beta is capped.
  // The [3] component of the expected velocity is modified such that
  // fluid momentum parallel to magnetic field is conserved with these
  // changes in fluid enthalpy.
  // Hence, parallel component of spatial velocity is decreased as following:
  // new v_parallel = x/(W_perp*(1+sqrt(1+x^2)))
  // where x = (2 * v_parallel * W^2 / W_perp) * (old enthalpy)/(new enthalpy)
  // enthalpy, h = rho + rho * epsilon + P
  // for given hard coded numbers:
  // W = 1.4433756729740643, W_perp = 1.0101525445522106
  // old_enthalpy/new_enthalpy = 0.5102040816326531
  // v_parallel = (v_i B^i)/B = 0.7071067811865476
  // magnetic field magnitude, B = sqrt(8) * 10^-4
  // new_v^i = v^i + (new_v_parallel-v_parallel)* B^i /B
  if (use_magnetization_limiting) {
    expected_spatial_velocity.get(0)[3] = 0.37297189902720029;
    get(expected_lorentz_factor)[3] = 1.1937085153735216;
  }

  variable_fixer(&density, &specific_internal_energy, &spatial_velocity,
                 &lorentz_factor, &pressure, &temperature, electron_fraction,
                 magnetic_field, spatial_metric, equation_of_state);

  CHECK_ITERABLE_APPROX(density, expected_density);
  CHECK_ITERABLE_APPROX(pressure, expected_pressure);
  CHECK_ITERABLE_APPROX(specific_internal_energy,
                        expected_specific_internal_energy);

  Approx approx = Approx::custom().epsilon(1.e-9);
  CHECK_ITERABLE_CUSTOM_APPROX(lorentz_factor, expected_lorentz_factor, approx);
  CHECK_ITERABLE_CUSTOM_APPROX(spatial_velocity, expected_spatial_velocity,
                               approx);
}

template <size_t Dim>
void test_variable_fixer(
    const VariableFixing::FixToAtmosphere<Dim>& variable_fixer,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state,
    const bool use_kappa_limiting, const double min_temperature) {
  CAPTURE(use_kappa_limiting);
  CAPTURE(min_temperature);
  // This test assumes magnetization limiting is disabled.
  REQUIRE(variable_fixer.magnetization_limiting() == std::nullopt);

  Scalar<DataVector> density{DataVector{2.e-12, 2.e-11, 4.e-12, 2.e-11}};
  Scalar<DataVector> specific_internal_energy{DataVector{
      2., 3., 3.,
      get(equation_of_state
              .specific_internal_energy_from_density_and_temperature(
                  Scalar<double>(get(density)[3]), Scalar<double>(-1e-5)))}};
  auto pressure = equation_of_state.pressure_from_density_and_energy(
      density, specific_internal_energy);
  auto temperature = equation_of_state.temperature_from_density_and_energy(
      density, specific_internal_energy);
  const Scalar<DataVector> electron_fraction{
      DataVector{get(density).size(), 0.5}};
  auto magnetic_field =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  magnetic_field.get(0) = DataVector{0.0, 0.0, 0.0, 0.0};

  Scalar<DataVector> lorentz_factor{DataVector{
      5. / 3., 7.0710678118654752, 1.8898223650461359, 7.0710678118654752}};
  CHECK(get(lorentz_factor).size() == get(density).size());
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.);
  spatial_velocity.get(0) = DataVector{0.8, 0.7, 0.6, 0.7};
  CHECK(spatial_velocity.get(0).size() == get(density).size());
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(density, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    spatial_metric.get(i, i) = 2.;
  }
  variable_fixer(&density, &specific_internal_energy, &spatial_velocity,
                 &lorentz_factor, &pressure, &temperature, electron_fraction,
                 magnetic_field, spatial_metric, equation_of_state);

  Scalar<DataVector> expected_density{
      DataVector{1.e-12, 2.e-11, 4.e-12, 2.e-11}};
  const auto compute_temperature = [&equation_of_state, &expected_density,
                                    &min_temperature](const size_t i) {
    const double density_lower_bound = 3.e-12;
    const double density_upper_bound = 3.e-11;
    const double epsilon_kappa_pm = 1.1;
    const bool below =
        get(expected_density)[i] < epsilon_kappa_pm * density_lower_bound;
    const double p_min = get(equation_of_state.pressure_from_density_and_energy(
        Scalar<double>{get(expected_density)[i]},
        equation_of_state.specific_internal_energy_from_density_and_temperature(
            Scalar<double>{get(expected_density)[i]},
            Scalar<double>{min_temperature})));
    return get(equation_of_state.temperature_from_density_and_energy(
        Scalar<double>{get(expected_density)[i]},
        equation_of_state.specific_internal_energy_from_density_and_pressure(
            Scalar<double>{get(expected_density)[i]},
            Scalar<double>{
                p_min * (1.0 + 0.01 * (below ? 1.0
                                             : (get(expected_density)[i] -
                                                density_lower_bound) /
                                                   (density_upper_bound -
                                                    density_lower_bound)))})));
  };

  Scalar<DataVector> expected_specific_internal_energy{
      use_kappa_limiting
          ? (min_temperature == 0.0 ? DataVector{4, 0.}
                                    :  // do more complicated kappa limiting
                 get(equation_of_state
                         .specific_internal_energy_from_density_and_temperature(
                             expected_density,
                             Scalar<DataVector>{DataVector{
                                 min_temperature, compute_temperature(1),
                                 compute_temperature(2), 0.0}})))
          : DataVector{
                0.0, 3.0, 3.0,
                get(equation_of_state
                        .specific_internal_energy_from_density_and_temperature(
                            Scalar<double>(get(expected_density)[3]),
                            Scalar<double>(0.)))}};
  auto expected_pressure = equation_of_state.pressure_from_density_and_energy(
      expected_density, expected_specific_internal_energy);
  auto expected_temperature =
      equation_of_state.temperature_from_density_and_energy(
          expected_density, expected_specific_internal_energy);
  Scalar<DataVector> expected_lorentz_factor{DataVector{
      1., 7.0710678118654752, 1.0000000001020408, 7.0710678118654752}};
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.);
  expected_spatial_velocity.get(0)[1] = 0.7;
  // The [2] component of the expected velocity is:
  //     velocity *= max_velocity_magnitude_ * (rho - rho_cut) /
  //                 (trans_rho - rho_cut) / |v^i|
  expected_spatial_velocity.get(0)[2] = 0.6 * (4.e-12 - 3.e-12) /
                                        (1.e-11 - 3.e-12) * 1.e-4 /
                                        sqrt(0.6 * 0.6 * 2.);
  // The [3] component is the same as the [1] component
  expected_spatial_velocity.get(0)[3] = expected_spatial_velocity.get(0)[1];

  CHECK(get(expected_lorentz_factor).size() == get(expected_density).size());
  CHECK_ITERABLE_APPROX(density, expected_density);
  CHECK_ITERABLE_APPROX(pressure, expected_pressure);
  CHECK_ITERABLE_APPROX(temperature, expected_temperature);
  CHECK_ITERABLE_APPROX(specific_internal_energy,
                        expected_specific_internal_energy);
  CHECK_ITERABLE_APPROX(lorentz_factor, expected_lorentz_factor);
  CHECK_ITERABLE_APPROX(spatial_velocity, expected_spatial_velocity);
}

template <size_t Dim>
void test_variable_fixer(
    const VariableFixing::FixToAtmosphere<Dim>& variable_fixer,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state) {
  // This tests checks the magnetization limitng and assumes kappa limiting is
  // disabled.

  REQUIRE(variable_fixer.kappa_limiting() == std::nullopt);

  // 1st case: rho, P needed to be increased
  // 2nd case: only rho need to be increased
  // 3rd case: only P need to be increased
  // 4th case: no need for change.
  Scalar<DataVector> density{DataVector{4.e-10, 4.e-10, 8.e-10, 8.e-10}};
  Scalar<DataVector> specific_internal_energy{
      DataVector{7.5e-2, 1.5, 3.75e-2, 1.5}};
  CHECK(get(specific_internal_energy).size() == get(density).size());

  auto pressure = equation_of_state.pressure_from_density_and_energy(
      density, specific_internal_energy);
  auto temperature = equation_of_state.temperature_from_density_and_energy(
      density, specific_internal_energy);
  const Scalar<DataVector> electron_fraction{
      DataVector{get(density).size(), 0.5}};
  auto magnetic_field =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  // comoing magnetic field squared, b^2 = 7.84 e-4 given values of spatial
  // metric and spatial_velocity specified below.
  magnetic_field.get(0) = DataVector{get(density).size(), 2.e-4};

  Scalar<DataVector> lorentz_factor{
      DataVector{get(density).size(), 1.4433756729740643}};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.);
  spatial_velocity.get(0) = DataVector{get(density).size(), 0.5};
  spatial_velocity.get(1) = DataVector{get(density).size(), 0.1};
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(density, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    spatial_metric.get(i, i) = 2.;
  }
  Scalar<DataVector> expected_density{
      DataVector{7.84e-10, 7.84e-10, 8.e-10, 8.e-10}};
  Scalar<DataVector> expected_pressure{
      DataVector{3.92e-11, 4.e-10, 3.92e-11, 8.e-10}};
  auto expected_specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density_and_pressure(
          expected_density, expected_pressure);
  Scalar<DataVector> expected_lorentz_factor{
      DataVector{1.1937085153218085, 1.3358820890287881, 1.4171496689248413,
                 1.4433756729740643}};
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  expected_spatial_velocity.get(0) = DataVector{
      {0.3729718989864436, 0.4580635119850747, 0.4909529983426436, 0.5}};
  expected_spatial_velocity.get(1) = DataVector{get(density).size(), 0.1};

  variable_fixer(&density, &specific_internal_energy, &spatial_velocity,
                 &lorentz_factor, &pressure, &temperature, electron_fraction,
                 magnetic_field, spatial_metric, equation_of_state);

  CHECK_ITERABLE_APPROX(density, expected_density);
  CHECK_ITERABLE_APPROX(pressure, expected_pressure);
  CHECK_ITERABLE_APPROX(specific_internal_energy,
                        expected_specific_internal_energy);

  //   Approx approx = Approx::custom().epsilon(1.e-14);
  CHECK_ITERABLE_APPROX(lorentz_factor, expected_lorentz_factor);
  CHECK_ITERABLE_APPROX(spatial_velocity, expected_spatial_velocity);
}

template <size_t Dim>
void test_variable_fixer() {
  using Vlo =
      typename VariableFixing::FixToAtmosphere<Dim>::VelocityLimitingOptions;
  using Klo =
      typename VariableFixing::FixToAtmosphere<Dim>::KappaLimitingOptions;
  using Mlo = typename VariableFixing::FixToAtmosphere<
      Dim>::MagnetizationLimitingOptions;
  // Test for representative 1-d equation of state
  const VariableFixing::FixToAtmosphere<Dim> variable_fixer_klo{
      1.e-12, 3.e-12, Vlo{0.0, 1.e-4, 3.e-12, 1.e-11},
      Klo{3.e-12, 1.e-3, 3.e-11, 0.01, std::nullopt, false}, std::nullopt};
  const VariableFixing::FixToAtmosphere<Dim> variable_fixer{
      1.e-12, 3.e-12, Vlo{0.0, 1.e-4, 3.e-12, 1.e-11}, std::nullopt,
      std::nullopt};
  const VariableFixing::FixToAtmosphere<Dim> variable_fixer_mlo{
      1.e-12, 3.e-12, Vlo{0.0, 1.e-4, 3.e-12, 1.e-11}, std::nullopt,
      Mlo{1.e2, 1.e3}};
  EquationsOfState::PolytropicFluid<true> polytrope{1.0, 2.0};
  test_variable_fixer<Dim>(variable_fixer, polytrope);
  test_variable_fixer<Dim>(variable_fixer_klo, polytrope);
  test_variable_fixer<Dim>(variable_fixer_mlo, polytrope);

  test_serialization(variable_fixer);
  test_serialization(variable_fixer_klo);
  test_serialization(variable_fixer_mlo);

  const auto check_velocity_limiting_options = [](const auto& var_fixer) {
    CHECK(var_fixer.density_of_atmosphere() == 1.e-12);
    CHECK(var_fixer.density_cutoff() == 3.e-12);
    REQUIRE(var_fixer.velocity_limiting().has_value());
    CHECK(var_fixer.velocity_limiting().value().atmosphere_max_velocity == 0.);
    CHECK(var_fixer.velocity_limiting().value().near_atmosphere_max_velocity ==
          1.e-4);
    CHECK(var_fixer.velocity_limiting().value().atmosphere_density_cutoff ==
          3.e-12);
    CHECK(var_fixer.velocity_limiting().value().transition_density_bound ==
          1.e-11);
  };
  check_velocity_limiting_options(variable_fixer);
  check_velocity_limiting_options(variable_fixer_klo);
  check_velocity_limiting_options(variable_fixer_mlo);

  CHECK(not variable_fixer.kappa_limiting().has_value());
  REQUIRE(variable_fixer_klo.kappa_limiting().has_value());
  CHECK(variable_fixer_klo.kappa_limiting()->density_lower_bound == 3.e-12);
  CHECK(variable_fixer_klo.kappa_limiting()->eplison_kappa_minus == 1.e-3);
  CHECK(variable_fixer_klo.kappa_limiting()->density_upper_bound == 3.e-11);
  CHECK(variable_fixer_klo.kappa_limiting()->epsilon_kappa_max == 0.01);
  CHECK(variable_fixer_klo.kappa_limiting()->min_temperature == std::nullopt);
  CHECK(variable_fixer_klo.kappa_limiting()->limit_above_density_upper_bound ==
        false);

  const auto fixer_from_options =
      TestHelpers::test_creation<VariableFixing::FixToAtmosphere<Dim>>(
          "DensityOfAtmosphere: 1.0e-12\n"
          "DensityCutoff: 3.0e-12\n"
          "VelocityLimiting:\n"
          "  AtmosphereMaxVelocity: 0\n"
          "  NearAtmosphereMaxVelocity: 1.0e-4\n"
          "  AtmosphereDensityCutoff: 3.0e-12\n"
          "  TransitionDensityBound: 1.0e-11\n"
          "KappaLimiting: Disabled\n"
          "MagnetizationLimiting: Disabled\n");
  const auto fixer_from_options_klo =
      TestHelpers::test_creation<VariableFixing::FixToAtmosphere<Dim>>(
          "DensityOfAtmosphere: 1.0e-12\n"
          "DensityCutoff: 3.0e-12\n"
          "VelocityLimiting:\n"
          "  AtmosphereMaxVelocity: 0\n"
          "  NearAtmosphereMaxVelocity: 1.0e-4\n"
          "  AtmosphereDensityCutoff: 3.0e-12\n"
          "  TransitionDensityBound: 1.0e-11\n"
          "KappaLimiting:\n"
          "  DensityLowerBound: 3.0e-12\n"
          "  EplisonKappaMinus: 1.0e-3\n"
          "  DensityUpperBound: 3.0e-11\n"
          "  EpsilonKappaMax: 0.01\n"
          "  MinTemperature: 1.0e-3\n"
          "  LimitAboveDensityUpperBound: False\n"
          "MagnetizationLimiting: Disabled\n");
  const auto fixer_from_options_mlo =
      TestHelpers::test_creation<VariableFixing::FixToAtmosphere<Dim>>(
          "DensityOfAtmosphere: 1.0e-12\n"
          "DensityCutoff: 3.0e-12\n"
          "VelocityLimiting:\n"
          "  AtmosphereMaxVelocity: 0\n"
          "  NearAtmosphereMaxVelocity: 1.0e-4\n"
          "  AtmosphereDensityCutoff: 3.0e-12\n"
          "  TransitionDensityBound: 1.0e-11\n"
          "KappaLimiting: Disabled\n"
          "MagnetizationLimiting:\n"
          "  MagnetizationBound: 1.0e2\n"
          "  InversePlasmaBetaBound: 1.0e3\n");
  test_variable_fixer(fixer_from_options, polytrope);
  test_variable_fixer(fixer_from_options_klo, polytrope);
  test_variable_fixer(fixer_from_options_mlo, polytrope);

  // Test for representative 2-d equation of state
  EquationsOfState::IdealFluid<true> ideal_fluid{5.0 / 3.0};
  test_variable_fixer<Dim>(variable_fixer, ideal_fluid, false, 0.0);
  test_variable_fixer<Dim>(variable_fixer_klo, ideal_fluid, true, 0.0);
  test_variable_fixer<Dim>(variable_fixer_mlo, ideal_fluid);

  test_variable_fixer<Dim>(fixer_from_options, ideal_fluid, false, 1.0e-3);
  test_variable_fixer<Dim>(fixer_from_options_klo, ideal_fluid, true, 1.0e-3);
  test_variable_fixer<Dim>(fixer_from_options_mlo, ideal_fluid);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.FixToAtmosphere",
                  "[VariableFixing][Unit]") {
  test_fix_reconstructed_state_to_atmosphere();
  //   test_variable_fixer<1>();
  //   test_variable_fixer<2>();
  test_variable_fixer<3>();
}
