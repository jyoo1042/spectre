// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
// check that pressure equals energy density at upper bound of specific
// internal energy for adiabatic index > 2
void check_dominant_energy_condition_at_bound() {
  const auto seed = std::random_device{}();
  MAKE_GENERATOR(generator, seed);
  CAPTURE(seed);
  auto distribution = std::uniform_real_distribution<>{2.0, 3.0};
  const double adiabatic_index = distribution(generator);
  const double rest_mass_density = distribution(generator);
  const auto eos = EquationsOfState::IdealFluid<true>{adiabatic_index};
  const double specific_internal_energy =
      eos.specific_internal_energy_upper_bound(rest_mass_density);
  const double pressure = get(eos.pressure_from_density_and_energy(
      Scalar<double>{rest_mass_density},
      Scalar<double>{specific_internal_energy}));
  const double energy_density =
      rest_mass_density * (1.0 + specific_internal_energy);
  CAPTURE(rest_mass_density);
  CAPTURE(specific_internal_energy);
  CHECK(approx(pressure) == energy_density);
}

template <bool IsRelativistic>
void check_bounds() {
  const auto eos = EquationsOfState::IdealFluid<IsRelativistic>{1.5};
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(0.0 == eos.specific_internal_energy_lower_bound(1.0));
  if constexpr (IsRelativistic) {
    CHECK(1.0 == eos.specific_enthalpy_lower_bound());
  } else {
    CHECK(0.0 == eos.specific_enthalpy_lower_bound());
  }
  const double max_double = std::numeric_limits<double>::max();
  CHECK(max_double == eos.rest_mass_density_upper_bound());
  CHECK(max_double == eos.specific_internal_energy_upper_bound(1.0));
  CHECK(eos.baryon_mass() ==
        approx(hydro::units::geometric::default_baryon_mass));
}

template <bool IsRelativistic>
void check_nonzero_bounds() {
  const auto seed = std::random_device{}();
  MAKE_GENERATOR(generator, seed);
  CAPTURE(seed);
  const auto adiabatic_index = 1.5;
  CAPTURE(adiabatic_index);
  auto distribution = std::uniform_real_distribution<>{1.e-9, 1.e-5};
  auto min_temp = distribution(generator);
  CAPTURE(min_temp);
  const auto eos =
      EquationsOfState::IdealFluid<IsRelativistic>{adiabatic_index, min_temp};
  CHECK(min_temp == eos.temperature_lower_bound());
  CHECK(min_temp / (adiabatic_index - 1) ==
        eos.specific_internal_energy_lower_bound(1.0));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.IdealFluid",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  register_derived_classes_with_charm<EoS::EquationOfState<true, 2>>();
  register_derived_classes_with_charm<EoS::EquationOfState<false, 2>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};
  const double d_for_size = std::numeric_limits<double>::signaling_NaN();
  const DataVector dv_for_size(5);
  TestHelpers::EquationsOfState::test_get_clone(
      EoS::IdealFluid<true>(5.0 / 3.0));
  const auto eos = EoS::IdealFluid<true>{5.0 / 3.0};
  const auto other_eos = EoS::IdealFluid<true>{4.0 / 3.0};
  const auto other_type_eos = EoS::PolytropicFluid<true>{100.0, 2.0};
  CHECK(eos == eos);
  CHECK(eos != other_eos);
  CHECK(eos != other_type_eos);
  CHECK(*eos.promote_to_3d_eos() ==
        EoS::Equilibrium3D<EoS::IdealFluid<true>>(eos));
  CHECK(not eos.is_barotropic());
  CHECK(not other_eos.is_barotropic());

  TestHelpers::EquationsOfState::check(EoS::IdealFluid<true>{5.0 / 3.0},
                                       "IdealFluid", "ideal_fluid", d_for_size,
                                       5.0 / 3.0);
  TestHelpers::EquationsOfState::check(EoS::IdealFluid<true>{4.0 / 3.0},
                                       "IdealFluid", "ideal_fluid", dv_for_size,
                                       4.0 / 3.0);
  TestHelpers::EquationsOfState::check(EoS::IdealFluid<false>{5.0 / 3.0},
                                       "IdealFluid", "ideal_fluid", d_for_size,
                                       5.0 / 3.0);
  TestHelpers::EquationsOfState::check(EoS::IdealFluid<false>{4.0 / 3.0},
                                       "IdealFluid", "ideal_fluid", dv_for_size,
                                       4.0 / 3.0);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 2>>>(
          {"IdealFluid:\n"
           "  AdiabaticIndex: 1.6666666666666667\n"
           "  MinTemperature: 0.0\n"}),
      "IdealFluid", "ideal_fluid", d_for_size, 5.0 / 3.0);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 2>>>(
          {"IdealFluid:\n"
           "  AdiabaticIndex: 1.3333333333333333\n"
           "  MinTemperature: 0.0\n"}),
      "IdealFluid", "ideal_fluid", dv_for_size, 4.0 / 3.0);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 2>>>(
          {"IdealFluid:\n"
           "  AdiabaticIndex: 1.6666666666666667\n"
           "  MinTemperature: 0.0\n"}),
      "IdealFluid", "ideal_fluid", d_for_size, 5.0 / 3.0);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 2>>>(
          {"IdealFluid:\n"
           "  AdiabaticIndex: 1.3333333333333333\n"
           "  MinTemperature: 0.0\n"}),
      "IdealFluid", "ideal_fluid", dv_for_size, 4.0 / 3.0);

  check_bounds<true>();
  check_bounds<false>();
  check_nonzero_bounds<true>();
  check_nonzero_bounds<false>();
  check_dominant_energy_condition_at_bound();
}
