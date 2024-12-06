// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/Nada.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {

Nada::Nada(CkMigrateMessage* msg) : InitialData(msg) {}

Nada::Nada(const double bh_mass, const double bh_dimless_spin,
           const double inner_edge_radius, const double outer_edge_radius,
           const double polytropic_constant, const double polytropic_exponent)
    : bh_mass_(bh_mass),
      bh_spin_a_(bh_mass * bh_dimless_spin),
      inner_edge_radius_(bh_mass * inner_edge_radius),
      outer_edge_radius_(bh_mass * outer_edge_radius),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      background_spacetime_{
          bh_mass_, {{0.0, 0.0, bh_dimless_spin}}, {{0.0, 0.0, 0.0}}} {}

std::unique_ptr<evolution::initial_data::InitialData> Nada::get_clone() const {
  return std::make_unique<Nada>(*this);
}

void Nada::pup(PUP::er& p) {
  InitialData::pup(p);
  p | bh_mass_;
  p | bh_spin_a_;
  p | inner_edge_radius_;
  p | outer_edge_radius_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
Nada::IntermediateVariables<DataType>::IntermediateVariables(
    const tnsr::I<DataType, 3>& x) {
  r_squared = square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>> Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  // initialized to 0 first
  auto rest_mass_density = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(vars, [&rest_mass_density](const size_t s) {
    get_element(get(rest_mass_density), s) = 1.0;
  });

  return {std::move(rest_mass_density)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>> Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> /* vars */) const {
  auto ye = make_with_value<Scalar<DataType>>(x, 0.1);

  return {std::move(ye)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));
  auto pressure = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(vars, [&pressure, &rest_mass_density, this](const size_t s) {
    get_element(get(pressure), s) =
        get(equation_of_state_.pressure_from_density(
            Scalar<double>{get_element(get(rest_mass_density), s)}));
  });
  return {std::move(pressure)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));
  auto specific_internal_energy = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(vars, [&specific_internal_energy, &rest_mass_density,
                        this](const size_t s) {
    get_element(get(specific_internal_energy), s) =
        get(equation_of_state_.specific_internal_energy_from_density(
            Scalar<double>{get_element(get(rest_mass_density), s)}));
  });
  return {std::move(specific_internal_energy)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));

  auto temperature = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(
      vars, [&temperature, &rest_mass_density, this](const size_t s) {
        get_element(get(temperature), s) =
            polytropic_constant_ * pow(get_element(get(rest_mass_density), s),
                                       polytropic_exponent_ - 1.0);
      });
  return {std::move(temperature)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>> Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>,
    gsl::not_null<IntermediateVariables<DataType>*> /* vars */) const {
  auto spatial_velocity = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  return {std::move(spatial_velocity)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>> Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> /* vars */) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>> Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> /* vars */) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
Nada::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> /* vars */) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType, typename Func>
void Nada::variables_impl(gsl::not_null<IntermediateVariables<DataType>*> vars,
                          Func f) const {
  // using spherical kerr schild hence r = sqrt(x^2+y^2+z^2)
  const DataType& r_squared = vars->r_squared;

  // fill in the disk matter only when in between inner and outer edge radius
  for (size_t s = 0; s < get_size(r_squared); ++s) {
    const double r_squared_s = get_element(r_squared, s);

    if (sqrt(r_squared_s) >= inner_edge_radius_ &&
        sqrt(r_squared_s) <= outer_edge_radius_) {
      f(s);
    }
  }
}

PUP::able::PUP_ID Nada::my_PUP_ID = 0;
bool operator==(const Nada& lhs, const Nada& rhs) {
  return lhs.bh_mass_ == rhs.bh_mass_ and lhs.bh_spin_a_ == rhs.bh_spin_a_ and
         lhs.inner_edge_radius_ == rhs.inner_edge_radius_ and
         lhs.outer_edge_radius_ == rhs.outer_edge_radius_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const Nada& lhs, const Nada& rhs) { return not(lhs == rhs); }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template class Nada::IntermediateVariables<DTYPE(data)>;                    \
  template tuples::TaggedTuple<hydro::Tags::RestMassDensity<DTYPE(data)>>     \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::RestMassDensity<DTYPE(data)>> /*meta*/,         \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<hydro::Tags::ElectronFraction<DTYPE(data)>>    \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::ElectronFraction<DTYPE(data)>> /*meta*/,        \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<hydro::Tags::Pressure<DTYPE(data)>>            \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::Pressure<DTYPE(data)>> /*meta*/,                \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::SpecificInternalEnergy<DTYPE(data)>>                       \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/,  \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<hydro::Tags::Temperature<DTYPE(data)>>         \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::Temperature<DTYPE(data)>> /*meta*/,             \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<hydro::Tags::MagneticField<DTYPE(data), 3>>    \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3,                   \
                                            Frame::Inertial>> /*meta*/,       \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::DivergenceCleaningField<DTYPE(data)>>                      \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::DivergenceCleaningField<DTYPE(data)>> /*meta*/, \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DTYPE(data), 3>>  \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> /*meta*/,      \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;   \
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>>       \
  Nada::variables(                                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/,           \
      gsl::not_null<Nada::IntermediateVariables<DTYPE(data)>*> vars) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace grmhd::AnalyticData
