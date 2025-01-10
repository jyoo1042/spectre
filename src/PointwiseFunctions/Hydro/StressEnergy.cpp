// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/StressEnergy.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename DataType>
tnsr::A<DataType, 3> four_velocity(const tnsr::I<DataType, 3>& spatial_velocity,
                                   const tnsr::I<DataType, 3>& shift,
                                   const Scalar<DataType>& lorentz_factor,
                                   const Scalar<DataType>& lapse) {
  tnsr::A<DataType, 3> result{};
  get<0>(result) = get(lorentz_factor) / get(lapse);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i + 1) = get(lorentz_factor) *
                        (spatial_velocity.get(i) - shift.get(i) / get(lapse));
  }
  return result;
}

template tnsr::A<DataVector, 3> four_velocity(const tnsr::I<DataVector, 3>&,
                                              const tnsr::I<DataVector, 3>&,
                                              const Scalar<DataVector>&,
                                              const Scalar<DataVector>&);

template tnsr::A<double, 3> four_velocity(const tnsr::I<double, 3>&,
                                          const tnsr::I<double, 3>&,
                                          const Scalar<double>&,
                                          const Scalar<double>&);
}  // namespace

namespace hydro {

template <typename DataType>
void energy_density(gsl::not_null<Scalar<DataType>*> result,
                    const Scalar<DataType>& rest_mass_density,
                    const Scalar<DataType>& specific_enthalpy,
                    const Scalar<DataType>& pressure,
                    const Scalar<DataType>& lorentz_factor,
                    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
                    const Scalar<DataType>& comoving_magnetic_field_squared) {
  *result = rest_mass_density;
  get(*result) *= get(specific_enthalpy);
  get(*result) += get(comoving_magnetic_field_squared);
  get(*result) -= square(get(magnetic_field_dot_spatial_velocity));
  get(*result) *= square(get(lorentz_factor));
  get(*result) -= get(pressure);
  get(*result) -= 0.5 * get(comoving_magnetic_field_squared);
}

template <typename DataType>
void momentum_density(
    gsl::not_null<tnsr::I<DataType, 3>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy,
    const tnsr::I<DataType, 3>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, 3>& magnetic_field,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& comoving_magnetic_field_squared) {
  get<0>(*result) = (get(rest_mass_density) * get(specific_enthalpy) +
                     get(comoving_magnetic_field_squared) -
                     square(get(magnetic_field_dot_spatial_velocity))) *
                    square(get(lorentz_factor));
  get<1>(*result) = get<0>(*result);
  get<2>(*result) = get<0>(*result);
  for (size_t d = 0; d < 3; ++d) {
    result->get(d) *= spatial_velocity.get(d);
    result->get(d) -=
        get(magnetic_field_dot_spatial_velocity) * magnetic_field.get(d);
  }
}

template <typename DataType>
void stress_trace(gsl::not_null<Scalar<DataType>*> result,
                  const Scalar<DataType>& rest_mass_density,
                  const Scalar<DataType>& specific_enthalpy,
                  const Scalar<DataType>& pressure,
                  const Scalar<DataType>& spatial_velocity_squared,
                  const Scalar<DataType>& lorentz_factor,
                  const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
                  const Scalar<DataType>& comoving_magnetic_field_squared) {
  get(*result) =
      3. * get(pressure) +
      get(rest_mass_density) * get(specific_enthalpy) *
          (square(get(lorentz_factor)) - 1.) +
      get(comoving_magnetic_field_squared) *
          (square(get(lorentz_factor)) * get(spatial_velocity_squared) + 0.5) -
      square(get(magnetic_field_dot_spatial_velocity)) *
          (square(get(lorentz_factor)) * get(spatial_velocity_squared) + 1.);
}

template <typename DataType>
void stress_energy_tensor(
    gsl::not_null<tnsr::AA<DataType, 3>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure, const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& lapse,
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const tnsr::I<DataType, 3>& spatial_velocity,
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& magnetic_field,
    const tnsr::ii<DataType, 3>& spatial_metric,
    const tnsr::II<DataType, 3>& inverse_spatial_metric) {
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto magnetic_field_dot_spatial_velocity =
      dot_product(magnetic_field, spatial_velocity, spatial_metric);

  const auto comoving_magnetic_field_v = comoving_magnetic_field(
      spatial_velocity, magnetic_field, magnetic_field_dot_spatial_velocity,
      lorentz_factor, shift, lapse);

  const auto four_velocity_v =
      four_velocity(spatial_velocity, shift, lorentz_factor, lapse);

  const auto rho_h_star =
      (get(rest_mass_density) +
       get(rest_mass_density) * get(specific_internal_energy)) +
      get(pressure) ;

  const auto p_star =
      get(pressure) ;

  get<0, 0>(*result) = (rho_h_star * square(get<0>(four_velocity_v))) +
                       (p_star * get<0, 0>(inverse_spacetime_metric)) ;

  get<1, 0>(*result) =
      (rho_h_star * get<1>(four_velocity_v) * get<0>(four_velocity_v)) +
      (p_star * get<1, 0>(inverse_spacetime_metric)) ;

  get<1, 1>(*result) = (rho_h_star * square(get<1>(four_velocity_v))) +
                       (p_star * get<1, 1>(inverse_spacetime_metric)) ;

  get<2, 0>(*result) =
      (rho_h_star * get<2>(four_velocity_v) * get<0>(four_velocity_v)) +
      (p_star * get<2, 0>(inverse_spacetime_metric)) ;

  get<2, 1>(*result) =
      (rho_h_star * get<2>(four_velocity_v) * get<1>(four_velocity_v)) +
      (p_star * get<2, 1>(inverse_spacetime_metric)) ;

  get<2, 2>(*result) = (rho_h_star * square(get<2>(four_velocity_v))) +
                       (p_star * get<2, 2>(inverse_spacetime_metric)) ;

  get<3, 0>(*result) =
      (rho_h_star * get<3>(four_velocity_v) * get<0>(four_velocity_v)) +
      (p_star * get<3, 0>(inverse_spacetime_metric));

  get<3, 1>(*result) =
      (rho_h_star * get<3>(four_velocity_v) * get<1>(four_velocity_v)) +
      (p_star * get<3, 1>(inverse_spacetime_metric)) ;

  get<3, 2>(*result) =
      (rho_h_star * get<3>(four_velocity_v) * get<2>(four_velocity_v)) +
      (p_star * get<3, 2>(inverse_spacetime_metric)) ;

  get<3, 3>(*result) = (rho_h_star * square(get<3>(four_velocity_v))) +
                       (p_star * get<3, 3>(inverse_spacetime_metric)) ;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                              \
  template void energy_density(                                             \
      gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&,      \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&);                                          \
  template void momentum_density(                                           \
      gsl::not_null<tnsr::I<DTYPE(data), 3>*>, const Scalar<DTYPE(data)>&,  \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);              \
  template void stress_trace(                                               \
      gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&,      \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);              \
  template void stress_energy_tensor(                                       \
      gsl::not_null<tnsr::AA<DTYPE(data), 3>*>, const Scalar<DTYPE(data)>&, \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const tnsr::I<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&,       \
      const tnsr::ii<DTYPE(data), 3>&, const tnsr::II<DTYPE(data), 3>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector))

#undef DTYPE
#undef INSTANTIATION

}  // namespace hydro
