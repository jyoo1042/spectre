// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>

#include "PointwiseFunctions/Hydro/MRIWavelength.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {
template <typename DataType>
void mri_wavelength(
    const gsl::not_null<tnsr::I<DataType, 3>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& relativistic_specific_enthalpy,
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& angular_velocity,
    const tnsr::I<DataType, 3>& coordinates,
    const tnsr::A<DataType, 3>& comoving_magnetic_field){

  const gsl::not_null<tnsr::I<tnsr::I<DataType>

  for(size_t i = 0; i < 3; ++i) {
    result->get(i) = ((2 * M_PI) / (sqrt((get(rest_mass_density) *
                     get(relativistic_specific_enthalpy)) +
                     square(get(comoving_magnetic_field_magnitude))) *
                     get(angular_velocity)) *
                     ();
}

template <typename DataType>
Scalar<DataType> mri_wavelength(
    const gsl::not_null<tnsr::I<DataType, 3>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& relativistic_specific_enthalpy,
    const Scalar<DataType>& comoving_magnetic_field_magnitude
    const Scalar<DataType>& angular_velocity,
    const tnsr::A<DataVector, Dim, Fr>& comoving_magnetic_field){
  Scalar<DataType> result{};
  mri_wavelength(make_not_null(&result), rest_mass_density,
                 relativistic_specific_enthalpy,
                 comoving_magnetic_field_magnitude,
                 angular_velocity, comoving_magnetic_field);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template void mri_wavelength(                                             \
      const gsl::not_null<tnsr::I<DTYPE(data),3>*> result,                  \
      const Scalar<DTYPE(data)>& rest_mass_density,                         \
      const Scalar<DTYPE(data)>& relativistic_specific_enthalpy,            \
      const Scalar<DTYPE(data)>& comoving_magnetic_field_magnitude;         \
      const Scalar<DTYPE(data)>& angular_velocity                           \
      const tnsr::A<DTYPE(data), 3>& comoving_magnetic_field)               \
  template tnsr::I<DTYPE(data),3> mri_wavelength(                           \
      const Scalar<DTYPE(data)>& rest_mass_density,                         \
      const Scalar<DTYPE(data)>& relativistic_specific_enthalpy,            \
      const Scalar<DTYPE(data)>& comoving_magnetic_field_magnitude;         \
      const Scalar<DTYPE(data)>& angular_velocity                           \
      const tnsr::A<DTYPE(data), 3>& comoving_magnetic_field)               \

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace hydro
