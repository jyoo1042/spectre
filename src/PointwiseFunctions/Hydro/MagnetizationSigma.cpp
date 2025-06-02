// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/MagnetizationSigma.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {
template <typename DataType>
void magnetization_sigma(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& rest_mass_density) {
  get(*result) =
      square(get(comoving_magnetic_field_magnitude)) / get(rest_mass_density);
}

template <typename DataType>
Scalar<DataType> magnetization_sigma(
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& rest_mass_density) {
  Scalar<DataType> result{};
  magnetization_sigma(make_not_null(&result), comoving_magnetic_field_magnitude,
                      rest_mass_density);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template void magnetization_sigma(                                \
      gsl::not_null<Scalar<DTYPE(data)>*> result,                   \
      const Scalar<DTYPE(data)>& comoving_magnetic_field_magnitude, \
      const Scalar<DTYPE(data)>& rest_mass_density);                \
  template Scalar<DTYPE(data)> magnetization_sigma(                 \
      const Scalar<DTYPE(data)>& comoving_magnetic_field_magnitude, \
      const Scalar<DTYPE(data)>& rest_mass_density);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
}  // namespace hydro
