// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/MRIQualityFactor.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {
template <typename DataType>
void mri_quality_factor(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) {
  get(*result) = 1.0 + get(specific_internal_energy) +
                 get(pressure) / get(rest_mass_density);
}

template <typename DataType>
Scalar<DataType> mri_quality_factor(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) {
  Scalar<DataType> result{};
  mri_quality_factor(make_not_null(&result), rest_mass_density,
                                 specific_internal_energy, pressure);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                   \
  template void mri_quality_factor(                            \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,        \
      const Scalar<DTYPE(data)>& rest_mass_density,            \
      const Scalar<DTYPE(data)>& specific_internal_energy,     \
      const Scalar<DTYPE(data)>& pressure);                    \
  template Scalar<DTYPE(data)> mri_quality_factor(             \
      const Scalar<DTYPE(data)>& rest_mass_density,            \
      const Scalar<DTYPE(data)>& specific_internal_energy,     \
      const Scalar<DTYPE(data)>& pressure);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace hydro
