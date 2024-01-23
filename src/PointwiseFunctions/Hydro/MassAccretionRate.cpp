// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "MassAccretionRate.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace hydro {

template <typename DataType, size_t Dim, typename Frame>
void mass_accretion_rate(
    const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> result,
    const Scalar<DataType>& rest_mass_density,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const Scalar<DataType>& lapse) {
  for (size_t i = 0; i < Dim; ++i) {
    result->get(i) =
        get(rest_mass_density) * (get(lapse) * spatial_velocity.get(i));
  }
}

template <typename DataType, size_t Dim, typename Frame>
tnsr::I<DataType, Dim, Frame> mass_accretion_rate(
    const Scalar<DataType>& rest_mass_density,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const Scalar<DataType>& lapse) {
  tnsr::I<DataType, Dim, Frame> result{};
  mass_accretion_rate(make_not_null(&result), rest_mass_density,
                      spatial_velocity, lapse);
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INSTANTIATE(_, data)                                                 \
  template void mass_accretion_rate(                                         \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>     \
          result,                                                            \
      const Scalar<DTYPE(data)>& rest_mass_density,                          \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity,  \
      const Scalar<DTYPE(data)>& lapse);                                     \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> mass_accretion_rate( \
      const Scalar<DTYPE(data)>& rest_mass_density,                          \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity,  \
      const Scalar<DTYPE(data)>& lapse);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace hydro
