// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "MagneticFlux.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace hydro {

template <typename DataType, size_t Dim, typename Fr>
void magnetic_flux(const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
                   const tnsr::I<DataType, Dim, Fr>& magnetic_field,
                   const Scalar<DataType>& sqrt_det_spatial_metric) {
  for (size_t i = 0; i < Dim; ++i) {
    result->get(i) = 0.5 * get(sqrt_det_spatial_metric) * magnetic_field.get(i);
  }
}

template <typename DataType, size_t Dim, typename Fr>
tnsr::I<DataType, Dim, Fr> magnetic_flux(
    const tnsr::I<DataType, Dim, Fr>& magnetic_field,
    const Scalar<DataType>& sqrt_det_spatial_metric) {
  tnsr::I<DataType, Dim, Fr> result{};
  magnetic_flux(make_not_null(&result), magnetic_field,
                sqrt_det_spatial_metric);
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INSTANTIATE(_, data)                                              \
  template void magnetic_flux(                                            \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>  \
          result,                                                         \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& magnetic_field, \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric);                \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> magnetic_flux(    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& magnetic_field, \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace hydro
