// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/DiskDiagnostics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename DataType>
tnsr::ij<DataType, 3> conversion_factor1(
    const Scalar<DataType>& radius, const tnsr::I<DataType, 3>& coordinates) {
  const auto one_over_radius = 1 / get(radius);
  auto result = make_with_value<tnsr::ij<DataType, 3>>(radius, 0.0);
  result.get(0, 0) = -one_over_radius * coordinates.get(0) * coordinates.get(1);
  result.get(0, 1) = one_over_radius * square(coordinates.get(0));
  result.get(1, 0) = -one_over_radius * square(coordinates.get(1));
  result.get(1, 1) = one_over_radius * coordinates.get(0) * coordinates.get(1);
  result.get(2, 0) = -one_over_radius * coordinates.get(1) * coordinates.get(2);
  result.get(2, 1) = one_over_radius * coordinates.get(0) * coordinates.get(2);
  return result;
}

template <typename DataType>
tnsr::i<DataType, 3> conversion_factor2(
    const Scalar<DataType>& radius, const tnsr::I<DataType, 3>& coordinates) {
  const auto one_over_radius = 1 / get(radius);
  auto result = make_with_value<tnsr::i<DataType, 3>>(radius, 0.0);
  result.get(0) = one_over_radius * coordinates.get(0);
  result.get(1) = one_over_radius * coordinates.get(1);
  result.get(2) = one_over_radius * coordinates.get(2);
  return result;
}

}  // namespace

namespace hydro {

template <typename DataType>
void edot(const gsl::not_null<Scalar<DataType>*> result,
          const tnsr::AA<DataType, 3>& stress_energy_tensor,
          const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates) {
  // Preallocate to minimize number of allocations.
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
                        ::Tags::TempAb<1, 3, Frame::Inertial, DataType>,
                        ::Tags::Tempaa<2, 3, Frame::Inertial, DataType>>>
      buffer(get_size(get(lapse)));

  auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& lowered_stress_energy_tensor =
      get<::Tags::TempAb<1, 3, Frame::Inertial, DataType>>(buffer);
  auto& spacetime_metric_v =
      get<::Tags::Tempaa<2, 3, Frame::Inertial, DataType>>(buffer);

  for (size_t i = 0; i < 3; ++i) {
    if (i == 0) {
      get(radius) = square(coordinates.get(0));
    } else {
      get(radius) = square(coordinates.get(i));
    }
  }
  get(radius) = sqrt(get(radius));

  gr::spacetime_metric(make_not_null(&spacetime_metric_v), lapse, shift,
                       spatial_metric);

  tenex::evaluate<ti::A, ti::c>(
      make_not_null(&lowered_stress_energy_tensor),
      stress_energy_tensor(ti::A, ti::B) * spacetime_metric_v(ti::b, ti::c));

  const auto conversion_factor1_v =
      conversion_factor1<DataType>(radius, coordinates);

  get(*result) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) += lowered_stress_energy_tensor.get(i + 1, j + 1) *
                      conversion_factor1_v.get(i, j);
    }
  }
}

template <typename DataType>
void ldot(const gsl::not_null<Scalar<DataType>*> result,
          const tnsr::AA<DataType, 3>& stress_energy_tensor,
          const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates) {
  // Preallocate to minimize number of allocations.
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
                        ::Tags::TempAb<1, 3, Frame::Inertial, DataType>,
                        ::Tags::Tempaa<2, 3, Frame::Inertial, DataType>>>
      buffer(get_size(get(lapse)));

  auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& lowered_stress_energy_tensor =
      get<::Tags::TempAb<1, 3, Frame::Inertial, DataType>>(buffer);
  auto& spacetime_metric_v =
      get<::Tags::Tempaa<2, 3, Frame::Inertial, DataType>>(buffer);

  for (size_t i = 0; i < 3; ++i) {
    if (i == 0) {
      get(radius) = square(coordinates.get(0));
    } else {
      get(radius) = square(coordinates.get(i));
    }
  }
  get(radius) = sqrt(get(radius));

  gr::spacetime_metric(make_not_null(&spacetime_metric_v), lapse, shift,
                       spatial_metric);

  tenex::evaluate<ti::A, ti::c>(
      make_not_null(&lowered_stress_energy_tensor),
      stress_energy_tensor(ti::A, ti::B) * spacetime_metric_v(ti::b, ti::c));

  const auto conversion_factor2_v =
      conversion_factor2<DataType>(radius, coordinates);

  get(*result) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    get(*result) += lowered_stress_energy_tensor.get(i + 1, 0) *
                    conversion_factor2_v.get(i);
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                              \
  template void edot(                                                       \
      gsl::not_null<Scalar<DTYPE(data)>*>, const tnsr::AA<DTYPE(data), 3>&, \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const tnsr::ii<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&);     \
  template void ldot(                                                       \
      gsl::not_null<Scalar<DTYPE(data)>*>, const tnsr::AA<DTYPE(data), 3>&, \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const tnsr::ii<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector))

#undef DTYPE
#undef INSTANTIATION
}  // namespace hydro
