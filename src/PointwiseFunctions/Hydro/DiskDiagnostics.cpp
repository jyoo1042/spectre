// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/DiskDiagnostics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
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
// T^r_phi * sqrt(-g)
template <typename DataType>
void ldot(const gsl::not_null<Scalar<DataType>*> result,
          const tnsr::AA<DataType, 3>& stress_energy_tensor,
          const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates) {
  // Preallocate to minimize number of allocations.
  // TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
  //                       ::Tags::TempScalar<1, DataType>,
  //                       ::Tags::TempAb<2, 3, Frame::Inertial, DataType>,
  //                       ::Tags::Tempaa<3, 3, Frame::Inertial, DataType>>>
  //     buffer(get_size(get(lapse)));

  // auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  // auto& sqrt_det_g = get<::Tags::TempScalar<1, DataType>>(buffer);
  // auto& lowered_stress_energy_tensor =
  //     get<::Tags::TempAb<2, 3, Frame::Inertial, DataType>>(buffer);
  // auto& spacetime_metric_v =
  //     get<::Tags::Tempaa<3, 3, Frame::Inertial, DataType>>(buffer);

  // get(radius) = square(coordinates.get(0));
  // for (size_t i = 1; i < 3; ++i) {
  //   get(radius) += square(coordinates.get(i));
  // }
  // get(radius) = sqrt(get(radius));

  // get(sqrt_det_g) = get(determinant_and_inverse(spatial_metric).first);
  // get(sqrt_det_g) = sqrt(get(sqrt_det_g));
  // get(sqrt_det_g) *= get(lapse);

  // gr::spacetime_metric(make_not_null(&spacetime_metric_v), lapse, shift,
  //                      spatial_metric);

  // // T^i_j
  // tenex::evaluate<ti::A, ti::c>(
  //     make_not_null(&lowered_stress_energy_tensor),
  //     stress_energy_tensor(ti::A, ti::B) * spacetime_metric_v(ti::b, ti::c));

  // const auto conversion_factor1_v =
  //     conversion_factor1<DataType>(radius, coordinates);

  // *result = make_with_value<Scalar<DataType>>(lapse, 0.0);
  // for (size_t i = 0; i < 3; ++i) {
  //   for (size_t j = 0; j < 3; ++j) {
  //     get(*result) += lowered_stress_energy_tensor.get(i + 1, j + 1) *
  //                     conversion_factor1_v.get(i, j);
  //   }
  // }
  // get(*result) *= get(sqrt_det_g);
  TempBuffer<tmpl::list<
      ::Tags::TempScalar<0, DataType>, ::Tags::TempScalar<1, DataType>,
      ::Tags::TempScalar<2, DataType>, ::Tags::TempScalar<3, DataType>,

      ::Tags::TempI<4, 3, Frame::Inertial, DataType>>>
      buffer(get_size(get(lapse)));
  auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& sin_theta_squared = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& cos_theta_squared = get<::Tags::TempScalar<2, DataType>>(buffer);
  auto& arb_func = get<::Tags::TempScalar<3, DataType>>(buffer);
  auto& massflux = get<::Tags::TempI<4, 3, Frame::Inertial, DataType>>(buffer);

  get(radius) = square(coordinates.get(0));
  for (size_t i = 1; i < 3; ++i) {
    get(radius) += square(coordinates.get(i));
  }
  get(radius) = sqrt(get(radius));

  get(sin_theta_squared) = square(coordinates.get(0));
  get(sin_theta_squared) += square(coordinates.get(1));
  get(sin_theta_squared) /= square(get(radius));

  get(cos_theta_squared) = square(coordinates.get(2));
  get(cos_theta_squared) /= square(get(radius));

  get(arb_func) = get(sin_theta_squared) * -0.3 + get(cos_theta_squared) * 0.4 +
                  square(get(sin_theta_squared)) * -0.8 +
                  square(get(cos_theta_squared)) * -.6;

  for (size_t i = 0; i < 3; ++i) {
    massflux.get(i) = coordinates.get(i) / get(radius);
  }

  const auto conversion_factor2_v =
      conversion_factor2<DataType>(radius, coordinates);

  *result = make_with_value<Scalar<DataType>>(radius, 1.0);
  // for (size_t i = 0; i < 3; ++i) {
  //   get(*result) += massflux.get(i) * conversion_factor2_v.get(i);
  // }
  get(*result) = get(arb_func);
}

// T^r_t * sqrt(-g)
template <typename DataType>
void edot(const gsl::not_null<Scalar<DataType>*> result,
          const tnsr::AA<DataType, 3>& stress_energy_tensor,
          const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates) {
  // // Preallocate to minimize number of allocations.
  // TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
  //                       ::Tags::TempScalar<1, DataType>,
  //                       ::Tags::TempAb<2, 3, Frame::Inertial, DataType>,
  //                       ::Tags::Tempaa<3, 3, Frame::Inertial, DataType>>>
  //     buffer(get_size(get(lapse)));

  // auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  // auto& sqrt_det_g = get<::Tags::TempScalar<1, DataType>>(buffer);
  // auto& lowered_stress_energy_tensor =
  //     get<::Tags::TempAb<2, 3, Frame::Inertial, DataType>>(buffer);
  // auto& spacetime_metric_v =
  //     get<::Tags::Tempaa<3, 3, Frame::Inertial, DataType>>(buffer);

  // get(radius) = square(coordinates.get(0));
  // for (size_t i = 1; i < 3; ++i) {
  //   get(radius) += square(coordinates.get(i));
  // }
  // get(radius) = sqrt(get(radius));

  // get(sqrt_det_g) = get(determinant_and_inverse(spatial_metric).first);
  // get(sqrt_det_g) = sqrt(get(sqrt_det_g));
  // get(sqrt_det_g) *= get(lapse);

  // gr::spacetime_metric(make_not_null(&spacetime_metric_v), lapse, shift,
  //                      spatial_metric);

  // // T^i_j
  // tenex::evaluate<ti::A, ti::c>(
  //     make_not_null(&lowered_stress_energy_tensor),
  //     stress_energy_tensor(ti::A, ti::B) * spacetime_metric_v(ti::b, ti::c));

  // const auto conversion_factor2_v =
  //     conversion_factor2<DataType>(radius, coordinates);

  // *result = make_with_value<Scalar<DataType>>(lapse, 0.0);
  // for (size_t i = 0; i < 3; ++i) {
  //   get(*result) += lowered_stress_energy_tensor.get(i + 1, 0) *
  //                   conversion_factor2_v.get(i);
  // }
  // get(*result) *= -1.0 * get(sqrt_det_g);
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
                        ::Tags::TempScalar<1, DataType>,
                        ::Tags::TempI<2, 3, Frame::Inertial, DataType>>>
      buffer(get_size(get(lapse)));
  auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& sin_theta_squared = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& massflux = get<::Tags::TempI<2, 3, Frame::Inertial, DataType>>(buffer);

  get(radius) = square(coordinates.get(0));
  for (size_t i = 1; i < 3; ++i) {
    get(radius) += square(coordinates.get(i));
  }
  get(radius) = sqrt(get(radius));

  get(sin_theta_squared) = square(coordinates.get(0));
  get(sin_theta_squared) += square(coordinates.get(1));
  get(sin_theta_squared) /= square(get(radius));

  for (size_t i = 0; i < 3; ++i) {
    massflux.get(i) = coordinates.get(i) / get(radius);
  }

  const auto conversion_factor2_v =
      conversion_factor2<DataType>(radius, coordinates);

  *result = make_with_value<Scalar<DataType>>(radius, 1.0);
  // for (size_t i = 0; i < 3; ++i) {
  //   get(*result) += massflux.get(i) * conversion_factor2_v.get(i);
  // }
  get(*result) *= pow<20>(get(sin_theta_squared));
}

// rho * u^r * sqrt(-g)
template <typename DataType>
void mdot(const gsl::not_null<Scalar<DataType>*> result,
          const Scalar<DataType>& rest_mass_density,
          const Scalar<DataType>& lorentz_factor,
          const tnsr::I<DataType, 3>& spatial_velocity,
          const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates) {
  // Preallocate to minimize number of allocations.
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
                        ::Tags::TempScalar<1, DataType>,
                        ::Tags::TempI<2, 3, Frame::Inertial, DataType>>>
      buffer(get_size(get(lapse)));
  auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& cos_theta_squared = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& massflux = get<::Tags::TempI<2, 3, Frame::Inertial, DataType>>(buffer);

  get(radius) = square(coordinates.get(0));
  for (size_t i = 1; i < 3; ++i) {
    get(radius) += square(coordinates.get(i));
  }
  get(radius) = sqrt(get(radius));

  get(cos_theta_squared) = square(coordinates.get(2));
  get(cos_theta_squared) /= square(get(radius));

  for (size_t i = 0; i < 3; ++i) {
    massflux.get(i) = get(cos_theta_squared) * coordinates.get(i) / get(radius);
  }

  const auto conversion_factor2_v =
      conversion_factor2<DataType>(radius, coordinates);

  *result = make_with_value<Scalar<DataType>>(lapse, 1.0);
  // for (size_t i = 0; i < 3; ++i) {
  //   get(*result) += massflux.get(i) * conversion_factor2_v.get(i);
  // }
  get(*result) *= get(cos_theta_squared);
}

// b^r * sqrt(gamma)
template <typename DataType>
void bdot(const gsl::not_null<Scalar<DataType>*> result,
          const tnsr::I<DataType, 3>& magnetic_field,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates) {
  // Preallocate to minimize number of allocations.
  // TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
  //                       ::Tags::TempScalar<1, DataType>>>
  //     buffer(get_size(get<0>(magnetic_field)));
  // auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  // auto& sqrt_det_gamma = get<::Tags::TempScalar<1, DataType>>(buffer);

  // get(radius) = square(coordinates.get(0));
  // for (size_t i = 1; i < 3; ++i) {
  //   get(radius) += square(coordinates.get(i));
  // }
  // get(radius) = sqrt(get(radius));

  // get(sqrt_det_gamma) = get(determinant_and_inverse(spatial_metric).first);
  // get(sqrt_det_gamma) = sqrt(get(sqrt_det_gamma));

  // const auto conversion_factor2_v =
  //     conversion_factor2<DataType>(radius, coordinates);

  // *result = make_with_value<Scalar<DataType>>(radius, 0.0);
  // for (size_t i = 0; i < 3; ++i) {
  //   get(*result) += magnetic_field.get(i) * conversion_factor2_v.get(i);
  // }
  // get(*result) = 0.5 * abs(get(*result) * get(sqrt_det_gamma));
  // Preallocate to minimize number of allocations.
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
                        ::Tags::TempScalar<1, DataType>,
                        ::Tags::TempI<2, 3, Frame::Inertial, DataType>>>
      buffer(get_size(magnetic_field.get(0)));
  auto& radius = get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& sin_theta_squared = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& massflux = get<::Tags::TempI<2, 3, Frame::Inertial, DataType>>(buffer);

  get(radius) = square(coordinates.get(0));
  for (size_t i = 1; i < 3; ++i) {
    get(radius) += square(coordinates.get(i));
  }
  get(radius) = sqrt(get(radius));

  get(sin_theta_squared) = square(coordinates.get(0));
  get(sin_theta_squared) += square(coordinates.get(1));
  get(sin_theta_squared) /= square(get(radius));

  for (size_t i = 0; i < 3; ++i) {
    massflux.get(i) = get(sin_theta_squared) * coordinates.get(i) / get(radius);
  }

  const auto conversion_factor2_v =
      conversion_factor2<DataType>(radius, coordinates);

  *result = make_with_value<Scalar<DataType>>(radius, 1.0);
  // for (size_t i = 0; i < 3; ++i) {
  //   get(*result) += massflux.get(i) * conversion_factor2_v.get(i);
  // }
  get(*result) *= get(sin_theta_squared);
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
      const tnsr::ii<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&);     \
  template void mdot(                                                       \
      gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&,      \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const tnsr::ii<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&);     \
  template void bdot(                                                       \
      gsl::not_null<Scalar<DTYPE(data)>*>, const tnsr::I<DTYPE(data), 3>&,  \
      const tnsr::ii<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector))

#undef DTYPE
#undef INSTANTIATION
}  // namespace hydro
