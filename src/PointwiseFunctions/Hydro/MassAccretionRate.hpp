// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare gr::Tags::Lapse
// IWYU pragma: no_forward_declare gr::Tags::Shift
// IWYU pragma: no_forward_declare gr::Tags::SqrtDetSpatialMetric
// IWYU pragma: no_forward_declare hydro::Tags::LorentzFactor
// IWYU pragma: no_forward_declare hydro::Tags::RestMassDensity
// IWYU pragma: no_forward_declare hydro::Tags::SpatialVelocity

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {
/// @{
/// Computes FIXME: write up docu later.
template <typename DataType, size_t Dim, typename Frame>
void mass_accretion_rate(gsl::not_null<tnsr::I<DataType, Dim, Frame>*> result,
                         const Scalar<DataType>& rest_mass_density,
                         const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
                         const Scalar<DataType>& lapse);

template <typename DataType, size_t Dim, typename Frame>
tnsr::I<DataType, Dim, Frame> mass_accretion_rate(
    const Scalar<DataType>& rest_mass_density,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const Scalar<DataType>& lapse);
/// @}

namespace Tags {
/// FIXME: write up docu
///
/// Can be retrieved using `hydro::Tags::MassAccretionRate`
template <typename DataType, size_t Dim, typename Frame>
struct MassAccretionRateCompute : MassAccretionRate<DataType, Dim, Frame>,
                                  db::ComputeTag {
  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, Dim, Frame>,
                 ::gr::Tags::Lapse<DataType>>;

  using return_type = tnsr::I<DataType, Dim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::I<DataType, Dim, Frame>*>, const Scalar<DataType>&,
      const tnsr::I<DataType, Dim, Frame>&, const Scalar<DataType>&)>(
      &mass_accretion_rate<DataType, Dim, Frame>);

  using base = MassAccretionRate<DataType, Dim, Frame>;
};
}  // namespace Tags
}  // namespace hydro
