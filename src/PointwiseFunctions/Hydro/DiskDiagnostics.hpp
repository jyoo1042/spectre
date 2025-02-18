// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
///// \endcond

namespace hydro {
template <typename DataType>
void edot(gsl::not_null<Scalar<DataType>*> result,
          const tnsr::AA<DataType, 3>& stress_energy_tensor,
          const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates);

template <typename DataType>
void ldot(gsl::not_null<Scalar<DataType>*> result,
          const tnsr::AA<DataType, 3>& stress_energy_tensor,
          const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
          const tnsr::ii<DataType, 3>& spatial_metric,
          const tnsr::I<DataType, 3>& coordinates);

namespace Tags {

template <typename DataType>
struct Edot : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct Ldot : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Can be retrieved using `hydro::Tags::Edot`
template <typename DataType, typename OutputCoordsTag>
struct EdotCompute : Edot<DataType>, db::ComputeTag {
  using base = Edot<DataType>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<StressEnergyTensor<DataType>, ::gr::Tags::Lapse<DataType>,
                 ::gr::Tags::Shift<DataType, 3, Frame::Inertial>,
                 ::gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>,
                 OutputCoordsTag>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*> result,
      const tnsr::AA<DataType, 3>& stress_energy_tensor,
      const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
      const tnsr::ii<DataType, 3>& spatial_metric,
      const tnsr::I<DataType, 3>& coordinates)>(&hydro::edot<DataType>);
};

/// Can be retrieved using `hydro::Tags::Ldot`
template <typename DataType, typename OutputCoordsTag>
struct LdotCompute : Ldot<DataType>, db::ComputeTag {
  using base = Ldot<DataType>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<StressEnergyTensor<DataType>, ::gr::Tags::Lapse<DataType>,
                 ::gr::Tags::Shift<DataType, 3, Frame::Inertial>,
                 ::gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>,
                 OutputCoordsTag>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*> result,
      const tnsr::AA<DataType, 3>& stress_energy_tensor,
      const Scalar<DataType>& lapse, const tnsr::I<DataType, 3>& shift,
      const tnsr::ii<DataType, 3>& spatial_metric,
      const tnsr::I<DataType, 3>& coordinates)>(&hydro::ldot<DataType>);
};

}  // namespace Tags
}  // namespace hydro
