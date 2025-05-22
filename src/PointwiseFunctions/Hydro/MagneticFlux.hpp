// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {
/// @{
/// Computes the vector \f$J^i\f$ in \f$\Phi{B} = \int J^i s_i d^2S\f$,
/// representing the magnetic flux through a surface with normal \f$s_i\f$.
///
/// Note that the integral is understood
/// as a flat-space integral: all metric factors are included in \f$J^i\f$.
/// In particular, if the integral is done over a Strahlkorper, the
/// `gr::surfaces::euclidean_area_element` of the Strahlkorper should be used,
/// and \f$s_i\f$ is
/// the normal one-form to the Strahlkorper normalized with the flat metric,
/// \f$s_is_j\delta^{ij}=1\f$.
///
/// The formula is
/// \f$ J^i = 0.5 * \sqrt{\gamma} * B^i \f$,
/// \f$B^i\f$ is the magnetic field,
/// \f$\gamma\f$ is the determinant of the 3-metric \f$\gamma_{ij}\f$.
/// @}
template <typename DataType, size_t Dim, typename Fr>
void magnetic_flux(gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
                   const tnsr::I<DataType, Dim, Fr>& magnetic_field,
                   const Scalar<DataType>& sqrt_det_spatial_metric);

template <typename DataType, size_t Dim, typename Fr>
tnsr::I<DataType, Dim, Fr> magnetic_flux(
    const tnsr::I<DataType, Dim, Fr>& magnetic_field,
    const Scalar<DataType>& sqrt_det_spatial_metric);
namespace Tags {
template <typename DataType, size_t Dim, typename Fr>
struct MagneticFlux : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "MagneticFlux"; }
};

/// Compute item for magnetic flux vector \f$J^i\f$.
///
/// Can be retrieved using `hydro::Tags::MagneticFlux`
template <typename DataType, size_t Dim, typename Fr>
struct MagneticFluxCompute : MagneticFlux<DataType, Dim, Fr>, db::ComputeTag {
  using argument_tags =
      tmpl::list<hydro::Tags::MagneticField<DataType, Dim, Fr>,
                 ::gr::Tags::SqrtDetSpatialMetric<DataType>>;

  using return_type = tnsr::I<DataType, Dim, Fr>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::I<DataType, Dim, Fr>*>,
      const tnsr::I<DataType, Dim, Fr>&, const Scalar<DataType>&)>(
      &magnetic_flux<DataType, Dim, Fr>);

  using base = MagneticFlux<DataType, Dim, Fr>;
};
}  // namespace Tags
}  // namespace hydro
