// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
///// \endcond

namespace hydro {
/// @{
/*!
 * \brief Computes the inverse plasma beta
 *
 * The inverse plasma beta \f$\beta^{-1} = b^2 / \rho\f$, where
 * \f$b^2\f$ is the square of the comoving magnetic field amplitude
 * and \f$\rho\f$ is the rest mass density.
 */

template <typename DataType>
void magnetization_sigma(
    gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& rest_mass_density);

template <typename DataType>
Scalar<DataType> magnetization_sigma(
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& rest_mass_density);
/// @}

namespace Tags {
template <typename DataType>
struct MagnetizationSigma : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Can be retrieved using `hydro::Tags::MagnetizationSigma`
template <typename DataType>
struct MagnetizationSigmaCompute : MagnetizationSigma<DataType>,
                                   db::ComputeTag {
  using base = MagnetizationSigma<DataType>;
  using return_type = Scalar<DataType>;

  using argument_tags = tmpl::list<ComovingMagneticFieldMagnitude<DataType>,
                                   RestMassDensity<DataType>>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataType>*>,
                           const Scalar<DataType>&, const Scalar<DataType>&)>(
          &magnetization_sigma<DataType>);
};
}  // namespace Tags
}  // namespace hydro
