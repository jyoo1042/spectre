// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace gr {
/*!
 * \brief Contains helper functions for transforming tensors for Kerr
 * spacetime in Spherical-Kerr-Schild coordinates.
 */
class SphericalKerrSchildCoords {
 public:
  SphericalKerrSchildCoords() = default;
  SphericalKerrSchildCoords(const SphericalKerrSchildCoords& /*rhs*/) = default;
  SphericalKerrSchildCoords& operator=(
      const SphericalKerrSchildCoords& /*rhs*/) = default;
  SphericalKerrSchildCoords(SphericalKerrSchildCoords&& /*rhs*/) = default;
  SphericalKerrSchildCoords& operator=(SphericalKerrSchildCoords&& /*rhs*/) =
      default;
  ~SphericalKerrSchildCoords() = default;

  SphericalKerrSchildCoords(double bh_mass, double bh_dimless_spin);

  // Transforms a spatial vector from spherical
  // coordinates to Cartesian for Spherical-Kerr-Schild coordinates.
  // If applied on points on the z-axis, the vector to transform
  // must have a vanishing \f$v^\vartheta\f$ in order
  // for the transformation to be single-valued.
  template <typename DataType>
  tnsr::I<DataType, 3, Frame::Inertial> cartesian_from_spherical_ks(
      const tnsr::I<DataType, 3, Frame::NoFrame>& spatial_vector,
      const tnsr::I<DataType, 3, Frame::Inertial>& cartesian_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/);

 private:
  friend bool operator==(const SphericalKerrSchildCoords& lhs,
                         const SphericalKerrSchildCoords& rhs);

  // The spatial components of the Jacobian of the transformation from
  // spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
  // for Spherical-Kerr-Schild.
  template <typename DataType>
  tnsr::Ij<DataType, 3, Frame::NoFrame> jacobian_matrix(
      const DataType& x, const DataType& y, const DataType& z) const;

  double spin_a_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const SphericalKerrSchildCoords& lhs,
                const SphericalKerrSchildCoords& rhs);

}  // namespace gr
