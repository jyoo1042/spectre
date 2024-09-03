// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::AnalyticData {
/*!
 * \brief Torus made by removing two polar cones from a spherical shell
 *
 * Maps source coordinates \f$(\xi, \eta, \zeta)\f$ to
 * \f{align}
 * \vec{x}(\xi, \eta, \zeta) =
 * \begin{bmatrix}
 *  r \sin\theta\cos\phi \\
 *  r \sin\theta\sin\phi \\
 *  r \cos\theta
 * \end{bmatrix}
 * \f}
 *
 * where
 * \f{align}
 *  r & = r_\mathrm{min}\frac{1-\xi}{2} + r_\mathrm{max}\frac{1+\xi}{2}, \\
 *  \eta_\mathrm{new} & = a_\mathrm{comp} \eta^3 + (1-a) \eta
 *  \theta & = \pi/2 - (\pi/2 - \theta_\mathrm{min}) \eta_\mathrm{new}, \\
 *  \phi   & = f_\mathrm{torus} \pi \zeta.
 * \f}
 *
 *  - $r_\mathrm{min}$ and $r_\mathrm{max}$ are inner and outer radius of torus.
 *  - $\theta_\mathrm{min}\in(0,\pi/2)$ is the minimum polar angle (measured
 *    from +z axis) of torus, which is equal to the half of the apex angle of
 *    the removed polar cones.
 *  - $f_\mathrm{torus}\in(0, 1)$ is azimuthal fraction that the torus covers.
 *  - $a_\mathrm{comp}\in[0,1)$ sets the level of equatorial compression
 *    for theta, with zero being none.
 *
 */
class SphericalTorus {
 public:
  static constexpr size_t dim = 3;

  struct RadialRange {
    using type = std::array<double, 2>;
    static constexpr Options::String help =
        "Radial extent of the torus, "
        "[min_radius, max_radius] ";
  };

  struct MinPolarAngle {
    using type = double;
    static constexpr Options::String help =
        "Half of the apex angle of excised polar cones. "
        "Polar angle (measured from +z axis) of torus has range "
        "[MinPolarAngle, pi - MinPolarAngle]";
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 0.5 * M_PI; }
  };

  struct FractionOfTorus {
    using type = double;
    static constexpr Options::String help =
        "Fraction of (azimuthal) orbit covered. Azimuthal angle has range "
        "[- pi * FractionOfTorus, pi * FractionOfTorus].";
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  struct CompressionLevel {
    using type = double;
    static constexpr Options::String help =
        "Level of Equatorial Compression for the polar angle.";
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  static constexpr Options::String help =
      "Torus made by removing polar cones from a spherical shell";

  using options =
      tmpl::list<RadialRange, MinPolarAngle, FractionOfTorus, CompressionLevel>;

  SphericalTorus(const std::array<double, 2>& radial_range,
                 const double min_polar_angle, const double fraction_of_torus,
                 const double compression_level,
                 const Options::Context& context = {});

  SphericalTorus(double r_min, double r_max, double min_polar_angle,
                 double fraction_of_torus = 1.0, double compression_level = 0.0,
                 const Options::Context& context = {});

  SphericalTorus() = default;

  template <typename T>
  tnsr::I<T, 3> operator()(const tnsr::I<T, 3>& source_coords) const;

  tnsr::I<double, 3> inverse(const tnsr::I<double, 3>& target_coords) const;

  template <typename T>
  Jacobian<T, 3, Frame::BlockLogical, Frame::Inertial> jacobian(
      const tnsr::I<T, 3>& source_coords) const;

  template <typename T>
  InverseJacobian<T, 3, Frame::BlockLogical, Frame::Inertial> inv_jacobian(
      const tnsr::I<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ijj<T, 3, Frame::NoFrame> hessian(
      const tnsr::I<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ijk<T, 3, Frame::NoFrame> derivative_of_inv_jacobian(
      const tnsr::I<T, 3>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  bool is_identity() const { return false; }

 private:
  template <typename T>
  T radius(const T& x) const;

  template <typename T>
  void radius(gsl::not_null<T*> r, const T& x) const;

  template <typename T>
  T radius_inverse(const T& r) const;

  template <typename T>
  T cubic_compression(const T& x) const;

  double cubic_inversion(const double& x) const;

  friend bool operator==(const SphericalTorus& lhs, const SphericalTorus& rhs);

  double r_min_ = std::numeric_limits<double>::signaling_NaN();
  double r_max_ = std::numeric_limits<double>::signaling_NaN();
  double pi_over_2_minus_theta_min_ =
      std::numeric_limits<double>::signaling_NaN();
  double fraction_of_torus_ = std::numeric_limits<double>::signaling_NaN();
  double compression_level_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const SphericalTorus& lhs, const SphericalTorus& rhs);
}  // namespace grmhd::AnalyticData
