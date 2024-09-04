// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/SphericalTorus.hpp"

#include <pup.h>

#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"

namespace grmhd::AnalyticData {

SphericalTorus::SphericalTorus(const double r_min, const double r_max,
                               const double min_polar_angle,
                               const double fraction_of_torus,
                               const double compression_level,
                               const Options::Context& context)
    : r_min_(r_min),
      r_max_(r_max),
      pi_over_2_minus_theta_min_(M_PI_2 - min_polar_angle),
      fraction_of_torus_(fraction_of_torus),
      compression_level_(compression_level) {
  if (r_min_ <= 0.0) {
    PARSE_ERROR(context, "Minimum radius must be positive.");
  }
  if (r_max_ <= r_min_) {
    PARSE_ERROR(context, "Maximum radius must be greater than minimum radius.");
  }
  if (pi_over_2_minus_theta_min_ <= 0.0) {
    PARSE_ERROR(context, "Minimum polar angle should be less than pi/2.");
  }
  if (pi_over_2_minus_theta_min_ >= M_PI_2) {
    PARSE_ERROR(context, "Minimum polar angle should be positive.");
  }
  if (fraction_of_torus_ <= 0.0) {
    PARSE_ERROR(context, "Fraction of torus included must be positive.");
  }
  if (fraction_of_torus_ > 1.0) {
    PARSE_ERROR(context, "Fraction of torus included must be at most 1.");
  }
  if (compression_level_ < 0.0) {
    PARSE_ERROR(context, "Compression level must be non-negative.");
  }
  // a = 1 can lead to 1/0 when computing inv_jacobian so we simply
  // avoid this issue by restricting to a <1.
  if (compression_level_ >= 1.0) {
    PARSE_ERROR(context, "Compression level must be less than 1.");
  }
}

SphericalTorus::SphericalTorus(const std::array<double, 2>& radial_range,
                               const double min_polar_angle,
                               const double fraction_of_torus,
                               const double compression_level,
                               const Options::Context& context)
    : SphericalTorus(radial_range[0], radial_range[1], min_polar_angle,
                     fraction_of_torus, compression_level, context) {}

//
//  * r       : radius
//  * theta   : polar angle is measured from +z axis.
//  * phi     : azimuthal angle
//
template <typename T>
tnsr::I<T, 3> SphericalTorus::operator()(
    const tnsr::I<T, 3>& source_coords) const {
  tnsr::I<T, 3> result;
  const auto r = radius(get<0>(source_coords));
  // intermediate variable for equatorial compression
  const auto intermediate_val = cubic_compression(get<1>(source_coords));
  const auto theta = M_PI_2 - (pi_over_2_minus_theta_min_ * intermediate_val);
  const auto phi = M_PI * fraction_of_torus_ * get<2>(source_coords);
  get<0>(result) = r * sin(theta) * cos(phi);
  get<1>(result) = r * sin(theta) * sin(phi);
  get<2>(result) = r * cos(theta);
  return result;
}

tnsr::I<double, 3> SphericalTorus::inverse(
    const tnsr::I<double, 3>& target_coords) const {
  using std::abs;

  tnsr::I<double, 3> result;
  const double r = std::hypot(get<0>(target_coords), get<1>(target_coords),
                              get<2>(target_coords));
  const double theta =
      std::atan2(std::hypot(get<0>(target_coords), get<1>(target_coords)),
                 get<2>(target_coords));
  const double phi = std::atan2(get<1>(target_coords), get<0>(target_coords));
  get<0>(result) = radius_inverse(r);

  get<1>(result) = (M_PI_2 - theta) / pi_over_2_minus_theta_min_;
  // if we are using compression (compression_level_ !=0), there is an extra
  // step of inverting back to original eta before the cubic compression map.
  if (compression_level_ != 0.0) {
    get<1>(result) = cubic_inversion(get<1>(result));
  }
  get<2>(result) = phi / (M_PI * fraction_of_torus_);
  return result;
}

template <typename T>
Jacobian<T, 3, Frame::BlockLogical, Frame::Inertial> SphericalTorus::jacobian(
    const tnsr::I<T, 3>& source_coords) const {
  auto jacobian =
      make_with_value<Jacobian<T, 3, Frame::BlockLogical, Frame::Inertial>>(
          get<0>(source_coords), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // jacobian for storing temp variables as below.

  auto& sin_theta = get<2, 1>(jacobian);
  auto& cos_theta = get<2, 0>(jacobian);
  // intermediate variable for equatorial compression
  get<2, 2>(jacobian) = cubic_compression(get<1>(source_coords));
  get<2, 2>(jacobian) =
      M_PI_2 - (pi_over_2_minus_theta_min_ * get<2, 2>(jacobian));
  sin_theta = sin(get<2, 2>(jacobian));
  cos_theta = cos(get<2, 2>(jacobian));

  auto& sin_phi = get<1, 1>(jacobian);
  auto& cos_phi = get<1, 2>(jacobian);
  get<2, 2>(jacobian) = M_PI * fraction_of_torus_ * get<2>(source_coords);
  sin_phi = sin(get<2, 2>(jacobian));
  cos_phi = cos(get<2, 2>(jacobian));

  auto& r = get<2, 2>(jacobian);
  radius(make_not_null(&r), get<0>(source_coords));

  // Note : execution order matters here since we are overwriting each temp
  // variables with jacobian values corresponding to the slot.
  get<0, 0>(jacobian) = 0.5 * (r_max_ - r_min_) * sin_theta * cos_phi;
  get<0, 1>(jacobian) = -pi_over_2_minus_theta_min_ * r * cos_theta * cos_phi;
  get<0, 2>(jacobian) = -M_PI * fraction_of_torus_ * r * sin_theta * sin_phi;
  get<1, 0>(jacobian) = 0.5 * (r_max_ - r_min_) * sin_theta * sin_phi;
  get<1, 1>(jacobian) = -pi_over_2_minus_theta_min_ * r * cos_theta * sin_phi;
  get<1, 2>(jacobian) = M_PI * fraction_of_torus_ * r * sin_theta * cos_phi;
  get<2, 0>(jacobian) = 0.5 * (r_max_ - r_min_) * cos_theta;
  get<2, 1>(jacobian) = pi_over_2_minus_theta_min_ * r * sin_theta;

  // an extra factor of partial deriv. from equatorial compression map
  get<2, 2>(jacobian) =
      3.0 * compression_level_ * pow<2>(get<1>(source_coords)) +
      (1.0 - compression_level_);
  get<0, 1>(jacobian) *= get<2, 2>(jacobian);
  get<1, 1>(jacobian) *= get<2, 2>(jacobian);
  get<2, 1>(jacobian) *= get<2, 2>(jacobian);
  // Set it back to 0
  get<2, 2>(jacobian) = 0.0;
  return jacobian;
}

template <typename T>
InverseJacobian<T, 3, Frame::BlockLogical, Frame::Inertial>
SphericalTorus::inv_jacobian(const tnsr::I<T, 3>& source_coords) const {
  auto inv_jacobian = make_with_value<
      InverseJacobian<T, 3, Frame::BlockLogical, Frame::Inertial>>(
      get<0>(source_coords), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // jacobian for storing temp variables as below.

  auto& sin_theta = get<1, 2>(inv_jacobian);
  auto& cos_theta = get<0, 2>(inv_jacobian);
  // intermediate variable for equatorial compression
  get<2, 2>(inv_jacobian) = cubic_compression(get<1>(source_coords));
  get<2, 2>(inv_jacobian) =
      M_PI_2 - (pi_over_2_minus_theta_min_ * get<2, 2>(inv_jacobian));
  cos_theta = cos(get<2, 2>(inv_jacobian));
  sin_theta = sin(get<2, 2>(inv_jacobian));

  auto& sin_phi = get<1, 1>(inv_jacobian);
  auto& cos_phi = get<2, 1>(inv_jacobian);
  get<2, 2>(inv_jacobian) = M_PI * fraction_of_torus_ * get<2>(source_coords);
  cos_phi = cos(get<2, 2>(inv_jacobian));
  sin_phi = sin(get<2, 2>(inv_jacobian));

  auto& r = get<2, 2>(inv_jacobian);
  radius(make_not_null(&r), get<0>(source_coords));

  // Note : execution order matters here since we are overwriting each temp
  // variables with jacobian values corresponding to the slot.
  get<0, 0>(inv_jacobian) = 2.0 / (r_max_ - r_min_) * sin_theta * cos_phi;
  get<0, 1>(inv_jacobian) = 2.0 / (r_max_ - r_min_) * sin_theta * sin_phi;
  get<1, 0>(inv_jacobian) =
      -(1.0 / pi_over_2_minus_theta_min_) * cos_theta * cos_phi / r;
  get<2, 0>(inv_jacobian) =
      -(1.0 / (M_PI * fraction_of_torus_)) * sin_phi / (r * sin_theta);
  get<1, 1>(inv_jacobian) =
      -(1.0 / pi_over_2_minus_theta_min_) * cos_theta * sin_phi / r;
  get<2, 1>(inv_jacobian) =
      (1.0 / (M_PI * fraction_of_torus_)) * cos_phi / (r * sin_theta);
  get<0, 2>(inv_jacobian) = 2.0 / (r_max_ - r_min_) * cos_theta;
  get<1, 2>(inv_jacobian) = (1.0 / pi_over_2_minus_theta_min_) * sin_theta / r;

  // an extra factor of partial deriv. from equatorial compression map.
  get<2, 2>(inv_jacobian) =
      (3.0 * compression_level_ * pow<2>(get<1>(source_coords)) +
       (1.0 - compression_level_));
  get<1, 0>(inv_jacobian) /= get<2, 2>(inv_jacobian);
  get<1, 1>(inv_jacobian) /= get<2, 2>(inv_jacobian);
  get<1, 2>(inv_jacobian) /= get<2, 2>(inv_jacobian);

  // set it back to 0.
  get<2, 2>(inv_jacobian) = 0.0;

  return inv_jacobian;
}

template <typename T>
tnsr::Ijj<T, 3, Frame::NoFrame> SphericalTorus::hessian(
    const tnsr::I<T, 3>& source_coords) const {
  auto hessian = make_with_value<tnsr::Ijj<T, 3, Frame::NoFrame>>(
      get<0>(source_coords), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // hessian for storing temp variables as below.

  const double theta_factor = pi_over_2_minus_theta_min_;
  const double phi_factor = M_PI * fraction_of_torus_;

  // these two slots are zeros (not used)
  // these two corresponds to the derivative and double derivative of the
  // cubic function we use for the equatorial compression.
  auto& eta_prime = get<2, 0, 0>(hessian);
  auto& eta_double_prime = get<2, 0, 2>(hessian);
  eta_prime = 3 * compression_level_ * pow<2>(get<1>(source_coords)) + 1 -
              compression_level_;
  eta_double_prime = 6.0 * compression_level_ * get<1>(source_coords);

  // these two slots are zeros (not used)
  auto& cos_theta = get<2, 1, 2>(hessian);
  auto& sin_theta = get<2, 2, 2>(hessian);
  get<0, 0, 0>(hessian) = cubic_compression(get<1>(source_coords));
  get<0, 0, 0>(hessian) = M_PI_2 - (theta_factor * get<0, 0, 0>(hessian));
  cos_theta = cos(get<0, 0, 0>(hessian));
  sin_theta = sin(get<0, 0, 0>(hessian));

  // these two slots are NOT zeros, but will be overwritten with hessian later
  auto& cos_phi = get<2, 0, 1>(hessian);
  auto& sin_phi = get<2, 1, 1>(hessian);
  get<0, 0, 0>(hessian) = phi_factor * get<2>(source_coords);
  cos_phi = cos(get<0, 0, 0>(hessian));
  sin_phi = sin(get<0, 0, 0>(hessian));

  // these two slots are zeros (not used)
  auto& r_factor = get<0, 0, 0>(hessian);
  auto& r = get<1, 0, 0>(hessian);
  r_factor = 0.5 * (r_max_ - r_min_);
  radius(make_not_null(&r), get<0>(source_coords));

  // Note : execution order matters here
  get<0, 0, 1>(hessian) =
      -r_factor * theta_factor * cos_theta * cos_phi * eta_prime;
  get<0, 0, 2>(hessian) = -r_factor * phi_factor * sin_theta * sin_phi;
  get<0, 1, 1>(hessian) =
      -square(theta_factor * eta_prime) * r * sin_theta * cos_phi -
      theta_factor * r * cos_theta * cos_phi * eta_double_prime;
  get<0, 1, 2>(hessian) =
      phi_factor * theta_factor * r * cos_theta * sin_phi * eta_prime;
  get<0, 2, 2>(hessian) = -square(phi_factor) * r * sin_theta * cos_phi;
  get<1, 0, 1>(hessian) =
      -r_factor * theta_factor * cos_theta * sin_phi * eta_prime;
  get<1, 0, 2>(hessian) = r_factor * phi_factor * sin_theta * cos_phi;
  get<1, 1, 1>(hessian) =
      -square(theta_factor * eta_prime) * r * sin_theta * sin_phi -
      theta_factor * r * cos_theta * sin_phi * eta_double_prime;
  get<1, 1, 2>(hessian) =
      -phi_factor * theta_factor * r * cos_theta * cos_phi * eta_prime;
  get<1, 2, 2>(hessian) = -square(phi_factor) * r * sin_theta * sin_phi;
  get<2, 0, 1>(hessian) = r_factor * theta_factor * sin_theta * eta_prime;
  get<2, 1, 1>(hessian) = -square(theta_factor * eta_prime) * r * cos_theta +
                          theta_factor * r * sin_theta * eta_double_prime;

  // remove temp vars and restore to zero
  get<0, 0, 0>(hessian) = 0.0;
  get<1, 0, 0>(hessian) = 0.0;
  get<2, 0, 0>(hessian) = 0.0;
  get<2, 0, 2>(hessian) = 0.0;
  get<2, 1, 2>(hessian) = 0.0;
  get<2, 2, 2>(hessian) = 0.0;

  return hessian;
}

template <typename T>
tnsr::Ijk<T, 3, Frame::NoFrame> SphericalTorus::derivative_of_inv_jacobian(
    const tnsr::I<T, 3>& source_coords) const {
  auto result = make_with_value<tnsr::Ijk<T, 3, Frame::NoFrame>>(
      get<0>(source_coords), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // `result` for storing temp variables as below.

  const double theta_factor = pi_over_2_minus_theta_min_;
  const double phi_factor = M_PI * fraction_of_torus_;

  auto& cos_phi = get<1, 2, 2>(result);
  auto& sin_phi = get<2, 2, 0>(result);
  get<0, 0, 0>(result) = M_PI * fraction_of_torus_ * get<2>(source_coords);
  cos_phi = cos(get<0, 0, 0>(result));
  sin_phi = sin(get<0, 0, 0>(result));

  auto& cos_theta = get<2, 2, 1>(result);
  auto& sin_theta = get<2, 2, 2>(result);
  get<0, 0, 0>(result) = cubic_compression(get<1>(source_coords));
  get<0, 0, 0>(result) =
      M_PI_2 - (pi_over_2_minus_theta_min_ * get<0, 0, 0>(result));
  cos_theta = cos(get<0, 0, 0>(result));
  sin_theta = sin(get<0, 0, 0>(result));

  // these two corresponds to the derivative and double derivative of the
  // cubic function we use for the equatorial compression.
  auto& eta_prime = get<0, 2, 0>(result);
  auto& eta_double_prime = get<0, 2, 2>(result);
  eta_prime = 3.0 * compression_level_ * pow<2>(get<1>(source_coords)) + 1.0 -
              compression_level_;
  eta_double_prime = 6.0 * compression_level_ * get<1>(source_coords);

  auto& r = get<0, 0, 0>(result);
  auto& r_factor = get<0, 1, 0>(result);
  radius(make_not_null(&r), get<0>(source_coords));
  r_factor = 0.5 * (r_max_ - r_min_);

  get<0, 0, 1>(result) =
      -theta_factor / r_factor * cos_theta * cos_phi * eta_prime;
  get<0, 0, 2>(result) = -phi_factor / r_factor * sin_theta * sin_phi;
  get<0, 1, 1>(result) =
      -theta_factor / r_factor * cos_theta * sin_phi * eta_prime;
  get<0, 1, 2>(result) = phi_factor / r_factor * sin_theta * cos_phi;
  get<0, 2, 1>(result) = theta_factor / r_factor * sin_theta * eta_prime;
  get<1, 0, 0>(result) =
      r_factor / theta_factor * cos_theta * cos_phi / (square(r) * eta_prime);
  get<1, 0, 1>(result) =
      (-cos_phi / r) * ((sin_theta) - ((cos_theta * eta_double_prime) /
                                       (square(eta_prime) * theta_factor)));
  get<1, 0, 2>(result) =
      phi_factor / theta_factor * cos_theta * sin_phi / (r * eta_prime);
  get<1, 1, 0>(result) =
      r_factor / theta_factor * cos_theta * sin_phi / (square(r) * eta_prime);
  get<1, 1, 1>(result) =
      (-sin_phi / r) * ((sin_theta) - ((cos_theta * eta_double_prime) /
                                       (square(eta_prime) * theta_factor)));
  get<1, 1, 2>(result) =
      -phi_factor / theta_factor * cos_theta * cos_phi / (r * eta_prime);
  get<1, 2, 0>(result) =
      -r_factor / theta_factor * sin_theta / (square(r) * eta_prime);
  get<1, 2, 1>(result) =
      (-1.0 / r) * (cos_theta + (sin_theta * eta_double_prime /
                                 (square(eta_prime) * theta_factor)));
  get<2, 0, 0>(result) =
      r_factor / phi_factor * sin_phi / (square(r) * sin_theta);
  get<2, 0, 1>(result) = -theta_factor / phi_factor * cos_theta * sin_phi *
                         eta_prime / (r * square(sin_theta));
  get<2, 0, 2>(result) = -cos_phi / (r * sin_theta);
  get<2, 1, 0>(result) =
      -r_factor / phi_factor * cos_phi / (square(r) * sin_theta);
  get<2, 1, 1>(result) = theta_factor / phi_factor * cos_theta * cos_phi *
                         eta_prime / (r * square(sin_theta));
  get<2, 1, 2>(result) = -sin_phi / (r * sin_theta);

  // remove temp vars and restore to zero
  get<0, 0, 0>(result) = 0.0;
  get<0, 1, 0>(result) = 0.0;
  get<0, 2, 0>(result) = 0.0;
  get<0, 2, 2>(result) = 0.0;
  get<1, 2, 2>(result) = 0.0;
  get<2, 2, 0>(result) = 0.0;
  get<2, 2, 1>(result) = 0.0;
  get<2, 2, 2>(result) = 0.0;

  return result;
}

void SphericalTorus::pup(PUP::er& p) {
  size_t version = 1;  // was 0 before
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | r_min_;
    p | r_max_;
    p | pi_over_2_minus_theta_min_;
    p | fraction_of_torus_;
    p | compression_level_;
  }
}

template <typename T>
T SphericalTorus::radius(const T& x) const {
  return 0.5 * r_min_ * (1.0 - x) + 0.5 * r_max_ * (1.0 + x);
}

template <typename T>
void SphericalTorus::radius(const gsl::not_null<T*> r, const T& x) const {
  *r = 0.5 * r_min_ * (1.0 - x) + 0.5 * r_max_ * (1.0 + x);
}

template <typename T>
T SphericalTorus::radius_inverse(const T& r) const {
  return ((r - r_min_) - (r_max_ - r)) / (r_max_ - r_min_);
}

bool operator==(const SphericalTorus& lhs, const SphericalTorus& rhs) {
  return lhs.r_min_ == rhs.r_min_ and lhs.r_max_ == rhs.r_max_ and
         lhs.pi_over_2_minus_theta_min_ == rhs.pi_over_2_minus_theta_min_ and
         lhs.fraction_of_torus_ == rhs.fraction_of_torus_;
}

template <typename T>
T SphericalTorus::cubic_compression(const T& x) const {
  return compression_level_ * pow<3>(x) + (1.0 - compression_level_) * x;
}

double SphericalTorus::cubic_inversion(const double& x) const {
  // using p.228 of Numerical Recipe (cubic equation)
  const double Q = (compression_level_ - 1.0) / (3.0 * compression_level_);
  const double R = -0.5 * x / (compression_level_);
  const double A =
      -sgn(R) * pow(abs(R) + sqrt(square(R) - pow<3>(Q)), 1.0 / 3.0);
  double B;
  if (A != 0.0) {
    B = Q / A;
  } else {
    B = 0.0;
  }
  return A + B;
}

bool operator!=(const SphericalTorus& lhs, const SphericalTorus& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::I<DTYPE(data), 3> SphericalTorus::operator()(                \
      const tnsr::I<DTYPE(data), 3>& source_coords) const;                    \
  template Jacobian<DTYPE(data), 3, Frame::BlockLogical, Frame::Inertial>     \
  SphericalTorus::jacobian(const tnsr::I<DTYPE(data), 3>& source_coords)      \
      const;                                                                  \
  template InverseJacobian<DTYPE(data), 3, Frame::BlockLogical,               \
                           Frame::Inertial>                                   \
  SphericalTorus::inv_jacobian(const tnsr::I<DTYPE(data), 3>& source_coords)  \
      const;                                                                  \
  template tnsr::Ijj<DTYPE(data), 3, Frame::NoFrame> SphericalTorus::hessian( \
      const tnsr::I<DTYPE(data), 3>& source_coords) const;                    \
  template tnsr::Ijk<DTYPE(data), 3, Frame::NoFrame>                          \
  SphericalTorus::derivative_of_inv_jacobian(                                 \
      const tnsr::I<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE
#undef DTYPE

}  // namespace grmhd::AnalyticData
