// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"

/// \cond
namespace EquationsOfState {
template <bool, size_t>
class EquationOfState;
}  // namespace EquationsOfState
namespace grmhd::ValenciaDivClean {
class PrimitiveFromConservativeOptions;
}  // namespace grmhd::ValenciaDivClean
/// \endcond

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {

/*!
 * \brief Compute the primitive variables from the conservative variables using
 * the scheme of [Palenzuela et al, Phys. Rev. D 92, 044045
 * (2015)](https://doi.org/10.1103/PhysRevD.92.044045).
 *
 * In the notation of the Palenzuela paper, `tau` is \f$D q\f$,
 * `momentum_density_squared` is \f$r^2 D^2\f$,
 * `momentum_density_dot_magnetic_field` is \f$t D^{\frac{3}{2}}\f$,
 * `magnetic_field_squared` is \f$s D\f$, and
 * `rest_mass_density_times_lorentz_factor` is \f$D\f$.
 * Furthermore, the returned `PrimitiveRecoveryData.rho_h_w_squared` is \f$x
 * D\f$.  Note also that \f$h\f$ in the Palenzuela paper is the specific
 * enthalpy times the rest mass density.
 *
 * In terms of the conservative variables (in our notation):
 * \f{align*}
 * q = & \frac{{\tilde \tau}}{{\tilde D}} \\
 * r = & \frac{\gamma^{mn} {\tilde S}_m {\tilde S}_n}{{\tilde D}^2} \\
 * t^2 = & \frac{({\tilde B}^m {\tilde S}_m)^2}{{\tilde D}^3 \sqrt{\gamma}} \\
 * s = & \frac{\gamma_{mn} {\tilde B}^m {\tilde B}^n}{{\tilde D}\sqrt{\gamma}}
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$,
 * \f${\tilde \tau}\f$, and \f${\tilde B}^i\f$ are a generalized mass-energy
 * density, momentum density, specific internal energy density, and magnetic
 * field, and \f$\gamma\f$ and \f$\gamma^{mn}\f$ are the determinant and inverse
 * of the spatial metric \f$\gamma_{mn}\f$.
 */
class PalenzuelaEtAl {
 public:
  template <bool EnforcePhysicality, typename EosType>
  static std::optional<PrimitiveRecoveryData> apply(
      double /*initial_guess_pressure*/, double tau,
      double momentum_density_squared,
      double momentum_density_dot_magnetic_field, double magnetic_field_squared,
      double rest_mass_density_times_lorentz_factor, double electron_fraction,
      const EosType& equation_of_state,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options);

  static const std::string name() { return "PalenzuelaEtAl"; }

 private:
  static constexpr size_t max_iterations_ = 100;
  static constexpr double absolute_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
  static constexpr double relative_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
};
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes
