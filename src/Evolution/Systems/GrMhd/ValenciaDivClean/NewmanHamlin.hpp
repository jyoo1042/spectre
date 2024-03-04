// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
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
 * the scheme of [Newman and Hamlin, SIAM J. Sci. Comput., 36(4)
 * B661-B683 (2014)](https://epubs.siam.org/doi/10.1137/140956749).
 *
 * In the Newman and Hamlin paper, `tau` is \f$e - \rho W\f$,
 * `momentum_density_squared` is\f${\cal M}^2\f$,
 * `momentum_density_dot_magnetic_field` is \f${\cal T}\f$,
 * `magnetic_field_squared` is \f${\cal B}^2\f$, and
 * `rest_mass_density_times_lorentz_factor` is \f${\tilde \rho}\f$.
 * Furthermore, the returned `PrimitiveRecoveryData.rho_h_w_squared` is \f${\cal
 * L}\f$.
 *
 * In terms of the conservative variables (in our notation):
 * \f{align}
 * e = & \frac{{\tilde D} + {\tilde \tau}}{\sqrt{\gamma}} \\
 * {\cal M}^2 = & \frac{\gamma^{mn} {\tilde S}_m {\tilde S}_n}{\gamma} \\
 * {\cal T} = & \frac{{\tilde B}^m {\tilde S}_m}{\gamma} \\
 * {\cal B}^2 = & \frac{\gamma_{mn} {\tilde B}^m {\tilde B}^n}{\gamma} \\
 * {\tilde \rho} = & \frac{\tilde D}{\sqrt{\gamma}}
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$,
 * \f${\tilde \tau}\f$, and \f${\tilde B}^i\f$ are a generalized mass-energy
 * density, momentum density, specific internal energy density, and magnetic
 * field, and \f$\gamma\f$ and \f$\gamma^{mn}\f$ are the determinant and inverse
 * of the spatial metric \f$\gamma_{mn}\f$.
 */
class NewmanHamlin {
 public:
  template <bool EnforcePhysicality, typename EosType>
  static std::optional<PrimitiveRecoveryData> apply(
      double initial_guess_for_pressure, double tau,
      double momentum_density_squared,
      double momentum_density_dot_magnetic_field, double magnetic_field_squared,
      double rest_mass_density_times_lorentz_factor, double electron_fraction,
      const EosType& equation_of_state,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options);

  static const std::string name() { return "Newman Hamlin"; }

 private:
  static constexpr size_t max_iterations_ = 50;
  static constexpr double relative_tolerance_ = 1.e-10;
};
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes
