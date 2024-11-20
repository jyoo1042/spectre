// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace VariableFixing {

/*!
 * \ingroup VariableFixingGroup
 * \brief Fix the primitive variables to an atmosphere in low density regions
 *
 * If the rest mass density is below \f$\rho_{\textrm{cutoff}}\f$
 * (DensityCutoff), it is set to \f$\rho_{\textrm{atm}}\f$
 * (DensityOfAtmosphere), and the pressure, and specific internal energy (for
 * one-dimensional equations of state) are adjusted to satisfy the equation of
 * state.  For a two-dimensional equation of state, the specific internal energy
 * is set to zero.
 */
template <size_t Dim>
class FixToAtmosphere {
 public:
  /// \brief Rest mass density of the atmosphere
  struct DensityOfAtmosphere {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {"Density of atmosphere"};
  };
  /// \brief Rest mass density at which to impose the atmosphere. Should be
  /// greater than or equal to the density of the atmosphere.
  struct DensityCutoff {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Density to impose atmosphere at. Must be >= rho_atm"};
  };

  /*!
   * \brief Limit the velocity in and near the atmosphere.
   *
   * Let $v_{\max}$ be the maximum magnitude of the
   * velocity near the atmosphere, which we typically set to $10^{-4}$.
   * We let $v_{\mathrm{atm}}$ be the maximum magnitude of the velocity
   * in the atmosphere, which we typically set to $0$. We then define
   * the maximum magnitude of the spatial velocity to be
   *
   * \f{align*}{
   *   \tilde{v}
   *   &=\begin{cases}
   *     v_{\mathrm{atm}}, & \mathrm{if}\; \rho < \rho_{v^-} \   \
   *     v_{\mathrm{atm}} + \left(v_{\max} - v_{\mathrm{atm}}\right)
   *     \frac{\rho - \rho_{v^-}}{\rho_{v^+} - \rho_{v^-}},
   *                       & \mathrm{if}\;\rho_{v^-} \le \rho < \rho_{v^+}
   *   \end{cases}
   * \f}
   *
   * We then rescale the velocity by
   *
   * \f{align*}{
   *   v^i\to v^i\frac{\tilde{v}}{\sqrt{v^i\gamma_{ij}v^j}}.
   * \f}
   */
  struct VelocityLimitingOptions {
    struct AtmosphereMaxVelocity {
      using type = double;
      static constexpr Options::String help = {
          "The maximum velocity magnitude IN the atmosphere. Typically set to "
          "0."};
    };

    struct NearAtmosphereMaxVelocity {
      using type = double;
      static constexpr Options::String help = {
          "The maximum velocity magnitude NEAR the atmosphere. Typically set "
          "to 1e-4."};
    };

    struct AtmosphereDensityCutoff {
      using type = double;
      static constexpr Options::String help = {
          "The rest mass density cutoff below which the velocity magnitude is "
          "limited to AtmosphereMaxVelocity. Typically set to "
          "(10 or 20)*DensityOfAtmosphere."};
    };

    struct TransitionDensityBound {
      using type = double;
      static constexpr Options::String help = {
          "The rest mass density above which no velocity limiting is done. "
          "Between "
          "this value and AtmosphereDensityCutoff a linear transition in the "
          "maximum magnitude of the velocity between AtmosphereMaxVelocity and "
          "NearAtmosphereMaxVelocity is done. Typically set to "
          "10*AtmosphereDensityCutoff."};
    };
    using options = tmpl::list<AtmosphereMaxVelocity, NearAtmosphereMaxVelocity,
                               AtmosphereDensityCutoff, TransitionDensityBound>;
    static constexpr Options::String help = {
        "Limit the velocity in and near the atmosphere."};

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);

    bool operator==(const VelocityLimitingOptions& rhs) const;
    bool operator!=(const VelocityLimitingOptions& rhs) const;

    double atmosphere_max_velocity{
        std::numeric_limits<double>::signaling_NaN()};
    double near_atmosphere_max_velocity{
        std::numeric_limits<double>::signaling_NaN()};
    double atmosphere_density_cutoff{
        std::numeric_limits<double>::signaling_NaN()};
    double transition_density_bound{
        std::numeric_limits<double>::signaling_NaN()};
  };
  struct VelocityLimiting {
    using type = Options::Auto<VelocityLimitingOptions>;
    static constexpr Options::String help = VelocityLimitingOptions::help;
  };

  /// \brief Additional flooring treatment applied on highly magnetized region.
  /// Prescribes an upperbound on magnetization and plasma beta.
  /// `MagneticFieldTreatment`
  struct MagneticFieldTreatment {
    static constexpr Options::String help = {
        "Additinoal treatment for highly magnetized region."
        "These parameters sets bound on magnetization and plasma beta. "
        "Both values should be supplied."};
  };
  /// \brief Upperbound for magnetization = b^2/rho.
  /// 'MagnetizationBound'
  struct MagnetizationBound {
    static std::string name() { return "MagnetizationBound"; }
    static constexpr Options::String help{"Bound for Magneizations."};
    using type = Options::Auto<double, Options::AutoLabel::None>;
    using group = MagneticFieldTreatment;
  };
  /// \brief Upperbound for plasma beta = b^2/(2*P).
  /// 'PlasmaBetaBound'
  struct PlasmaBetaBound {
    static std::string name() { return "PlasmaBetaBound"; }
    static constexpr Options::String help{"Bound for Plasma Beta."};
    using type = Options::Auto<double, Options::AutoLabel::None>;
    using group = MagneticFieldTreatment;
  };

  using options =
      tmpl::list<DensityOfAtmosphere, DensityCutoff, VelocityLimiting,MagnetizationBound, PlasmaBetaBound>;
  static constexpr Options::String help = {
      "If the rest mass density is below DensityCutoff, it is set\n"
      "to DensityOfAtmosphere, and the pressure, and specific internal energy\n"
      "(for one-dimensional equations of state) are\n"
      "adjusted to satisfy the equation of state. For a two-dimensional\n"
      "equation of state, the specific internal energy is set to zero.\n"
      "In addition, the spatial velocity is set to zero, and the Lorentz\n"
      "factor is set to one.\n"};

  FixToAtmosphere(double density_of_atmosphere, double density_cutoff,
                  std::optional<VelocityLimitingOptions> velocity_limiting,
                  std::optional<double> magnetization_bound,
                  std::optional<double> plasma_beta_bound,
                  const Options::Context& context = {});

  FixToAtmosphere() = default;
  FixToAtmosphere(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere& operator=(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere(FixToAtmosphere&& /*rhs*/) = default;
  FixToAtmosphere& operator=(FixToAtmosphere&& /*rhs*/) = default;
  ~FixToAtmosphere() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::Temperature<DataVector>>;
  using argument_tags = tmpl::list<hydro::Tags::ElectronFraction<DataVector>,
                                   hydro::Tags::MagneticField<DataVector, Dim>,
                                   gr::Tags::SpatialMetric<DataVector, Dim>,
                                   hydro::Tags::EquationOfStateBase>;

  // for use in `db::mutate_apply`
  template <size_t ThermodynamicDim>
  void operator()(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& electron_fraction,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state) const;

 private:
  template <size_t ThermodynamicDim>
  void set_density_to_atmosphere(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& electron_fraction,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state,
      size_t grid_index) const;

  void apply_velocity_limit(
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      const Scalar<DataVector>& rest_mass_density,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      size_t grid_index) const;

  template <size_t ThermodynamicDim>
  void high_magnetiziation_treatment(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      const Scalar<DataVector>& electron_fraction,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      const double comoving_magnetic_field_squared,
      const double magnetic_field_squared, const double magnetic_field_dot_v,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state,
      size_t grid_index) const;

  template <size_t SpatialDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const FixToAtmosphere<SpatialDim>& lhs,
                         const FixToAtmosphere<SpatialDim>& rhs);

  double density_of_atmosphere_{std::numeric_limits<double>::signaling_NaN()};
  double density_cutoff_{std::numeric_limits<double>::signaling_NaN()};
  std::optional<VelocityLimitingOptions> velocity_limiting_{std::nullopt};
  std::optional<double> magnetization_bound_;
  std::optional<double> plasma_beta_bound_;
};

template <size_t Dim>
bool operator!=(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs);

}  // namespace VariableFixing
