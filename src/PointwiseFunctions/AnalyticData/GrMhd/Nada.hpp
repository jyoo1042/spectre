// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Temperature.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

class Nada : public virtual evolution::initial_data::InitialData,
             public MarkAsAnalyticData {
 protected:
  template <typename DataType>
  struct IntermediateVariables;

 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  /// The mass of the black hole, \f$M\f$.
  struct BhMass {
    using type = double;
    static constexpr Options::String help = {"The mass of the black hole."};
    static type lower_bound() { return 0.0; }
  };
  /// The dimensionless black hole spin, \f$\chi = a/M\f$.
  struct BhDimlessSpin {
    using type = double;
    static constexpr Options::String help = {
        "The dimensionless black hole spin."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };
  /// The radial coordinate of the inner edge of the disk, in units of \f$M\f$.
  struct InnerEdgeRadius {
    using type = double;
    static constexpr Options::String help = {
        "The radial coordinate of the inner edge of the disk."};
  };
  /// The radial coordinate of the outer edge of the disk, in units of \f$M\f$.
  struct OuterEdgeRadius {
    using type = double;
    static constexpr Options::String help = {
        "The radial coordinate of the outer edge of the disk."};
  };
  /// The polytropic constant of the fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() { return 0.; }
  };
  /// The polytropic exponent of the fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() { return 1.; }
  };

  using options =
      tmpl::list<BhMass, BhDimlessSpin, InnerEdgeRadius, OuterEdgeRadius,
                 PolytropicConstant, PolytropicExponent>;
  static constexpr Options::String help = {"Stationary disk at rest."};

  Nada() = default;
  Nada(const Nada& /*rhs*/) = default;
  Nada& operator=(const Nada& /*rhs*/) = default;
  Nada(Nada&& /*rhs*/) = default;
  Nada& operator=(Nada&& /*rhs*/) = default;
  ~Nada() override = default;

  Nada(double bh_mass, double bh_dimless_spin, double inner_edge_radius,
       double outer_edge_radius, double polytropic_constant,
       double polytropic_exponent);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit Nada(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Nada);
  /// \endcond

  template <typename DataType>
  using tags =
      tmpl::append<hydro::grmhd_tags<DataType>,
                   typename gr::Solutions::SphericalKerrSchild::tags<DataType>>;

  /// @{
  /// The variables in Cartesian Spherical-Kerr-Schild coordinates at x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    // Can't store IntermediateVariables as member variable because we need to
    // be threadsafe.
    IntermediateVariables<DataType> vars(x);
    return {std::move(
        get<Tags>(variables(x, tmpl::list<Tags>{}, make_not_null(&vars))))...};
  }

  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    // Can't store IntermediateVariables as member variable because we need to
    // be threadsafe.
    IntermediateVariables<DataType> vars(x);
    return variables(x, tmpl::list<Tag>{}, make_not_null(&vars));
  }
  /// @}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

 protected:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
      gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
      gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  // Grab the metric variables
  template <typename DataType, typename Tag,
            Requires<not tmpl::list_contains_v<
                tmpl::push_back<hydro::grmhd_tags<DataType>,
                                hydro::Tags::SpecificEnthalpy<DataType>,
                                hydro::Tags::SpatialVelocity<DataType, 3>,
                                hydro::Tags::LorentzFactor<DataType>>,
                Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(
      const tnsr::I<DataType, 3>& x, tmpl::list<Tag> /*meta*/,
      gsl::not_null<IntermediateVariables<DataType>*> vars) const {
    return {get<Tag>(background_spacetime_.variables(
        x, 0.0, tmpl::list<Tag>{},
        make_not_null(&vars->sph_kerr_schild_cache)))};
  }

  template <typename DataType, typename Func>
  void variables_impl(gsl::not_null<IntermediateVariables<DataType>*> vars,
                      Func f) const;

  template <typename DataType>
  struct IntermediateVariables {
    explicit IntermediateVariables(const tnsr::I<DataType, 3>& x);

    DataType r_squared{};
    gr::Solutions::SphericalKerrSchild::IntermediateVars<DataType,
                                                         Frame::Inertial>
        sph_kerr_schild_cache =
            gr::Solutions::SphericalKerrSchild::IntermediateVars<
                DataType, Frame::Inertial>(0);
  };

  friend bool operator==(const Nada& lhs, const Nada& rhs);

  double bh_mass_ = std::numeric_limits<double>::signaling_NaN();
  double bh_spin_a_ = std::numeric_limits<double>::signaling_NaN();
  double inner_edge_radius_ = std::numeric_limits<double>::signaling_NaN();
  double outer_edge_radius_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::SphericalKerrSchild background_spacetime_{};
};
bool operator!=(const Nada& lhs, const Nada& rhs);

}  // namespace grmhd::AnalyticData
