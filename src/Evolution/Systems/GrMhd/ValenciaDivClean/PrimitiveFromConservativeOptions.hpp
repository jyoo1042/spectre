// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::ValenciaDivClean {

enum class PrimitiveInconsistencyFix { None, ScaleDown, SetToCap };

std::ostream& operator<<(std::ostream& os,
                         PrimitiveInconsistencyFix primitive_inconsistency_fix);

/// Options to be passed to the Con2Prim algorithm.
/// Currently, we simply set a threshold for tildeD
/// below which the inversion is not performed and
/// the density is set to atmosphere values.
class PrimitiveFromConservativeOptions {
 public:
  struct CutoffDForInversion {
    static std::string name() { return "CutoffDForInversion"; }
    static constexpr Options::String help{
        "Value of density times Lorentz factor below which we skip "
        "conservative to primitive inversion."};
    using type = double;
    static type lower_bound() { return 0.0; }
  };

  struct DensityWhenSkippingInversion {
    static std::string name() { return "DensityWhenSkippingInversion"; }
    static constexpr Options::String help{
        "Value of density when we skip conservative to primitive inversion."};
    using type = double;
    static type lower_bound() { return 0.0; }
  };

  struct KastaunMaxLorentzFactor {
    static constexpr Options::String help{
        "The maximum Lorentz allowed during primitive recovery when using the "
        "Kastaun schemes."};
    using type = double;
    static type lower_bound() { return 1.0; }
  };

  struct PrimitiveInconsistencyFixOption {
    static std::string name() { return "PrimitiveInconsistencyFix"; }
    static constexpr Options::String help{
        "Policy for handling inconsistencies between the recovered Lorentz "
        "factor and the spatial velocity reconstructed from the conserved "
        "variables. 'None' leaves the recovered primitives unchanged. "
        "'ScaleDown' rescales the spatial velocity to match the recovered "
        "Lorentz factor. "
        "'SetToCap' sets the Lorentz factor to the Kastaun max Lorentz factor "
        "and scales the spatial velocity accordingly."};
    using type = PrimitiveInconsistencyFix;
    static type default_value() { return PrimitiveInconsistencyFix::None; }
  };

  using options =
      tmpl::list<CutoffDForInversion, DensityWhenSkippingInversion,
                 KastaunMaxLorentzFactor, PrimitiveInconsistencyFixOption>;

  static constexpr Options::String help{
      "Options given to conservative to primitive inversion."};

  PrimitiveFromConservativeOptions() = default;

  PrimitiveFromConservativeOptions(
      double cutoff_d_for_inversion, double density_when_skipping_inversion,
      double kastaun_max_lorentz_factor,
      PrimitiveInconsistencyFix primitive_inconsistency_fix =
          PrimitiveInconsistencyFix::None,
      const Options::Context& context = {});

  void pup(PUP::er& p);

  double cutoff_d_for_inversion() const { return cutoff_d_for_inversion_; }
  double density_when_skipping_inversion() const {
    return density_when_skipping_inversion_;
  }
  double kastaun_max_lorentz_factor() const {
    return kastaun_max_lorentz_factor_;
  }

  PrimitiveInconsistencyFix primitive_inconsistency_fix() const {
    return primitive_inconsistency_fix_;
  }

 private:
  friend bool operator==(const PrimitiveFromConservativeOptions& lhs,
                         const PrimitiveFromConservativeOptions& rhs);

  double cutoff_d_for_inversion_ = std::numeric_limits<double>::signaling_NaN();
  double density_when_skipping_inversion_ =
      std::numeric_limits<double>::signaling_NaN();
  double kastaun_max_lorentz_factor_ =
      std::numeric_limits<double>::signaling_NaN();
  PrimitiveInconsistencyFix primitive_inconsistency_fix_ =
      PrimitiveInconsistencyFix::None;
};

bool operator!=(const PrimitiveFromConservativeOptions& lhs,
                const PrimitiveFromConservativeOptions& rhs);

}  // namespace grmhd::ValenciaDivClean

template <>
struct Options::create_from_yaml<
    grmhd::ValenciaDivClean::PrimitiveInconsistencyFix> {
  template <typename Metavariables>
  static grmhd::ValenciaDivClean::PrimitiveInconsistencyFix create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
grmhd::ValenciaDivClean::PrimitiveInconsistencyFix
Options::create_from_yaml<grmhd::ValenciaDivClean::PrimitiveInconsistencyFix>::
    create<void>(const Options::Option& options);
