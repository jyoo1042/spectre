// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"

#include <cmath>
#include <limits>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>

#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

namespace {

using PrimitiveInconsistencyFix =
    grmhd::ValenciaDivClean::PrimitiveInconsistencyFix;

std::vector<PrimitiveInconsistencyFix> known_inconsistency_fixes() {
  return {PrimitiveInconsistencyFix::None, PrimitiveInconsistencyFix::ScaleDown,
          PrimitiveInconsistencyFix::SetToCap};
}
}  // namespace

namespace grmhd::ValenciaDivClean {

std::ostream& operator<<(
    std::ostream& os,
    const PrimitiveInconsistencyFix primitive_inconsistency_fix) {
  switch (primitive_inconsistency_fix) {
    case PrimitiveInconsistencyFix::None:
      os << "None";
      break;
    case PrimitiveInconsistencyFix::ScaleDown:
      os << "ScaleDown";
      break;
    case PrimitiveInconsistencyFix::SetToCap:
      os << "SetToCap";
      break;
    default:
      ERROR(
          "An unknown PrimitiveInconsistencyFix was passed to the stream "
          "operator: "
          << static_cast<int>(primitive_inconsistency_fix));
  }
  return os;
}

PrimitiveFromConservativeOptions::PrimitiveFromConservativeOptions(
    const double cutoff_d_for_inversion,
    const double density_when_skipping_inversion,
    const double kastaun_max_lorentz_factor,
    const PrimitiveInconsistencyFix primitive_inconsistency_fix,
    const Options::Context& context)
    : cutoff_d_for_inversion_(cutoff_d_for_inversion),
      density_when_skipping_inversion_(density_when_skipping_inversion),
      kastaun_max_lorentz_factor_(kastaun_max_lorentz_factor),
      primitive_inconsistency_fix_(primitive_inconsistency_fix) {
  using std::sqrt;
  if (kastaun_max_lorentz_factor_ > sqrt(std::numeric_limits<double>::max())) {
    PARSE_ERROR(context, "The Kastaun max lorentz factor must be smaller than "
                             << sqrt(std::numeric_limits<double>::max())
                             << " but is " << kastaun_max_lorentz_factor_);
  }
}

void PrimitiveFromConservativeOptions::pup(PUP::er& p) {
  p | cutoff_d_for_inversion_;
  p | density_when_skipping_inversion_;
  p | kastaun_max_lorentz_factor_;
  p | primitive_inconsistency_fix_;
}

bool operator==(const PrimitiveFromConservativeOptions& lhs,
                const PrimitiveFromConservativeOptions& rhs) {
  return (lhs.cutoff_d_for_inversion_ == rhs.cutoff_d_for_inversion_) and
         (lhs.density_when_skipping_inversion_ ==
          rhs.density_when_skipping_inversion_) and
         lhs.kastaun_max_lorentz_factor_ == rhs.kastaun_max_lorentz_factor_ and
         lhs.primitive_inconsistency_fix_ == rhs.primitive_inconsistency_fix_;
}

bool operator!=(const PrimitiveFromConservativeOptions& lhs,
                const PrimitiveFromConservativeOptions& rhs) {
  return not(lhs == rhs);
}

}  // namespace grmhd::ValenciaDivClean

template <>
grmhd::ValenciaDivClean::PrimitiveInconsistencyFix
Options::create_from_yaml<grmhd::ValenciaDivClean::PrimitiveInconsistencyFix>::
    create<void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();

  for (const auto inconsistency_fix : known_inconsistency_fixes()) {
    if (type_read == get_output(inconsistency_fix)) {
      return inconsistency_fix;
    }
  }

  using ::operator<<;
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << type_read
                                     << "\" to "
                                        "PrimitiveInconsistencyFix.\n"
                                        "Must be one of "
                                     << known_inconsistency_fixes() << ".");
}
