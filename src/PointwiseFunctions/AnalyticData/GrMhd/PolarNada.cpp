// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/PolarNada.hpp"

#include <pup.h>
#include <utility>

namespace grmhd::AnalyticData {

PolarNada::PolarNada(Nada nada, grmhd::AnalyticData::SphericalTorus torus_map)
    : nada_(std::move(nada)), torus_map_(std::move(torus_map)) {}

std::unique_ptr<evolution::initial_data::InitialData> PolarNada::get_clone()
    const {
  return std::make_unique<PolarNada>(*this);
}

PolarNada::PolarNada(CkMigrateMessage* msg) : nada_(msg) {}

void PolarNada::pup(PUP::er& p) {
  p | nada_;
  p | torus_map_;
}

PUP::able::PUP_ID PolarNada::my_PUP_ID = 0;  // NOLINT

bool operator==(const PolarNada& lhs, const PolarNada& rhs) {
  return lhs.nada_ == rhs.nada_ and lhs.torus_map_ == rhs.torus_map_;
}

bool operator!=(const PolarNada& lhs, const PolarNada& rhs) {
  return not(lhs == rhs);
}

}  // namespace grmhd::AnalyticData
