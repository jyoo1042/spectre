// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"

#include "Evolution/DgSubcell/CombineVolumeGhostData.hpp"

// removed Actions/Test_ReconstructionCommunication for now
// put it back into cmakelist later.

namespace evolution::dg::subcell {
namespace {

std::array<DataVector, 3> generate_volume_ghost_data(
    Index<2> volume_extents, size_t ghost_size,
    Direction<2> direction_to_extend) {
  std::array<DataVector, 3> result;
  DataVector volume_data = DataVector{
      {1. / 6, 3. / 6, 5. / 6, 1. / 6, 3. / 6, 5. / 6, 1. / 6, 3. / 6, 5. / 6,
       1. / 6, 1. / 6, 1. / 6, 3. / 6, 3. / 6, 3. / 6, 5. / 6, 5. / 6, 5. / 6}};
  // for 1st direction lower
  //   DataVector ghost_data =
  //       DataVector{{-5. / 6, -3. / 6, -5. / 6, -3. / 6, -5. / 6, -3. / 6,
  //       1./6, 1./6, 3./6, 3./6, 5./6, 5./6}};
  //   DataVector extended_data = DataVector{{-5. / 6, -3. / 6, 1. / 6, 3. /
  //   6, 5. / 6, -5. / 6, -3. / 6, 1. / 6,
  //        3. / 6, 5. / 6, -5. / 6, -3. / 6, 1. / 6, 3. / 6, 5. / 6,
  //        1./6, 1./6, 1./6, 1./6, 1./6, 3./6, 3./6, 3./6, 3./6, 3./6,
  //        5./6, 5./6, 5./6, 5./6, 5./6}};
  // for 2nd direction upper
  DataVector ghost_data =
      DataVector{{1. / 6, 3. / 6, 5. / 6, 1. / 6, 3. / 6, 5. / 6, 7. / 6,
                  7. / 6, 7. / 6, 9. / 6, 9. / 6, 9. / 6}};
  DataVector extended_data = DataVector{
      {1. / 6, 3. / 6, 5. / 6, 1. / 6, 3. / 6, 5. / 6, 1. / 6, 3. / 6,
       5. / 6, 1. / 6, 3. / 6, 5. / 6, 1. / 6, 3. / 6, 5. / 6, 1. / 6,
       1. / 6, 1. / 6, 3. / 6, 3. / 6, 3. / 6, 5. / 6, 5. / 6, 5. / 6,
       7. / 6, 7. / 6, 7. / 6, 9. / 6, 9. / 6, 9. / 6}};

  result[0] = volume_data;
  result[1] = ghost_data;
  result[2] = extended_data;

  return result;
}

void test() {
  const Index<2> volume_extents = Index<2>{3};
  const Index<2> ghost_extents = Index<2>{2};
  const size_t ghost_zone_size = 2;
  const Direction<2> direction_to_extend{1, Side::Upper};

  const std::array<DataVector, 3> yolo =
      generate_volume_ghost_data(volume_extents, 2, direction_to_extend);

  const DataVector volume_data = yolo[0];
  const DataVector ghost_data = yolo[1];
  const DataVector extended_data = yolo[2];

  CAPTURE(volume_data);
  CAPTURE(ghost_data);
  CAPTURE(extended_data);
  const DataVector new_result =
      subcell::combine_data(volume_data, ghost_data, volume_extents,
                            ghost_zone_size, direction_to_extend);
  CAPTURE(new_result);
  if (extended_data.size() != new_result.size()) {
    ERROR("combined data size does not match expected size");
  }

  for (size_t i = 0; i < extended_data.size(); ++i) {
    if (extended_data[i] != new_result[i]) {
      CAPTURE(i);
      ERROR("the combined data does not match expected");
    }
  }
}

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.Evolution.Subcell.CombineVolumeGhostData",
                  "[Evolution][Unit]") {
  test();
}

}  // namespace
}  // namespace evolution::dg::subcell
