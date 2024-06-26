// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/CombineVolumeGhostData.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"  //FIXME:
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell {
namespace {

template <size_t Dim>
Index<Dim> expanded_index(size_t index, const Index<Dim>& extents);

template <>
Index<1> expanded_index(size_t index, const Index<1>& extents) {
  return Index<1>{{index}};
}

template <>
Index<2> expanded_index(size_t index, const Index<2>& extents) {
  size_t ix = index % extents[0];
  size_t iy = (index - ix) / extents[0];
  return Index<2>{{ix, iy}};
}

template <>
Index<3> expanded_index(size_t index, const Index<3>& extents) {
  size_t ix = index % extents[0];
  size_t iy = ((index - ix) / extents[0]) % extents[1];
  size_t iz = ((((index - ix) / extents[0]) - iy) / extents[1]);
  return Index<3>{{ix, iy, iz}};
}

template <size_t Dim>
void sort_copy_data(gsl::not_null<DataVector*> extended_subcell_vars,
                    const DataVector& volume_subcell_vars,
                    const DataVector& ghost_subcell_vars,
                    size_t component_offset_ext, size_t component_offset_vol,
                    size_t component_offset_gho,
                    const Index<Dim>& extended_extents,
                    const Index<Dim>& volume_extents,
                    const Index<Dim>& ghost_extents,
                    const Direction<Dim>& direction_to_extend) {
  bool lower_side = (direction_to_extend.side() == Side::Lower);
  for (size_t nv = 0; nv < volume_extents.product(); ++nv) {
    Index<Dim> tmp_idx = expanded_index(nv, volume_extents);
    for (size_t d = 0; d < Dim; ++d) {
      if (d == direction_to_extend.dimension()) {
        if (lower_side) {
          tmp_idx[d] += ghost_extents[d];
        }
      }
    }
    size_t ne = collapsed_index(tmp_idx, extended_extents);
    (*extended_subcell_vars)[ne + component_offset_ext] =
        volume_subcell_vars[nv + component_offset_vol];
  }

  for (size_t ng = 0; ng < ghost_extents.product(); ++ng) {
    Index<Dim> tmp_idx = expanded_index(ng, ghost_extents);
    for (size_t d = 0; d < Dim; ++d) {
      if (d == direction_to_extend.dimension()) {
        if (!lower_side) {
          tmp_idx[d] += volume_extents[d];
        }
      }
    }
    size_t ne = collapsed_index(tmp_idx, extended_extents);
    (*extended_subcell_vars)[ne + component_offset_ext] =
        ghost_subcell_vars[ng + component_offset_gho];
  }
}

}  // namespace
template <size_t Dim>
DataVector combine_data(const DataVector& volume_subcell_vars,
                        const DataVector& ghost_subcell_vars,
                        const Index<Dim>& subcell_extents,
                        const size_t ghost_zone_size,
                        const Direction<Dim>& direction_to_extend) {
  // number of original volume subcell points and number of components
  const size_t num_vol_pts = subcell_extents.product();
  const size_t number_of_components = volume_subcell_vars.size() / num_vol_pts;

  // ghost extents
  Index<Dim> ghost_extents{0};
  for (size_t d = 0; d < Dim; ++d) {
    if (d == direction_to_extend.dimension()) {
      ghost_extents[d] = ghost_zone_size;
    } else {
      ghost_extents[d] = subcell_extents[d];
    }
  }
  const size_t num_gho_pts = ghost_extents.product();

  // new extended extents
  Index<Dim> extended_extents{static_cast<size_t>(0)};
  for (size_t d = 0; d < Dim; ++d) {
    if (d == direction_to_extend.dimension()) {
      extended_extents[d] = subcell_extents[d] + ghost_zone_size;
    } else {
      extended_extents[d] = subcell_extents[d];
    }
  }
  const size_t num_ext_pts = extended_extents.product();

  // create new extended size datavector should be
  DataVector result = DataVector{num_ext_pts * number_of_components};

  // loop through component index
  for (size_t component_index = 0; component_index < number_of_components;
       ++component_index) {
    const size_t component_offset_vol = component_index * num_vol_pts;
    const size_t component_offset_gho = component_index * num_gho_pts;
    const size_t component_offset_ext = component_index * num_ext_pts;

    sort_copy_data(&result, volume_subcell_vars, ghost_subcell_vars,
                   component_offset_ext, component_offset_vol,
                   component_offset_gho, extended_extents, subcell_extents,
                   ghost_extents, direction_to_extend);
  }

  return result;
}
// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template DataVector combine_data(const DataVector&, const DataVector&,  \
                                   const Index<DIM(data)>&, const size_t, \
                                   const Direction<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace evolution::dg::subcell
