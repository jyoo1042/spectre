// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Index;
/// \endcond

/*!
 * \brief Combine the subcell variables and the ghost variables in
 * a given direction for neighbors to perform reconstruction.
 */

namespace evolution::dg::subcell {
template <size_t Dim>
DataVector combine_data(const DataVector& volume_subcell_vars,
                        const DataVector& ghost_subcell_vars,
                        const Index<Dim>& subcell_extents,
                        const size_t number_of_ghost_points,
                        const Direction<Dim>& direction_to_extend);
}  // namespace evolution::dg::subcell
