// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class ElementId;
namespace PUP {
class er;
}  // namespace PUP
namespace intrp {
template <size_t Dim>
class Irregular;
}  // namespace intrp
/// \endcond

namespace interpolators_detail {
template <size_t Dim>
struct ProblemoBook {
  static constexpr size_t dim = Dim;
  // FIXME: this will be int to store the side
  // better way would be to actually just store Direction
  // note this is the direction in which one needs to extend,
  // ot the direction in which we cannot send the ghost data
  // without further work.
  Direction<Dim> direction_to_extend;
  ProblemoBook() = default;
  ProblemoBook(Direction<Dim> direction_to_extend_v)
      : direction_to_extend(direction_to_extend_v) {
    ASSERT(direction_to_extend.dimension() < dim,
           "Problematic Dimension cannot be greater than the dimension "
           "of the problem.");
  }
  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | direction_to_extend; };
};

// template struct ProblemoBook<1>;
// template struct ProblemoBook<2>;
// template struct ProblemoBook<3>;

}  // namespace interpolators_detail

namespace evolution::dg::subcell::Tags {
/*!
 * \brief An `intrp::Irregular` from our FD grid to our neighbors' FD grid.
 *
 * Values are only set if the neighboring elements' logical coordinate axes
 * are not aligned with ours. I.e., on block boundaries.
 */
template <size_t Dim>
struct InterpolatorsFromFdToNeighborFd : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
};

/*!
 * \brief An `intrp::Irregular` from our DG grid to our neighbors' FD grid.
 *
 * Values are only set if the neighboring elements' logical coordinate axes
 * are not aligned with ours. I.e., on block boundaries.
 */
template <size_t Dim>
struct InterpolatorsFromDgToNeighborFd : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
};

/*!
 * \brief An `intrp::Irregular` from our neighbors' DG grid to our FD grid.
 *
 * Values are only set if the neighboring elements' logical coordinate axes
 * are not aligned with ours. I.e., on block boundaries.
 */
template <size_t Dim>
struct InterpolatorsFromNeighborDgToFd : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
};

/*!
 * \brief This will be a book-keeping that records whether there the neighbor's
 * ghost goes out of bound and the direction in which it does.
 *
 */
template <size_t Dim>
struct ProblemoChest : db::SimpleTag {
  using type = DirectionMap<Dim, interpolators_detail::ProblemoBook<Dim>>;
};
}  // namespace evolution::dg::subcell::Tags
