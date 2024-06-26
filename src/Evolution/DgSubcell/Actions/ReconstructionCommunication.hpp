// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Structure/TrimMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/CombineVolumeGhostData.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/NeighborRdmpAndVolumeData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Reconstructor.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"  //FIXME:
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Sets the local data from the relaxed discrete maximum principle
 * troubled-cell indicator and sends ghost zone data to neighboring elements.
 *
 * The action proceeds as follows:
 *
 * 1. Determine in which directions we have neighbors
 * 2. Slice the variables provided by GhostDataMutator to send to our neighbors
 *    for ghost zones
 * 3. Send the ghost zone data, appending the max/min for the TCI at the end of
 *    the `DataVector` we are sending.
 *
 * \warning This assumes the RDMP TCI data in the DataBox has been set, it does
 * not calculate it automatically. The reason is this way we can only calculate
 * the RDMP data when it's needed since computing it can be pretty expensive.
 *
 * Some notes:
 * - In the future we will need to send the cell-centered fluxes to do
 *   high-order FD without additional reconstruction being necessary.
 *
 * GlobalCache:
 * - Uses:
 *   - `ParallelComponent` proxy
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Mesh<Dim>`
 *   - `subcell::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `Tags::TimeStepId`
 *   - `Tags::Next<Tags::TimeStepId>`
 *   - `subcell::Tags::ActiveGrid`
 *   - `System::variables_tag`
 *   - `subcell::Tags::DataForRdmpTci`
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::GhostDataForReconstruction<Dim>`
 */
template <size_t Dim, typename GhostDataMutator, bool LocalTimeStepping,
          bool UseNodegroupDgElements>
struct SendDataForReconstruction {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Dim, UseNodegroupDgElements>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        not LocalTimeStepping,
        "DG-subcell does not yet support local time stepping. The "
        "reconstruction data must be sent using dense output sometimes, and "
        "not at all other times. However, the data for the RDMP TCI should be "
        "sent along with the data for reconstruction each time.");
    static_assert(UseNodegroupDgElements ==
                      Parallel::is_dg_element_collection_v<ParallelComponent>,
                  "The action SendDataForReconstruction is told by the "
                  "template parameter UseNodegroupDgElements that it is being "
                  "used with a DgElementCollection, but the ParallelComponent "
                  "is not a DgElementCollection. You need to change the "
                  "template parameter on the SendDataForReconstruction action "
                  "in your action list.");

    ASSERT(db::get<Tags::ActiveGrid>(box) == ActiveGrid::Subcell,
           "The SendDataForReconstruction action can only be called when "
           "Subcell is the active scheme.");
    using flux_variables = typename Metavariables::system::flux_variables;

    db::mutate<Tags::GhostDataForReconstruction<Dim>>(
        [](const auto ghost_data_ptr) {
          // Clear the previous neighbor data and add current local data
          ghost_data_ptr->clear();
        },
        make_not_null(&box));

    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const size_t ghost_zone_size =
        db::get<evolution::dg::subcell::Tags::Reconstructor>(box)
            .ghost_zone_size();

    // Optimization note: could save a copy+allocation if we moved
    // all_sliced_data when possible before sending.
    //
    // Note: RDMP size doesn't help here since we need to slice data after
    // anyway, so no way to save an allocation through that.
    const auto& cell_centered_flux =
        db::get<Tags::CellCenteredFlux<flux_variables, Dim>>(box);
    DataVector volume_data_to_slice = db::mutate_apply(
        GhostDataMutator{}, make_not_null(&box),
        cell_centered_flux.has_value() ? cell_centered_flux.value().size()
                                       : 0_st);
    if (cell_centered_flux.has_value()) {
      std::copy(
          cell_centered_flux.value().data(),
          std::next(
              cell_centered_flux.value().data(),
              static_cast<std::ptrdiff_t>(cell_centered_flux.value().size())),
          std::next(
              volume_data_to_slice.data(),
              static_cast<std::ptrdiff_t>(volume_data_to_slice.size() -
                                          cell_centered_flux.value().size())));
    }

    // FIXME: currently, we are feeding in element.internal_boundaries() to
    // slice data. This could include the directions which are problematic
    // for which we don't really want to do anything.
    // For now, we use 'continue' in the below to avoid doing anything in
    // the problematic directions (aka not process and send)
    // In the future, probably just feed in an unordered set which does not
    // include the problematic directions to the slice_data() in the first
    // place to avoid doing extra work than necessary.

    const auto problemo_chest =
        db::get<evolution::dg::subcell::Tags::ProblemoChest<Dim>>(box);
    std::unordered_set<Direction<Dim>> dirs_to_work;
    for (const auto& internal_dir : element.internal_boundaries()) {
      if (problemo_chest.contains(internal_dir)) {
        continue;
      } else {
        dirs_to_work.insert(internal_dir);
      }
    }
    // CAPTURE_FOR_ERROR(element.internal_boundaries());
    // CAPTURE_FOR_ERROR(dirs_to_work);

    // ERROR("before slice_data");

    const DirectionMap<Dim, DataVector> all_sliced_data = slice_data(
        volume_data_to_slice, subcell_mesh.extents(), ghost_zone_size,
        dirs_to_work, 0,
        db::get<
            evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>>(
            box));

    // ERROR("wow line 204 after slice data");

    std::unordered_set<Direction<Dim>> sliced_data_keys;
    for (auto& kv : all_sliced_data) {
      sliced_data_keys.insert(kv.first);
    }
    std::unordered_set<Direction<Dim>> pc_keys;
    for (auto& kv : problemo_chest) {
      pc_keys.insert(kv.first);
    }

    CAPTURE_FOR_ERROR(sliced_data_keys);
    CAPTURE_FOR_ERROR(dirs_to_work);
    CAPTURE_FOR_ERROR(pc_keys);
    CAPTURE_FOR_ERROR(element);
    for (const auto& dir : dirs_to_work) {
      if (!all_sliced_data.contains(dir)) {
        ERROR("something wrong at slice data stage");
      }
    }
    // if (element.id()==ElementId<Dim>{"[B1,(L2I0,L2I0,L2I0)]"}){
    //   ERROR("line 231!");
    // }
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const RdmpTciData& rdmp_tci_data = db::get<Tags::DataForRdmpTci>(box);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id = [&box]() {
      if (LocalTimeStepping) {
        return db::get<::Tags::Next<::Tags::TimeStepId>>(box);
      } else {
        return db::get<::Tags::TimeStepId>(box);
      }
    }();

    const int tci_decision =
        db::get<evolution::dg::subcell::Tags::TciDecision>(box);
    // Compute and send actual variables
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      // FIXME:
      // skipping the direction which is inside the DirectionlMap problemo_chest
      // containing problematic directions as keys
      if (problemo_chest.contains(direction)) {
        continue;
      }

      // continue on the with the loop for non-problematic directions.

      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      ASSERT(neighbors_in_direction.size() == 1,
             "AMR is not yet supported when using DG-subcell. Note that this "
             "condition could be relaxed to support AMR only where the "
             "evolution is using DG without any changes to subcell.");

      for (const ElementId<Dim>& neighbor : neighbors_in_direction) {
        const size_t rdmp_size = rdmp_tci_data.max_variables_values.size() +
                                 rdmp_tci_data.min_variables_values.size();
        const auto& sliced_data_in_direction = all_sliced_data.at(direction);

        // Allocate with subcell data and rdmp data
        DataVector subcell_data_to_send{sliced_data_in_direction.size() +
                                        rdmp_size};
        // Note: Currently we interpolate our solution to our neighbor FD grid
        // even when grid points align but are oriented differently. There's a
        // possible optimization for the rare (almost never?) edge case where
        // two blocks have the same ghost zone coordinates but have different
        // orientations (e.g. RotatedBricks). Since this shouldn't ever happen
        // outside of tests, we currently don't bother with it. If we wanted to,
        // here's the code:
        //
        // if (not orientation.is_aligned()) {
        //   std::array<size_t, Dim> slice_extents{};
        //   for (size_t d = 0; d < Dim; ++d) {
        //     gsl::at(slice_extents, d) = subcell_mesh.extents(d);
        //   }
        //   gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;
        //   // Need a view so we only get the subcell data and not the rdmp
        //   // data
        //   DataVector subcell_data_to_send_view{
        //       subcell_data_to_send.data(),
        //       subcell_data_to_send.size() - rdmp_size};
        //   orient_variables(make_not_null(&subcell_data_to_send_view),
        //                  sliced_data_in_direction, Index<Dim>{slice_extents},
        //                  orientation);
        // } else { std::copy(...); }
        //
        // FIXME:  get rid of this block later: purely for debugging
        // if (sliced_data_in_direction[0] == 1042.0) {
        //   CAPTURE_FOR_ERROR(sliced_data_in_direction);
        //   CAPTURE_FOR_ERROR(neighbor);
        //   CAPTURE_FOR_ERROR(element.id());
        //   ERROR("wow 1042!!!!");
        // }

        // Copy over data since it's already oriented from interpolation
        std::copy(sliced_data_in_direction.begin(),
                  sliced_data_in_direction.end(), subcell_data_to_send.begin());

        // Copy rdmp data to end of subcell_data_to_send
        std::copy(
            rdmp_tci_data.max_variables_values.cbegin(),
            rdmp_tci_data.max_variables_values.cend(),
            std::prev(subcell_data_to_send.end(), static_cast<int>(rdmp_size)));
        std::copy(rdmp_tci_data.min_variables_values.cbegin(),
                  rdmp_tci_data.min_variables_values.cend(),
                  std::prev(subcell_data_to_send.end(),
                            static_cast<int>(
                                rdmp_tci_data.min_variables_values.size())));

        evolution::dg::BoundaryData<Dim> data{
            subcell_mesh,
            dg_mesh.slice_away(direction.dimension()),
            std::move(subcell_data_to_send),
            std::nullopt,
            next_time_step_id,
            tci_decision};

        Parallel::receive_data<
            evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                Dim, Parallel::is_dg_element_collection_v<ParallelComponent>>>(
            receiver_proxy[neighbor], time_step_id,
            std::pair{DirectionalId<Dim>{direction_from_neighbor, element.id()},
                      std::move(data)});
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// FIXME:
// send data (modified sliced data where I don't send anyhting for problematic
// direction) new action (receive data -> inbox -> checks to make sure it got
// everyhitng all direction except problematic -> interpolation using the ghost
// data plus yourself (volume subcell plus the neighbordat)) receive data
// (untouched)

/*!
 * \brief Receive the subcell data from our neighbor for all non-problematic
 * directions (non-block boundary).
 * Interpolate and send data for all problematic directions.
 *
 */
template <size_t Dim, typename GhostDataMutator, bool LocalTimeStepping,
          bool UseNodegroupDgElements>
struct ReceiveAndSendDataForReconstruction {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Dim, UseNodegroupDgElements>>;
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto problemo_chest =
        db::get<evolution::dg::subcell::Tags::ProblemoChest<Dim>>(box);
    if (problemo_chest.empty()) {
      // We have no other stuffs to resend. Everything should have been
      // sent alright with the previous action, so just continue without
      // doing any work
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    // need to subtract number of problematic directions
    const auto number_of_expected_messages =
        element.neighbors().size() - problemo_chest.size();
    // CAPTURE_FOR_ERROR(element);
    // ERROR("number_of_expected_msgs : " << number_of_expected_messages <<"
    // !\n");

    std::unordered_set<DirectionalId<Dim>> expected_keys;
    for (const auto& dir : element.internal_boundaries()) {
      if (problemo_chest.contains(dir)) {
        continue;
      } else {
        expected_keys.insert(DirectionalId<Dim>{
            dir, *((element.neighbors()).at(dir).ids().begin())});
      }
    }

    if (UNLIKELY(number_of_expected_messages == 0)) {
      // We have no neighbors, so just continue without doing any work
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto interpolants = db::get<
        evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>>(
        box);

    using ::operator<<;
    const auto& current_time_step_id = db::get<::Tags::TimeStepId>(box);
    std::map<TimeStepId,
             DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>>& inbox =
        tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Metavariables::volume_dim,
            Parallel::is_dg_element_collection_v<ParallelComponent>>>(inboxes);
    const auto& received = inbox.find(current_time_step_id);
    // Check we have at least some data from correct time, and then check
    // we have received all data
    if (received == inbox.end() or
        !std::all_of(expected_keys.begin(), expected_keys.end(),
                     [&received](const auto& key) {
                       return received->second.find(key) !=
                              received->second.end();
                     })) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    // need a conditional to make sure I have received all the data that was
    // needed at this stage for interpolation.
    // this should be fine for now. //FIXME:

    const size_t ghost_zone_size =
        db::get<evolution::dg::subcell::Tags::Reconstructor>(box)
            .ghost_zone_size();
    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);
    const Index<Dim> subcell_extents = subcell_mesh.extents();

    // FIXME: copy over received data
    const DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>&
        received_data = inbox[current_time_step_id];

    ASSERT(
        received_data.size() >= number_of_expected_messages,
        "received_data size: " << received_data.size()
                               << " not equal to expected number of messages: "
                               << number_of_expected_messages << " !");

    DirectionalIdMap<Dim, size_t> ghost_cell_data_size;
    for (const auto& [key, value] : received_data) {
      ghost_cell_data_size[key] = value.ghost_cell_data.value().size();
    }

    // CAPTURE_FOR_ERROR(element);
    // CAPTURE_FOR_ERROR(ghost_cell_data_size);
    // ERROR("yolo");

    // for now for the volume subcell data just copy paste the block in
    // send data for reconstruction action.
    // don't need to bother doing the flux variables bs here
    using flux_variables = typename Metavariables::system::flux_variables;
    const auto& cell_centered_flux =
        db::get<Tags::CellCenteredFlux<flux_variables, Dim>>(box);
    DataVector volume_data_to_slice = db::mutate_apply(
        GhostDataMutator{}, make_not_null(&box),
        cell_centered_flux.has_value() ? cell_centered_flux.value().size()
                                       : 0_st);
    if (cell_centered_flux.has_value()) {
      std::copy(
          cell_centered_flux.value().data(),
          std::next(
              cell_centered_flux.value().data(),
              static_cast<std::ptrdiff_t>(cell_centered_flux.value().size())),
          std::next(
              volume_data_to_slice.data(),
              static_cast<std::ptrdiff_t>(volume_data_to_slice.size() -
                                          cell_centered_flux.value().size())));
    }
    // for sending stuffs
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const RdmpTciData& rdmp_tci_data = db::get<Tags::DataForRdmpTci>(box);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id = [&box]() {
      if (LocalTimeStepping) {
        return db::get<::Tags::Next<::Tags::TimeStepId>>(box);
      } else {
        return db::get<::Tags::TimeStepId>(box);
      }
    }();
    const int tci_decision =
        db::get<evolution::dg::subcell::Tags::TciDecision>(box);

    const size_t num_pts = subcell_mesh.extents().product();
    const size_t number_of_components = volume_data_to_slice.size() / num_pts;

    // element's volume subcell in the ghost data variable structure
    // I need to access the recevied data in the direction I need to extend
    // and then I just need put them together

    for (const auto& [problematic_direction, problemo_book] : problemo_chest) {
      // direction_to_extend
      const Direction<Dim> dir_to_extend = problemo_book.direction_to_extend;
      // this assume only one neighbor per side
      ElementId<Dim> relevant_neighbor_id =
          *((element.neighbors()).at(dir_to_extend).ids().begin());
      ElementId<Dim> problematic_neighbor_id =
          *((element.neighbors()).at(problematic_direction).ids().begin());

      const auto& orientation =
          (element.neighbors()).at(problematic_direction).orientation();
      const auto direction_from_neighbor =
          orientation(problematic_direction.opposite());

      // sanity checks #1 : only one neighbor per direction
      ASSERT(element.neighbors().at(dir_to_extend).ids().size() == 1,
             "New implementation assumes only one neighbor per direction.");
      // sanity checks #2 : only internal boundary for the extension.
      ASSERT(element.internal_boundaries().contains(dir_to_extend),
             "We are assuming directions to extend is one of the internal "
             "boundaries.");

      std::unordered_set<DirectionalId<Dim>> received_data_keys;
      for (auto& kv : received_data) {
        received_data_keys.insert(kv.first);
      }

      std::unordered_set<Direction<Dim>> pc_keys;
      for (auto& kv : problemo_chest) {
        pc_keys.insert(kv.first);
      }
      CAPTURE_FOR_ERROR(problematic_direction);
      CAPTURE_FOR_ERROR(element);
      CAPTURE_FOR_ERROR(relevant_neighbor_id);
      CAPTURE_FOR_ERROR(dir_to_extend);
      CAPTURE_FOR_ERROR(received_data_keys);
      CAPTURE_FOR_ERROR(pc_keys);
      // sanity checks #3 : we are assuming dir to extend has been filled.
      ASSERT((received_data.contains(
                 DirectionalId<Dim>{dir_to_extend, relevant_neighbor_id})),
             "Received data must have an entry in direction to extend.");

      // sanity checks #4 : we are assuming ghost data has been received.
      ASSERT(((received_data.at(
                   DirectionalId<Dim>{dir_to_extend, relevant_neighbor_id}))
                  .ghost_cell_data.has_value()),
             "Ghost data must not be empty for this action");
      // write a free fuction that combiens volume data and ghost data
      // and outputs combined datavector

      const DataVector relevant_ghost_data =
          received_data
              .at(DirectionalId<Dim>{dir_to_extend, relevant_neighbor_id})
              .ghost_cell_data.value();
      auto interpolant = (interpolants.at(DirectionalId<Dim>{
                              problematic_direction, problematic_neighbor_id}))
                             .value();
      const DataVector combined_data =
          combine_data(volume_data_to_slice, relevant_ghost_data,
                       subcell_extents, ghost_zone_size, dir_to_extend);
      const size_t result_size =
          ghost_zone_size * subcell_mesh.extents()
                                .slice_away(problematic_direction.dimension())
                                .product();
      DataVector result = DataVector{result_size * number_of_components};
      auto result_span = gsl::make_span(result.data(), result.size());
      interpolant.interpolate(
          make_not_null(&result_span),
          gsl::make_span(combined_data.data(), combined_data.size()));
      const size_t rdmp_size = rdmp_tci_data.max_variables_values.size() +
                               rdmp_tci_data.min_variables_values.size();

      DataVector subcell_data_to_send{result.size() + rdmp_size};
      std::copy(result.begin(), result.end(), subcell_data_to_send.begin());
      // Copy rdmp data to end of subcell_data_to_send
      std::copy(
          rdmp_tci_data.max_variables_values.cbegin(),
          rdmp_tci_data.max_variables_values.cend(),
          std::prev(subcell_data_to_send.end(), static_cast<int>(rdmp_size)));
      std::copy(rdmp_tci_data.min_variables_values.cbegin(),
                rdmp_tci_data.min_variables_values.cend(),
                std::prev(subcell_data_to_send.end(),
                          static_cast<int>(
                              rdmp_tci_data.min_variables_values.size())));
      evolution::dg::BoundaryData<Dim> data{
          subcell_mesh,
          dg_mesh.slice_away(problematic_direction.dimension()),
          std::move(subcell_data_to_send),
          std::nullopt,
          next_time_step_id,
          tci_decision};
      Parallel::receive_data<
          evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
              Dim, Parallel::is_dg_element_collection_v<ParallelComponent>>>(
          receiver_proxy[problematic_neighbor_id], time_step_id,
          std::pair{DirectionalId<Dim>{direction_from_neighbor, element.id()},
                    std::move(data)});
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
/*!
 * \brief Receive the subcell data from our neighbor, and accumulate the data
 * from the relaxed discrete maximum principle troubled-cell indicator.
 *
 * Note:
 * - Since we only care about the min/max over all neighbors and ourself at the
 *   past time, we accumulate all data immediately into the `RdmpTciData`.
 * - If the neighbor is using DG and therefore sends boundary correction data
 *   then that is added into the `evolution::dg::Tags::MortarData` tag
 * - The next `TimeStepId` is recorded, but we do not yet support local time
 *   stepping.
 * - This action will never care about what variables are sent for
 *   reconstruction. It is only responsible for receiving the data and storing
 *   it in the `NeighborData`.
 *
 * GlobalCache:
 * -Uses: nothing
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Element<Dim>`
 *   - `Tags::TimeStepId`
 *   - `domain::Tags::Mesh<Dim>`
 *   - `subcell::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `Tags::Next<Tags::TimeStepId>`
 *   - `subcell::Tags::ActiveGrid`
 *   - `System::variables_tag`
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::GhostDataForReconstruction<Dim>`
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `evolution::dg::Tags::MortarData`
 *   - `evolution::dg::Tags::MortarNextTemporalId`
 */
template <size_t Dim>
struct ReceiveDataForReconstruction {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto number_of_expected_messages = element.neighbors().size();
    if (UNLIKELY(number_of_expected_messages == 0)) {
      // We have no neighbors, so just continue without doing any work
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    using ::operator<<;
    const auto& current_time_step_id = db::get<::Tags::TimeStepId>(box);
    std::map<TimeStepId,
             DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>>& inbox =
        tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Metavariables::volume_dim,
            Parallel::is_dg_element_collection_v<ParallelComponent>>>(inboxes);
    const auto& received = inbox.find(current_time_step_id);
    // Check we have at least some data from correct time, and then check that
    // we have received all data
    if (received == inbox.end() or
        received->second.size() != number_of_expected_messages) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    // Now that we have received all the data, copy it over as needed.
    DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>> received_data =
        std::move(inbox[current_time_step_id]);
    inbox.erase(current_time_step_id);

    // DirectionalIdMap<Dim,size_t>mortar_shits;
    // DirectionalIdMap<Dim,size_t>ghost_cell_data_size;
    // for (const auto& [key,value]:received_data){
    //   ghost_cell_data_size[key] = value.ghost_cell_data.value().size();
    //   if (value.boundary_correction_data.has_value()) {
    //   mortar_shits[key] = value.boundary_correction_data.value().size();}
    //   else {
    //     mortar_shits[key] = 0;
    //   }}
    // CAPTURE_FOR_ERROR(ghost_cell_data_size);
    // CAPTURE_FOR_ERROR(mortar_shits);

    // ERROR("fuck ");
    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);
    const auto& mortar_meshes = get<evolution::dg::Tags::MortarMesh<Dim>>(box);

    db::mutate<Tags::GhostDataForReconstruction<Dim>, Tags::DataForRdmpTci,
               evolution::dg::Tags::MortarData<Dim>,
               evolution::dg::Tags::MortarNextTemporalId<Dim>,
               domain::Tags::NeighborMesh<Dim>,
               evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
        [&element,
         ghost_zone_size =
             db::get<evolution::dg::subcell::Tags::Reconstructor>(box)
                 .ghost_zone_size(),
         &received_data, &subcell_mesh, &mortar_meshes](
            const gsl::not_null<DirectionalIdMap<Dim, GhostData>*>
                ghost_data_ptr,
            const gsl::not_null<RdmpTciData*> rdmp_tci_data_ptr,
            const gsl::not_null<
                DirectionalIdMap<Dim, evolution::dg::MortarDataHolder<Dim>>*>
                mortar_data,
            const gsl::not_null<DirectionalIdMap<Dim, TimeStepId>*>
                mortar_next_time_step_id,
            const gsl::not_null<DirectionalIdMap<Dim, Mesh<Dim>>*>
                neighbor_mesh,
            const auto neighbor_tci_decisions,
            const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
                neighbor_dg_to_fd_interpolants) {
          // Remove neighbor meshes for neighbors that don't exist anymore
          domain::remove_nonexistent_neighbors(neighbor_mesh, element);

          for (auto& received_mortar_data : received_data) {
            const auto& mortar_id = received_mortar_data.first;
            if (!mortar_next_time_step_id->contains(mortar_id)) {
              CAPTURE_FOR_ERROR(*mortar_next_time_step_id);
              ERROR("wtf " << mortar_id);
            }
          }
          // Get the next time step id, and also the fluxes data if the neighbor
          // is doing DG.
          for (auto& received_mortar_data : received_data) {
            const auto& mortar_id = received_mortar_data.first;
            try {
              mortar_next_time_step_id->at(mortar_id) =
                  received_mortar_data.second.validity_range;
            } catch (std::exception& e) {
              ERROR("Failed retrieving the MortarId: ("
                    << mortar_id.direction() << ',' << mortar_id.id()
                    << ") from the mortar_next_time_step_id. Got exception: "
                    << e.what());
            }
            if (received_mortar_data.second.boundary_correction_data
                    .has_value()) {
              mortar_data->at(mortar_id).neighbor().face_mesh =
                  received_mortar_data.second.interface_mesh;
              mortar_data->at(mortar_id).neighbor().mortar_mesh =
                  mortar_meshes.at(mortar_id);
              mortar_data->at(mortar_id).neighbor().mortar_data = std::move(
                  *received_mortar_data.second.boundary_correction_data);
            }
            // Set new neighbor mesh
            neighbor_mesh->insert_or_assign(
                mortar_id,
                received_mortar_data.second.volume_mesh_ghost_cell_data);
          }

          ASSERT(ghost_data_ptr->empty(),
                 "Should have no elements in the neighbor data when "
                 "receiving neighbor data");
          const size_t number_of_rdmp_vars =
              rdmp_tci_data_ptr->max_variables_values.size();
          ASSERT(rdmp_tci_data_ptr->min_variables_values.size() ==
                     number_of_rdmp_vars,
                 "The number of RDMP variables for which we have a maximum "
                 "and minimum should be the same, but we have "
                     << number_of_rdmp_vars << " for the max and "
                     << rdmp_tci_data_ptr->min_variables_values.size()
                     << " for the min.");

          for (const auto& [direction, neighbors_in_direction] :
               element.neighbors()) {
            for (const auto& neighbor : neighbors_in_direction) {
              DirectionalId<Dim> directional_element_id{direction, neighbor};
              ASSERT(ghost_data_ptr->count(directional_element_id) == 0,
                     "Found neighbor already inserted in direction "
                         << direction << " with ElementId " << neighbor);
              ASSERT(received_data[directional_element_id]
                         .ghost_cell_data.has_value(),
                     "Received subcell data message that does not contain any "
                     "actual subcell data for reconstruction.");
              // Collect the max/min of u(t^n) for the RDMP as we receive data.
              // This reduces the memory footprint.

              evolution::dg::subcell::insert_neighbor_rdmp_and_volume_data(
                  rdmp_tci_data_ptr, ghost_data_ptr,
                  *received_data[directional_element_id].ghost_cell_data,
                  number_of_rdmp_vars, directional_element_id,
                  neighbor_mesh->at(directional_element_id), element,
                  subcell_mesh, ghost_zone_size,
                  neighbor_dg_to_fd_interpolants);
              ASSERT(neighbor_tci_decisions->contains(directional_element_id),
                     "The NeighorTciDecisions should contain the neighbor ("
                         << directional_element_id.direction() << ", "
                         << directional_element_id.id() << ") but doesn't");
              neighbor_tci_decisions->at(directional_element_id) =
                  received_data[directional_element_id].tci_status;
            }
          }
        },
        make_not_null(&box),
        db::get<
            evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>>(
            box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::subcell::Actions
