// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/ValidateSelection.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

// FIXME: got clang-tidy "nested namespace can be concatenated"
namespace dg::Events {
/// \cond
template <size_t VolumeDim, typename Tensors,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename ArraySectionIdTag = void>
class ObserveInterpolatedData;
/// \endcond

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe volume tensor fields interpolated to a new mesh slice.
 *
 * A class that writes volume quantities to an h5 file during the simulation.
 * The observed quantitites are specified in the `VariablesToObserve` option.
 * Any `Tensor` in the `db::DataBox` can be observed but must be listed in the
 * `Tensors` template parameter. Any additional compute tags that hold a
 * `Tensor` can also be added to the `Tensors` template parameter. Finally,
 * `Variables` and other non-tensor compute tags can be listed in the
 * `NonTensorComputeTags` to facilitate observing. Note that the
 * `InertialCoordinates` are always observed.
 *
 * FIXME: add more details here.
 *
 *
 * \note The `NonTensorComputeTags` are intended to be used for `Variables`
 * compute tags like `Tags::DerivCompute`
 *
 * \par Array sections
 * This event supports sections (see `Parallel::Section`). Set the
 * `ArraySectionIdTag` template parameter to split up observations into subsets
 * of elements. The `observers::Tags::ObservationKey<ArraySectionIdTag>` must be
 * available in the DataBox. It identifies the section and is used as a suffix
 * for the path in the output file.
 */
template <size_t VolumeDim, typename... Tensors,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
class ObserveInterpolatedData<VolumeDim, tmpl::list<Tensors...>,
                              tmpl::list<NonTensorComputeTags...>,
                              ArraySectionIdTag> : public Event {
 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveInterpolatedData(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveInterpolatedData);  // NOLINT
  /// \endcond

  struct VariablesToObserve {
    static constexpr Options::String help = "Subset of variables to observe";
    using type = std::vector<std::string>;
    static size_t lower_bound_on_size() { return 1; }
  };

  struct InterpDim {
    static constexpr Options::String help = "Dimension to take the interpolate";
    using type = size_t;
  };

  struct InterpVal {
    static constexpr Options::String help =
        "Value at which to do interpolation";
    using type = double;
  };

  /// The floating point type/precision with which to write the data to disk.
  ///
  /// Must be specified once for all data or individually for each variable
  /// being observed.
  struct FloatingPointTypes {
    static constexpr Options::String help =
        "The floating point type/precision with which to write the data to "
        "disk.\n\n"
        "Must be specified once for all data or individually  for each "
        "variable being observed.";
    using type = std::vector<FloatingPointType>;
    static size_t upper_bound_on_size() { return sizeof...(Tensors); }
    static size_t lower_bound_on_size() { return 1; }
  };

  /// The floating point type/precision with which to write the coordinates to
  /// disk.
  struct CoordinatesFloatingPointType {
    static constexpr Options::String help =
        "The floating point type/precision with which to write the coordinates "
        "to disk.";
    using type = FloatingPointType;
  };

  using options =
      tmpl::list<SubfileName, CoordinatesFloatingPointType, FloatingPointTypes,
                 VariablesToObserve, InterpDim, InterpVal>;

  static constexpr Options::String help =
      "Observe volume tensor fields.\n"
      "\n"
      "Writes volume quantities:\n"
      " * InertialCoordinates\n"
      " * Tensors listed in the 'VariablesToObserve' option\n";

  ObserveInterpolatedData() = default;

  ObserveInterpolatedData(
      const std::string& subfile_name,
      FloatingPointType coordinates_floating_point_type,
      const std::vector<FloatingPointType>& floating_point_types,
      const std::vector<std::string>& variables_to_observe,
      const size_t interp_dim, const double interp_val,
      const Options::Context& context = {});

  using compute_tags_for_observation_box =
      tmpl::list<Tensors..., NonTensorComputeTags...>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox,
                                   ::Events::Tags::ObserverMesh<VolumeDim>>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<DataBoxType, ComputeTagsList>& box,
                  const Mesh<VolumeDim>& mesh,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& array_index,
                  const ParallelComponent* const component,
                  const ObservationValue& observation_value) const {
    // Skip observation on elements that are not part of a section
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return;
    }
    call_operator_impl(subfile_path_ + *section_observation_key,
                       variables_to_observe_, interp_dim_, interp_val_, mesh,
                       box, cache, array_index, component, observation_value);
  }

  // We factor out the work into a static member function so it can  be shared
  // with other field observing events, like the one that deals with DG-subcell
  // where there are two grids. This is to avoid copy-pasting all of the code.
  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  static void call_operator_impl(
      const std::string& subfile_path,
      const std::unordered_map<std::string, FloatingPointType>&
          variables_to_observe,
      const size_t interp_dim, const double interp_val,
      const Mesh<VolumeDim>& mesh,
      const ObservationBox<DataBoxType, ComputeTagsList>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& element_id,
      const ParallelComponent* const /*meta*/,
      const ObservationValue& observation_value) {
    // For each element, we find the block logical coordinate
    // of the upper and lower side in the interpo_dim.
    // We only need to use the element for which the interp_val
    // falls within the range.

    const double lower_bound_bl =
        element_id.segment_id(interp_dim).endpoint(Side::Lower);
    const double upper_bound_bl =
        element_id.segment_id(interp_dim).endpoint(Side::Upper);
    if ((lower_bound_bl < interp_val) and (upper_bound_bl >= interp_val)) {
      // Create a 3 dimensional new_mesh corresponding to a single point in
      // interp_dim times 2-dimensional slice of the original Mesh.
      // This is used for the sake of creating a RegularGrid Interpolant.
      std::array<size_t, VolumeDim> target_extents{};
      std::array<Spectral::Basis, VolumeDim> bases{};
      std::array<Spectral::Quadrature, VolumeDim> quadratures{};
      for (size_t d = 0; d < VolumeDim; ++d) {
        if (d == interp_dim) {
          gsl::at(target_extents, d) = 1;
          gsl::at(quadratures, d) = Spectral::Quadrature::Gauss;
        } else {
          gsl::at(target_extents, d) = mesh.extents(d);
          gsl::at(bases, d) = mesh.basis(d);
          gsl::at(quadratures, d) = mesh.quadrature(d);
        }
      }
      const auto new_mesh =
          Mesh<VolumeDim>{{target_extents}, {bases}, {quadratures}};

      // We need to convert the block logical coordinates
      // to corresponding element logical coordinate for the given
      // element for the next step.
      const double elm_interp_val =
          (interp_val - ((upper_bound_bl + lower_bound_bl) / 2.)) /
          ((upper_bound_bl - lower_bound_bl) / 2.);

      auto override_target_mesh_with_1d_coords =
          make_array<VolumeDim>(DataVector{});
      gsl::at(override_target_mesh_with_1d_coords, interp_dim) =
          DataVector{elm_interp_val};
      const intrp::RegularGrid<VolumeDim> interpolant(
          mesh, mesh, override_target_mesh_with_1d_coords);

      // // Remove tensor types, only storing individual components.
      std::vector<TensorComponent> components;
      // // This is larger than we need if we are only observing some
      // // tensors, but that's not a big deal and calculating the correct
      // // size is nontrivial.
      components.reserve(alg::accumulate(
          std::initializer_list<size_t>{std::decay_t<decltype(value(
              typename Tensors::type{}))>::size()...},
          0_st));

      const auto record_tensor_component_impl =
          [&components, &interpolant](
              const auto& tensor, const FloatingPointType floating_point_type,
              const std::string& tag_name) {
            for (size_t i = 0; i < tensor.size(); ++i) {
              const auto tensor_component = interpolant.interpolate(tensor[i]);
              if (floating_point_type == FloatingPointType::Float) {
                components.emplace_back(
                    tag_name + tensor.component_suffix(i),
                    std::vector<float>{tensor_component.begin(),
                                       tensor_component.end()});
              } else {
                components.emplace_back(tag_name + tensor.component_suffix(i),
                                        tensor_component);
              }
            }
          };
      const auto record_tensor_components =
          [&box, &record_tensor_component_impl,
           &variables_to_observe](const auto tensor_tag_v) {
            using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
            const std::string tag_name = db::tag_name<tensor_tag>();
            if (const auto var_to_observe = variables_to_observe.find(tag_name);
                var_to_observe != variables_to_observe.end()) {

              const auto& tensor = get<tensor_tag>(box);
              if (not has_value(tensor)) {
                // This will only print a warning the first time it's called on
                // a node.
                [[maybe_unused]] static bool t =
                    ObserveInterpolatedData::print_warning_about_optional<
                        tensor_tag>();
                return;
              }
              const auto floating_point_type = var_to_observe->second;
              record_tensor_component_impl(value(tensor), floating_point_type,
                                           tag_name);
            }
          };
      EXPAND_PACK_LEFT_TO_RIGHT(
          record_tensor_components(tmpl::type_<Tensors>{}));

      // // Send data to volume observer
      auto& local_observer = *Parallel::local_branch(
          Parallel::get_parallel_component<observers::Observer<Metavariables>>(
              cache));

      Parallel::simple_action<observers::Actions::ContributeVolumeData>(
          local_observer,
          observers::ObservationId(observation_value.value,
                                   subfile_path + ".vol"),
          subfile_path,
          Parallel::make_array_component_id<ParallelComponent>(element_id),
          ElementVolumeData{element_id, std::move(components), new_mesh});
    }
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    if ((db::get<domain::Tags::Element<VolumeDim>>(box)
             .id()
             .segment_id(interp_dim_)
             .endpoint(Side::Lower) < interp_val_) and
        (db::get<domain::Tags::Element<VolumeDim>>(box)
             .id()
             .segment_id(interp_dim_)
             .endpoint(Side::Upper)) >= interp_val_) {
      return {{observers::TypeOfObservation::Volume,
               observers::ObservationKey(
                   subfile_path_ + section_observation_key.value() + ".vol")}};
    } else {
      return std::nullopt;
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
    p | variables_to_observe_;
    p | interp_dim_;
    p | interp_val_;
  }

 private:
  template <typename Tag>
  static bool print_warning_about_optional() {
    Parallel::printf(
        "Warning: ObserveInterpolatedData is trying to dump the tag %s "
        "but it is stored as a std::optional and has not been "
        "evaluated. This most commonly occurs when you are "
        "trying to either observe an analytic solution or errors when "
        "no analytic solution is available.\n",
        db::tag_name<Tag>());
    return false;
  }

  std::string subfile_path_;
  std::unordered_map<std::string, FloatingPointType> variables_to_observe_{};
  size_t interp_dim_{};
  double interp_val_{};
};

template <size_t VolumeDim, typename... Tensors,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveInterpolatedData<VolumeDim, tmpl::list<Tensors...>,
                        tmpl::list<NonTensorComputeTags...>,
                        ArraySectionIdTag>::
    ObserveInterpolatedData(
        const std::string& subfile_name,
        const FloatingPointType coordinates_floating_point_type,
        const std::vector<FloatingPointType>& floating_point_types,
        const std::vector<std::string>& variables_to_observe,
        const size_t interp_dim, const double interp_val,
        const Options::Context& context)
    : subfile_path_("/" + subfile_name),
      variables_to_observe_([&context, &floating_point_types,
                             &variables_to_observe]() {
        if (floating_point_types.size() != 1 and
            floating_point_types.size() != variables_to_observe.size()) {
          PARSE_ERROR(context, "The number of floating point types specified ("
                                   << floating_point_types.size()
                                   << ") must be 1 or the number of variables "
                                      "specified for observing ("
                                   << variables_to_observe.size() << ")");
        }
        std::unordered_map<std::string, FloatingPointType> result{};
        for (size_t i = 0; i < variables_to_observe.size(); ++i) {
          result[variables_to_observe[i]] = floating_point_types.size() == 1
                                                ? floating_point_types[0]
                                                : floating_point_types[i];
          ASSERT(
              result.at(variables_to_observe[i]) == FloatingPointType::Float or
                  result.at(variables_to_observe[i]) ==
                      FloatingPointType::Double,
              "Floating point type for variable '"
                  << variables_to_observe[i]
                  << "' must be either Float or Double.");
        }
        return result;
      }()),
      interp_dim_(interp_dim),
      interp_val_(interp_val) {
  ASSERT(
      (... or (db::tag_name<Tensors>() == "InertialCoordinates")),
      "There is no tag with name 'InertialCoordinates' specified "
      "for the observer. Please make sure you specify a tag in the 'Tensors' "
      "list that has the 'db::tag_name()' 'InertialCoordinates'.");
  db::validate_selection<tmpl::list<Tensors...>>(variables_to_observe, context);
  variables_to_observe_["InertialCoordinates"] =
      coordinates_floating_point_type;
}

/// \cond
template <size_t VolumeDim, typename... Tensors,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveInterpolatedData<VolumeDim, tmpl::list<Tensors...>,
                                          tmpl::list<NonTensorComputeTags...>,
                                          ArraySectionIdTag>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace dg::Events
