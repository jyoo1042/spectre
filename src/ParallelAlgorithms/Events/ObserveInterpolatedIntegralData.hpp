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

#include "DataStructures/DataBox/DataBox.hpp"
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
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
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
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace dg::Events {
namespace detail {
using ObserveInterpolatedReductionData = Parallel::ReductionData<
    // Observation value (Time)
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    // MassAccretionRate Mdot
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // MagneticFlux Phi_BH
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // Total Energy Flux Edot
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // AngularMomentum Flux Ldot
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // SurfaceIntegral of the given tag
    Parallel::ReductionDatum<double, funcl::Plus<>>>;
}  // namespace detail
/// \cond
template <size_t VolumeDim, typename Tensors,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename ArraySectionIdTag = void>
class ObserveInterpolatedIntegralData;
/// \endcond

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe volume tensor fields interpolated to a new mesh slice and
 * the corresponding surface integral.
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
class ObserveInterpolatedIntegralData<VolumeDim, tmpl::list<Tensors...>,
                                      tmpl::list<NonTensorComputeTags...>,
                                      ArraySectionIdTag> : public Event {
 public:
  using ReductionData = Events::detail::ObserveInterpolatedReductionData;

  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveInterpolatedIntegralData(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveInterpolatedIntegralData);  // NOLINT
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

  ObserveInterpolatedIntegralData() = default;

  ObserveInterpolatedIntegralData(
      const std::string& subfile_name,
      FloatingPointType coordinates_floating_point_type,
      const std::vector<FloatingPointType>& floating_point_types,
      const std::vector<std::string>& variables_to_observe,
      const size_t interp_dim, const double interp_val,
      const Options::Context& context = {});

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

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
    // this will be surface integral of a given tag for debugging purpose.
    double surface_integral_value = 0.0;
    double magnetic_flux = 0.0;
    double mdot = 0.0;
    double ldot = 0.0;
    double edot = 0.0;

    const DataVector det_jacobian =
        1. /
        get(get<::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                       Frame::Inertial>>(box));
    const auto jac =
        get<::Events::Tags::ObserverJacobian<VolumeDim, Frame::ElementLogical,
                                             Frame::Inertial>>(box);

    const double lower_bound_bl =
        element_id.segment_id(interp_dim).endpoint(Side::Lower);
    const double upper_bound_bl =
        element_id.segment_id(interp_dim).endpoint(Side::Upper);

    const double x0 = element_id.segment_id(0).endpoint(Side::Lower);
    const double y0 = element_id.segment_id(1).endpoint(Side::Lower);
    const double z0 = element_id.segment_id(2).endpoint(Side::Lower);

    // we only need to work on the elements where the intepolation surface
    // lies in between
    if ((lower_bound_bl < interp_val) and (upper_bound_bl >= interp_val)) {
      // (D-1) dimensional mesh with original mesh sliced away in the
      // interpolation dimension.
      const auto new_mesh = mesh.slice_away(interp_dim);

      // Actually need the element logical value corresponding to
      // interpolation target for next step
      // This step of course assumes, we are already in the element in which
      // the interpolation target lies.
      // checked this formula on Oct 11 once again!!.
      const double elm_interp_val =
          (interp_val - ((upper_bound_bl + lower_bound_bl) / 2.)) /
          ((upper_bound_bl - lower_bound_bl) / 2.);

      // when sliced away the target_points has values -1 (lowest val) for
      // interp_dim. set it to appropriate element value as computed above.
      auto target_points = data_on_slice(logical_coordinates(mesh),
                                         mesh.extents(), interp_dim, 0);
      target_points.get(interp_dim) = elm_interp_val;
      const intrp::Irregular<VolumeDim> interpolant(mesh, target_points);

      // we need to convert the above target points in element logical frame
      // to inertial frame so that we can evaluate the metric quantities
      auto ones =
          make_with_value<tnsr::I<DataVector, VolumeDim, Frame::Inertial>>(
              target_points, 1.0);
      auto target_points_inertial =
          make_with_value<tnsr::I<DataVector, VolumeDim, Frame::Inertial>>(
              target_points, x0);

      get<1>(target_points_inertial) = y0;
      get<2>(target_points_inertial) = z0;
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = 0; j < VolumeDim; ++j) {
          target_points_inertial.get(i) +=
              jac.get(i, j)[0] * (target_points.get(j) + ones.get(j));
        }
      }
      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;
      auto* initial_data_ptr =
          &Parallel::get<evolution::initial_data::Tags::InitialData>(cache);
      // all the metric related tags that we need for computing
      // MassAccretionRate: shift, lapse, sqrt_det_spatial_metric
      using metric_tags =
          tmpl::list<gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
                     gr::Tags::SqrtDetSpatialMetric<DataVector>,
                     gr::Tags::Lapse<DataVector>>;
      using metric_tuples = tuples::tagged_tuple_from_typelist<metric_tags>;
      auto metric_quantities =
          call_with_dynamic_type<metric_tuples, derived_classes>(
              initial_data_ptr,
              [&target_points_inertial](const auto* const data_or_solution) {
                return evolution::Initialization::initial_data(
                    *data_or_solution, target_points_inertial, 0.0,
                    metric_tags{});
              });

      const auto rho = get(get<hydro::Tags::RestMassDensity<DataVector>>(box));
      const auto vr = get<0>(
          get<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>(
              box));
      const auto br = get<0>(
          get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(box));
      const auto lf = get(get<hydro::Tags::LorentzFactor<DataVector>>(box));
      const auto record_tensor_component_impl =
          [&interpolant, &surface_integral_value, &magnetic_flux, &mdot, &edot,
           &ldot, &new_mesh, &det_jacobian, &rho, &vr, &br, &lf,
           &metric_quantities](const auto& tensor) {
            const auto new_tensor = interpolant.interpolate(tensor[0]);
            const auto new_rho = interpolant.interpolate(rho);
            const auto new_det_jacobian = interpolant.interpolate(det_jacobian);
            const auto new_vr = interpolant.interpolate(vr);
            const auto new_br = interpolant.interpolate(br);
            const auto new_lf = interpolant.interpolate(lf);

            // this is where we perform surface intgral of the "new_tensor"
            // basically whichever tensor integrated at the surface
            const double surface_integral_value_contribution =
                definite_integral(new_tensor * new_det_jacobian, new_mesh);
            surface_integral_value += surface_integral_value_contribution;

            // this is where we compute Mdot
            // here we directly compute all the metric quantities and
            // interpolate the rest

            const auto shift =
                get<0>(get<gr::Tags::Shift<DataVector, 3, Frame::Inertial>>(
                    metric_quantities));
            const auto lapse =
                get(get<gr::Tags::Lapse<DataVector>>(metric_quantities));
            const auto detsqrtsm =
                get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
                    metric_quantities));
            // we zero out potential outflow
            DataVector integrand =
                new_rho * new_lf * detsqrtsm * (lapse * new_vr - shift);
            for (size_t i = 0; i < integrand.size(); i++) {
              if (integrand[i] > 0) {
                integrand[i] = 0;
              }
            }
            // mdot
            // integral rho * u^r sqrt(-g) dphi dtheta
            const double mdot_contribution =
                definite_integral(integrand * new_det_jacobian, new_mesh);
            mdot += mdot_contribution;

            // magnetic flux
            // 1/2 integral *F^(tr) sqrt(-g) dphi dtheta
            // note B^r = alpha * (*F^(tr)) and sqrt(-g) = alpha * gamma
            // so we only need gamma * B^r
            DataVector magnetic_flux_integrand = detsqrtsm * abs(new_br);
            const double magnetic_flux_contribution =
                0.5 * definite_integral(
                          magnetic_flux_integrand * new_det_jacobian, new_mesh);
            magnetic_flux += magnetic_flux_contribution;
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
                // This will only print a warning the first time it's called
                // on a node.
                [[maybe_unused]] static bool t =
                    ObserveInterpolatedIntegralData::
                        print_warning_about_optional<tensor_tag>();
                return;
              }
              const auto floating_point_type = var_to_observe->second;
              record_tensor_component_impl(value(tensor));
            }
          };
      EXPAND_PACK_LEFT_TO_RIGHT(
          record_tensor_components(tmpl::type_<Tensors>{}));
    }
    // Send data to volume observer
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(observation_value.value,
                                 subfile_path + ".dat"),
        Parallel::make_array_component_id<ParallelComponent>(element_id),
        subfile_path,
        std::vector<std::string>{observation_value.name, "Mdot", "MagneticFlux",
                                 "Edot", "Ldot", "SurfaceIntegral"},
        ReductionData{observation_value.value, std::move(mdot),
                      std::move(magnetic_flux), std::move(edot),
                      std::move(ldot), std::move(surface_integral_value)});
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
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(
                 subfile_path_ + section_observation_key.value() + ".dat")}};
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
ObserveInterpolatedIntegralData<VolumeDim, tmpl::list<Tensors...>,
                                tmpl::list<NonTensorComputeTags...>,
                                ArraySectionIdTag>::
    ObserveInterpolatedIntegralData(
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
  // variables_to_observe_["InertialCoordinates"] =
  //     coordinates_floating_point_type;
}

/// \cond
template <size_t VolumeDim, typename... Tensors,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveInterpolatedIntegralData<
    VolumeDim, tmpl::list<Tensors...>, tmpl::list<NonTensorComputeTags...>,
    ArraySectionIdTag>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace dg::Events
