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
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
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
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
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

#include "Utilities/ErrorHandling/CaptureForError.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace {

template <typename DataType>
void four_velocity(const gsl::not_null<tnsr::A<DataType, 3>*> result,
                   const tnsr::I<DataType, 3>& spatial_velocity,
                   const tnsr::I<DataType, 3>& shift,
                   const Scalar<DataType>& lorentz_factor,
                   const Scalar<DataType>& lapse) {
  get<0>(*result) = get(lorentz_factor) / get(lapse);
  for (size_t i = 0; i < 3; ++i) {
    result->get(i + 1) =
        get<0>(*result) * (get(lapse) * spatial_velocity.get(i) - shift.get(i));
  }
}

std::array<DataVector, 4> NewEFlux(
    const Scalar<DataVector>& rho, const Scalar<DataVector>& energy,
    const Scalar<DataVector>& pressure,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& comoving_magnetic_field_magnitude,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric_v) {
  const size_t num_pts = get_size(get(rho));
  TempBuffer<tmpl::list<
      ::Tags::TempScalar<0, DataVector>, ::Tags::TempScalar<1, DataVector>,
      ::Tags::TempScalar<2, DataVector>, ::Tags::TempScalar<3, DataVector>,
      ::Tags::TempScalar<4, DataVector>, ::Tags::TempScalar<5, DataVector>,
      ::Tags::TempScalar<6, DataVector>, ::Tags::TempScalar<7, DataVector>,
      ::Tags::TempScalar<8, DataVector>, ::Tags::TempScalar<9, DataVector>,
      ::Tags::TempScalar<10, DataVector>, ::Tags::TempScalar<11, DataVector>,
      ::Tags::TempScalar<12, DataVector>,
      ::Tags::TempA<13, 3, Frame::Inertial, DataVector>,
      ::Tags::TempA<14, 3, Frame::Inertial, DataVector>,
      ::Tags::Tempa<15, 3, Frame::Inertial, DataVector>,
      ::Tags::Tempa<16, 3, Frame::Inertial, DataVector>>>
      buffer(num_pts);

  auto& rho_h = get<::Tags::TempScalar<0, DataVector>>(buffer);
  auto& rho_h_2 = get<::Tags::TempScalar<1, DataVector>>(buffer);
  auto& magnetic_field_dot_spatial_velocity =
      get<::Tags::TempScalar<2, DataVector>>(buffer);
  auto& shift_dot_spatial_velocity =
      get<::Tags::TempScalar<3, DataVector>>(buffer);
  auto& velocity_squared = get<::Tags::TempScalar<4, DataVector>>(buffer);
  auto& new_lorentz_factor = get<::Tags::TempScalar<5, DataVector>>(buffer);
  auto& shift_dot_magnetic_field =
      get<::Tags::TempScalar<6, DataVector>>(buffer);

  auto& u_t = get<::Tags::TempScalar<7, DataVector>>(buffer);
  auto& b_t = get<::Tags::TempScalar<8, DataVector>>(buffer);

  auto& result1 = get<::Tags::TempScalar<9, DataVector>>(buffer);
  auto& result2 = get<::Tags::TempScalar<10, DataVector>>(buffer);
  auto& result3 = get<::Tags::TempScalar<11, DataVector>>(buffer);
  auto& result4 = get<::Tags::TempScalar<12, DataVector>>(buffer);

  auto& comoving_magnetic_field_v =
      get<::Tags::TempA<13, 3, Frame::Inertial, DataVector>>(buffer);
  auto& four_velocity_v =
      get<::Tags::TempA<14, 3, Frame::Inertial, DataVector>>(buffer);
  auto& comoving_magnetic_field_one_form_v =
      get<::Tags::Tempa<15, 3, Frame::Inertial, DataVector>>(buffer);
  auto& four_velocity_one_form_v =
      get<::Tags::Tempa<16, 3, Frame::Inertial, DataVector>>(buffer);

  dot_product(make_not_null(&magnetic_field_dot_spatial_velocity),
              magnetic_field, spatial_velocity, spatial_metric);

  dot_product(make_not_null(&shift_dot_spatial_velocity), shift,
              spatial_velocity, spatial_metric);

  dot_product(make_not_null(&velocity_squared), spatial_velocity,
              spatial_velocity, spatial_metric);

  dot_product(make_not_null(&shift_dot_magnetic_field), shift, magnetic_field,
              spatial_metric);

  hydro::lorentz_factor(make_not_null(&new_lorentz_factor), velocity_squared);

  hydro::comoving_magnetic_field(make_not_null(&comoving_magnetic_field_v),
                                 spatial_velocity, magnetic_field,
                                 magnetic_field_dot_spatial_velocity,
                                 lorentz_factor, shift, lapse);

  four_velocity(make_not_null(&four_velocity_v), spatial_velocity, shift,
                lorentz_factor, lapse);

  tenex::evaluate<ti::b>(
      make_not_null(&comoving_magnetic_field_one_form_v),
      comoving_magnetic_field_v(ti::A) * spacetime_metric_v(ti::a, ti::b));

  tenex::evaluate<ti::b>(
      make_not_null(&four_velocity_one_form_v),
      four_velocity_v(ti::A) * spacetime_metric_v(ti::a, ti::b));

  // - alpha W + beta * u
  // = W (-alpha + beta * v)
  get(u_t) =
      get(lorentz_factor) * (-get(lapse) + get(shift_dot_spatial_velocity));
  // - alpha W (v*B) + (beta * b)
  // = (v*B) * u_t  + (beta * B) / W
  get(b_t) = (get(magnetic_field_dot_spatial_velocity) * get(u_t)) +
             (get(shift_dot_magnetic_field) / get(lorentz_factor));

  get(rho_h) = (get(rho) + get(rho) * get(energy) + get(pressure));
  get(rho_h_2) = get(rho) * (1. + 4. / 3. * get(energy));

  // this computed T^r_t from newly computed u^mu u_mu b^mu b_nu
  // where one forms are computed by using spacetime metric
  get(result1) = (get(rho_h) + square(get(comoving_magnetic_field_magnitude))) *
                     get<1>(four_velocity_v) *
                     get<0>(four_velocity_one_form_v) -
                 get<1>(comoving_magnetic_field_v) *
                     get<0>(comoving_magnetic_field_one_form_v);
  // this just computed the one form components from the formula above
  get(result2) = (get(rho_h) + square(get(comoving_magnetic_field_magnitude))) *
                     get<1>(four_velocity_v) * get(u_t) -
                 get<1>(comoving_magnetic_field_v) * get(b_t);

  // this is same as result 1 but uses rho_h_2 instead where we compute stuff
  // from energy
  get(result3) =
      (get(rho_h_2) + square(get(comoving_magnetic_field_magnitude))) *
          get<1>(four_velocity_v) * get<0>(four_velocity_one_form_v) -
      get<1>(comoving_magnetic_field_v) *
          get<0>(comoving_magnetic_field_one_form_v);

  // this is same as result 2 but uses rho_h_2 instead where we compute stuff
  // from energy
  get(result4) =
      (get(rho_h_2) + square(get(comoving_magnetic_field_magnitude))) *
          get<1>(four_velocity_v) * get(u_t) -
      get<1>(comoving_magnetic_field_v) * get(b_t);

  // consistency check:
  for (size_t i = 0; i < num_pts; ++i) {
    if (abs(get(lorentz_factor)[i] - get(new_lorentz_factor)[i]) > 1.e-5) {
      ERROR("Lorentz factor and spatial velocity are not consistent!\n"
            << std::setprecision(17)
            << "Lorentz factor in box = " << get(lorentz_factor)[i] << "\n"
            << "Lorentz factor calculated = " << get(new_lorentz_factor)[i]
            << "\n"
            << "rest mass density = " << get(rho)[i] << "\n"
            << "energy = " << get(energy)[i] << "\n"
            << "pressure = " << get(pressure)[i] << "\n"
            << "comoving magnetic field magnitude = "
            << get(comoving_magnetic_field_magnitude)[i] << "\n"
            << "magnetic field dot v = "
            << get(magnetic_field_dot_spatial_velocity)[i] << "\n"
            << "result1 = " << get(result1)[i] << "\n"
            << "result2 = " << get(result2)[i] << "\n"
            << "result3 = " << get(result3)[i] << "\n"
            << "result4 = " << get(result4)[i] << "\n");
    }
  }

  const std::array<DataVector, 4> result{
      {get(result1), get(result2), get(result3), get(result4)}};
  return result;
}
}  // namespace

namespace dg::Events {
namespace detail {
using ObserveInterpolatedReductionData = Parallel::ReductionData<
    // Observation value (Time)
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    // Mdot: rho u^r sqrt(-g)
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // Edot: -get<1,0>(T^mu nu g_nu rho) sqrt(-g)
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // Ldot: get<1,3>(T^mu nu g_nu rho) sqrt(-g)
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // Phi_B: 0.5 |B^r| gamma
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
    // computed by computing integrand outside and then interpolated as whole
    double mdot = 0.0;
    double edot = 0.0;
    double ldot = 0.0;
    double phib = 0.0;

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

      // get hydro stuffs from the box
      const auto& rho = get<hydro::Tags::RestMassDensity<DataVector>>(box);
      const auto& energy =
          get<hydro::Tags::SpecificInternalEnergy<DataVector>>(box);
      const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(box);
      const auto& lorentz_factor =
          get<hydro::Tags::LorentzFactor<DataVector>>(box);
      const auto& comoving_magnetic_field_magnitude =
          get<hydro::Tags::ComovingMagneticFieldMagnitude<DataVector>>(box);
      const auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>(
              box);
      const auto& magnetic_field =
          get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(box);

      // get metric tags from the box
      const auto& shift =
          get<gr::Tags::Shift<DataVector, 3, Frame::Inertial>>(box);
      const auto& lapse = get<gr::Tags::Lapse<DataVector>>(box);
      const auto& spatial_metric =
          get<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>(box);
      const auto& inverse_spatial_metric =
          get<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>(
              box);
      const auto& gamma = get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(box);

      tnsr::AA<DataVector, 3, Frame::Inertial> stress_energy_tensor_v{};
      hydro::stress_energy_tensor(make_not_null(&stress_energy_tensor_v), rho,
                                  energy, pressure, lorentz_factor, lapse,
                                  comoving_magnetic_field_magnitude,
                                  spatial_velocity, shift, magnetic_field,
                                  spatial_metric, inverse_spatial_metric);

      const auto spacetime_metric_v =
          gr::spacetime_metric(lapse, shift, spatial_metric);

      tnsr::Ab<DataVector, 3, Frame::Inertial> lowered_stress_energy_tensor_v{};
      tenex::evaluate<ti::A, ti::c>(
          make_not_null(&lowered_stress_energy_tensor_v),
          stress_energy_tensor_v(ti::A, ti::B) *
              spacetime_metric_v(ti::b, ti::c));

      const DataVector sqrt_g = get(lapse) * get(gamma);

      // this is the regular integrand computed from the
      // lowered component of Stress Energy Tensor
      // using StressEnergy.cpp and spacetime_metric_v

      const DataVector mdot_integrand =
          get(rho) * get(lorentz_factor) * get(gamma) *
          (get(lapse) * get<0>(spatial_velocity) - get<0>(shift));
      const DataVector edot_integrand =
          -sqrt_g * get<1, 0>(lowered_stress_energy_tensor_v);
      const DataVector ldot_integrand =
          sqrt_g * get<1, 3>(lowered_stress_energy_tensor_v);
      const DataVector phib_integrand =
          0.5 * get(gamma) * abs(get<0>(magnetic_field));

      const auto record_tensor_component_impl =
          [&interpolant, &new_mesh, &det_jacobian, &mdot, &edot, &ldot, &phib,
           &mdot_integrand, &edot_integrand, &ldot_integrand,
           &phib_integrand](const auto& tensor) {
            const auto mdot_integrand_interpolated =
                interpolant.interpolate(mdot_integrand);
            const auto edot_integrand_interpolated =
                interpolant.interpolate(edot_integrand);
            const auto ldot_integrand_interpolated =
                interpolant.interpolate(ldot_integrand);
            const auto phib_integrand_interpolated =
                interpolant.interpolate(phib_integrand);

            const auto det_jacobian_interpolated =
                interpolant.interpolate(det_jacobian);

            const double mdot_contribution = definite_integral(
                mdot_integrand_interpolated * det_jacobian_interpolated,
                new_mesh);
            mdot += mdot_contribution;

            const double edot_contribution = definite_integral(
                edot_integrand_interpolated * det_jacobian_interpolated,
                new_mesh);
            edot += edot_contribution;

            const double ldot_contribution = definite_integral(
                ldot_integrand_interpolated * det_jacobian_interpolated,
                new_mesh);
            ldot += ldot_contribution;

            const double phib_contribution = definite_integral(
                phib_integrand_interpolated * det_jacobian_interpolated,
                new_mesh);
            phib += phib_contribution;
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
        std::vector<std::string>{observation_value.name, "mdot", "edot", "ldot",
                                 "phib"},
        ReductionData{observation_value.value, std::move(mdot), std::move(edot),
                      std::move(ldot), std::move(phib)});
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
