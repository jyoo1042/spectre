// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_set>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/HighOrderFluxCorrection.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct Scalar0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Vector0 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
double test(const fd::DerivativeOrder correction_order,
            const bool aligned_coordinates) {
  CAPTURE(aligned_coordinates);
  CAPTURE(Dim);
  if (not aligned_coordinates) {
    ASSERT(Dim == 2 or Dim == 3,
           "A test for a non-aligned grid is not provided for 1-D.");
  };
  const size_t max_degree =
      correction_order == fd::DerivativeOrder::OneHigherThanRecons
          ? 6
          : (correction_order ==
                     fd::DerivativeOrder::OneHigherThanReconsButFiveToFour
                 ? 4
                 : static_cast<size_t>(correction_order));
  const size_t points_per_dimension = static_cast<size_t>(max_degree) + 2;
  const size_t stencil_width = max_degree + 1;
  const size_t number_of_ghost_points = (stencil_width - 1) / 2 + 1;
  CAPTURE(points_per_dimension);

  using FluxTags = tmpl::list<Scalar0, Vector0<Dim>>;
  using Scalar0Flux = ::Tags::Flux<Scalar0, tmpl::size_t<Dim>, Frame::Inertial>;
  using Vector0Flux =
      ::Tags::Flux<Vector0<Dim>, tmpl::size_t<Dim>, Frame::Inertial>;
  using FluxVars =
      Variables<db::wrap_tags_in<::Tags::Flux, FluxTags, tmpl::size_t<Dim>,
                                 Frame::Inertial>>;
  using CorrectionVars = Variables<FluxTags>;

  const Mesh<Dim> subcell_mesh{points_per_dimension,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * static_cast<double>(i);
  }

  std::unique_ptr<
      domain::CoordinateMapBase<Frame::ElementLogical, Frame::Inertial, Dim>>
      coordinate_map = nullptr;
  if (aligned_coordinates) {
    coordinate_map = std::make_unique<
        domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                              domain::CoordinateMaps::Identity<Dim>>>();
  } else {
    if constexpr (Dim == 2) {
      coordinate_map = std::make_unique<domain::CoordinateMap<
          Frame::ElementLogical, Frame::Inertial,
          domain::CoordinateMaps::Rotation<2>,
          domain::CoordinateMaps::ProductOf2Maps<
              domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine>>>(
          domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
              domain::CoordinateMaps::Rotation<2>(2. * M_PI_4 / 3.),
              domain::CoordinateMaps::ProductOf2Maps<
                  domain::CoordinateMaps::Affine,
                  domain::CoordinateMaps::Affine>{
                  domain::CoordinateMaps::Affine{-1., 1., -2., 2.},
                  domain::CoordinateMaps::Affine{-1., 1., -0.3, 1.7}}));
    } else if constexpr (Dim == 3) {
      coordinate_map = std::make_unique<
          domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                domain::CoordinateMaps::BulgedCube>>(
          domain::CoordinateMaps::BulgedCube(5., 0.1, false));
    }
  }
  const auto inertial_coords = (*coordinate_map)(logical_coords);
  const auto inv_jacobian = coordinate_map->inv_jacobian(logical_coords);
  const auto det_inv_jacobian = determinant(inv_jacobian);

  // Compute polynomial on cell centers in FD cluster of points
  const auto set_polynomial = Overloader{
      [max_degree](const gsl::not_null<FluxVars*> vars_ptr,
                   const auto& local_inertial_coords) {
        (void)max_degree;
        for (size_t storage_index = 0;
             storage_index < get<Scalar0Flux>(*vars_ptr).size();
             ++storage_index) {
          get<Scalar0Flux>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Scalar0Flux>(*vars_ptr)[storage_index] +=
                  pow(local_inertial_coords.get(i), degree);
            }
          }
        }
        for (size_t storage_index = 0;
             storage_index < get<Vector0Flux>(*vars_ptr).size();
             ++storage_index) {
          get<Vector0Flux>(*vars_ptr)[storage_index] =
              1.0 + 0.3 * static_cast<double>(storage_index);
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0Flux>(*vars_ptr)[storage_index] +=
                  pow(local_inertial_coords.get(i), degree);
            }
          }
        }
      },
      [max_degree](const gsl::not_null<CorrectionVars*> vars_ptr,
                   const auto& local_inertial_coords) {
        (void)max_degree;
        for (size_t storage_index = 0;
             storage_index < get<Scalar0>(*vars_ptr).size(); ++storage_index) {
          get<Scalar0>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Scalar0>(*vars_ptr)[storage_index] +=
                  pow(local_inertial_coords.get(i), degree);
            }
          }
        }
        for (size_t storage_index = 0;
             storage_index < get<Vector0<Dim>>(*vars_ptr).size();
             ++storage_index) {
          get<Vector0<Dim>>(*vars_ptr)[storage_index] =
              100.0 + 11.0 * static_cast<double>(storage_index);
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*vars_ptr)[storage_index] +=
                  pow(local_inertial_coords.get(i), degree);
            }
          }
        }
      }};
  const auto set_polynomial_divergence =
      [max_degree](const gsl::not_null<CorrectionVars*> d_vars_ptr,
                   const auto& local_inertial_coords) {
        (void)max_degree;
        get(get<Scalar0>(*d_vars_ptr)) = 0.0;
        for (size_t i = 0; i < Dim; ++i) {
          // constant deriv is zero
          get<Vector0<Dim>>(*d_vars_ptr).get(i) = 0.0;
        }
        // Compute divergence
        for (size_t deriv_dim = 0; deriv_dim < Dim; ++deriv_dim) {
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            get(get<Scalar0>(*d_vars_ptr)) +=
                degree * pow(local_inertial_coords.get(deriv_dim), degree - 1);
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*d_vars_ptr).get(i) +=
                  degree *
                  pow(local_inertial_coords.get(deriv_dim), degree - 1);
            }
          }
        }
      };
  std::optional<FluxVars> volume_vars(subcell_mesh.number_of_grid_points());
  set_polynomial(&(volume_vars.value()), inertial_coords);

  CorrectionVars expected_divergence(subcell_mesh.number_of_grid_points());
  set_polynomial_divergence(&expected_divergence, inertial_coords);

  // Compute the polynomial at the cell center for the neighbor data that we
  // "received".
  //
  // We do this by computing the solution in our entire neighbor, then using
  // slice_data to get the subset of points that are needed.
  DirectionMap<Dim, FluxVars> neighbor_data{};
  DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>
      reconstruction_ghost_data{};

  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_inertial_coords =
        (*coordinate_map)(neighbor_logical_coords);
    FluxVars neighbor_vars(subcell_mesh.number_of_grid_points(), 0.0);
    set_polynomial(&neighbor_vars, neighbor_inertial_coords);

    const auto sliced_data = evolution::dg::subcell::slice_data(
        neighbor_vars, subcell_mesh.extents(), number_of_ghost_points,
        std::unordered_set{direction.opposite()}, 0, {});
    CAPTURE(number_of_ghost_points);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    REQUIRE(sliced_data.at(direction.opposite()).size() %
                FluxVars::number_of_independent_components ==
            0);
    neighbor_data[direction].initialize(
        sliced_data.at(direction.opposite()).size() /
        FluxVars::number_of_independent_components);
    std::copy(sliced_data.at(direction.opposite()).begin(),
              sliced_data.at(direction.opposite()).end(),
              neighbor_data[direction].data());

    const DirectionalId<Dim> mortar_id{direction, ElementId<Dim>{0}};
    reconstruction_ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    reconstruction_ghost_data[mortar_id]
        .neighbor_ghost_data_for_reconstruction() =
        DataVector{sliced_data.at(direction.opposite()).size()};
    std::copy(sliced_data.at(direction.opposite()).begin(),
              sliced_data.at(direction.opposite()).end(),
              reconstruction_ghost_data[mortar_id]
                  .neighbor_ghost_data_for_reconstruction()
                  .data());
  }

  std::array<CorrectionVars, Dim> second_order_corrections{};
  std::array<
      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>,
      Dim>
      correction_inv_jacobians{};
  std::array<Scalar<DataVector>, Dim> correction_det_inv_jacobians{};

  for (size_t i = 0; i < Dim; ++i) {
    // Compare to analytic solution on the faces.
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<Dim>(points_per_dimension);
    gsl::at(extents, i) = points_per_dimension + 1;
    gsl::at(quadrature, i) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    auto face_logical_coords = logical_coordinates(face_centered_mesh);
    for (size_t j = 1; j < Dim; ++j) {
      face_logical_coords.get(j) += 4.0 * static_cast<double>(j);
    }
    const auto face_inertial_coords = (*coordinate_map)(face_logical_coords);

    gsl::at(second_order_corrections, i)
        .initialize(face_centered_mesh.number_of_grid_points());
    set_polynomial(make_not_null(&gsl::at(second_order_corrections, i)),
                   face_inertial_coords);

    gsl::at(correction_inv_jacobians, i) =
        coordinate_map->inv_jacobian(face_logical_coords);
    gsl::at(correction_det_inv_jacobians, i) =
        determinant(gsl::at(correction_inv_jacobians, i));

    // We use n_i F^i in the code, so need to negate to get sign to agree.
    gsl::at(second_order_corrections, i) *= -1.0;
  }

  // If not using an aligned coordinate map, we must convert the boundary term
  // to a local Cartesian flux on the logical grid. Assuming a flat spacetime,
  // we do this by computing G^{(\hat{i})} = J \pdv{\xi^\hat{i}}{x^j} G^{(i)}.
  // In this case, we are `cheating' a bit because the polynomial is defined
  // such that the flux in each direction of the inertial frame is the same.
  // This is exploited in the coordinate transformation.
  if (not aligned_coordinates) {
    const auto inertial_second_order_corrections = second_order_corrections;
    for (size_t storage_index = 0;
         storage_index <
         get(get<Scalar0>(gsl::at(second_order_corrections, 0))).size();
         ++storage_index) {
      for (size_t i = 0; i < Dim; ++i) {
        get(get<Scalar0>(gsl::at(second_order_corrections, i)))[storage_index] =
            0.;
        for (size_t j = 0; j < Dim; ++j) {
          get(get<Scalar0>(
              gsl::at(second_order_corrections, i)))[storage_index] +=
              gsl::at(correction_inv_jacobians, i).get(i, j)[storage_index] *
              get(get<Scalar0>(gsl::at(inertial_second_order_corrections,
                                       i)))[storage_index] /
              get(gsl::at(correction_det_inv_jacobians, i))[storage_index];
          get<Vector0<Dim>>(gsl::at(second_order_corrections, i))
              .get(j)[storage_index] = 0.;
          for (size_t k = 0; k < Dim; ++k) {
            get<Vector0<Dim>>(gsl::at(second_order_corrections, i))
                .get(j)[storage_index] +=
                gsl::at(correction_inv_jacobians, i).get(i, k)[storage_index] *
                get<Vector0<Dim>>(gsl::at(inertial_second_order_corrections, i))
                    .get(k)[storage_index] /
                get(gsl::at(correction_det_inv_jacobians, i))[storage_index];
          }
        }
      }
    }
  }

  std::array<std::vector<std::uint8_t>, Dim> reconstruction_order_storage{};
  std::array<gsl::span<std::uint8_t>, Dim> reconstruction_order{};
  if (correction_order == fd::DerivativeOrder::OneHigherThanRecons or
      correction_order ==
          fd::DerivativeOrder::OneHigherThanReconsButFiveToFour) {
    Index<Dim> recons_extents = subcell_mesh.extents();
    recons_extents[0] += 2;
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(reconstruction_order_storage, i) =
          std::vector<std::uint8_t>(recons_extents.product(), 5);
      gsl::at(reconstruction_order, i) =
          gsl::span(gsl::at(reconstruction_order_storage, i).data(),
                    gsl::at(reconstruction_order_storage, i).size());
    }
  }

  // The unnormalized normal vector is n_j = d \xi^{\hat i}/dx^j with "i"
  // the current face. When normalizing, we adopt a sign convention which
  // is compatible with DG.
  std::array<tnsr::i<DataVector, Dim, Frame::Inertial>, Dim> lower_conormal;
  std::array<tnsr::i<DataVector, Dim, Frame::Inertial>, Dim> upper_conormal;
  if (not aligned_coordinates) {
    Index<Dim> recons_extents = subcell_mesh.extents();
    recons_extents[0] += 2;
    const size_t reconstructed_num_pts = recons_extents.product();
    tnsr::i<DataVector, Dim, Frame::Inertial> lower_outward_conormal{
        reconstructed_num_pts, 0.0};
    const Mesh<Dim> dg_mesh{points_per_dimension, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
    auto dg_logical_coords = logical_coordinates(dg_mesh);

    // Make the logical coordinates different in each direction
    for (size_t i = 1; i < Dim; ++i) {
      dg_logical_coords.get(i) += 4.0 * static_cast<double>(i);
    }

    const auto dg_inv_jacobian =
        coordinate_map->inv_jacobian(dg_logical_coords);
    const auto dg_det_inv_jacobian = determinant(dg_inv_jacobian);
    for (size_t i = 0; i < Dim; ++i) {
      // Build extents of mesh shifted by half a grid cell in direction i
      const unsigned long& num_subcells_1d = subcell_mesh.extents(0);
      Index<Dim> face_mesh_extents;
      if constexpr (Dim == 2) {
        face_mesh_extents =
            Index<2>(std::array<size_t, 2>{num_subcells_1d, num_subcells_1d});
      } else if constexpr (Dim == 3) {
        face_mesh_extents = Index<3>(std::array<size_t, 3>{
            num_subcells_1d, num_subcells_1d, num_subcells_1d});
      }
      face_mesh_extents[i] = num_subcells_1d + 1;

      for (size_t j = 0; j < Dim; j++) {
        lower_outward_conormal.get(j) =
            evolution::dg::subcell::fd::project_to_faces(
                dg_inv_jacobian.get(i, j), dg_mesh, face_mesh_extents, i);
      }
      const auto det_inv_jacobian_face =
          evolution::dg::subcell::fd::project_to_faces(
              get(dg_det_inv_jacobian), dg_mesh, face_mesh_extents, i);

      const Scalar<DataVector> normalization{sqrt(
          get(dot_product(lower_outward_conormal, lower_outward_conormal)))};
      for (size_t j = 0; j < Dim; j++) {
        lower_outward_conormal.get(j) =
            lower_outward_conormal.get(j) / get(normalization);
      }
      tnsr::i<DataVector, Dim, Frame::Inertial> upper_outward_conormal{
          reconstructed_num_pts, 0.0};
      for (size_t j = 0; j < Dim; j++) {
        upper_outward_conormal.get(j) = -lower_outward_conormal.get(j);
        gsl::at(upper_conormal, i).get(j) =
            upper_outward_conormal.get(j) *
            (get(normalization) / det_inv_jacobian_face);
        gsl::at(lower_conormal, i).get(j) =
            lower_outward_conormal.get(j) *
            (get(normalization) / det_inv_jacobian_face);
      }
    }
  }

  // Now compute the Cartesian derivative of the high_order_corrections to
  // verify that it is computed sufficiently accurately.
  std::optional<std::array<CorrectionVars, Dim>> high_order_corrections{};
  ::fd::cartesian_high_order_flux_corrections(
      make_not_null(&high_order_corrections), volume_vars,
      second_order_corrections, correction_order, reconstruction_ghost_data,
      subcell_mesh, number_of_ghost_points, reconstruction_order,
      aligned_coordinates, lower_conormal, upper_conormal);

  CorrectionVars flux_divergence{subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t d = 0; d < Dim; ++d) {
    const auto& corrections_in_dim =
        high_order_corrections.has_value()
            ? gsl::at(high_order_corrections.value(), d)
            : gsl::at(second_order_corrections, d);
    // Note: assumes isotropic mesh
    const double one_over_delta_xi =
        -1.0 / (logical_coords.get(0)[1] - logical_coords.get(0)[0]);
    evolution::dg::subcell::add_cartesian_flux_divergence(
        make_not_null(&get(get<Scalar0>(flux_divergence))), one_over_delta_xi,
        get(det_inv_jacobian), get(get<Scalar0>(corrections_in_dim)),
        subcell_mesh.extents(), d);
    for (size_t i = 0; i < Dim; ++i) {
      evolution::dg::subcell::add_cartesian_flux_divergence(
          make_not_null(&get<Vector0<Dim>>(flux_divergence).get(i)),
          one_over_delta_xi, get(det_inv_jacobian),
          get<Vector0<Dim>>(corrections_in_dim).get(i), subcell_mesh.extents(),
          d);
    }
  }

  // With high-order corrections roundoff can accumulate.
  Approx custom_approx = Approx::custom().epsilon(5.e-12);
  if (not aligned_coordinates and Dim == 3) {
    // In the case of non aligned coordinates and 3-D, we are truly
    // testing a non-aligned grid which does not have a uniform
    // Jacobian. As such, the error is limited by the FD grid's
    // resolution of this non-uniformity. For the lower order tests,
    // we use very few grid points, resulting in a large error.
    custom_approx = Approx::custom().epsilon(2.e-2);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(get<Scalar0>(flux_divergence),
                               get<Scalar0>(expected_divergence),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Vector0<Dim>>(flux_divergence),
                               get<Vector0<Dim>>(expected_divergence),
                               custom_approx);

  CorrectionVars divergence_fractional_error(
      subcell_mesh.number_of_grid_points());
  get(get<Scalar0>(divergence_fractional_error)) =
      abs(get(get<Scalar0>(flux_divergence)) -
          get(get<Scalar0>(expected_divergence))) /
      get(get<Scalar0>(expected_divergence));
  for (size_t i = 0; i < Dim; ++i) {
    get<Vector0<Dim>>(divergence_fractional_error).get(i) =
        abs(get<Vector0<Dim>>(flux_divergence).get(i) -
            get<Vector0<Dim>>(expected_divergence).get(i)) /
        get(magnitude(get<Vector0<Dim>>(expected_divergence)));
  }

  // Test assertions
#ifdef SPECTRE_DEBUG
  if (correction_order != fd::DerivativeOrder::Two) {
    std::optional<std::array<CorrectionVars, Dim>>
        high_order_corrections_assert = make_array<Dim>(CorrectionVars{
            second_order_corrections[0].number_of_grid_points()});
    high_order_corrections_assert.value()[0].initialize(
        second_order_corrections[0].number_of_grid_points() * 2);
    CHECK_THROWS_WITH(
        ::fd::cartesian_high_order_flux_corrections(
            make_not_null(&high_order_corrections_assert), volume_vars,
            second_order_corrections, correction_order,
            reconstruction_ghost_data, subcell_mesh, number_of_ghost_points),
        Catch::Matchers::ContainsSubstring(
            "The high_order_corrections must all have size"));
  }
  if constexpr (Dim > 1) {
    auto second_order_corrections_copy = second_order_corrections;
    second_order_corrections_copy[0].initialize(
        second_order_corrections_copy[0].number_of_grid_points() * 2);
    CHECK_THROWS_WITH(
        ::fd::cartesian_high_order_flux_corrections(
            make_not_null(&high_order_corrections), volume_vars,
            second_order_corrections_copy, correction_order,
            reconstruction_ghost_data, subcell_mesh, number_of_ghost_points),
        Catch::Matchers::ContainsSubstring(
            "All second-order boundary corrections must be of the same size"));
  }
#endif  // SPECTRE_DEBUG

  return std::max(
      {max(get(get<Scalar0>(divergence_fractional_error))),
       max(get(magnitude(get<Vector0<Dim>>(divergence_fractional_error))))});
}

SPECTRE_TEST_CASE("Unit.FiniteDifference.CartesianHighOrderFluxCorrection",
                  "[Unit][NumericalAlgorithms]") {
  using DO = fd::DerivativeOrder;
  for (const fd::DerivativeOrder correction_order :
       {DO::Two, DO::Four, DO::Six, DO::Eight, DO::Ten, DO::OneHigherThanRecons,
        DO::OneHigherThanReconsButFiveToFour}) {
    CAPTURE(correction_order);
    test<1>(correction_order, true);
    test<2>(correction_order, true);
    test<3>(correction_order, true);
    test<2>(correction_order, false);
  }
  // Test that error improves as we increase the correction order. This part
  // could be improved because, currently, increasing the correction order
  // also increases the number of grid points in the test, as well as the
  // maximum order of the polynomial that the flux/divergence functions are
  // dependent on. There is still an improvement in fractional error with
  // correction order, but it's hard to necessarily isolate the effect of
  // the finite difference derivative order from these other effects as the test
  // currently stands.
  std::optional<double> previous_error{};
  for (const fd::DerivativeOrder correction_order :
       {DO::Two, DO::Four, DO::Six, DO::Eight, DO::Ten}) {
    CAPTURE(correction_order);
    const auto current_error = test<3>(correction_order, false);
    if (previous_error.has_value()) {
      CHECK(current_error < previous_error);
    }
    previous_error = current_error;
  }
}
}  // namespace
