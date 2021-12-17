// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Burgers/FiniteDifference/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.TimeDerivative",
                  "[Unit][Evolution]") {
  using evolved_vars_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, evolved_vars_tag>;

  DirectionMap<1, Neighbors<1>> neighbors{};
  for (size_t i = 0; i < 2; ++i) {
    neighbors[gsl::at(Direction<1>::all_directions(), i)] =
        Neighbors<1>{{ElementId<1>{i + 1, {}}}, {}};
  }
  const Element<1> element{ElementId<1>{0, {}}, neighbors};

  const size_t num_dg_pts = 5;
  const Mesh<1> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // Perform test with MC reconstruction & Rusanov riemann solver
  using ReconstructionForTest = fd::MonotisedCentral;
  using BoundaryCorrectionForTest = BoundaryCorrections::Rusanov;

  // Set the testing profile for U.
  // Here we use
  //   * U(x)   = x + 1
  const auto compute_test_solution = [](const auto& coords) {
    using tag = Tags::U;
    Variables<tmpl::list<tag>> vars{get<0>(coords).size(), 0.0};
    get(get<tag>(vars)) += coords.get(0) + 1.0;
    return vars;
  };
  auto logical_coords_subcell = logical_coordinates(subcell_mesh);
  const auto volume_vars_subcell =
      compute_test_solution(logical_coords_subcell);

  // set the ghost data from neighbor
  const ReconstructionForTest reconstructor{};
  evolution::dg::subcell::Tags::NeighborDataForReconstruction<1>::type
      neighbor_data = TestHelpers::Burgers::fd::compute_neighbor_data(
          subcell_mesh, logical_coords_subcell, element.neighbors(),
          reconstructor.ghost_zone_size(), compute_test_solution);

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<1>, evolution::dg::subcell::Tags::Mesh<1>,
      evolved_vars_tag, dt_variables_tag,
      evolution::dg::subcell::Tags::NeighborDataForReconstruction<1>,
      fd::Tags::Reconstructor, evolution::Tags::BoundaryCorrection<System>,
      domain::Tags::ElementMap<1, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<1, Frame::ElementLogical>,
      evolution::dg::Tags::MortarData<1>>>(
      element, subcell_mesh, volume_vars_subcell,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      neighbor_data,
      std::unique_ptr<fd::Reconstructor>{
          std::make_unique<ReconstructionForTest>()},
      std::unique_ptr<BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<BoundaryCorrectionForTest>()},
      ElementMap<1, Frame::Grid>{
          ElementId<1>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<1>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<1>{}),
      logical_coords_subcell,
      typename evolution::dg::Tags::MortarData<1>::type{});

  {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<1>{});
    InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
        cell_centered_logical_to_grid_inv_jacobian{};
    const auto cell_centered_logical_to_inertial_inv_jacobian =
        coordinate_map.inv_jacobian(logical_coords_subcell);
    for (size_t i = 0; i < cell_centered_logical_to_grid_inv_jacobian.size();
         ++i) {
      cell_centered_logical_to_grid_inv_jacobian[i] =
          cell_centered_logical_to_inertial_inv_jacobian[i];
    }
    subcell::TimeDerivative::apply(
        make_not_null(&box), cell_centered_logical_to_grid_inv_jacobian,
        determinant(cell_centered_logical_to_grid_inv_jacobian));
  }

  const auto& dt_vars = db::get<dt_variables_tag>(box);

  // Analytic time derivative of U for the testing profile
  //   * dt(U) = -(1+x)
  const auto compute_test_derivative = [](const auto& coords) {
    using tag = ::Tags::dt<Tags::U>;
    Variables<tmpl::list<tag>> dt_expected{get<0>(coords).size(), 0.0};
    get(get<tag>(dt_expected)) += -1.0 - coords.get(0);
    return dt_expected;
  };

  CHECK_ITERABLE_APPROX(get<::Tags::dt<Tags::U>>(dt_vars),
                        get<::Tags::dt<Tags::U>>(
                            compute_test_derivative(logical_coords_subcell)));
}
}  // namespace Burgers