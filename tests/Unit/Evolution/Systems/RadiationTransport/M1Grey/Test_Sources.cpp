// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Sources.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.Sources", "[Unit][M1Grey]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RadiationTransport/M1Grey"};

  pypp::check_with_random_values<1>(&RadiationTransport::M1Grey::ComputeSources<
                                        neutrinos::ElectronNeutrinos<0>>::apply,
                                    "Sources",
                                    {"source_tilde_e", "source_tilde_s"},
                                    {{{0.0, 1.0}}}, DataVector{5});
}
