// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.StressEnergy",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/Python/");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(&energy_density<DataVector>,
                                    "Test_StressEnergy", {"energy_density"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(&momentum_density<DataVector>,
                                    "Test_StressEnergy", {"momentum_density"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(&stress_trace<DataVector>,
                                    "Test_StressEnergy", {"stress_trace"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &stress_energy_tensor<DataVector>, "Test_StressEnergy",
      {"stress_energy_tensor"}, {{{0.0, 1.0}}}, used_for_size);
}

}  // namespace hydro
