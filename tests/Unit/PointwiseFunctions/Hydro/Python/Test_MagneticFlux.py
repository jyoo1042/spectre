# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import sys
import unittest

import numpy as np
import numpy.testing as npt

import spectre.DataStructures.Tensor.Frame as fr
import spectre.PointwiseFunctions.Hydro as hydro
from spectre import Informer
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TestFunctions import *


class TestMagneticFlux(unittest.TestCase):
    def test_magnetic_flux(self):
        spatial_velocity = tnsr.I[DataVector, 3, fr.Grid](
            num_points=5, fill=-1.0
        )
        sqrt_det_spatial_metric = Scalar[DataVector](num_points=5, fill=1.8)

        bindings = hydro.magnetic_flux(
            magnetic_field,
            sqrt_det_spatial_metric,
        )

        alternative = magnetic_flux(
            np.array(magnetic_field),
            np.array(sqrt_det_spatial_metric)[0],
        )

        assert type(bindings) == tnsr.I[DataVector, 3, fr.Grid]
        np.testing.assert_allclose(bindings, alternative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
