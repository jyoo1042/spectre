# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def conversion_factor1(radius, coordinates):
    result = np.zeros([3, 3])
    one_over_radius = 1.0 / radius
    result[0, 0] = -one_over_radius * coordinates[0] * coordinates[1]
    result[0, 1] = one_over_radius * (coordinates[0] ** 2)
    result[1, 0] = -one_over_radius * (coordinate[1] ** 2)
    result[1, 1] = one_over_radius * coordinates[0] * coordinates[1]
    result[2, 0] = -one_over_radius * coordinates[1] * coordinates[2]
    result[2, 1] = one_over_radius * coordinates[0] * coordinates[2]
    return result


def converseion_factor2(radius, coordinates):
    result = np.zeros([3, 1])
    one_over_radius = 1.0 / radius
    result[0] = one_over_radius * coordinates[0]
    result[1] = one_over_radius * coordinates[1]
    result[2] = one_over_radius * coordinates[2]
    return result


def edot(stress_energy_tensor, lapse, shfit, spatial_metric, coordinates):
    return result
