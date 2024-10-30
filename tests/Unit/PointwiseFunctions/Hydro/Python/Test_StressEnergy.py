# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def inverse_spacetime_metric(lapse, shift, inverse_spatial_metric):
    result = np.zeros([4, 4])
    result[0, 0] = -1.0 / (lapse * lapse)
    for i in range(3):
        result[0, i + 1] = -shift[i] * result[0, 0]
        result[i + 1, 0] = result[0, i + 1]
    for i in range(3):
        for j in range(3):
            result[i + 1, j + 1] = inverse_spatial_metric[i, j] + (
                shift[i] * shift[j] * result[0, 0]
            )
    return result


def four_velocity(spatial_velocity, shift, lorentz_factor, lapse):
    result = np.zeros(4)
    result[0] = lorentz_factor / lapse
    result[1:] = lorentz_factor * (spatial_velocity - shift / lapse)
    return result


def comoving_magnetic_field(
    spatial_velocity,
    magnetic_field,
    magnetic_field_dot_spatial_velocity,
    lorentz_factor,
    shift,
    lapse,
):
    result = np.zeros(4)
    result[0] = magnetic_field_dot_spatial_velocity / lapse
    result[1:] = (
        magnetic_field
        + lapse
        * result[0]
        * (lorentz_factor * (spatial_velocity - shift / lapse))
    ) / lorentz_factor
    return result


def energy_density(
    rest_mass_density,
    specific_enthalpy,
    pressure,
    lorentz_factor,
    magnetic_field_dot_spatial_velocity,
    comoving_magnetic_field_squared,
):
    return (
        rest_mass_density * specific_enthalpy * lorentz_factor**2
        - pressure
        + comoving_magnetic_field_squared * (lorentz_factor**2 - 0.5)
        - (lorentz_factor * magnetic_field_dot_spatial_velocity) ** 2
    )


def momentum_density(
    rest_mass_density,
    specific_enthalpy,
    spatial_velocity,
    lorentz_factor,
    magnetic_field,
    magnetic_field_dot_spatial_velocity,
    comoving_magnetic_field_squared,
):
    return (
        rest_mass_density
        * specific_enthalpy
        * lorentz_factor**2
        * spatial_velocity
        + comoving_magnetic_field_squared
        * lorentz_factor**2
        * spatial_velocity
        - magnetic_field_dot_spatial_velocity * magnetic_field
        - magnetic_field_dot_spatial_velocity**2
        * lorentz_factor**2
        * spatial_velocity
    )


def stress_trace(
    rest_mass_density,
    specific_enthalpy,
    pressure,
    spatial_velocity_squared,
    lorentz_factor,
    magnetic_field_dot_spatial_velocity,
    comoving_magnetic_field_squared,
):
    return (
        3.0 * pressure
        + rest_mass_density * specific_enthalpy * (lorentz_factor**2 - 1.0)
        + comoving_magnetic_field_squared
        * (lorentz_factor**2 * spatial_velocity_squared + 0.5)
        - magnetic_field_dot_spatial_velocity**2
        * (lorentz_factor**2 * spatial_velocity_squared + 1.0)
    )


def stress_energy_tensor(
    rest_mass_density,
    specific_internal_energy,
    pressure,
    lorentz_factor,
    lapse,
    comoving_magnetic_field_magnitude,
    spatial_velocity,
    shift,
    magnetic_field,
    spatial_metric,
    inverse_spatial_metric,
):
    inverse_spacetime_metric_v = inverse_spacetime_metric(
        lapse, shift, inverse_spatial_metric
    )

    magnetic_field_dot_spatial_velocity = np.dot(
        np.dot(spatial_metric, magnetic_field), spatial_velocity
    )
    comoving_magnetic_field_v = comoving_magnetic_field(
        spatial_velocity,
        magnetic_field,
        magnetic_field_dot_spatial_velocity,
        lorentz_factor,
        shift,
        lapse,
    )
    four_velocity_v = four_velocity(
        spatial_velocity, shift, lorentz_factor, lapse
    )
    rho_h_star = (
        (rest_mass_density + rest_mass_density * specific_internal_energy)
        + pressure
        + comoving_magnetic_field_magnitude**2
    )
    p_star = pressure + 0.5 * (comoving_magnetic_field_magnitude**2)

    result = np.zeros([4, 4])
    result[0, 0] = (
        (rho_h_star * (four_velocity_v[0] ** 2))
        + (p_star * inverse_spacetime_metric_v[0, 0])
        - (comoving_magnetic_field_v[0] ** 2)
    )
    result[1, 0] = (
        (rho_h_star * four_velocity_v[0] * four_velocity_v[1])
        + (p_star * inverse_spacetime_metric_v[0, 1])
        - (comoving_magnetic_field_v[0] * comoving_magnetic_field_v[1])
    )
    result[1, 1] = (
        (rho_h_star * four_velocity_v[1] ** 2)
        + (p_star * inverse_spacetime_metric_v[1, 1])
        - (comoving_magnetic_field_v[1] ** 2)
    )
    result[0, 1] = result[1, 0]
    result[2, 0] = (
        (rho_h_star * four_velocity_v[2] * four_velocity_v[0])
        + (p_star * inverse_spacetime_metric_v[2, 0])
        - (comoving_magnetic_field_v[2] * comoving_magnetic_field_v[0])
    )
    result[0, 2] = result[2, 0]
    result[2, 1] = (
        (rho_h_star * four_velocity_v[2] * four_velocity_v[1])
        + (p_star * inverse_spacetime_metric_v[2, 1])
        - (comoving_magnetic_field_v[2] * comoving_magnetic_field_v[1])
    )
    result[1, 2] = result[2, 1]
    result[2, 2] = (
        (rho_h_star * four_velocity_v[2] ** 2)
        + (p_star * inverse_spacetime_metric_v[2, 2])
        - (comoving_magnetic_field_v[2] ** 2)
    )
    result[3, 0] = (
        (rho_h_star * four_velocity_v[3] * four_velocity_v[0])
        + (p_star * inverse_spacetime_metric_v[3, 0])
        - (comoving_magnetic_field_v[3] * comoving_magnetic_field_v[0])
    )
    result[0, 3] = result[3, 0]
    result[3, 1] = (
        (rho_h_star * four_velocity_v[3] * four_velocity_v[1])
        + (p_star * inverse_spacetime_metric_v[3, 1])
        - (comoving_magnetic_field_v[3] * comoving_magnetic_field_v[1])
    )
    result[1, 3] = result[3, 1]
    result[3, 2] = (
        (rho_h_star * four_velocity_v[3] * four_velocity_v[2])
        + (p_star * inverse_spacetime_metric_v[3, 2])
        - (comoving_magnetic_field_v[3] * comoving_magnetic_field_v[2])
    )
    result[2, 3] = result[3, 2]
    result[3, 3] = (
        (rho_h_star * four_velocity_v[3] ** 2)
        + (p_star * inverse_spacetime_metric_v[3, 3])
        - (comoving_magnetic_field_v[3] ** 2)
    )
    return result
