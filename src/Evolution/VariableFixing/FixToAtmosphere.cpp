// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

#include "Utilities/ErrorHandling/CaptureForError.hpp"

namespace VariableFixing {

template <size_t Dim>
FixToAtmosphere<Dim>::FixToAtmosphere(
    const double density_of_atmosphere, const double density_cutoff,
    const double transition_density_cutoff, const double max_velocity_magnitude,
    const std::optional<double> magnetization_bound,
    const std::optional<double> plasma_beta_bound,
    const Options::Context& context)
    : density_of_atmosphere_(density_of_atmosphere),
      density_cutoff_(density_cutoff),
      transition_density_cutoff_(transition_density_cutoff),
      max_velocity_magnitude_(max_velocity_magnitude),
      magnetization_bound_(magnetization_bound),
      plasma_beta_bound_(plasma_beta_bound) {
  if (density_of_atmosphere_ > density_cutoff_) {
    PARSE_ERROR(context, "The cutoff density ("
                             << density_cutoff_
                             << ") must be greater than or equal to the "
                                "density value in the atmosphere ("
                             << density_of_atmosphere_ << ')');
  }
  if (transition_density_cutoff_ < density_of_atmosphere_ or
      transition_density_cutoff_ > 10.0 * density_of_atmosphere_) {
    PARSE_ERROR(context, "The transition density must be in ["
                             << density_of_atmosphere_ << ", "
                             << 10 * density_of_atmosphere_ << "], but is "
                             << transition_density_cutoff_);
  }
  if (transition_density_cutoff_ <= density_cutoff_) {
    PARSE_ERROR(context, "The transition density cutoff ("
                             << transition_density_cutoff_
                             << ") must be bigger than the density cutoff ("
                             << density_cutoff_ << ")");
  }
  if (magnetization_bound.has_value() != plasma_beta_bound.has_value()) {
    PARSE_ERROR(context,
                "Both input values should be assigned for "
                "magnetization bound and plasma beta bound "
                "or they should be both None.");
  }
  if (magnetization_bound.has_value()) {
    if (magnetization_bound < 0.0) {
      PARSE_ERROR(context,
                  "Upper bound of magnetization bound "
                  "must be positive, but is  "
                      << magnetization_bound.value());
    }
  }
  if (plasma_beta_bound.has_value()) {
    if (plasma_beta_bound < 0.0) {
      PARSE_ERROR(context,
                  "Upper bound of plsma beta bound "
                  "must be positive, but is  "
                      << plasma_beta_bound.value());
    }
  }
}

template <size_t Dim>
// NOLINTNEXTLINE(google-runtime-references)
void FixToAtmosphere<Dim>::pup(PUP::er& p) {
  p | density_of_atmosphere_;
  p | density_cutoff_;
  p | transition_density_cutoff_;
  p | max_velocity_magnitude_;
  p | magnetization_bound_;
  p | plasma_beta_bound_;
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void FixToAtmosphere<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const {
  for (size_t i = 0; i < rest_mass_density->get().size(); i++) {
    if (UNLIKELY(rest_mass_density->get()[i] < density_cutoff_)) {
      set_density_to_atmosphere(rest_mass_density, specific_internal_energy,
                                temperature, pressure, electron_fraction,
                                equation_of_state, i);
      for (size_t d = 0; d < Dim; ++d) {
        spatial_velocity->get(d)[i] = 0.0;
      }
      get(*lorentz_factor)[i] = 1.0;
    } else if (UNLIKELY(rest_mass_density->get()[i] <
                        transition_density_cutoff_)) {
      set_to_magnetic_free_transition(spatial_velocity, lorentz_factor,
                                      *rest_mass_density, spatial_metric, i);
    }

    // For 2D & 3D EoS, we also need to limit the temperature / energy
    if constexpr (ThermodynamicDim > 1) {
      bool changed_temperature = false;
      if (const double min_temperature =
              equation_of_state.temperature_lower_bound();
          get(*temperature)[i] < min_temperature) {
          get(*temperature)[i] = min_temperature;
        changed_temperature = true;
      }

      // We probably need a better maximum temperature as well, but this is not
      // as well defined. To be discussed once implementation needs improvement.
      if (const double max_temperature =
              equation_of_state.temperature_upper_bound();
          get(*temperature)[i] > max_temperature) {
        get(*temperature)[i] = max_temperature;
        changed_temperature = true;
      }

      if (changed_temperature) {
        if constexpr (ThermodynamicDim == 2) {
          specific_internal_energy->get()[i] =
              get(equation_of_state
                      .specific_internal_energy_from_density_and_temperature(
                          Scalar<double>{rest_mass_density->get()[i]},
                          Scalar<double>{get(*temperature)[i]}));
          pressure->get()[i] =
              get(equation_of_state.pressure_from_density_and_energy(
                  Scalar<double>{rest_mass_density->get()[i]},
                  Scalar<double>{specific_internal_energy->get()[i]}));
        } else {
          specific_internal_energy->get()[i] =
              get(equation_of_state
                      .specific_internal_energy_from_density_and_temperature(
                          Scalar<double>{rest_mass_density->get()[i]},
                          Scalar<double>{get(*temperature)[i]},
                          Scalar<double>{get(electron_fraction)[i]}));
          pressure->get()[i] =
              get(equation_of_state.pressure_from_density_and_temperature(
                  Scalar<double>{rest_mass_density->get()[i]},
                  Scalar<double>{temperature->get()[i]},
                  Scalar<double>{get(electron_fraction)[i]}));
        }
      }
    }
    if (magnetization_bound_.has_value() and plasma_beta_bound_.has_value()) {
      const double sigma_bound = magnetization_bound_.value();
      const double beta_bound = plasma_beta_bound_.value();
      double magnetic_field_squared = 0.0;
      double magnetic_field_dot_v = 0.0;

      for (size_t j = 0; j < Dim; ++j) {
        for (size_t k = 0; k < Dim; ++k) {
          magnetic_field_squared += magnetic_field.get(j)[i] *
                                    magnetic_field.get(k)[i] *
                                    spatial_metric.get(j, k)[i];

          magnetic_field_dot_v += magnetic_field.get(j)[i] *
                                  spatial_velocity->get(k)[i] *
                                  spatial_metric.get(j, k)[i];
        }
      }

      double comoving_magnetic_field_squared =
          (magnetic_field_squared / (square(get(*lorentz_factor)[i]))) +
          square(magnetic_field_dot_v);
      if (get(*rest_mass_density)[i] <
              comoving_magnetic_field_squared / sigma_bound or
          get(*pressure)[i] <
              comoving_magnetic_field_squared / (2.0 * beta_bound)) {
        high_magnetiziation_treatment(
            rest_mass_density, specific_internal_energy, temperature, pressure,
            spatial_velocity, lorentz_factor, electron_fraction, magnetic_field,
            spatial_metric, comoving_magnetic_field_squared,
            magnetic_field_squared, magnetic_field_dot_v, equation_of_state, i);
      }
    }
  }
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void FixToAtmosphere<Dim>::set_density_to_atmosphere(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const Scalar<DataVector>& electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const size_t grid_index) const {
  const Scalar<double> atmosphere_density{density_of_atmosphere_};
  rest_mass_density->get()[grid_index] = get(atmosphere_density);
  get(*temperature)[grid_index] = equation_of_state.temperature_lower_bound();

  if constexpr (ThermodynamicDim == 1) {
    pressure->get()[grid_index] =
        get(equation_of_state.pressure_from_density(atmosphere_density));
    specific_internal_energy->get()[grid_index] =
        get(equation_of_state.specific_internal_energy_from_density(
            atmosphere_density));
  } else {
    const Scalar<double> atmosphere_temperature{get(*temperature)[grid_index]};
    if constexpr (ThermodynamicDim == 2) {
      specific_internal_energy->get()[grid_index] =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      atmosphere_density, atmosphere_temperature));
      pressure->get()[grid_index] =
          get(equation_of_state.pressure_from_density_and_energy(
              atmosphere_density,
              Scalar<double>{specific_internal_energy->get()[grid_index]}));
    } else {
      specific_internal_energy->get()[grid_index] =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      Scalar<double>{get(*rest_mass_density)[grid_index]},
                      Scalar<double>{get(*temperature)[grid_index]},
                      Scalar<double>{get(electron_fraction)[grid_index]}));
      pressure->get()[grid_index] =
          get(equation_of_state.pressure_from_density_and_temperature(
              Scalar<double>{get(*rest_mass_density)[grid_index]},
              Scalar<double>{get(*temperature)[grid_index]},
              Scalar<double>{get(electron_fraction)[grid_index]}));
    }
  }
}

template <size_t Dim>
void FixToAtmosphere<Dim>::set_to_magnetic_free_transition(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const Scalar<DataVector>& rest_mass_density,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const size_t grid_index) const {
  double magnitude_of_velocity = 0.0;
  for (size_t j = 0; j < Dim; ++j) {
    magnitude_of_velocity += spatial_velocity->get(j)[grid_index] *
                             spatial_velocity->get(j)[grid_index] *
                             spatial_metric.get(j, j)[grid_index];
    for (size_t k = j + 1; k < Dim; ++k) {
      magnitude_of_velocity += 2.0 * spatial_velocity->get(j)[grid_index] *
                               spatial_velocity->get(k)[grid_index] *
                               spatial_metric.get(j, k)[grid_index];
    }
  }
  magnitude_of_velocity = sqrt(magnitude_of_velocity);
  const double scale_factor =
      (get(rest_mass_density)[grid_index] - density_cutoff_) /
      (transition_density_cutoff_ - density_cutoff_);
  if (const double max_mag_of_velocity = scale_factor * max_velocity_magnitude_;
      magnitude_of_velocity > max_mag_of_velocity) {
    for (size_t j = 0; j < Dim; ++j) {
      spatial_velocity->get(j)[grid_index] *=
          max_mag_of_velocity / magnitude_of_velocity;
    }
    get(*lorentz_factor)[grid_index] =
        1.0 / sqrt(1.0 - max_mag_of_velocity * max_mag_of_velocity);
  }
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void FixToAtmosphere<Dim>::high_magnetiziation_treatment(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const double comoving_magnetic_field_squared,
    const double magnetic_field_squared, const double magnetic_field_dot_v,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    size_t grid_index) const {
  using std::max;
  using std::min;
  const double sigma_bound = magnetization_bound_.value();
  const double beta_bound = plasma_beta_bound_.value();
  // old rest mass density * specific enthalpy before we
  // apply flooring on rest mass density, pressure, and specific internal
  // energy based on magnetic field strength.
  const double old_wg = get(*rest_mass_density)[grid_index] +
                        get(*rest_mass_density)[grid_index] *
                            get(*specific_internal_energy)[grid_index] +
                        get(*pressure)[grid_index];
  // Increment rest_mass_density and temperature until magnetization and beta
  // are bounded above by some prescribed values. this is so that we are not in
  // extremly high magnetized regions in our simulation which could lead to
  // failure with PrimitiveRecovery.

  get(*rest_mass_density)[grid_index] =
      max(get(*rest_mass_density)[grid_index],
          comoving_magnetic_field_squared / sigma_bound);
  get(*pressure)[grid_index] =
      max(get(*pressure)[grid_index],
          comoving_magnetic_field_squared / (2.0 * beta_bound));

  const Scalar<double> updated_density{get(*rest_mass_density)[grid_index]};
  // Since all the EoS functions take either temperature or
  // specific_internal_energy recast the incrementation in pressure into
  // incrementation in temperature.
  get(*temperature)[grid_index] =
      get(*pressure)[grid_index] / get(*rest_mass_density)[grid_index];
  const Scalar<double> updated_temperature{get(*temperature)[grid_index]};

  // re-adjust the other thermodynamics variable in accordance
  // with changes in rest_mass_density and temperature (pressure)
  const bool eos_is_barotropic = equation_of_state.is_barotropic();

  if constexpr (ThermodynamicDim == 1) {
    pressure->get()[grid_index] = get(equation_of_state.pressure_from_density(
        updated_density));
    specific_internal_energy->get()[grid_index] =
        get(equation_of_state.specific_internal_energy_from_density(
            updated_density));
  } else {
    if constexpr (ThermodynamicDim == 2) {
      specific_internal_energy->get()[grid_index] =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      updated_density,
                      updated_temperature));
    } else {
      if (eos_is_barotropic) {
        // for now, for barotropic runs, just apply the sigma bounds
        // and recompute other quantities from updates rest_mass_density
        get(*pressure)[grid_index] =
            get(equation_of_state.pressure_from_density_and_temperature(
                updated_density, updated_temperature,
                Scalar<double>{get(electron_fraction)[grid_index]}));
        get(*temperature)[grid_index] =
            get(*pressure)[grid_index] / get(*rest_mass_density)[grid_index];
        specific_internal_energy->get()[grid_index] =
            get(equation_of_state
                    .specific_internal_energy_from_density_and_temperature(
                        updated_density, updated_temperature,
                        Scalar<double>{get(electron_fraction)[grid_index]}));
      } else {
        specific_internal_energy->get()[grid_index] =
            get(equation_of_state
                    .specific_internal_energy_from_density_and_temperature(
                        updated_density, updated_temperature,
                        Scalar<double>{get(electron_fraction)[grid_index]}));
      }
    }
  }

  // With changes in rest mass density, pressure, and specific internal energy,
  // enthalpy is changed. In order to preserve fluid momentum, parallel to
  // to magnetic field, we need to decrease the parallel component of spatial
  // velocity. To do this, we follow the so called "drift-frame flooring".

  double velocity_squared = 0.0;
  for (size_t j = 0; j < Dim; ++j) {
    velocity_squared += spatial_velocity->get(j)[grid_index] *
                        spatial_velocity->get(j)[grid_index] *
                        spatial_metric.get(j, j)[grid_index];
    for (size_t k = j + 1; k < Dim; ++k) {
      velocity_squared += 2.0 * spatial_velocity->get(j)[grid_index] *
                          spatial_velocity->get(k)[grid_index] *
                          spatial_metric.get(j, k)[grid_index];
    }
  }
  // compute new rho * h
  const double new_wg = get(*rest_mass_density)[grid_index] +
                        get(*rest_mass_density)[grid_index] *
                            get(*specific_internal_energy)[grid_index] +
                        get(*pressure)[grid_index];
  CAPTURE_FOR_ERROR(velocity_squared);
  // We only need to do this if non-zero velocity and if rest mass density
  // times specific enthalpy has been increased.
  // The latter should be always true the way that we applied flooring but
  // we do this for sanity check.

  if (velocity_squared > 1.e-15 && new_wg > old_wg) {
    const double magnetic_field_magnitude = sqrt(magnetic_field_squared);
    const double v_parallel = magnetic_field_dot_v / magnetic_field_magnitude;
    const double lorentz_factor_v = get(*lorentz_factor)[grid_index];
    const double lorentz_factor_perp =
        1.0 / sqrt(square(v_parallel) + (1.0 / (square(lorentz_factor_v))));

    CAPTURE_FOR_ERROR(magnetic_field_magnitude);
    CAPTURE_FOR_ERROR(v_parallel);
    CAPTURE_FOR_ERROR(lorentz_factor_v);
    CAPTURE_FOR_ERROR(lorentz_factor_perp);
    // if (lorentz_factor_perp > lorentz_factor_v) {
    //   ERROR(
    //       "lorentz factor of perpendicular-only velocity is bigger "
    //       "than that of total velocity!");
    // }
    // rest_mass_density time specific_enthalpy
    const double rho_h = get(*rest_mass_density)[grid_index] +
                         get(*pressure)[grid_index] +
                         get(*specific_internal_energy)[grid_index];

    const double x =
        (2 * v_parallel * square(lorentz_factor_v) / lorentz_factor_perp) *
        (old_wg / new_wg);

    CAPTURE_FOR_ERROR(magnetic_field_dot_v);
    CAPTURE_FOR_ERROR(rho_h);
    CAPTURE_FOR_ERROR(x);

    const double new_v_parallel =
        (x / lorentz_factor_perp) / (1.0 + sqrt(1.0 + square(x)));
    CAPTURE_FOR_ERROR(new_v_parallel);

    if (abs(new_v_parallel) > abs(v_parallel)) {
      ERROR(
          "the parallel component of the velocity is increased "
          "instead of being reduced!!");
    }

    // readjust the spatial velocity
    for (size_t j = 0; j < Dim; ++j) {
      spatial_velocity->get(j)[grid_index] +=
          (new_v_parallel - v_parallel) * magnetic_field.get(j)[grid_index] /
          magnetic_field_magnitude;
    }

    double new_velocity_squared = 0.0;
    for (size_t j = 0; j < Dim; ++j) {
      new_velocity_squared += spatial_velocity->get(j)[grid_index] *
                              spatial_velocity->get(j)[grid_index] *
                              spatial_metric.get(j, j)[grid_index];
      for (size_t k = j + 1; k < Dim; ++k) {
        new_velocity_squared += 2.0 * spatial_velocity->get(j)[grid_index] *
                                spatial_velocity->get(k)[grid_index] *
                                spatial_metric.get(j, k)[grid_index];
      }
    }
    CAPTURE_FOR_ERROR(new_velocity_squared);
    // readjust the loretnz_factor
    get(*lorentz_factor)[grid_index] = 1.0 / sqrt(1.0 - new_velocity_squared);
    const double new_lorentz_factor = get(*lorentz_factor)[grid_index];
    CAPTURE_FOR_ERROR(new_lorentz_factor);
    // if (new_lorentz_factor > lorentz_factor_v) {
    //   ERROR(
    //       "the lorentz factor new velocityis increased "
    //       "instead of being reduced!!");
    // }
  }
}

template <size_t Dim>
bool operator==(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return lhs.density_of_atmosphere_ == rhs.density_of_atmosphere_ and
         lhs.density_cutoff_ == rhs.density_cutoff_ and
         lhs.transition_density_cutoff_ == rhs.transition_density_cutoff_ and
         lhs.max_velocity_magnitude_ == rhs.max_velocity_magnitude_ and
         lhs.magnetization_bound_ == rhs.magnetization_bound_ and
         lhs.plasma_beta_bound_ == rhs.plasma_beta_bound_;
}

template <size_t Dim>
bool operator!=(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                     \
  template class FixToAtmosphere<DIM(data)>;                       \
  template bool operator==(const FixToAtmosphere<DIM(data)>& lhs,  \
                           const FixToAtmosphere<DIM(data)>& rhs); \
  template bool operator!=(const FixToAtmosphere<DIM(data)>& lhs,  \
                           const FixToAtmosphere<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define INSTANTIATION(r, data)                                                \
  template void FixToAtmosphere<DIM(data)>::operator()(                       \
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,             \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,      \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>   \
          spatial_velocity,                                                   \
      const gsl::not_null<Scalar<DataVector>*> lorentz_factor,                \
      const gsl::not_null<Scalar<DataVector>*> pressure,                      \
      const gsl::not_null<Scalar<DataVector>*> temperature,                   \
      const Scalar<DataVector>& electron_fraction,                            \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& magnetic_field,  \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric, \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&        \
          equation_of_state) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2, 3))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
