import astropy.constants as const
import pickle
import astropy.units as u
from scipy.interpolate import interp1d
from loop.magnetic_field import MagneticField
from maths.interpolater import NumbaInterpolator
import os
import warnings
import numpy as np
import pydrad.parse.parse
from scipy.interpolate import PchipInterpolator

class LoopProperties:
    def __init__(self, input_file, *, hydrogen_fraction=None, hydrogen_metadata=None):
        if hydrogen_fraction is None:
            hydrogen_fraction, inferred = self.hydrogen_from_profile(input_file)
            hydrogen_metadata = hydrogen_metadata or inferred
        elif hydrogen_metadata is None:
            hydrogen_metadata = {'source': 'explicit hydrogen_fraction override'}
        self.hydrogen_metadata = hydrogen_metadata
        self.s_array, self.ne_interp, self.nh_interp, self.rho_interp, self.T_interp, self.h_ionisation_interp, self.pe_interp, self.sound_speed_interp, self.velocity_interp = self.get_loop_properties(input_file, hydrogen_fraction)
        self.magnetic_field = MagneticField(self)

    @classmethod
    def from_hydrad(cls, profile, root, profile_index, hydrogen_history=None):
        """Use saved H II, a supplied cache, or the fully ionised closure when ne = nH."""
        if hydrogen_history is None:
            return cls(profile)
        from hydrogen import HydrogenHistory
        history = (HydrogenHistory.load(hydrogen_history)
                   if isinstance(hydrogen_history, (str, os.PathLike)) else hydrogen_history)
        if not isinstance(history, HydrogenHistory):
            raise TypeError('hydrogen_history must be a HydrogenHistory or cache path')
        fraction = history.fraction(profile_index, profile.coordinate.to_value(u.cm),
                                    root=root, time=profile.time.to_value(u.s))
        return cls(profile, hydrogen_fraction=fraction,
                   hydrogen_metadata={'source': 'reconstruction', **history.metadata})

    @staticmethod
    def calculate_density(number_density):
        return number_density * const.m_p.cgs

    @staticmethod
    def estimate_mass_density(ne, ni, helium_abundance=0.1):
        """
        Estimate mass density ρ from electron and ion number densities using a fixed He/H number ratio.
        Assumes hydrogen + helium plasma; electrons contribute negligible mass.
        helium_abundance: n_He / n_H (≈ 0.1 for solar).
        """
        A = helium_abundance
        # rho_from_ne = const.m_p.cgs * ne * (1 + 4*A) / (1 + 2*A)
        rho_from_ni = const.m_p.cgs * ni * (1 + 4*A) / (1 + A)
        return rho_from_ni

    def B(self, s):
        return self.magnetic_field(s)

    @staticmethod
    def _profile_attr(profile, new_name, old_name):
        # pydrad >= 0.2.dev19 renamed ion_* to hydrogen_* and level_population_hydrogen_6 to hydrogen_II_fraction
        try:
            return getattr(profile, new_name)
        except AttributeError:
            return getattr(profile, old_name)

    @staticmethod
    def hydrogen_from_profile(profile):
        """Saved H II if present. Otherwise fully ionised hydrogen when ne = nH.

        The second case is the quasi-neutral closure of a fully ionised HYDRAD run
        (Bradshaw & Cargill 2013). It is not an LTE chromosphere.
        """
        try:
            fraction = np.asarray(LoopProperties._profile_attr(
                profile, 'hydrogen_II_fraction', 'level_population_hydrogen_6').value, dtype=float)
        except (ValueError, AttributeError, TypeError):
            fraction = None
        if fraction is not None and fraction.dtype.kind in 'fc':
            return fraction, {'source': 'HYDRAD level_population_hydrogen_6'}
        ne = np.asarray(profile.electron_density.to_value(u.cm**-3), dtype=float)
        nh = np.asarray(LoopProperties._profile_attr(
            profile, 'hydrogen_density', 'ion_density').to_value(u.cm**-3), dtype=float)
        if ne.shape != profile.coordinate.shape or not np.allclose(ne, nh, rtol=1e-5, atol=0.0):
            raise ValueError(
                'No hydrogen populations available, and ne differs from nH, so the '
                'quasi-neutral fully ionised closure does not apply. Supply hydrogen_history.'
            )
        warnings.warn(
            'No hydrogen populations in this HYDRAD profile. Using fully ionised '
            'hydrogen because ne = nH (Bradshaw & Cargill 2013).',
            UserWarning,
            stacklevel=2,
        )
        return np.ones(ne.shape, dtype=float), {
            'source': 'quasi-neutral',
            'assumption': 'fully ionised hydrogen, ne = nH',
            'reference': 'Bradshaw & Cargill 2013, ApJ, 770, 12',
        }

    @staticmethod
    def get_loop_properties(input_file, hydrogen_fraction):
        if isinstance(input_file, pydrad.parse.parse.Profile):
            print('Loading from pydrad profile')
            h_ionisation = np.asarray(hydrogen_fraction, dtype=float)
            if (h_ionisation.shape != input_file.coordinate.shape
                    or not np.all(np.isfinite(h_ionisation))
                    or np.any((h_ionisation < 0) | (h_ionisation > 1))):
                raise ValueError('Hydrogen fraction must match the profile grid and lie in [0, 1]')
            ne_array = input_file.electron_density.to(u.cm**-3)
            get = LoopProperties._profile_attr
            nh_array = get(input_file, 'hydrogen_density', 'ion_density').to(u.cm**-3)
            T_array = get(input_file, 'hydrogen_temperature', 'ion_temperature').to(u.MK)
            s_array = input_file.coordinate.to(u.cm)
            pe_array = get(input_file, 'hydrogen_pressure', 'ion_pressure').to(u.dyne/u.cm**2)
            sound_speed_array = input_file.sound_speed.to(u.cm/u.s)
            velocity_array = input_file.velocity.to(u.cm/u.s)
            # import pdb; pdb.set_trace()
            
        else:
            raise TypeError('LoopProperties requires a pydrad Profile')


        # Extract necessary variables

        # Calculate derived quantities
        rho_array = LoopProperties.estimate_mass_density(ne_array, nh_array)
        s_array_cgs = s_array.to(u.cm).value
        ne_array_cgs = ne_array.to(u.cm**-3).value
        nh_array_cgs = nh_array.to(u.cm**-3).value
        rho_array_cgs = rho_array.to(u.g/u.cm**3).value
        T_array_cgs = T_array.to(u.K).value
        pe_array_cgs = pe_array.to(u.erg/u.cm**3).value
        sound_speed_array_cgs = sound_speed_array.to(u.cm/u.s).value
        velocity_array_cgs = velocity_array.to(u.cm/u.s).value

        # Create interpolation functions
        # ne_interp = interp1d(s_array_cgs, ne_array_cgs, kind='cubic', fill_value='extrapolate')
        # nh_interp = interp1d(s_array_cgs, nh_array_cgs, kind='cubic', fill_value='extrapolate')
        # rho_interp = interp1d(s_array_cgs, rho_array_cgs, kind='cubic', fill_value='extrapolate')
        # T_interp = interp1d(s_array_cgs, T_array_cgs, kind='cubic', fill_value='extrapolate')
        # h_ionisation_interp = interp1d(s_array_cgs, h_ionisation, kind='cubic', fill_value='extrapolate')
        # pe_interp = interp1d(s_array_cgs, pe_array_cgs, kind='cubic', fill_value='extrapolate')
        # sound_speed_interp = interp1d(s_array_cgs, sound_speed_array_cgs, kind='cubic', fill_value='extrapolate')
        # velocity_interp = interp1d(s_array_cgs, velocity_array_cgs, kind='cubic', fill_value='extrapolate')
        # Use PchipInterpolator directly (not as a kind argument)
        ne_interp = PchipInterpolator(s_array_cgs, ne_array_cgs, extrapolate=True)
        nh_interp = PchipInterpolator(s_array_cgs, nh_array_cgs, extrapolate=True)
        rho_interp = PchipInterpolator(s_array_cgs, rho_array_cgs, extrapolate=True)
        T_interp = PchipInterpolator(s_array_cgs, T_array_cgs, extrapolate=True)
        h_ionisation_interp = PchipInterpolator(s_array_cgs, h_ionisation, extrapolate=True)
        pe_interp = PchipInterpolator(s_array_cgs, pe_array_cgs, extrapolate=True)
        sound_speed_interp = PchipInterpolator(s_array_cgs, sound_speed_array_cgs, extrapolate=True)
        velocity_interp = PchipInterpolator(s_array_cgs, velocity_array_cgs, extrapolate=True)
        return s_array, ne_interp, nh_interp, rho_interp, T_interp, h_ionisation_interp, pe_interp, sound_speed_interp, velocity_interp
    
    @property
    def loop_length(self):
        """Get the total length of the loop."""
        return self.s_array[-1] - self.s_array[0]

    # Define useful plasma properties functions
    def rho(self, s):
        return self.rho_interp(s) * u.g/u.cm**3

    def T(self, s):
        return self.T_interp(s) * u.K

    def ne(self, s):
        return self.ne_interp(s) * u.cm**-3
    
    def nh(self, s):
        return self.nh_interp(s) * u.cm**-3
    
    def pe(self, s):
        return self.pe_interp(s) * u.dyne/u.cm**2
    
    def h_ionisation(self, s):
        s_cm = s.to_value(u.cm) if isinstance(s, u.Quantity) else np.asarray(s)
        # Hold the end-cell values outside the centers; extrapolating fractions
        # can otherwise produce negative neutral/proton collision rates.
        return self.h_ionisation_interp(np.clip(s_cm, self.s_array[0].to_value(u.cm), self.s_array[-1].to_value(u.cm)))

    def P(self, s):
        n_e_s = self.ne(s)
        T_s = self.T(s)
        return 2 * n_e_s * const.k_B * T_s

    def sound_speed(self, s):
        return self.sound_speed_interp(s) * u.cm/u.s
    
    def velocity(self, s):
        return self.velocity_interp(s) * u.cm/u.s
