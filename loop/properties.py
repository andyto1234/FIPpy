import astropy.constants as const
import pickle
import astropy.units as u
from scipy.interpolate import interp1d
from loop.magnetic_field import MagneticField
from maths.interpolater import NumbaInterpolator
import os
import pydrad.parse.parse
from scipy.interpolate import PchipInterpolator

class LoopProperties:
    def __init__(self, input_file):
        self.s_array, self.ne_interp, self.nh_interp, self.rho_interp, self.T_interp, self.h_ionisation_interp, self.pe_interp, self.sound_speed_interp, self.velocity_interp = self.get_loop_properties(input_file)
        self.magnetic_field = MagneticField(self)

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
    def get_loop_properties(input_file):
        if type(input_file) == pydrad.parse.parse.Profile:
            print('Loading from pydrad profile')
            h_ionisation = 1-input_file.level_population_hydrogen_1.value
            ne_array = input_file.electron_density.to(u.cm**-3)
            nh_array = input_file.ion_density.to(u.cm**-3)
            T_array = input_file.ion_temperature.to(u.MK)
            s_array = input_file.coordinate.to(u.cm)
            pe_array = input_file.ion_pressure.to(u.dyne/u.cm**2)
            sound_speed_array = input_file.sound_speed.to(u.cm/u.s)
            velocity_array = input_file.velocity.to(u.cm/u.s)
            # import pdb; pdb.set_trace()
            
        else:
            print('file type not compatible ')


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
        return self.h_ionisation_interp(s)

    def P(self, s):
        n_e_s = self.ne(s)
        T_s = self.T(s)
        return 2 * n_e_s * const.k_B * T_s

    def sound_speed(self, s):
        return self.sound_speed_interp(s) * u.cm/u.s
    
    def velocity(self, s):
        return self.velocity_interp(s) * u.cm/u.s