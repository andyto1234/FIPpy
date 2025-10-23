from scipy.integrate import solve_ivp
# from physics.constants import u
import astropy.units as u
from loop.magnetic_field import MagneticField
from loop.heights import ScaleHeightCalculator
from physics.speed import AlfvenSpeedCalculator
from plasmapy.formulary import beta
import numpy as np
# from line_profiler import profile
from .slowmode import solve_dvz

class WaveAnalyzer:
    """Class to handle wave analysis calculations and properties."""
    
    def __init__(self, loop_properties):
        self.loop_properties = loop_properties
        # self.frequency = frequency
        # self.omega = self.frequency_to_angular_frequency(self.frequency)
        self.B = MagneticField(loop_properties)
        self.alfven_speed_calculator = AlfvenSpeedCalculator(self.B, self.loop_properties.rho)
        self.density_scale_height_calculator = ScaleHeightCalculator(self.loop_properties.rho)
        self.alfven_scale_height_calculator = ScaleHeightCalculator(self.alfven_speed_calculator.calculate)

    @staticmethod
    def frequency_to_angular_frequency(frequency):
        return 2 * np.pi * frequency * u.rad

    def VA(self, s):
        """Calculate Alfvén speed at position s."""
        return AlfvenSpeedCalculator(self.B, self.loop_properties.rho).calculate(s)

    def HD(self, s):
        """Calculate density scale height at position s."""
        return self.density_scale_height_calculator.calculate(s)

    def HA(self, s):
        """Calculate Alfvén scale height at position s."""
        return self.alfven_scale_height_calculator.calculate(s)

    def solve_slow_mode_dvz(self, s, w_A, dv_A):
        """
        Solve for slow mode vertical velocity perturbation at position s.
        
        Parameters:
            s: Position along loop [astropy quantity with length units]
            w_A: Alfvén wave angular frequency [rad/s]
            dv_A: Alfvén wave amplitude [cm/s]
            
        Returns:
            dv_z: Slow mode vertical velocity perturbation [cm/s] (complex)
        """
        
        # Get required parameters at position s
        V_A = self.VA(s).to(u.cm/u.s).value
        c_s = self.loop_properties.sound_speed(s).to(u.cm/u.s).value
        # c_s = self.calculate_sound_speed(s).value
        L_rho = np.abs(self.HD(s).to(u.cm).value)
        w_A = self.frequency_to_angular_frequency(w_A).value
        # # Convert inputs to cgs if needed
        if hasattr(w_A, 'unit'):
            w_A = w_A.to(u.rad/u.s).value
        if hasattr(dv_A, 'unit'):
            dv_A = dv_A.to(u.cm/u.s).value
        
        return solve_dvz(w_A, dv_A, c_s, L_rho, V_A) * u.cm/u.s
    # @profile
    def solve_wave_equations(self, amplitude, frequency):
        """Solve the wave equations using ODE solver."""
        
        I_plus_0 = amplitude.to(u.cm/u.s)
        y0 = [I_plus_0.value, 0.0, 0.0, 0.0]
        omega = self.frequency_to_angular_frequency(frequency)

        # finding where beta < 1
        valid_indices = self._get_valid_indices()
        # print(valid_indices)
        # import pdb; pdb.set_trace()
        s_array_unique, _ = np.unique(self.loop_properties.s_array[valid_indices], return_index=True)
        
        s_eval = s_array_unique.to(u.cm).value
        print("Solving ODE ")
        sol = self._solve_ode(s_eval, y0, omega.value)
        print("ODE solved")
        return sol

    def solve_slow_mode_along_loop(self, w_A, dv_A, s_positions=None):
        """
        Calculate slow mode velocity perturbation along the loop.
        
        Parameters:
            w_A: Alfvén wave angular frequency [rad/s]
            dv_A: Alfvén wave amplitude [cm/s]
            s_positions: Array of positions to evaluate (optional, uses loop s_array if None)
            
        Returns:
            dict: Contains 's_array', 'dv_z_real', 'dv_z_imag', 'dv_z_magnitude'
        """
        if s_positions is None:
            s_positions = self.loop_properties.s_array
        
        dv_z_array = []
        valid_positions = []
        # Vectorized calculation for all positions at once
        # Handle both astropy quantities and plain numpy values for s_positions
        if hasattr(s_positions, 'value'):
            s_values = s_positions.value
            s_unit = s_positions.unit
        else:
            s_values = s_positions
            s_unit = u.cm
        
        # Ensure s_values has units for the solve_slow_mode_dvz method
        if not hasattr(s_values, 'unit'):
            s_values = s_values * s_unit
        
        # Handle dv_A as array - each position gets corresponding dv_A value
        if hasattr(dv_A, '__len__') and len(dv_A) > 1:
            # dv_A is an array, use element-wise calculation
            if len(dv_A) != len(s_positions):
                raise ValueError(f"Length of dv_A ({len(dv_A)}) must match length of s_positions ({len(s_positions)})")
            
            dv_z_results = []
            for i, (s_val, dv_val) in enumerate(zip(s_values, dv_A)):
                if not hasattr(s_val, 'unit'):
                    s_val = s_val * s_unit
                dv_z = self.solve_slow_mode_dvz(s_val, w_A, dv_val)
                if hasattr(dv_z, 'value'):
                    dv_z_results.append(dv_z.value)
                else:
                    dv_z_results.append(dv_z)
            
            dv_z_array = np.array(dv_z_results)
        else:
            # dv_A is scalar, use same value for all positions
            dv_z_results = []
            for s_val in s_values:
                if not hasattr(s_val, 'unit'):
                    s_val = s_val * s_unit
                dv_z = self.solve_slow_mode_dvz(s_val, w_A, dv_A)
                if hasattr(dv_z, 'value'):
                    dv_z_results.append(dv_z.value)
                else:
                    dv_z_results.append(dv_z)
            
            dv_z_array = np.array(dv_z_results)
        
        return np.abs(dv_z_array) * u.cm/u.s
        # }

    def _get_valid_indices(self):
        """Get indices where plasma beta is less than 1.2."""

        B_field = self.B(self.loop_properties.s_array).to(u.G).value
        thermal_pressure = self.loop_properties.pe(self.loop_properties.s_array).to(u.erg/u.cm**3).value
        p_mag = (B_field**2 / (8 * np.pi))  # magnetic pressure in cgs
        plasma_beta = thermal_pressure/p_mag
        return np.where(plasma_beta < 1)[0]
        # return np.where(beta(
        #     self.loop_properties.T(self.loop_properties.s_array),
        #     self.loop_properties.ne(self.loop_properties.s_array),
        #     self.B(self.loop_properties.s_array)
        # ) < 0.8)[0]

        #     self.loop_properties.pe(self.loop_properties.s_array),
            # return np.where(beta(self.loop_properties.T(self.loop_properties.s_array),self.loop_properties.ne(self.loop_properties.s_array),self.B(self.loop_properties.s_array)) < 0.8)[0]

    # @profile
    def _solve_ode(self, s_eval, y0, omega):
        """Internal method to solve the ODE system."""
        def ode_system(s, y):
            """Define the ODE system for wave propagation."""
            
            V_A_s, H_D_s, H_A_s, u_s, _s = self._ode_feeder(s)  # Now returns u_s
            # u_s = 0.0  # Remove this line - now using actual velocity

            # I_plus_R, I_plus_I, I_minus_R, I_minus_I = y
            # # Denominators for wave equations
            # denom_plus = u_s - V_A_s
            # denom_minus = u_s + V_A_s

            # # Wave equations in purely numeric form
            # dI_plus_R_ds = ((u_s + V_A_s) * (I_plus_R / (4 * H_D_s) + I_minus_R / (2 * H_A_s)) + omega * I_plus_I) / denom_plus
            # dI_plus_I_ds = ((u_s + V_A_s) * (I_plus_I / (4 * H_D_s) + I_minus_I / (2 * H_A_s)) - omega * I_plus_R) / denom_plus
            # dI_minus_R_ds = ((u_s - V_A_s) * (I_minus_R / (4 * H_D_s) + I_plus_R / (2 * H_A_s)) + omega * I_minus_I) / denom_minus
            # dI_minus_I_ds = ((u_s - V_A_s) * (I_minus_I / (4 * H_D_s) + I_plus_I / (2 * H_A_s)) - omega * I_minus_R) / denom_minus

            return ode_system_numba(s, y, V_A_s, H_D_s, H_A_s, u_s, omega)
        
        return solve_ivp(
            ode_system,
            [s_eval[0], s_eval[-1]],
            y0,
            t_eval=s_eval,
            method='Radau',
            atol=1e-6,
            rtol=1e-6
        )    # Define ODE system
    def _ode_feeder(self, s):
        # Convert input s to astropy quantity for safety
        s_cgs = s * u.cm
        # Get values with units then extract numeric values
        V_A_s = self.VA(s_cgs).to(u.cm/u.s).value # Alfven speed
        H_D_s = self.HD(s_cgs).to(u.cm).value # density scale height
        H_A_s = self.HA(s_cgs).to(u.cm).value # Alfven speed scale height
        u_s = self.loop_properties.velocity(s_cgs).to(u.cm/u.s).value # Background flow velocity
        return V_A_s, H_D_s, H_A_s, u_s, s
    
from numba import njit

@njit
def ode_system_numba(s, y, V_A_s, H_D_s, H_A_s, u_s, omega):
    I_plus_R, I_plus_I, I_minus_R, I_minus_I = y
    dI_plus_R_ds = ((u_s + V_A_s) * (I_plus_R / (4.0 * H_D_s) + I_minus_R / (2.0 * H_A_s)) + omega * I_plus_I) / (u_s - V_A_s)
    dI_plus_I_ds = ((u_s + V_A_s) * (I_plus_I / (4.0 * H_D_s) + I_minus_I / (2.0 * H_A_s)) - omega * I_plus_R) / (u_s - V_A_s)
    dI_minus_R_ds = ((u_s - V_A_s) * (I_minus_R / (4.0 * H_D_s) + I_plus_R / (2.0 * H_A_s)) + omega * I_minus_I) / (u_s + V_A_s)
    dI_minus_I_ds = ((u_s - V_A_s) * (I_minus_I / (4.0 * H_D_s) + I_plus_I / (2.0 * H_A_s)) - omega * I_minus_R) / (u_s + V_A_s)
    
    return np.array([dI_plus_R_ds, dI_plus_I_ds, dI_minus_R_ds, dI_minus_I_ds])
    # def _ode_system(self, s, y, omega):
    #     """Define the ODE system for wave propagation."""
    #     I_plus_R, I_plus_I, I_minus_R, I_minus_I = y
    #     V_A_s, H_D_s, H_A_s, _s = self._ode_feeder(s)
    #     u_s = 0.0  # Background flow speed

    #     # Denominators for wave equations
    #     denom_plus = u_s - V_A_s
    #     denom_minus = u_s + V_A_s

    #     # Wave equations in purely numeric form
    #     dI_plus_R_ds = ((u_s + V_A_s) * (I_plus_R / (4 * H_D_s) + I_minus_R / (2 * H_A_s)) + omega * I_plus_I) / denom_plus
    #     dI_plus_I_ds = ((u_s + V_A_s) * (I_plus_I / (4 * H_D_s) + I_minus_I / (2 * H_A_s)) - omega * I_plus_R) / denom_plus
    #     dI_minus_R_ds = ((u_s - V_A_s) * (I_minus_R / (4 * H_D_s) + I_plus_R / (2 * H_A_s)) + omega * I_minus_I) / denom_minus
    #     dI_minus_I_ds = ((u_s - V_A_s) * (I_minus_I / (4 * H_D_s) + I_plus_I / (2 * H_A_s)) - omega * I_minus_R) / denom_minus

    #     return [dI_plus_R_ds, dI_plus_I_ds, dI_minus_R_ds, dI_minus_I_ds]

def solve_wave_equations_from_apex(self, amplitude, frequency):
    """
    Solve wave equations from apex using symmetry (faster).
    """
    valid_indices = self._get_valid_indices()
    s_array_valid = self.loop_properties.s_array[valid_indices]
    s_eval_full = s_array_valid.to(u.cm).value
    
    apex_idx = len(s_array_valid) // 2
    I_0 = amplitude.to(u.cm/u.s).value
    y0_apex = [I_0, 0.0, I_0, 0.0]
    omega = self.frequency_to_angular_frequency(frequency)
    
    # Solve only right side (apex → right footpoint)
    s_right = s_eval_full[apex_idx:]
    sol_right = self._solve_ode(s_right, y0_apex, omega.value)
    
    # Extract footpoint amplitude
    I_plus_right = (sol_right.y[0, -1] + 1j * sol_right.y[1, -1]) * u.cm/u.s
    
    # By symmetry: left footpoint has same amplitude
    # (I_minus on left = I_plus on right, by symmetry)
    I_minus_left = I_plus_right  # Same by symmetry
    
    average_amplitude = np.abs(I_plus_right)  # They're equal
    
    return {
        'sol_right': sol_right,
        's_right': s_right * u.cm,
        'footpoint_amplitude': average_amplitude,  # Use this for FIP bias
        'left_footpoint_amplitude': np.abs(I_minus_left),
        'right_footpoint_amplitude': np.abs(I_plus_right)
    }

def _solve_ode_reversed(self, s_eval, y0, omega):
    """Solve ODE in reverse direction (for left side of apex)."""
    # Modify signs appropriately for reversed integration
    # ... similar to _solve_ode but handle direction