from scipy.integrate import simpson 
import numpy as np
import matplotlib.pyplot as plt
import io
import astropy.units as u

class FIPBiasCalculator:
    def __init__(self, loop_properties, acceleration, collision_components, coordinates):
        self.loop_properties = loop_properties
        self.acceleration = acceleration.to(u.cm/u.s**2) # [cm/s^2]
        self.collision_components = collision_components
        self.coordinates = coordinates.to(u.cm) # [cm]
        # self.wave = WaveSolutionExtractor.(loop_properties, frequency, amplitude)

    def _integral_terms(self):
        ionisation_fraction = self.collision_components.ionisation_fraction
        v_s = self.collision_components.v_s # element specific turbulent speed
        v_ion = self.collision_components.v_ion
        v_eff = self.collision_components.v_eff
        acceleration = self.acceleration
        # Constants
        k_B = 1.3807e-16  # Boltzmann constant [erg/K]
        amu = 1.66e-24    # atomic mass unit [g]
        
        # v_st in [cm²/s²]
        v_st = v_s**2+(k_B * self.loop_properties.T(self.coordinates).value /
                        (self.collision_components.mass_number * amu))        # v_st = v_s**2
        # v_st = (k_B * self.loop_properties.T(self.coordinates).value /
        #                 (self.collision_components.mass_number * amu))        # v_st = v_s**2
        integral = ionisation_fraction * acceleration.value * v_eff / (v_st * v_ion)

        return integral
    
    def calculate_fip_bias(self, z0, z1, direction='forward'):
        """
        Calculate FIP bias ratio between heights z0 [cm] and z1 [cm]
        
        Parameters:
        -----------
        z0 : astropy.units.Quantity
            Starting height
        z1 : astropy.units.Quantity
            Ending height
        direction : str
            'forward' for left-to-right integration
            'reverse' for right-to-left integration
        """
        # Convert inputs to Quantity if they aren't already
        z0 = u.Quantity(z0, u.cm)
        z1 = u.Quantity(z1, u.cm)

        # Get integrand terms
        integrand = self._integral_terms()
        
        # Find indices corresponding to z0 and z1
        idx0 = np.argmin(np.abs(self.coordinates.value - z0.value))
        idx1 = np.argmin(np.abs(self.coordinates.value - z1.value))
        
        # If the indices are the same, return 1.0 (no FIP bias between same points)
        if idx0 == idx1:
            return 1.0
            
        # For reverse direction, we need to:
        # 1. Negate the integrand (since we're integrating backwards)
        # 2. Keep the same ordering logic for the integration
        if direction == 'reverse':
            integrand = -integrand
        
        # Ensure proper integration order
        if idx1 < idx0:
            idx0, idx1 = idx1, idx0  # swap them
            
        # Include the endpoint in the integration
        integral_result = simpson(y=integrand[idx0:idx1+1], 
                                x=self.coordinates[idx0:idx1+1])
        
        # Calculate final result
        fip_bias = np.exp(2 * integral_result)
        
        return fip_bias
    def calculate_height_profile(self):
        """Calculate FIP bias ratio from base to each height."""
        fip_bias_profile = np.ones_like(self.coordinates)  # Initialize with ones
        z_base = self.coordinates[0]
        
        # Calculate FIP bias from base to each height, starting from second point
        for i, z in enumerate(self.coordinates[1:], start=1):
            print(z_base, z)
            fip_bias_profile[i] = self.calculate_fip_bias(z_base, z)
        # Save the height profile to a pickle file
        output_dir = '/Users/andysh.to/Script/Python_Script/alfven_loop/test_data/output/'
        import os
        import pickle
        os.makedirs(output_dir, exist_ok=True)
        
        profile_data = {
            'coordinates_Mm': self.coordinates,
            'fip_bias_profile': fip_bias_profile
        }
        
        output_file = os.path.join(output_dir, 'fip_bias_height_profile.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(profile_data, f)
        return fip_bias_profile
    def plot_profile(self, save=True, filepath=None):
        """
        Create a plot of FIP bias vs height.
        
        Parameters:
        save (bool): If True, saves plot to BytesIO object. If False, returns the figure.
        
        Returns:
        BytesIO object or matplotlib figure depending on save parameter
        """
        fip_bias_profile = self.calculate_height_profile()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        s_Mm = (self.coordinates*u.Mm).value
        ax.semilogy(s_Mm, fip_bias_profile, 'k-', label='FIP Bias')
        
        ax.set_xlabel('Distance Along Loop (Mm)')
        ax.set_ylabel('FIP Bias Ratio')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(-1, 10)
        ax.set_title('FIP Bias Profile Along the Loop')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            if filepath:
                plt.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
                plt.close()
                return None
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                return buf.getvalue()
        else:
            return fig