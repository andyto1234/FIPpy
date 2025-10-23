from scipy.integrate import simpson 
import numpy as np
import matplotlib.pyplot as plt
import io
import astropy.units as u

class FIPBiasCalculator:
    def __init__(self, loop_properties, acceleration, collision_components, coordinates, slow_mode_velocity = None):
        self.loop_properties = loop_properties
        self.acceleration = acceleration.to(u.cm/u.s**2) # [cm/s^2]
        self.collision_components = collision_components
        self.coordinates = coordinates.to(u.cm) # [cm]
        if slow_mode_velocity is not None:
            self.slow_mode_velocity = slow_mode_velocity.to(u.cm/u.s).value # [cm/s]
        else:
            self.slow_mode_velocity = None
        # self.slow_mode_velocity = slow_mode_velocity.to(u.cm/u.s) # [cm/s]
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
        # v_st encompasses the thermal speed, turbulent speed and slow mode velocity
        v_st_squared = self.slow_mode_velocity**2 + v_s**2 + (k_B * self.loop_properties.T(self.coordinates).value /
                        (self.collision_components.mass_number * amu))        # v_st = v_s**2 
        # v_st_squared = self.slow_mode_velocity**2+(k_B * self.loop_properties.T(self.coordinates).value /
        #                 (self.collision_components.mass_number * amu))        # v_st = v_s**2 # removed v_s**2
        # v_st = (k_B * self.loop_properties.T(self.coordinates).value /
        #                 (self.collision_components.mass_number * amu))        # v_st = v_s**2
        integral = ionisation_fraction * acceleration.value * v_eff / (v_st_squared * v_ion)

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
        # print(integrand)
        # Find indices corresponding to z0 and z1
        idx0 = np.argmin(np.abs(self.coordinates.value - z0.value))
        idx1 = np.argmin(np.abs(self.coordinates.value - z1.value))
        
        # If the indices are the same, return 1.0 (no FIP bias between same points)
        # if idx0 == idx1:
        #     return 1.0
            
        # For reverse direction, we need to:
        # # 1. Negate the integrand (since we're integrating backwards)
        # # 2. Keep the same ordering logic for the integration
        # if direction == 'reverse':
        #     integrand = -integrand
        
        # # Ensure proper integration order
        # if idx1 < idx0:
        #     idx0, idx1 = idx1, idx0  # swap them
            
        # # Include the endpoint in the integration
        # integral_result = simpson(y=integrand[idx0:idx1+1], 
        #                         x=self.coordinates[idx0:idx1+1])
        
        # # Calculate final result
        # fip_bias = np.exp(2 * integral_result)
        
        # return fip_bias
        # Always integrate in the direction of increasing coordinate index
        # The physics is captured by the sign of the integrand and the integration limits
        if idx0 == idx1:
            return 1.0
        # print(direction)
        # import pdb; pdb.set_trace()
        # Always integrate from z0 to z1, but handle the direction properly
        if idx1 < idx0:
            # We're integrating backwards along the coordinate array
            # Swap indices and negate the integral
            integral_result = -simpson(y=integrand[idx1:idx0+1], 
                                    x=self.coordinates[idx1:idx0+1])
        else:
            # We're integrating forwards along the coordinate array
            integral_result = simpson(y=integrand[idx0:idx1+1], 
                                    x=self.coordinates[idx0:idx1+1])
        # Calculate final result
        fip_bias = np.exp(2 * integral_result)
        # fip_bias = np.exp(integral_result)
        return fip_bias
        
    def calculate_height_profile(self):
        """Calculate FIP bias ratio from base to each height."""
        # Initialize with ones as a regular numpy array (dimensionless)
        fip_bias_profile = np.ones(len(self.coordinates))
        z_base = self.coordinates[0]
        
        # Calculate FIP bias from base to each height, starting from second point
        for i, z in enumerate(self.coordinates[1:], start=1):
            # print(z_base, z)
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
    def plot_profile(self, save=True, filepath=None, inverse=False):
        """
        Create a plot of all initialization variables vs height.
        
        Parameters:
        save (bool): If True, saves plot to BytesIO object. If False, returns the figure.
        filepath (str): Optional filepath to save the plot
        inverse (bool): If True, integrates from right to left and inverts acceleration
        
        Returns:
        BytesIO object or matplotlib figure depending on save parameter
        """
        if inverse:
            # Calculate FIP bias from right to left (apex to base)
            fip_bias_profile = np.ones(len(self.coordinates))
            z_apex = self.coordinates[-1]  # Start from the rightmost point (apex)
            
            # Calculate FIP bias from apex to each height, going backwards
            for i in range(len(self.coordinates) - 2, -1, -1):
                z = self.coordinates[i]
                fip_bias_profile[i] = self.calculate_fip_bias(z_apex, z, direction='reverse')
        else:
            fip_bias_profile = self.calculate_height_profile()
        
        # Create subplots for all variables
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Convert coordinates to height in Mm
        heights_Mm = (self.coordinates / u.cm * u.Mm).value
        
        # Plot 1: FIP Bias Profile
        axes[0].semilogy(heights_Mm, fip_bias_profile, 'k-', label='FIP Bias')
        axes[0].set_xlabel('Height (Mm)')
        axes[0].set_ylabel('FIP Bias Ratio')
        title_suffix = ' (Inverse)' if inverse else ''
        axes[0].set_title(f'FIP Bias Profile{title_suffix}')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Ponderomotive Acceleration (invert if inverse=True)
        acceleration_values = -self.acceleration.value if inverse else self.acceleration.value
        axes[1].plot(heights_Mm, acceleration_values, 'r-', label='Ponderomotive Acceleration')
        axes[1].set_xlabel('Height (Mm)')
        axes[1].set_ylabel('Acceleration (cm/s²)')
        axes[1].set_title(f'Ponderomotive Acceleration{title_suffix}')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Ionization Fraction
        axes[2].semilogy(heights_Mm, self.collision_components.ionisation_fraction, 'b-', label='Ionization Fraction')
        axes[2].set_xlabel('Height (Mm)')
        axes[2].set_ylabel('Ionization Fraction')
        axes[2].set_title('Ionization Fraction')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot 4: Effective Velocity
        axes[3].semilogy(heights_Mm, self.collision_components.v_eff, 'g-', label='Effective Velocity')
        axes[3].set_xlabel('Height (Mm)')
        axes[3].set_ylabel('v_eff (cm/s)')
        axes[3].set_title('Effective Velocity')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        # Plot 5: Slow Mode Velocity (if available)
        if self.slow_mode_velocity is not None:
            axes[4].plot(heights_Mm, self.slow_mode_velocity, 'm-', label='Slow Mode Velocity')
            axes[4].set_xlabel('Height (Mm)')
            axes[4].set_ylabel('Slow Mode Velocity (cm/s)')
            axes[4].set_title('Slow Mode Velocity')
            axes[4].grid(True, alpha=0.3)
            axes[4].set_yscale('log')
            # axes[4].set_ylim(0.001, 100)
            axes[4].legend()
        else:
            axes[4].text(0.5, 0.5, 'Slow Mode Velocity\nNot Available', 
                        ha='center', va='center', transform=axes[4].transAxes)
            axes[4].set_title('Slow Mode Velocity')
        
        # Plot 6: Temperature and Density
        T = self.loop_properties.T(self.coordinates)
        ne = self.loop_properties.ne(self.coordinates)
        
        ax6_twin = axes[5].twinx()
        line1 = axes[5].semilogy(heights_Mm, ne, 'c-', label='Electron Density')
        line2 = ax6_twin.semilogy(heights_Mm, T, 'orange', label='Temperature')
        
        axes[5].set_xlabel('Height (Mm)')
        axes[5].set_ylabel('Electron Density (cm⁻³)', color='c')
        ax6_twin.set_ylabel('Temperature (K)', color='orange')
        axes[5].set_title('Temperature and Density')
        axes[5].grid(True, alpha=0.3)
        
        # Combine legends for the twin axis plot
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[5].legend(lines, labels, loc='best')
        
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