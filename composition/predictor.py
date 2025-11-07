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

    def _integral_terms(self, return_all=False):
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
        thermal_speed_squared = k_B * self.loop_properties.T(self.coordinates).value / (self.collision_components.mass_number * amu)
        v_st_squared = self.slow_mode_velocity**2 + v_s**2 + thermal_speed_squared
        integral = ionisation_fraction * acceleration.value * v_eff / (v_st_squared * v_ion)

        if return_all:
            return integral, ionisation_fraction, v_s, v_ion, v_eff, acceleration, thermal_speed_squared
        else:
            return integral
    
    def calculate_fip_bias(self, z0, z1, direction='forward', adaptive=True, tol=1e-6, max_refinements=5):
        """
        Calculate FIP bias ratio between heights z0 [cm] and z1 [cm] with adaptive grid refinement
        
        Parameters:
        -----------
        z0 : astropy.units.Quantity
            Starting height
        z1 : astropy.units.Quantity
            Ending height
        direction : str
            'forward' for left-to-right integration
            'reverse' for right-to-left integration
        adaptive : bool
            Whether to use adaptive grid refinement
        tol : float
            Tolerance for adaptive refinement
        max_refinements : int
            Maximum number of refinement levels
        """
        # Convert inputs to Quantity if they aren't already
        z0 = u.Quantity(z0, u.cm)
        z1 = u.Quantity(z1, u.cm)

        # Get integrand terms
        integrand = self._integral_terms()
        
        # Find indices corresponding to z0 and z1
        idx0 = np.argmin(np.abs(self.coordinates.value - z0.value))
        idx1 = np.argmin(np.abs(self.coordinates.value - z1.value))
        
        if idx0 == idx1:
            return 1.0
            
        # Ensure proper ordering
        if idx1 < idx0:
            idx0, idx1 = idx1, idx0
            
        if not adaptive:
            # Use simple Simpson's rule
            integral_result = simpson(y=integrand[idx0:idx1+1], 
                                    x=self.coordinates[idx0:idx1+1])
        else:
            # Use adaptive integration
            integral_result = self._adaptive_integrate(
                integrand, self.coordinates, idx0, idx1, tol, max_refinements
            )
        
        # Handle direction for backwards integration
        if direction == 'reverse' and z1.value < z0.value:
            integral_result = -integral_result
            
        # Calculate final result
        fip_bias = np.exp(2 * integral_result)
        return fip_bias
    
    def _adaptive_integrate(self, integrand, coordinates, idx0, idx1, tol, max_refinements):
        """
        Perform adaptive integration using recursive grid refinement
        """
        def integrate_segment(y_vals, x_vals, level=0):
            """Recursively integrate a segment with adaptive refinement"""
            if len(y_vals) < 3:
                # Not enough points for Simpson's rule, use trapezoidal
                return np.trapz(y_vals, x_vals)
                
            # Calculate integral with current grid
            integral_coarse = simpson(y_vals, x=x_vals)
            
            if level >= max_refinements:
                return integral_coarse
                
            # Refine grid by interpolating midpoints
            x_refined = []
            y_refined = []
            
            for i in range(len(x_vals)):
                x_refined.append(x_vals[i])
                y_refined.append(y_vals[i])
                
                if i < len(x_vals) - 1:
                    # Add midpoint
                    x_mid = (x_vals[i] + x_vals[i+1]) / 2
                    # Interpolate integrand at midpoint
                    y_mid = (y_vals[i] + y_vals[i+1]) / 2
                    x_refined.append(x_mid)
                    y_refined.append(y_mid)
            
            x_refined = np.array(x_refined)
            y_refined = np.array(y_refined)
            
            # Calculate integral with refined grid
            integral_fine = simpson(y_refined, x=x_refined)
            
            # Check convergence
            error = abs(integral_fine - integral_coarse)
            relative_error = error / (abs(integral_fine) + 1e-15)
            
            if relative_error < tol:
                return integral_fine
            else:
                # Need further refinement - split into segments
                mid_idx = len(x_vals) // 2
                
                # Left segment
                left_integral = integrate_segment(
                    y_vals[:mid_idx+1], x_vals[:mid_idx+1], level+1
                )
                
                # Right segment  
                right_integral = integrate_segment(
                    y_vals[mid_idx:], x_vals[mid_idx:], level+1
                )
                
                return left_integral + right_integral
        
        # Extract the segment to integrate
        y_segment = integrand[idx0:idx1+1]
        x_segment = coordinates[idx0:idx1+1].value
        
        return integrate_segment(y_segment, x_segment)
        
    # def calculate_height_profile(self):
    #     """Calculate FIP bias ratio from base to each height."""
    #     # Initialize with ones as a regular numpy array (dimensionless)
    #     fip_bias_profile = np.ones(len(self.coordinates))
    #     z_base = self.coordinates[0]
        
    #     # Calculate FIP bias from base to each height, starting from second point
    #     for i, z in enumerate(self.coordinates[1:], start=1):
    #         # print(z_base, z)
    #         fip_bias_profile[i] = self.calculate_fip_bias(z_base, z)
    #     # Save the height profile to a pickle file
    #     output_dir = '/Users/andysh.to/Script/Python_Script/alfven_loop/test_data/output/'
    #     import os
    #     import pickle
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     profile_data = {
    #         'coordinates_Mm': self.coordinates,
    #         'fip_bias_profile': fip_bias_profile
    #     }
        
    #     output_file = os.path.join(output_dir, 'fip_bias_height_profile.pkl')
    #     with open(output_file, 'wb') as f:
    #         pickle.dump(profile_data, f)
    #     return fip_bias_profile

    def calculate_height_profile(self):
        """Calculate FIP bias ratio from base to each height using cumulative integration."""
        # Get the integrand for all points at once (already vectorized)
        integrand = self._integral_terms()
        
        # Use cumulative trapezoidal integration (much faster than repeated Simpson's rule)
        from scipy.integrate import cumulative_trapezoid
        
        # Cumulative integral from base to each height
        cumulative_integral = cumulative_trapezoid(
            y=integrand, 
            x=self.coordinates.value, 
            initial=0.0
        )
        
        # Calculate FIP bias profile: exp(2 * integral)
        fip_bias_profile = np.exp(2 * cumulative_integral)
        
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