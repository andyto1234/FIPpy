import pickle
import sys
from pathlib import Path
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import io
import os
from matplotlib import ticker

# Notebook kernels often append this repo after the standard library, so
# "wave" binds to the stdlib WAV module. Search this repo first.
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
_loaded_wave = sys.modules.get("wave")
if _loaded_wave is not None and not hasattr(_loaded_wave, "__path__"):
    del sys.modules["wave"]

from loop.properties import LoopProperties
from loop.reader import HydradReader
from physics.saha import get_ionisation_fraction
from physics.collision import CollisionCalculator
from wave.solver import WaveAnalyzer
from wave.extractor import WaveSolutionExtractor
from composition.predictor import FIPBiasCalculator

def plot_ionization(results):
    # print(results)
    s_Mm = results['s_values'].to(u.Mm).value
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Ionization Fractions Along the Coronal Loop\nFrequency: {results["frequency"]:.3f}, Amplitude: {results["amplitude"]:.2f}, Loop Length: {results["loop_length"].to(u.Mm):.2f}', fontsize=12)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results['ionization_fractions'])))
    
    for (element, fractions), color in zip(results['ionization_fractions'].items(), colors):
        ax.semilogy(s_Mm, fractions, label=element, color=color)
        print(f"Element: {element}, Min: {np.min(fractions):.3f}, Max: {np.max(fractions):.3f}")
    
    ax.set_xlabel('Distance Along the Loop (Mm)')
    ax.set_ylabel('Ionization Fraction')
    ax.set_xlim(0, np.max(s_Mm))  # Set x-axis limit to the full range of s_Mm
    ax.set_ylim(0.1, 1.0)  # Adjust y-axis limit to show the full range of fractions
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
    ax.legend(title='Elements', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()

def setup_output_directory(output_dir, frequency, amplitude, flux_value=None, v_add=None):
    run_name = f'v_{frequency.value:.4f}_A_{amplitude.value:.2f}'
    if flux_value is not None:
        run_name += '_' + f'F_{flux_value.value:.1e}'.replace('+', '')
    elif v_add is not None:
        run_name += '_' + f'v_add_{v_add.value:.2f}'.replace('+', '')

    output_dir = os.path.join(os.path.abspath(output_dir), run_name) + os.sep
    image_dir = os.path.join(output_dir, 'images') + os.sep
    for directory in [output_dir, image_dir]:
        os.makedirs(directory, exist_ok=True)
    return output_dir, image_dir
    
def process_hydrad_data(input_file, frequency, amplitude, output_dir, strand_num=105, acoustic_flux=None, v_add=None, *, hydrogen_history=None):
    # Load data from input pickle file
    output_dir, image_dir = setup_output_directory(output_dir, frequency, amplitude, flux_value=acoustic_flux, v_add=v_add)

    hydrad_strand = HydradReader(input_file, strand_num=strand_num).read_strand()
    # print(type(hydrad_strand))
    loop_properties = LoopProperties.from_hydrad(hydrad_strand, input_file, strand_num, hydrogen_history)

    analyzer = WaveAnalyzer(loop_properties)
    sol = analyzer.solve_wave_equations(amplitude, frequency)

    extractor = WaveSolutionExtractor(loop_properties, sol)
    # returning all the wave calculatable components - e.g. acceleration, flux
    wave_components = extractor.process_solution()
    I_plus = wave_components.I_plus
    I_minus = wave_components.I_minus
    flux_right = wave_components.flux_right
    flux_left = wave_components.flux_left

    # import pdb; pdb.set_trace()
    s_values = wave_components.s_array
    
    # Find peaks of ponderomotive acceleration in first and second half
    accel_abs = np.abs(wave_components.ponderomotive_acceleration.value)
    mid_idx = len(s_values) // 2
    
    # Find peak indices in each half
    peak_idx_first = np.argmax(accel_abs[:mid_idx])
    peak_idx_second = mid_idx + np.argmax(accel_abs[mid_idx:])
    
    # Define interpolation window size
    WINDOW_SIZE = 3
    NUM_INTERP_POINTS = 20
    
    def create_interpolation_range(peak_idx, s_values, window_size):
        """Create interpolation range around a peak index."""
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(s_values) - 1, peak_idx + window_size)
        s_range = s_values[start_idx:end_idx + 1]
        # Create denser spacing around the peak
        peak_s = s_values[peak_idx].value
        s_start = s_range[0].value
        s_end = s_range[-1].value
        
        # Split range into two parts: before and after peak
        # Use more points closer to the peak
        n_before = NUM_INTERP_POINTS // 2
        n_after = NUM_INTERP_POINTS - n_before
        
        # Create non-uniform spacing with higher density near peak
        # Using quadratic spacing for smoother transition
        t_before = np.linspace(0, 1, n_before)**2
        t_after = np.linspace(0, 1, n_after)**2
        
        s_before = s_start + t_before * (peak_s - s_start)
        s_after = peak_s + t_after * (s_end - peak_s)
        
        s_interp = np.concatenate([s_before, s_after[1:]]) * s_range.unit
        return start_idx, end_idx, s_interp
    
    # Process first peak
    start_idx_first, end_idx_first, s_interp_first = create_interpolation_range(
        peak_idx_first, s_values, WINDOW_SIZE
    )
    
    # Process second peak
    start_idx_second, end_idx_second, s_interp_second = create_interpolation_range(
        peak_idx_second, s_values, WINDOW_SIZE
    )
    
    # Create interpolation functions for I_plus and I_minus
    from scipy.interpolate import interp1d
    
    def interpolate_complex_quantity(quantity, s_values, s_interp):
        """Interpolate a complex quantity at new positions."""
        real_interp = interp1d(s_values.value, quantity.real.value, kind='cubic')
        imag_interp = interp1d(s_values.value, quantity.imag.value, kind='cubic')
        return (real_interp(s_interp.value) + 1j * imag_interp(s_interp.value)) * quantity.unit
    
    # Interpolate at first peak
    I_plus_interp_first = interpolate_complex_quantity(I_plus, s_values, s_interp_first)
    I_minus_interp_first = interpolate_complex_quantity(I_minus, s_values, s_interp_first)
    
    # Interpolate at second peak
    I_plus_interp_second = interpolate_complex_quantity(I_plus, s_values, s_interp_second)
    I_minus_interp_second = interpolate_complex_quantity(I_minus, s_values, s_interp_second)
    
    # Reconstruct arrays with interpolated points
    # Order: [before first peak] + [first peak interpolated] + [between peaks] + 
    #        [second peak interpolated] + [after second peak]
    s_values = np.concatenate([
        s_values[:start_idx_first],
        s_interp_first,
        s_values[end_idx_first + 1:start_idx_second],
        s_interp_second,
        s_values[end_idx_second + 1:]
    ])
    
    I_plus = np.concatenate([
        I_plus[:start_idx_first],
        I_plus_interp_first,
        I_plus[end_idx_first + 1:start_idx_second],
        I_plus_interp_second,
        I_plus[end_idx_second + 1:]
    ])
    
    I_minus = np.concatenate([
        I_minus[:start_idx_first],
        I_minus_interp_first,
        I_minus[end_idx_first + 1:start_idx_second],
        I_minus_interp_second,
        I_minus[end_idx_second + 1:]
    ])
    
    # Interpolate ponderomotive acceleration at both peaks
    ponderomotive_accel_interp_first = interp1d(
        wave_components.s_array.value, 
        wave_components.ponderomotive_acceleration.value, 
        kind='cubic'
    )(s_interp_first.value) * wave_components.ponderomotive_acceleration.unit
    
    ponderomotive_accel_interp_second = interp1d(
        wave_components.s_array.value, 
        wave_components.ponderomotive_acceleration.value, 
        kind='cubic'
    )(s_interp_second.value) * wave_components.ponderomotive_acceleration.unit
    
    wave_components.ponderomotive_acceleration = np.concatenate([
        wave_components.ponderomotive_acceleration[:start_idx_first],
        ponderomotive_accel_interp_first,
        wave_components.ponderomotive_acceleration[end_idx_first + 1:start_idx_second],
        ponderomotive_accel_interp_second,
        wave_components.ponderomotive_acceleration[end_idx_second + 1:]
    ])
    # Add this line to sync s_array with s_values:

    wave_components.s_array = s_values
    # Recalculate derived quantities with new grid
    delta_B_over_sqrt_4pi_rho = (I_plus - I_minus) / 2
    delta_v = (I_plus + I_minus) / 2
    
    delta_B_over_sqrt_4pi_rho = (I_plus - I_minus) / 2
    delta_v = (I_plus + I_minus) / 2

    rho_s = loop_properties.rho(s_values)
    VA_s = analyzer.VA(s_values)
    B_s = analyzer.B(s_values)
    slow_mode_results = analyzer.solve_slow_mode_along_loop(frequency, delta_v, s_positions=s_values)

    # Initialize collision calculator to get collision freqencies
    # Initialize collision calculator and calculate FIP bias for multiple elements
    elements = ['Ca', 'Mg', 'Fe', 'Si', 'S', 'C', 'N', 'O', 'Ne', 'Ar']
    # elements = ['Ca', 'Mg', 'Fe', 'Si', 'S', 'C', 'O', 'Ar', 'Ne']
    fip_bias_heights = {}
    ionization_fractions = {}
    collision_frequencies = {}
    integral_terms = {}
    collision_components_dict = {}
    thermal_speed_squared_dict = {}
    sound_speed_dict = {}
    for element in elements:
        collision_calculator = CollisionCalculator(loop_properties, element)
        collision_components = collision_calculator.calculate_collision_components(s_values)
        
        # Store ionization fractions and collision components for this element
        ionization_fractions[element] = collision_components.ionisation_fraction
        # print(collision_components.ionisation_fraction)
        collision_frequencies[element] = collision_components
        collision_components_dict[element] = collision_components
        fip_bias_calculator = FIPBiasCalculator(
            loop_properties,
            wave_components.ponderomotive_acceleration, 
            collision_components, 
            s_values,
            slow_mode_results,
            acoustic_flux=acoustic_flux,
            v_add=v_add # artificial value to test FIP bias
        )
        # fip_bias_calculator = FIPBiasCalculator(loop_properties,
        #                                       wave_components.ponderomotive_acceleration, 
        #                                       collision_components, 
        #                                       wave_components.s_array,
        #                                       slow_mode_results)
        integrand, ionisation_fraction, v_s, v_ion, v_eff, acceleration, thermal_speed_squared, sound_speed  = fip_bias_calculator._integral_terms(return_all=True)
        integral_terms[element] = integrand
        thermal_speed_squared_dict[element] = thermal_speed_squared
        sound_speed_dict[element] = sound_speed
        # Define the loop apex (assuming it's at 6 Mm)
        apex_height = loop_properties.s_array[-1].to(u.cm)
        
        # Left side calculation (0.6 Mm to 3.5 Mm)
        left_start_height = (s_values[0]).to(u.cm)
        left_end_height = (15000 * u.km).to(u.cm)
        
        # Right side calculation (6 Mm to 6 Mm - 3000 km)
        right_start_height = apex_height
        right_end_height = (apex_height - 15000 * u.km).to(u.cm)

        # Find indices for the height ranges
        s_array_km = wave_components.s_array.to(u.km)
        start_idx = np.argmin(np.abs(s_array_km - left_start_height.to(u.km)))
        end_idx = np.argmin(np.abs(s_array_km - left_end_height.to(u.km)))
        right_idx = np.argmin(np.abs(s_array_km - right_start_height.to(u.km)))
        left_idx = np.argmin(np.abs(s_array_km - right_end_height.to(u.km)))


        # Calculate FIP bias for both sides using cumulative integration
        fip_bias_left = []
        fip_bias_right = []

        # Left side (forward integration)
        left_heights = wave_components.s_array[start_idx:end_idx+1]
        if len(left_heights) > 0:
            # Get integrand for the left region
            idx_start_in_full = start_idx
            idx_end_in_full = end_idx
            
            # Extract integrand for this region (already computed at line 235)
            integrand_segment = integrand[idx_start_in_full:idx_end_in_full+1]
            coords_segment = s_values[idx_start_in_full:idx_end_in_full+1]
            
            # Cumulative integration from left_start_height
            from scipy.integrate import cumulative_trapezoid
            cumulative_integral_left = cumulative_trapezoid(
                y=integrand_segment, 
                x=coords_segment.value, 
                initial=0.0
            )
            fip_bias_left = np.exp(2 * cumulative_integral_left)
        else:
            fip_bias_left = np.array([])
            
        # Right side (reverse integration from apex)
        right_heights = wave_components.s_array[left_idx:right_idx+1]
        if len(right_heights) > 0:
            # Extract integrand for the right region
            idx_start_in_full = left_idx
            idx_end_in_full = right_idx
            
            integrand_segment_right = integrand[idx_start_in_full:idx_end_in_full+1]
            coords_segment_right = s_values[idx_start_in_full:idx_end_in_full+1].value
            
            # For reverse integration, integrate backwards and negate
            # Reverse the arrays
            integrand_reversed = integrand_segment_right[::-1]
            coords_reversed = coords_segment_right[::-1]
            
            # Cumulative integration from right_start_height (apex) going backwards
            cumulative_integral_right = cumulative_trapezoid(
                y=-integrand_reversed,  # Negate for reverse direction
                x=coords_reversed,
                initial=0.0
            )
            fip_bias_right = np.exp(-2 * cumulative_integral_right[::-1])  # Reverse back to original order
        else:
            fip_bias_right = np.array([])
        # Calculate and plot height profile using FIPBiasCalculator
        # fip_bias_profile = fip_bias_calculator.calculate_height_profile()
        # fip_bias_plot = fip_bias_calculator.plot_profile(save=True, inverse=True)
        # output_dir = '/Users/andysh.to/Script/Python_Script/alfven_loop_SM/test_data/output/'

        # Save the plot to the output directory
        plot_filename = f'fip_bias_profile_{element}_f{frequency.value:.3f}_A{amplitude.value:.2f}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        # with open(plot_path, 'wb') as f:
        #     f.write(fip_bias_plot)
        # print(f'Saved FIP bias profile plot for {element}: {plot_path}')

        # Store results with corresponding heights
        fip_bias_heights[element] = {
            'left': {
                'heights': left_heights.to(u.Mm).value,
                'bias': fip_bias_left
            },
            'right': {
                'heights': right_heights.to(u.Mm).value,
                'bias': fip_bias_right
            }
        }
    # import pdb; pdb.set_trace()

    # Create plot of fractionation vs height with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot fractionation vs height on first subplot with different colors for each element
    colors = {'Ca': 'blue', 'Mg': 'red', 'Fe': 'orange', 'Si': 'green', 'S': 'purple', 'C': 'gray', 'O': 'pink', 'Ar': 'brown', 'Ne': 'black', 'N': 'cyan'}
    for element in elements:
        # Plot left side
        ax1.plot(fip_bias_heights[element]['left']['heights'], 
                fip_bias_heights[element]['left']['bias'], 
                color=colors[element], 
                label=f'{element} (Left)')
        
        # Plot right side
        ax1.plot(fip_bias_heights[element]['right']['heights'], 
                fip_bias_heights[element]['right']['bias'], 
                color=colors[element], 
                linestyle='--', 
                label=f'{element} (Right)')
    ax1.set_ylabel('FIP Bias Ratio')
    ax1.set_ylim(0, 10)
    ax1.set_title('FIP Bias vs Height')
    ax1.grid(True)
    # ax1.set_xscale('log')
    ax1.legend()
    
    # Plot ponderomotive acceleration on second subplot
    accel = wave_components.ponderomotive_acceleration[start_idx:end_idx+1].to(u.cm/u.s**2)
    heights = wave_components.s_array[start_idx:end_idx+1].to(u.Mm).value  # Define heights here
    ax2_twin = ax2.twinx()  # Create twin axis for slow mode results
    
    ax2.plot(heights, np.abs(accel), 'k-', label='Ponderomotive Acceleration')
    ax2.set_ylabel('Acceleration (cm/s²)')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    # Plot slow mode results on twin axis
    slow_mode_velocity = slow_mode_results[start_idx:end_idx+1].to(u.km/u.s)
    ax2_twin.plot(heights, slow_mode_velocity, 'r--', label='Slow Mode Velocity')
    
    # Plot sound speed on the same twin axis
    sound_speed = loop_properties.sound_speed(wave_components.s_array[start_idx:end_idx+1]).to(u.km/u.s)
    ax2_twin.plot(heights, sound_speed, 'g--', label='Sound Speed')
    
    ax2_twin.set_ylabel('Velocity (km/s)', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2_twin.set_yscale('log')
    
    # Plot acceleration and slow mode results at the other end (right side)
    accel_right = wave_components.ponderomotive_acceleration[left_idx:right_idx+1].to(u.cm/u.s**2)
    heights_right = wave_components.s_array[left_idx:right_idx+1].to(u.Mm).value
    
    ax2.plot(heights_right, np.abs(accel_right), 'k:', label='Ponderomotive Acceleration (Right)')
    
    # Plot slow mode results for right side on twin axis
    slow_mode_velocity_right = slow_mode_results[left_idx:right_idx+1].to(u.km/u.s)
    ax2_twin.plot(heights_right, slow_mode_velocity_right, 'r:', label='Slow Mode Velocity (Right)')
    
    # Plot sound speed for right side on twin axis
    sound_speed_right = loop_properties.sound_speed(wave_components.s_array[left_idx:right_idx+1]).to(u.km/u.s)
    ax2_twin.plot(heights_right, sound_speed_right, 'g:', label='Sound Speed (Right)')
    
    # Combine legends
    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    lines_twin, labels_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_ax2 + lines_twin, labels_ax2 + labels_twin, loc='best')

    # Plot density and temperature on third subplot
    ax3_T = ax3.twinx()  # Create twin axis for temperature
    
    # Get density and temperature for the height range
    ne = loop_properties.ne(wave_components.s_array[start_idx:end_idx+1])
    T = loop_properties.T(wave_components.s_array[start_idx:end_idx+1])
    
    # Plot density on left y-axis
    ax3.semilogy(heights, ne, 'b-', label='Electron Density')
    ax3.set_ylabel('Electron Density (cm⁻³)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    
    # Plot temperature on right y-axis
    ax3_T.semilogy(heights, T, 'r-', label='Temperature')
    ax3_T.set_ylabel('Temperature (K)', color='r')
    ax3_T.tick_params(axis='y', labelcolor='r')
    
    # Plot density and temperature for the right end too
    ne_right = loop_properties.ne(wave_components.s_array[left_idx:right_idx+1])
    T_right = loop_properties.T(wave_components.s_array[left_idx:right_idx+1])
    
    ax3.semilogy(heights_right, ne_right, 'b:', label='Electron Density (Right)')
    ax3_T.semilogy(heights_right, T_right, 'r:', label='Temperature (Right)')
    
    ax3.set_xlabel('Height (Mm)')
    ax3.grid(True)
    
    # Combine legends for density and temperature
    lines_ne, labels_ne = ax3.get_legend_handles_labels()
    lines_T, labels_T = ax3_T.get_legend_handles_labels()
    ax3.legend(lines_ne + lines_T, labels_ne + labels_T, loc='center right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(image_dir + f'fip_bias_vs_height_multi_{frequency.value:.3f}_{amplitude.value:.2f}_{strand_num}.png', dpi=300, bbox_inches='tight')
    print(f'Saved FIP bias vs height plot for {frequency.value:.3f}_{amplitude.value:.2f}_{strand_num}: {image_dir}fip_bias_vs_height_multi_{frequency.value:.3f}_{amplitude.value:.2f}_{strand_num}.png')
    plt.close()
    # import pdb; pdb.set_trace()
    flux_right = (0.25 * rho_s * np.abs(I_plus)**2 * VA_s).to(u.erg/u.cm**2/u.s)
    flux_left = (0.25 * rho_s * np.abs(I_minus)**2 * VA_s).to(u.erg/u.cm**2/u.s)
    flux_difference = flux_right - flux_left
    normalized_flux_difference = flux_difference / B_s

    C_LIGHT_CM_S = 2.99792458e+10 * u.cm / u.s
    delta_E_square = ((B_s**2 / 2) / C_LIGHT_CM_S**2) * (np.abs(I_plus)**2 + np.abs(I_minus)**2)
    delta_E_squared_over_B_squared = delta_E_square / (B_s**2)
    gradient = np.gradient(delta_E_squared_over_B_squared, s_values)
    acceleration_cm_s2 = 0.25 * gradient * C_LIGHT_CM_S**2


    # Prepare output data
    output_data = {
        # Position array
        's_values': s_values,
        # Wave properties
        'I_plus': I_plus,
        'I_minus': I_minus,
        'delta_B_over_sqrt_4pi_rho': delta_B_over_sqrt_4pi_rho,
        'delta_v': delta_v,
        'flux_right': flux_right,
        'flux_left': flux_left,
        'normalized_flux_difference': normalized_flux_difference,
        
        # Ponderomotive acceleration
        'ponderomotive_acceleration': wave_components.ponderomotive_acceleration,
        'acceleration_cm_s2': acceleration_cm_s2,
        
        # Plasma parameters
        'VA_s': VA_s,
        'ne': loop_properties.ne(s_values),
        'nh': loop_properties.nh(s_values),
        'h_ionisation': loop_properties.h_ionisation(s_values),
        'hydrogen_metadata': loop_properties.hydrogen_metadata,
        'T': loop_properties.T(s_values),
        'thermal_speed_squared': thermal_speed_squared_dict,
        'B': B_s,
        'rho': rho_s,
        'collision_components_dict': collision_components_dict, 
        # Slow mode results
        'slow_mode_velocity': slow_mode_results,
        'sound_speed': sound_speed_dict,
        # 'slow_mode_density_perturbation': slow_mode_results['density_perturbation'],
        # 'slow_mode_pressure_perturbation': slow_mode_results['pressure_perturbation'],
        'acoustic_flux': acoustic_flux,
        'v_add': v_add,
        # FIP bias results for each element
        'fip_bias_heights': fip_bias_heights,
        
        # Ionization fractions for each element
        'ionization_fractions': ionization_fractions,
        'ionisation_model': 'Saha–CHIANTI 11.0.2 hybrid (CIE table 1e3–1e9 K)',
        
        # Collision frequencies for each element (if available)
        'collision_frequencies': collision_frequencies,
        'integral_terms': integral_terms,
        # Wave parameters
        'frequency': frequency,
        'amplitude': amplitude,
        'loop_length': loop_properties.s_array[-1],
    }

    # Update filename to include v_add (around line 818-819)
    flux_str = f'_F{acoustic_flux.value:.1e}' if acoustic_flux is not None else ''
    v_add_str = f'_v_add_{v_add.to(u.km/u.s).value:.1f}' if v_add is not None else ''
    filename = f'test_loop_extra_properties_{frequency.value:.3f}_{amplitude.value:.2f}_{strand_num}{flux_str}{v_add_str}.pkl'
    with open(output_dir + filename, 'wb') as f:
        pickle.dump(output_data, f)
    return output_data

def save_results(output_data, output_file):
    # Get the directory of the output file
    output_dir = os.path.dirname(output_file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create plots and save them in the same directory as the output file
    full_loop_plot = plot_full_loop(output_data)
    full_loop_plot_path = os.path.join(output_dir, f'full_loop_plot_f{output_data["frequency"].value:.3f}_A{output_data["amplitude"].value:.2f}_L{output_data["loop_length"].to(u.Mm).value:.2f}.png')
    with open(full_loop_plot_path, 'wb') as f:
        f.write(full_loop_plot)

    chromosphere_A_plot = plot_chromosphere(output_data, 'A')
    chromosphere_A_plot_path = os.path.join(output_dir, f'chromosphere_A_f{output_data["frequency"].value:.3f}_A{output_data["amplitude"].value:.2f}_L{output_data["loop_length"].to(u.Mm).value:.2f}.png')
    with open(chromosphere_A_plot_path, 'wb') as f:
        f.write(chromosphere_A_plot)

    chromosphere_B_plot = plot_chromosphere(output_data, 'B')
    chromosphere_B_plot_path = os.path.join(output_dir, f'chromosphere_B_f{output_data["frequency"].value:.3f}_A{output_data["amplitude"].value:.2f}_L{output_data["loop_length"].to(u.Mm).value:.2f}.png')
    with open(chromosphere_B_plot_path, 'wb') as f:
        f.write(chromosphere_B_plot)


    # Add plot paths to output_data
    output_data['full_loop_plot_path'] = full_loop_plot_path
    output_data['chromosphere_A_plot_path'] = chromosphere_A_plot_path
    output_data['chromosphere_B_plot_path'] = chromosphere_B_plot_path
    # output_data['ionization_plot_path'] = ionization_plot_path

    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

def main(input_file, frequencies, amplitudes, output_dir, strand_num):
    for frequency in frequencies:
        for amplitude in amplitudes:
            
            results = process_hydrad_data(input_file, frequency, amplitude, output_dir, strand_num)
            output_file = os.path.join(output_dir, f'output_f{frequency.value:.3f}_A{amplitude.value:.2f}_L{results["loop_length"].to(u.Mm).value:.2f}.pkl')
            # save_results(results, output_file)

def plot_full_loop(results):
    s_Mm = results['s_values'].to(u.Mm).value

    fig, axs = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle(f'Wave and Plasma Properties Along the Full Coronal Loop\nFrequency: {results["frequency"]:.3f}, Amplitude: {results["amplitude"]:.2f}, Loop Length: {results["loop_length"].to(u.Mm):.2f}', fontsize=16)

    # Elsässer variables
    axs[0, 0].plot(s_Mm, results['delta_B_over_sqrt_4pi_rho'].real.to(u.km/u.s).value, 'k-', label='δB/(4πρ)^1/2 (real)')
    axs[0, 0].plot(s_Mm, results['delta_v'].real.to(u.km/u.s).value, 'k--', label='δv (real)')
    axs[0, 0].plot(s_Mm, results['delta_B_over_sqrt_4pi_rho'].imag.to(u.km/u.s).value, 'gray', linestyle='-', label='δB/(4πρ)^1/2 (imag)')
    axs[0, 0].plot(s_Mm, results['delta_v'].imag.to(u.km/u.s).value, 'gray', linestyle='--', label='δv (imag)')
    axs[0, 0].set_ylabel('Elsässer variables (km/s)')
    axs[0, 0].legend()
    axs[0, 0].set_title('Elsässer Variables')

    # Wave energy fluxes
    axs[0, 1].semilogy(s_Mm, results['flux_right'], 'k-', label='|Flux Right|')
    axs[0, 1].semilogy(s_Mm, results['flux_left'], 'k--', label='|Flux Left|')
    axs[0, 1].semilogy(s_Mm, results['normalized_flux_difference'], 'k:', label='|Flux Difference/B|')
    axs[0, 1].set_ylabel('Wave Energy Flux (erg/cm²/s)')
    axs[0, 1].legend()
    axs[0, 1].set_title('Wave Energy Fluxes')

    # Acceleration
    acceleration = results['acceleration_cm_s2']
    positive_mask = acceleration >= 0
    negative_mask = acceleration < 0

    # Introduce NaN where the mask is False to break lines
    positive_acceleration = np.where(positive_mask, acceleration, np.nan)
    negative_acceleration = np.where(negative_mask, np.abs(acceleration), np.nan)

    axs[1, 0].semilogy(s_Mm, positive_acceleration, 'r-', label='Positive Acceleration')
    axs[1, 0].semilogy(s_Mm, negative_acceleration, 'b-', label='Negative Acceleration')

    axs[1, 0].set_ylabel('|Acceleration| (cm/s²)')
    axs[1, 0].legend()
    axs[1, 0].set_title('Acceleration (Absolute Value)')

    # Alfvén speed
    axs[1, 1].plot(s_Mm, results['VA_s'].to(u.km/u.s).value, 'b-', label='|Alfvén Speed|')
    axs[1, 1].set_ylabel('Speed (km/s)')
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()
    axs[1, 1].set_title('Alfvén Speed (Absolute Value)')

    # Electron Density and Temperature
    ax_ne = axs[2, 0]
    ax_T = ax_ne.twinx()
    ax_ne.semilogy(s_Mm, results['ne'], 'g-', label='Electron Density')
    ax_ne.set_ylabel('Electron Density (cm⁻³)', color='g')
    ax_ne.tick_params(axis='y', labelcolor='g')
    ax_T.semilogy(s_Mm, results['T'], 'm-', label='Temperature')
    ax_T.set_ylabel('Temperature (K)', color='m')
    ax_T.tick_params(axis='y', labelcolor='m')
    ax_ne.set_title('Electron Density and Temperature')
    lines_ne, labels_ne = ax_ne.get_legend_handles_labels()
    lines_T, labels_T = ax_T.get_legend_handles_labels()
    ax_ne.legend(lines_ne + lines_T, labels_ne + labels_T, loc='upper left')

    # Magnetic Field
    axs[2, 1].semilogy(s_Mm, results['B'], 'b-', label='Magnetic Field')
    axs[2, 1].set_ylabel('Magnetic Field (G)')
    axs[2, 1].set_title('Magnetic Field')
    axs[2, 1].legend()

    for ax in axs.flatten():
        ax.set_xlabel('Distance Along the Loop (Mm)')
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf.getvalue()

def plot_chromosphere(results, region_type):
    s_Mm = results['s_values'].to(u.Mm).value
    mask = (s_Mm >= 0) & (s_Mm <= 110)
    s_Mm_filtered = s_Mm[mask]

    fig, axs = plt.subplots(3, 3, figsize=(24, 18))
    region_name = "Chromosphere A (Injected)" if region_type == 'A' else "Chromosphere B"
    fig.suptitle(f'Wave and Plasma Properties in the {region_name}\nFrequency: {results["frequency"]:.3f}, Amplitude: {results["amplitude"]:.2f}, Loop Length: {results["loop_length"].to(u.Mm):.2f}', fontsize=16)

    # Elsässer variables
    plot_elsasser_variables(axs[0, 0], s_Mm_filtered, results, mask)

    # Wave energy fluxes
    plot_wave_energy_fluxes(axs[0, 1], s_Mm_filtered, results, mask)

    # Acceleration
    plot_acceleration(axs[0, 2], s_Mm, results['acceleration_cm_s2'][mask])

    # Alfvén speed
    plot_alfven_speed(axs[1, 0], s_Mm_filtered, results['VA_s'][mask])

    # Electron Density and Temperature
    plot_density_and_temperature(axs[1, 1], s_Mm_filtered, results, mask)

    # Magnetic Field
    plot_magnetic_field(axs[1, 2], s_Mm_filtered, results['B'][mask])

    # Ionization fractions
    plot_ionization(axs[2, 0], s_Mm_filtered, results, mask)

    for ax in axs.flatten():
        ax.set_xlabel('Distance Along the Loop (Mm)')
        ax.tick_params(axis='x', rotation=0)
        ax.set_xlim(94, 100) if region_type == 'A' else ax.set_xlim(0, 6)

    # Remove empty subplots
    fig.delaxes(axs[2, 1])
    fig.delaxes(axs[2, 2])

    plt.tight_layout()
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()

def plot_elsasser_variables(ax, s_Mm, results, mask):
    ax.plot(s_Mm, results['delta_B_over_sqrt_4pi_rho'].real[mask].to(u.km/u.s).value, 'k-', label='δB/(4πρ)^1/2 (real)')
    ax.plot(s_Mm, results['delta_v'].real[mask].to(u.km/u.s).value, 'k--', label='δv (real)')
    ax.plot(s_Mm, results['delta_B_over_sqrt_4pi_rho'].imag[mask].to(u.km/u.s).value, 'gray', linestyle='-', label='δB/(4πρ)^1/2 (imag)')
    ax.plot(s_Mm, results['delta_v'].imag[mask].to(u.km/u.s).value, 'gray', linestyle='--', label='δv (imag)')
    ax.set_ylabel('Elsässer variables (km/s)')
    ax.legend()
    ax.set_title('Elsässer Variables')

def plot_wave_energy_fluxes(ax, s_Mm, results, mask):
    ax.semilogy(s_Mm, results['flux_right'][mask], 'k-', label='|Flux Right|')
    ax.semilogy(s_Mm, results['flux_left'][mask], 'k--', label='|Flux Left|')
    ax.semilogy(s_Mm, results['normalized_flux_difference'][mask], 'k:', label='|Flux Difference/B|')
    ax.set_ylabel('Wave Energy Flux (erg/cm²/s)')
    ax.legend()
    ax.set_title('Wave Energy Fluxes')

def plot_acceleration(ax, s_Mm, acceleration):
    positive_mask = acceleration >= 0
    negative_mask = acceleration < 0
    positive_acceleration = np.where(positive_mask, acceleration, np.nan)
    negative_acceleration = np.where(negative_mask, np.abs(acceleration), np.nan)

    ax.semilogy(s_Mm, positive_acceleration, 'r-', label='Positive Acceleration')
    ax.semilogy(s_Mm, negative_acceleration, 'b-', label='Negative Acceleration')
    ax.set_ylabel('|Acceleration| (cm/s²)')
    ax.legend()
    ax.set_title('Acceleration (Absolute Value)')

def plot_alfven_speed(ax, s_Mm, VA_s):
    ax.plot(s_Mm, VA_s.to(u.km/u.s).value, 'b-', label='|Alfvén Speed|')
    ax.set_ylabel('Speed (km/s)')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Alfvén Speed (Absolute Value)')

def plot_density_and_temperature(ax, s_Mm, results, mask):
    ax_T = ax.twinx()
    ax.semilogy(s_Mm, results['ne'][mask], 'g-', label='Electron Density')
    ax.set_ylabel('Electron Density (cm⁻³)', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax_T.semilogy(s_Mm, results['T'][mask], 'm-', label='Temperature')
    ax_T.set_ylabel('Temperature (K)', color='m')
    ax_T.tick_params(axis='y', labelcolor='m')
    ax.set_title('Electron Density and Temperature')
    lines_ne, labels_ne = ax.get_legend_handles_labels()
    lines_T, labels_T = ax_T.get_legend_handles_labels()
    ax.legend(lines_ne + lines_T, labels_ne + labels_T, loc='upper left')

def plot_ionization(ax, s_Mm, results, mask):
    ax.semilogy(s_Mm, results['ionization_fractions']['C'][mask], 'blue', linestyle='-', label='C')
    ax.semilogy(s_Mm, results['ionization_fractions']['O'][mask], 'green', linestyle='--', label='O')
    ax.semilogy(s_Mm, results['ionization_fractions']['Mg'][mask], 'orange', label='Mg')
    ax.semilogy(s_Mm, results['ionization_fractions']['Si'][mask], 'purple', label='Si')
    ax.semilogy(s_Mm, results['ionization_fractions']['S'][mask], 'royalblue', label='S')
    ax.semilogy(s_Mm, results['ionization_fractions']['Fe'][mask], 'red', label='Fe')
    ax.set_ylabel('Ionization Fraction')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 1.0001)
    ax.legend()
    ax.set_title('Ionization Fraction')

def plot_magnetic_field(ax, s_Mm, B):
    ax.semilogy(s_Mm, B, 'b-', label='Magnetic Field')
    ax.set_ylabel('Magnetic Field (G)')
    ax.set_title('Magnetic Field')
    ax.legend()

if __name__ == "__main__":
    import sys
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    if len(sys.argv) != 4:
        print("Usage: python script.py input_file output_directory strand_num") 
        print("Example: python main_fip_bias.py run_test/val_c_initial.pkl run_20241017_171935/output 105")
        print("Example: python main_fip_bias.py test_data/hydrad test_data/output 1300")
        sys.exit(1)
    
    input_file = sys.argv[1]
    strand_num = int(sys.argv[3])

    output_dir = '/'.join(input_file.split('/')[:-1]) + '/fip-bias-output'
    
    # Define frequencies and amplitudes
    # frequencies = [0.07,0.071,0.072,0.073,0.074,0.075] * u.s**-1 # resonance frequency for starting VAL-C
    # amplitudes = [0.27] * u.km/u.s  
    # strand_nums = [700]
    # frequencies = np.linspace(0.01, 1.0, 50) * u.s**-1
    frequencies = [0.206] * u.s**-1 # resonance frequency for starting VAL-C


    # amplitudes = [0.97,0.85,0.30,0.169476,0.123802] * u.km/u.s  
    amplitudes = [0.97] * u.km/u.s  
    # amplitudes = [0.97] * u.km/u.s  
    strand_nums = [700]

        # Add array of flux values to test
    acoustic_fluxes = [
        # 1e9 * u.erg / u.cm**2 / u.s,
        1e8 * u.erg / u.cm**2 / u.s,
        # 1e7 * u.erg / u.cm**2 / u.s,
        # 1e6 * u.erg / u.cm**2 / u.s,
        # 1e5 * u.erg / u.cm**2 / u.s,  # default
        # 1e3 * u.erg / u.cm**2 / u.s,
    ]
    
    # Array of additional turbulence values to test
    v_add_values = [
        0.0 * u.km/u.s,      # No additional turbulence (baseline)
        1.0 * u.km/u.s,
        2.0 * u.km/u.s,
        5.0 * u.km/u.s,
        10.0 * u.km/u.s,
        20.0 * u.km/u.s,
        50.0 * u.km/u.s,
    ]

    # Define wrapper function
    def process_strand_safe(strand_num, frequency, amplitude, flux, v_add):
        try:
            results = process_hydrad_data(
                input_file, frequency, amplitude, 
                input_file, strand_num, acoustic_flux=flux, v_add=v_add
            )
            return strand_num, frequency.value, amplitude.value, flux.value, v_add.value, True
        except Exception as e:
            print(f"Error processing strand {strand_num}, freq {frequency}, amp {amplitude}, flux {flux}, v_add {v_add}: {e}")
            import traceback
            traceback.print_exc()
            return strand_num, frequency.value, amplitude.value, flux.value, v_add.value, False
    
    # Use joblib with loky backend
    n_jobs = 5
    
    print(f"Starting parallel processing with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(process_strand_safe)(strand_num, frequency, amplitude, flux, v_add)
        for strand_num in strand_nums
        for frequency in frequencies
        for amplitude in amplitudes
        for flux in acoustic_fluxes
        for v_add in v_add_values
    )    
    # Check for failures
    failed = [r[0] for r in results if not r[1]]
    if failed:
        print(f"Failed strands ({len(failed)}): {failed}")
    else:
        print("All strands processed successfully!")


        # python main_fip_bias_batch.py /Users/andysh.to/Script/Python_Script/fip_bias_hydrad/hydards/HYDRAD_flare_LL6e9_0.009_train_with_H_thermal_heating_2000s_1s_step_4_heatings /Users/andysh.to/Script/Python_Script/fip_bias_hydrad/hydards/HYDRAD_flare_LL6e9_0.009_train_with_H_thermal_heating_2000s_1s_step_4_heatings/output 0
