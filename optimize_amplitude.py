import sys
import os
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from joblib import Parallel, delayed

# 1. Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_fip_bias_batch import process_hydrad_data

# 2. Configuration
INPUT_DIR = '/Users/andysh.to/Script/Python_Script/fip_bias_hydrad/hydards/HYDRAD_flare_LL6e9_0.009_train_with_H_thermal_heating_2000s_1s_step_4_heatings'
OUTPUT_BASE = '/Users/andysh.to/Script/Python_Script/alfven_loop_SM/results/optimized_flux_study/'

# Parameters
STRAND_NUM = 700
FREQUENCY = 0.206 * u.s**-1
TARGET_RATIO = 4.0
TARGET_HEIGHT = 57.5  # Mm

# Fluxes to test (Smooth curve)
ACOUSTIC_FLUXES = np.logspace(3, 9, 30) * u.erg / u.cm**2 / u.s

# Parallel processing config
N_JOBS = 8 # Adjust based on your CPU cores

def get_bias_at_height(fip_data, element, target_height_mm):
    """
    Finds the FIP bias for a given element at the target height
    by searching both 'left' and 'right' segments.
    """
    best_dist = float('inf')
    best_val = None
    
    for segment in ['left', 'right']:
        if segment in fip_data[element]:
            heights = fip_data[element][segment]['heights']
            biases = fip_data[element][segment]['bias']
            
            if len(heights) > 0:
                idx = np.argmin(np.abs(heights - target_height_mm))
                dist = np.abs(heights[idx] - target_height_mm)
                
                if dist < best_dist:
                    best_dist = dist
                    best_val = biases[idx]
    
    return best_val, best_dist

def run_simulation(amplitude_val, frequency, flux, strand_num, temp_dir):
    """Run a single simulation and return Ca/O ratio error."""
    amplitude = amplitude_val * u.km / u.s
    
    results = process_hydrad_data(
        INPUT_DIR,
        frequency,
        amplitude,
        temp_dir,
        strand_num=strand_num,
        acoustic_flux=flux
    )
    
    fip_data = results['fip_bias_heights']
    
    if 'Ca' in fip_data and 'O' in fip_data:
        ca_bias, _ = get_bias_at_height(fip_data, 'Ca', TARGET_HEIGHT)
        o_bias, _ = get_bias_at_height(fip_data, 'O', TARGET_HEIGHT)
        
        if ca_bias is None or o_bias is None:
            return None, None
        
        current_ratio = ca_bias / o_bias
        return current_ratio - TARGET_RATIO, results
    
    return None, None

def optimize_single_flux(flux_value):
    """
    Optimize amplitude for a single flux value.
    Returns a dict with results or None if failed.
    """
    flux = flux_value * u.erg / u.cm**2 / u.s
    flux_val_sci = f"{flux.value:.1e}"
    
    print(f"[Flux {flux_val_sci}] Starting optimization...")
    
    # Create temp folder for this flux
    temp_dir = os.path.join(OUTPUT_BASE, f"temp_opt_F{flux_val_sci}")
    os.makedirs(temp_dir, exist_ok=True)
    
    min_amp = 0.05
    max_amp = 3.0
    
    try:
        # Check bounds
        diff_min, _ = run_simulation(min_amp, FREQUENCY, flux, STRAND_NUM, temp_dir)
        diff_max, _ = run_simulation(max_amp, FREQUENCY, flux, STRAND_NUM, temp_dir)
        
        if diff_min is None or diff_max is None:
            print(f"[Flux {flux_val_sci}] Simulation failed at boundary check.")
            return None
            
        if np.sign(diff_min) == np.sign(diff_max):
            print(f"[Flux {flux_val_sci}] Solution not bracketed. Diff at min: {diff_min:.2f}, max: {diff_max:.2f}")
            return None
        
        # Define objective for brentq (closure over flux)
        def objective(amp_val):
            diff, _ = run_simulation(amp_val, FREQUENCY, flux, STRAND_NUM, temp_dir)
            if diff is None:
                raise ValueError("Simulation failed")
            return diff
        
        # Optimize
        optimal_amp = brentq(objective, min_amp, max_amp, xtol=0.01)
        print(f"[Flux {flux_val_sci}] Optimal amplitude: {optimal_amp:.4f} km/s")
        
        # Extract final data
        _, final_res = run_simulation(optimal_amp, FREQUENCY, flux, STRAND_NUM, temp_dir)
        fip_data = final_res['fip_bias_heights']
        
        row = {
            'flux': flux.value,
            'amplitude': optimal_amp,
            'frequency': FREQUENCY.value
        }
        
        for elem in fip_data.keys():
            val, _ = get_bias_at_height(fip_data, elem, TARGET_HEIGHT)
            if val is not None:
                row[f'bias_{elem}'] = val
        
        return row
        
    except Exception as e:
        print(f"[Flux {flux_val_sci}] Optimization failed: {e}")
        return None

def optimize_fluxes_parallel():
    print(f"=== Starting Parallel Optimization ===")
    print(f"Strand: {STRAND_NUM}")
    print(f"Frequency: {FREQUENCY.value:.3f} Hz")
    print(f"Target Ca/O Ratio: {TARGET_RATIO}")
    print(f"Target Height: {TARGET_HEIGHT} Mm")
    print(f"Number of fluxes: {len(ACOUSTIC_FLUXES)}")
    print(f"Parallel jobs: {N_JOBS}")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Extract flux values (without units) for parallel processing
    flux_values = [f.value for f in ACOUSTIC_FLUXES]
    
    # Run optimizations in parallel
    results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(optimize_single_flux)(fv) for fv in flux_values
    )
    
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    # Save results
    if results_list:
        df = pd.DataFrame(results_list)
        save_path = os.path.join(OUTPUT_BASE, "optimal_amplitudes_smooth.csv")
        df.to_csv(save_path, index=False)
        print(f"\nOptimization complete. Results saved to:\n{save_path}")
        print(f"\nSuccessful optimizations: {len(results_list)}/{len(ACOUSTIC_FLUXES)}")
        
        # Generate Plot
        plot_results(df, OUTPUT_BASE)
    else:
        print("\nNo results generated.")

def plot_results(df, save_dir):
    plt.figure(figsize=(10, 8))
    
    bias_cols = [c for c in df.columns if c.startswith('bias_')]
    df = df.sort_values('flux')
    
    for col in bias_cols:
        elem = col.replace('bias_', '')
        plt.loglog(df['flux'], df[col], 'o-', label=elem)
        
    plt.xlabel('Acoustic Flux (erg cm$^{-2}$ s$^{-1}$)')
    plt.ylabel(f'FIP Bias at {TARGET_HEIGHT} Mm')
    plt.title(f'FIP Bias vs Acoustic Flux (Optimized for Ca/O={TARGET_RATIO})')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "fip_bias_vs_flux.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    optimize_fluxes_parallel()