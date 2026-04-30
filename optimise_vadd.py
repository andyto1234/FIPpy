import sys
import os
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# 1. Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_fip_bias_batch import process_hydrad_data

# 2. Configuration
INPUT_DIR = '/Users/andysh.to/Script/Python_Script/fip_bias_hydrad/hydards/HYDRAD_flare_LL6e9_0.009_train_with_H_thermal_heating_2000s_1s_step_4_heatings'
OUTPUT_BASE = '/Users/andysh.to/Script/Python_Script/alfven_loop_SM/results/v_add_study/'

# Fixed parameters
STRAND_NUM = 700
FREQUENCY = 0.206 * u.s**-1
AMPLITUDE = 0.97 * u.km / u.s
ACOUSTIC_FLUX = 1e8 * u.erg / u.cm**2 / u.s
TARGET_HEIGHT = 57.5  # Mm

# v_add values to test (smooth curve from 0 to 50 km/s)
V_ADD_VALUES = np.arange(0, 51, 2) * u.km / u.s  # 0, 2, 4, ..., 50
# Parallel processing config
N_JOBS = 8  # Adjust based on your CPU cores

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

def run_single_v_add(v_add_value):
    """
    Run simulation for a single v_add value.
    Returns a dict with results or None if failed.
    """
    v_add = v_add_value * u.km / u.s
    v_add_str = f"{v_add.value:.2f}"
    
    print(f"[v_add {v_add_str} km/s] Running simulation...")
    
    # Create temp folder for this v_add
    temp_dir = os.path.join(OUTPUT_BASE, f"temp_v_add_{v_add_str}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Run simulation
        results = process_hydrad_data(
            INPUT_DIR,
            FREQUENCY,
            AMPLITUDE,
            temp_dir,
            strand_num=STRAND_NUM,
            acoustic_flux=ACOUSTIC_FLUX,
            v_add=v_add
        )
        
        fip_data = results['fip_bias_heights']
        
        row = {
            'v_add': v_add.value,
            'amplitude': AMPLITUDE.value,
            'frequency': FREQUENCY.value,
            'acoustic_flux': ACOUSTIC_FLUX.value
        }
        
        # Extract biases for all elements
        for elem in fip_data.keys():
            val, _ = get_bias_at_height(fip_data, elem, TARGET_HEIGHT)
            if val is not None:
                row[f'bias_{elem}'] = val
        
        print(f"[v_add {v_add_str} km/s] Done.")
        return row
        
    except Exception as e:
        print(f"[v_add {v_add_str} km/s] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_v_add_study_parallel():
    print(f"=== Starting v_add Parameter Study ===")
    print(f"Strand: {STRAND_NUM}")
    print(f"Frequency: {FREQUENCY.value:.3f} Hz")
    print(f"Amplitude: {AMPLITUDE.value:.2f} km/s")
    print(f"Acoustic Flux: {ACOUSTIC_FLUX.value:.1e} erg/cm²/s")
    print(f"Target Height: {TARGET_HEIGHT} Mm")
    print(f"Number of v_add values: {len(V_ADD_VALUES)}")
    print(f"Parallel jobs: {N_JOBS}")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Extract v_add values (without units) for parallel processing
    v_add_vals = [v.value for v in V_ADD_VALUES]
    
    # Run simulations in parallel
    results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single_v_add)(v) for v in v_add_vals
    )
    
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    # Save results
    if results_list:
        df = pd.DataFrame(results_list)
        save_path = os.path.join(OUTPUT_BASE, "v_add_study_results.csv")
        df.to_csv(save_path, index=False)
        print(f"\nStudy complete. Results saved to:\n{save_path}")
        print(f"\nSuccessful runs: {len(results_list)}/{len(V_ADD_VALUES)}")
        
        # Generate Plot
        plot_results(df, OUTPUT_BASE)
    else:
        print("\nNo results generated.")

def plot_results(df, save_dir):
    # Set plot parameters (matching your style)
    plt.rcParams['lines.linewidth'] = 2.6
    plt.rcParams['axes.linewidth'] = 1.9
    plt.rcParams['xtick.major.width'] = 1.9
    plt.rcParams['ytick.major.width'] = 1.9
    plt.rcParams['xtick.minor.width'] = 1.9
    plt.rcParams['ytick.minor.width'] = 1.9
    plt.rcParams['grid.linewidth'] = 1.8

    linewidths = 2.4
    title_size = 18
    label_size = 15
    legend_size = 12
    tick_size = 12

    # Element colors and linestyles
    element_colors = {
        'Ca': 'C0',  'Mg': 'C1',  'Fe': 'C2',  'Si': 'C3',
        'O': 'C4',   'S': 'C5',   'Ar': 'C6',  'C': 'C7',  'Ne': 'C8',  'N': 'C9'
    }

    element_linestyles = {
        'Ca': '-',   'Mg': '--',  'Fe': '-.',  'Si': ':',
        'O': '-',    'S': '--',   'Ar': '-.',  'C': ':',  'Ne': '-',  'N': '--'
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Sort by v_add
    df = df.sort_values('v_add')
    
    v_add = df['v_add'].values
    o_bias = df['bias_O'].values
    
    # Elements to plot (excluding O since it will be 1 after normalization)
    elements = ['Ca', 'Mg', 'Fe', 'Si', 'S', 'C', 'N', 'Ne', 'Ar']
    
    for element in elements:
        col = f'bias_{element}'
        if col in df.columns:
            bias = df[col].values
            bias_normalized = bias / o_bias
            
            color = element_colors.get(element, None)
            linestyle = element_linestyles.get(element, '-')
            
            ax.plot(v_add, bias_normalized, 
                    label=element, 
                    linestyle=linestyle, 
                    color=color, 
                    linewidth=linewidths)
    
    # Add horizontal line at y=1 (oxygen reference)
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='O (reference)')
    
    # Format axes
    ax.set_xlabel(r'$v_{\mathrm{add}}$ (km/s)', fontsize=label_size)
    ax.set_ylabel(f'FIP Bias (normalized by O) at {TARGET_HEIGHT} Mm', fontsize=label_size)
    ax.set_title(f'FIP Bias vs Turbulence ($v_{{\\mathrm{{add}}}}$)\n'
                 f'$F_{{ac}}$ = {ACOUSTIC_FLUX.value:.0e} erg/cm²/s, '
                 f'$I_0$ = {AMPLITUDE.value:.2f} km/s', fontsize=title_size)
    
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 6)
    
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=legend_size, ncol=2)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{save_dir}/fip_bias_vs_v_add_normalized_by_O.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_dir}/fip_bias_vs_v_add_normalized_by_O.png', bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {save_dir}/fip_bias_vs_v_add_normalized_by_O.png")

if __name__ == "__main__":
    run_v_add_study_parallel()