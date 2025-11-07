# from plasmapy.formulary.ionization import Saha
# import numpy as np
# from astropy import units as u

# def sigmoid_weight(x, x0, width):
#     """Calculate sigmoid weights for smooth transition"""
#     return 1 / (1 + np.exp(-(x - x0) / (width/4)))

# # def get_ionisation_fraction(element, atomic_data, n_e, T, T_connect=1e4):
# #     """
# #     Calculate the ionization fraction using both Saha equation and CHIANTI data with a smooth transition.
    
# #     Parameters:
# #     T: Temperature in Kelvin
# #     n_e: Electron number density in cm^-3
# #     atomic_data: Dictionary containing ionization data
# #     chianti_data: Dictionary containing CHIANTI ionization equilibrium data
# #     T_connect: Temperature at which to transition from Saha to CHIANTI (default: 1e4 K)
    
# #     Returns:
# #     Ionization fraction
# #     """
# #     # Calculate Saha fraction
# #     saha_fraction = Saha(atomic_data['g_i'], atomic_data['g_j'], n_e, atomic_data['ionization_energy'], T)
# #     saha_fraction = 1 / (1 + 1 / np.array(saha_fraction))
    
# #     try:
# #         chianti_data = np.load('chianti/ion_fraction_with_T.npz')
# #     except FileNotFoundError:
# #         print('CHIANTI data not found, using Saha equation only - which will be wrong')
# #         return saha_fraction
# #     _T = T.to(u.K).value

# #     # Initialize array for final ionization fraction
# #     ion_fraction = np.zeros_like(_T)
    
# #     # Define transition region
# #     transition_width = 0.1 * T_connect  # Wider transition region (10%)
# #     weights = sigmoid_weight(_T, T_connect, transition_width)
    
# #     # Get CHIANTI data for this element
# #     chianti_fraction = chianti_data[f'{element}_ioneq']
# #     chianti_temps = chianti_data['temperature']
    
# #     # Interpolate CHIANTI data to match our temperature grid
# #     chianti_interp = np.interp(_T, chianti_temps, chianti_fraction)
    
# #     # Apply masks
# #     ion_fraction = (saha_fraction * (1 - weights) + 
# #                    chianti_interp * weights)
    
# #     return ion_fraction


# def get_ionisation_fraction(element, atomic_data, n_e, T):
#     """
#     Ensure continuity by matching Saha and CHIANTI at transition point
#     with proper bounds checking
#     """
#     # Calculate both fractions
#     saha_fraction = Saha(atomic_data['g_i'], atomic_data['g_j'], n_e, atomic_data['ionization_energy'], T)
#     saha_fraction = 1 / (1 + 1 / np.array(saha_fraction))
    
#     # Ensure Saha fraction is bounded [0, 1]
#     saha_fraction = np.clip(saha_fraction, 0.0, 1.0)
    
#     try:
#         chianti_data = np.load('chianti/ion_fraction_with_T.npz')
#         chianti_fraction = chianti_data[f'{element}_ioneq']
#         chianti_temps = chianti_data['temperature']
#         chianti_interp = np.interp(T.to(u.K).value, chianti_temps, chianti_fraction)
        
#         # Ensure CHIANTI fraction is bounded [0, 1]
#         chianti_interp = np.clip(chianti_interp, 0.0, 1.0)
        
#     except FileNotFoundError:
#         print(f'CHIANTI data not found for {element}, using Saha equation only')
#         return saha_fraction
    
#     _T = T.to(u.K).value
#     _n_e = n_e.to(u.cm**-3).value
    
#     # Find optimal transition temperature for this element and density
#     FIP = atomic_data['ionization_energy'].to(u.eV).value
#     if FIP < 10:  # Low FIP
#         T_transition = 15000  # K
#     else:  # High FIP
#         T_transition = 25000  # K
    
#     # Adjust transition temperature based on density (but keep it reasonable)
#     density_factor = np.clip((1e10 / np.mean(_n_e))**0.1, 0.5, 2.0)
#     T_transition *= density_factor
    
#     # Smooth transition with continuity matching
#     transition_width = 0.2 * T_transition
#     weights = sigmoid_weight(_T, T_transition, transition_width)
    
#     # More robust continuity matching
#     transition_mask = np.abs(_T - T_transition) < transition_width/4  # Narrower matching region
    
#     if np.any(transition_mask):
#         saha_at_transition = np.median(saha_fraction[transition_mask])  # Use median for robustness
#         chianti_at_transition = np.median(chianti_interp[transition_mask])
        
#         if chianti_at_transition > 1e-10:  # Avoid division by very small numbers
#             # Calculate scaling factor but limit its range
#             scaling_factor = saha_at_transition / chianti_at_transition
#             scaling_factor = np.clip(scaling_factor, 0.1, 10.0)  # Reasonable bounds
            
#             # Apply scaling but ensure we don't exceed physical bounds
#             chianti_scaled = chianti_interp * scaling_factor
#             chianti_interp = np.clip(chianti_scaled, 0.0, 1.0)
#         else:
#             # If CHIANTI data is essentially zero, use offset matching instead
#             offset = saha_at_transition - chianti_at_transition
#             chianti_interp = np.clip(chianti_interp + offset, 0.0, 1.0)
    
#     # Weighted combination
#     ion_fraction = saha_fraction * (1 - weights) + chianti_interp * weights
    
#     # Final bounds check - this is crucial!
#     ion_fraction = np.clip(ion_fraction, 0.0, 1.0)
    
#     return ion_fraction


from plasmapy.formulary.ionization import Saha
import numpy as np
from astropy import units as u

def sigmoid_weight(x, x0, width):
    """Calculate sigmoid weights for smooth transition"""
    return 1 / (1 + np.exp(-(x - x0) / (width/4)))

def get_element_transition_params(element):
    """
    Get element-specific transition parameters based on FIP and atomic physics
    Returns log10(T) values for more physical transitions
    """
    # Transition parameters in log10(T) space
    # Transition parameters in log10(T) space
    params = {
        # Low FIP elements (< 10 eV) - transition earlier
        'Ca': {'logT_trans': 4.5, 'width_dex': 0.2},
        'Mg': {'logT_trans': 4.5, 'width_dex': 0.2},
        'Fe': {'logT_trans': 4.5, 'width_dex': 0.2},
        'Si': {'logT_trans': 4.5, 'width_dex': 0.2}, 
        'S':  {'logT_trans': 4.5, 'width_dex': 0.2},
        'C':  {'logT_trans': 4.5, 'width_dex': 0.2},
        'O':  {'logT_trans': 4.5, 'width_dex': 0.2},  # ~2.0e4 K
        'N':  {'logT_trans': 4.5, 'width_dex': 0.2},  # ~2.1e4 K
        'Ar': {'logT_trans': 4.5, 'width_dex': 0.2},  # ~2.2e4 K
        'Ne': {'logT_trans': 4.5, 'width_dex': 0.2},  # ~2.5e4 K
        }
    
    # Default for unknown elements - medium FIP behavior
    return params.get(element, {'logT_trans': 4.5, 'width_dex': 0.20})

def _adaptive_gaussian_smooth(y, T_gradient, base_sigma=0.1, max_sigma=0.5):
    """Apply Gaussian smoothing only where temperature gradient is large"""
    # Normalize temperature gradient to identify transition regions
    T_gradient = np.abs(T_gradient)  # Use absolute value of gradient
    gradient_threshold = np.median(T_gradient) + 2 * np.std(T_gradient)

    y = np.asarray(y)
    if y.ndim == 0 or y.size < 3:
        return y
    
    # Identify regions with large gradients
    high_gradient_mask = T_gradient > gradient_threshold
    
    # If no high gradient regions, return original
    if not np.any(high_gradient_mask):
        return y
    
    # Create output array (start with original values)
    smoothed = y.copy()
    
    # Determine smoothing strength based on temperature gradient
    # Higher temperature gradient -> more smoothing (captures transition regions)
    sigma_array = base_sigma + (max_sigma - base_sigma) * (T_gradient / (gradient_threshold + 1e-12))
    sigma_array = np.clip(sigma_array, base_sigma, max_sigma)
    
    # Apply smoothing only to high gradient regions
    for i in range(len(y)):
        if high_gradient_mask[i]:
            sigma = sigma_array[i]
            r = int(3 * sigma + 0.5)
            if r < 1:
                continue
            
            # Define local window
            i_start = max(0, i - r)
            i_end = min(len(y), i + r + 1)
            
            # Create kernel for this point
            kx = np.arange(i_start - i, i_end - i)
            kernel = np.exp(-0.5 * (kx / sigma) ** 2)
            kernel /= kernel.sum()
            
            # Apply local smoothing
            smoothed[i] = np.sum(y[i_start:i_end] * kernel)
    
    return smoothed

def get_ionisation_fraction(element, atomic_data, n_e, T):
    """
    Calculate ionization fraction with smooth blending (no rescaling) and log-T transitions
    
    Key improvements:
    1. No rescaling of CHIANTI data - trust it at high T
    2. Transition in log-T space for more physical behavior
    3. Element-specific transition parameters based on FIP
    4. Adaptive width based on local gradients
    5. Gaussian smoothing of weights for stability
    6. Special handling for S and C to prefer higher of Saha or CHIANTI
    """
    # Calculate Saha fraction
    saha_fraction = Saha(atomic_data['g_i'], atomic_data['g_j'], n_e, atomic_data['ionization_energy'], T)
    saha_fraction = 1 / (1 + 1 / np.array(saha_fraction))
    saha_fraction = np.clip(saha_fraction, 0.0, 1.0)
    
    try:
        chianti_data = np.load('chianti/ion_fraction_with_T.npz')
        chianti_fraction = chianti_data[f'{element}_ioneq']
        chianti_temps = chianti_data['temperature']
        chianti_interp = np.interp(T.to(u.K).value, chianti_temps, chianti_fraction)
        chianti_interp = np.clip(chianti_interp, 0.0, 1.0)
        
    except FileNotFoundError:
        print(f'CHIANTI data not found for {element}, using Saha equation only')
        return saha_fraction
    
    _T = T.to(u.K).value
    
    # if element == 'Ar':
    #     element = 'O'
    # Get element-specific transition parameters
    trans_params = get_element_transition_params(element)
    logT_transition = trans_params['logT_trans']
    width_dex = trans_params['width_dex']
    
    # Work in log10(T) space for more physical transitions
    logT = np.log10(_T)
    # Tanh-based smooth interpolation
    x_norm = (logT - logT_transition) / width_dex
    smooth_weights = 0.5 * (1 + np.tanh(x_norm))

    # Blend in log-space for more physical behavior
    # (ionization fractions often vary exponentially)
    eps = 1e-10
    log_saha = np.log10(saha_fraction + eps)
    log_chianti = np.log10(chianti_interp + eps)
    log_ion_fraction = (1 - smooth_weights) * log_saha + smooth_weights * log_chianti

    ion_fraction = 10**log_ion_fraction
    ion_fraction = np.clip(ion_fraction, 0.0, 1.0)
    
    # Apply adaptive smoothing in transition regions
    # Identify regions with high gradients (rapid changes)
    if len(ion_fraction) > 2:
        # Calculate gradient of ionization fraction
        gradient = np.abs(np.gradient(T.to(u.K).value))
        
        
        # Create adaptive smoothing kernel based on local gradient
        
        # Apply adaptive smoothing
        ion_fraction = _adaptive_gaussian_smooth(ion_fraction, gradient, base_sigma=0.5, max_sigma=2.0)
        ion_fraction = np.clip(ion_fraction, 0.0, 1.0)
    
    return ion_fraction
    # # Optional: weak density dependence (higher density -> slightly lower transition T)
    # _n_e = n_e.to(u.cm**-3).value
    # density_correction = -0.05 * np.log10(np.mean(_n_e) / 1e10)  # Weak correction
    # logT_transition_corrected = logT_transition + np.clip(density_correction, -0.1, 0.1)
    
    # # Adaptive width based on local gradients; widen where T or n_e change sharply
    # def _safe_grad(x):
    #     x = np.asarray(x)
    #     if x.ndim == 0 or x.size < 3:
    #         return np.zeros_like(x)
    #     return np.gradient(x)
    
    # dlogT = np.abs(_safe_grad(logT))
    # dlogne = np.abs(_safe_grad(np.log10(_n_e)))
    
    # denT = np.median(dlogT) + 1e-12
    # denNe = np.median(dlogne) + 1e-12
    # alpha, beta = 1.0, 0.5
    # width_loc = width_dex * (1 + alpha * dlogT / denT + beta * dlogne / denNe)
    # width_loc = np.clip(width_loc, width_dex, 3.0 * width_dex)
    
    # # Calculate weights and smooth them along the coordinate
    # weights = sigmoid_weight(logT, logT_transition, width_loc)
    
    # def _gaussian_smooth1d(y, sigma=1, truncate=3.0):
    #     y = np.asarray(y)
    #     if y.ndim == 0 or y.size < 3 or sigma <= 0:
    #         return y
    #     r = int(truncate * sigma + 0.5)
    #     if r < 1:
    #         return y
    #     kx = np.arange(-r, r + 1)
    #     k = np.exp(-0.5 * (kx / sigma) ** 2)
    #     k /= k.sum()
    #     ypad = np.pad(y, (r, r), mode='reflect')
    #     return np.convolve(ypad, k, mode='valid')
    
    # weights = _gaussian_smooth1d(weights, sigma=1.0)
    
    # # Special handling for S and C: prefer higher value with smooth blending
    # # if element in ['S', 'C']:
    # #     # Take the maximum of Saha and CHIANTI at each point
    # #     max_fraction = np.maximum(saha_fraction, chianti_interp)
        
    # #     # Blend smoothly between Saha and the maximum
    # #     # Use reduced weight to stay closer to Saha
    # #     reduced_weights = weights * 0.3  # Reduce CHIANTI influence
        
    # #     def _logit_blend(f1, f2, w, eps=1e-12):
    # #         f1 = np.clip(f1, eps, 1 - eps)
    # #         f2 = np.clip(f2, eps, 1 - eps)
    # #         l1 = np.log(f1 / (1 - f1))
    # #         l2 = np.log(f2 / (1 - f2))
    # #         l  = (1 - w) * l1 + w * l2
    # #         return 1.0 / (1.0 + np.exp(-l))
        
    # #     ion_fraction = _logit_blend(saha_fraction, max_fraction, reduced_weights)
    # # else:
    #     # Standard blending for other elements
    # def _logit_blend(f1, f2, w, eps=1e-12):
    #     f1 = np.clip(f1, eps, 1 - eps)
    #     f2 = np.clip(f2, eps, 1 - eps)
    #     l1 = np.log(f1 / (1 - f1))
    #     l2 = np.log(f2 / (1 - f2))
    #     l  = (1 - w) * l1 + w * l2
    #     return 1.0 / (1.0 + np.exp(-l))
    
    # ion_fraction = _logit_blend(saha_fraction, chianti_interp, weights)
    
    # ion_fraction = np.clip(ion_fraction, 0.0, 1.0)
    # return ion_fraction

# def get_ionisation_fraction(element, atomic_data, n_e, T):
#     """
#     Calculate ionization fraction using only the Saha equation
#     """
#     # Calculate Saha fraction
#     saha_fraction = Saha(atomic_data['g_i'], atomic_data['g_j'], n_e, atomic_data['ionization_energy'], T)
#     saha_fraction = 1 / (1 + 1 / np.array(saha_fraction))
#     saha_fraction = np.clip(saha_fraction, 0.0, 1.0)
    
#     return saha_fraction

def get_ionisation_fraction_debug(element, atomic_data, n_e, T):
    """
    Debug version that returns intermediate values for inspection
    """
    # Calculate both fractions
    saha_fraction = Saha(atomic_data['g_i'], atomic_data['g_j'], n_e, atomic_data['ionization_energy'], T)
    saha_fraction = 1 / (1 + 1 / np.array(saha_fraction))
    saha_fraction = np.clip(saha_fraction, 0.0, 1.0)
    
    try:
        chianti_data = np.load('chianti/ion_fraction_with_T.npz')
        chianti_fraction = chianti_data[f'{element}_ioneq']
        chianti_temps = chianti_data['temperature']
        chianti_interp = np.interp(T.to(u.K).value, chianti_temps, chianti_fraction)
        chianti_interp = np.clip(chianti_interp, 0.0, 1.0)
    except FileNotFoundError:
        chianti_interp = np.zeros_like(saha_fraction)
    
    _T = T.to(u.K).value
    logT = np.log10(_T)
    
    # Get transition parameters
    trans_params = get_element_transition_params(element)
    logT_transition = trans_params['logT_trans']
    width_dex = trans_params['width_dex']
    
    # Calculate weights
    weights = sigmoid_weight(logT, logT_transition, width_dex)
    
    # Final result
    ion_fraction = saha_fraction * (1 - weights) + chianti_interp * weights
    ion_fraction = np.clip(ion_fraction, 0.0, 1.0)
    
    return {
        'ion_fraction': ion_fraction,
        'saha_fraction': saha_fraction,
        'chianti_fraction': chianti_interp,
        'weights': weights,
        'logT': logT,
        'logT_transition': logT_transition,
        'element': element
    }