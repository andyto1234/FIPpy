import numpy as np 

def effective_collision_frequencies(v_ion, v_neutral, ionisation_fraction):
    """
    Calculate the effective collision frequency for a elemenetal species consisting of ions and neutrals.
    """
    v_eff = (v_ion*v_neutral)/(ionisation_fraction*v_neutral + (1-ionisation_fraction)*v_ion)
    return v_eff


def v_calculation(element,T, n_H, chi):
    """
    Calculate collision frequencies for ions and neutrals of a given element.
    Args:
    Z: atomic number of the element
    A: number of nucleons
    n_H: number density of hydrogen (1e10 cm^-3)
    chi: fraction of hydrogen which is ionized (Marsch et al. 1995) - Avrett nH1/nH?
    alpha: 
    r: (Table 1. Marsch et al. 1995)
    """
    Z, A = get_atomic_data(element)
    data = get_Marsch1995_data(element)
    alpha = data[3]
    r = data[2]
    v_ion = 1.1e4 * Z^2 / np.sqrt(A*(A+1)) * (1e4/T) ** (3/2) * n_H * chi + (20.74/np.sqrt(A*(A+1)))*(n_H)*(1-chi)
    v_neutral = 25.4*np.sqrt(alpha/(A*(A+1)))*n_H*chi + 6.09*(r**2)/np.sqrt(A*(A+1))*n_H*(1-chi)
    return v_ion, v_neutral
        

# def wave_calculation():
#     """
#     Laming et al. 2015 - page 29
    
#     """
#     I_plus = delta_v + delta_b/np.sqrt(4*np.pi*rho)
def ponderomotive_acceleration(v_ion, v_neutral, ionisation_fraction):
    """
    Laming et al. 2015 - page 25
    need to first get wave electric field - this requires the wave calculation
    """
    a = c**2/4 




def get_Marsch1995_data(element):
    # Dictionary to store element data from Marsch et al. 1995
    # element[4] = vj [km/s]
    elements = {
        'H': [13.60, 65, None, 0.667, 9.09, 8.59, None, None, 0.132],
        'He': [24.59, 227, 1.56, 0.205, 5.25, 38.3, 89.4, 0.950, 0.169],
        'C': [11.26, 27, 2.34, 1.76, 2.62, 8.35, 22.8, 0.836, 0.309],
        'N': [14.53, 68, 2.11, 1.10, 2.43, 14.7, 20.1, 0.827, 0.216],
        'O': [13.62, 62, 2.26, 0.802, 2.27, 13.0, 17.9, 0.817, 0.210],
        'Ne': [21.56, 81, 1.75, 0.396, 2.03, 19.2, 14.6, 0.806, 0.237],
        'Mg': [7.65, 0.78, 3.09, 10.6, 1.85, 1.06, 11.2, 0.794, 1.36],
        'Si': [8.15, 1.1, 2.91, 5.38, 1.72, 1.34, 9.99, 0.784, 1.22],
        'S': [10.36, 11.6, 2.58, 2.90, 1.61, 4.90, 8.97, 0.774, 0.423],
        'Ar': [15.76, 50, 2.07, 1.64, 1.51, 9.09, 8.10, 0.765, 0.254],
        'Ca': [6.11, 0.70, 3.64, 22.8, 1.44, 0.853, 6.78, 0.757, 1.22],
        'Fe': [7.87, 0.91, 3.09, 9.4, 1.21, 1.14, 5.13, 0.725, 1.26]
    }
    
    # Check if the element exists in our dictionary
    if element not in elements:
        return f"Element '{element}' not found in the database."
    
    # Get the data for the requested element
    data = elements[element]
    
    # Create a dictionary with labeled parameters
    parameters = {
        'FIP [eV]': data[0],
        'tauj [s]': data[1],
        'rjH [Å]': data[2],
        'alpha_j [Å²]': data[3],
        'vj [km/s]': data[4],
        'lambda_j [km]': data[5],
        'Hj [km]': data[6],
        'Hj+ [km]': data[7],
        'wj [km/s]': data[8]
    }
    
    return parameters

# # Example usage
# element = input("Enter an element symbol (e.g., 'Ca'): ")
# result = get_element_parameters(element)
# print(result)

def get_atomic_data(element):
    element_data = {
        'H':  (1, 1),
        'He': (2, 4),
        'C':  (6, 12),
        'N':  (7, 14),
        'O':  (8, 16),
        'Ne': (10, 20),
        'Mg': (12, 24),
        'Si': (14, 28),
        'S':  (16, 32),
        'Ar': (18, 40),
        'Ca': (20, 40),
        'Fe': (26, 56)
    }
    
    if element in element_data:
        atomic_number, mass_number = element_data[element]
        return f"Element: {element}\nAtomic Number: {atomic_number}\nNumber of Nucleons: {mass_number}"
    else:
        return f"Element '{element}' not found in the database."

# # Example usage
# element = input("Enter an element symbol (e.g., 'Ca'): ")
# result = get_atomic_data(element)
# print(result)

def alfven_wave_system(z, y, omega, u, VA, HD, HA):
    """
    Laming et al. 2015 - page 29 - eq 11
    Calculate the Alfven wave integration 
    I± = R± + iS±
    """
    R_plus, S_plus, R_minus, S_minus = y
    
    dR_plus_dz = ((u + VA) * (R_plus/(4*HD) + R_minus/(2*HA)) + omega * S_plus) / (u - VA)
    dS_plus_dz = ((u + VA) * (S_plus/(4*HD) + S_minus/(2*HA)) - omega * R_plus) / (u - VA)
    
    dR_minus_dz = ((u - VA) * (R_minus/(4*HD) + R_plus/(2*HA)) + omega * S_minus) / (u + VA)
    dS_minus_dz = ((u - VA) * (S_minus/(4*HD) + S_plus/(2*HA)) - omega * R_minus) / (u + VA)
    
    return [dR_plus_dz, dS_plus_dz, dR_minus_dz, dS_minus_dz]


def calculate_alfven_frequency(loop_length, alfven_speed):
    """
    Calculate the frequency and angular frequency of the fundamental mode Alfvén wave in a loop.
    
    Args:
    loop_length (float): The length of the magnetic loop (in meters)
    alfven_speed (float): The Alfvén speed in the loop (in meters per second)
    
    Returns:
    tuple: (frequency in Hz, angular frequency in rad/s)
    # Example usage:
    L = 1e8  # Loop length of 100 Mm (100,000 km) in meters
    VA = 1e6  # Alfvén speed of 1000 km/s in m/s

    f, omega = calculate_alfven_frequency(L, VA)
    """
    frequency = alfven_speed / (2 * loop_length)
    angular_frequency = 2 * np.pi * frequency
    
    return frequency, angular_frequency



def calculate_HB(Bz, z):
    """
    Calculate the magnetic field scale height HB.
    
    Args:
    Bz (array-like): Vertical component of the magnetic field
    z (array-like): Height coordinate
    
    Returns:
    array-like: Scale height HB
    """
    dBz_dz = np.gradient(Bz, z)
    dlnBz_dz = dBz_dz / Bz
    return 1 / dlnBz_dz

def calculate_HD(rho, z):
    """
    Calculate the density scale height HD.
    
    Args:
    rho (array-like): Density
    z (array-like): Height coordinate
    
    Returns:
    array-like: Scale height HD
    """
    drho_dz = np.gradient(rho, z)
    dlnrho_dz = drho_dz / rho
    return 1 / dlnrho_dz

def calculate_HA(VA, z):
    """
    Calculate the Alfvén speed scale height HA.
    
    Args:
    VA (array-like): Alfvén speed
    z (array-like): Height coordinate
    
    Returns:
    array-like: Scale height HA
    """
    dVA_dz = np.gradient(VA, z)
    dlnVA_dz = dVA_dz / VA
    return 1 / dlnVA_dz