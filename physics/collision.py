import numpy as np
from dataclasses import dataclass
import astropy.units as u
from physics.saha import get_ionisation_fraction

@dataclass
class CollisionComponents:
    v_ion: np.ndarray
    v_neutral: np.ndarray
    v_eff: np.ndarray
    ionisation_fraction: np.ndarray
    v_s: np.ndarray
    atomic_number : int
    mass_number : float
    

class CollisionCalculator:
    def __init__(self, loop_properties, element):
        self.loop_properties = loop_properties
        self.temperature = loop_properties.T
        self.element = element
        self.Marsch1995 = CollisionCalculator.get_Marsch1995_data(self.element)
        self.atomic_data = self.get_element_properties(self.element)
        
    @staticmethod
    def get_Marsch1995_data(element):
    # Dictionary to store element data from Marsch et al. 1995
    # element[4] = vj [km/s]
        elements = {
            # 'H':  [13.60,  65,   None,  0.667, 9.09, 8.59,  None, None, 0.132],
            # 'He': [24.59,  227,  1.56,  0.205, 5.25, 38.3,  89.4, 0.950, 0.169],
            # 'C':  [11.26,  27,   2.34,  1.76,  2.62, 8.35,  22.8, 0.836, 0.309],
            # 'N':  [14.53,  68,   2.11,  1.10,  2.43, 14.7,  20.1, 0.827, 0.216],
            # 'O':  [13.62,  62,   2.26,  0.802, 2.27, 13.0,  17.9, 0.817, 0.210],
            # 'Ne': [21.56,  81,   1.75,  0.396, 2.03, 19.2,  14.6, 0.806, 0.237],
            # 'Mg': [7.65,   0.78, 3.09,  10.6,  1.85, 1.06,  11.2, 0.794, 1.36],
            # 'Si': [8.15,   1.1,  2.91,  5.38,  1.72, 1.34,  9.99, 0.784, 1.22],
            # 'S':  [10.36,  11.6, 2.58,  2.90,  1.61, 4.90,  8.97, 0.774, 0.423],
            # 'Ar': [15.76,  50,   2.07,  1.64,  1.51, 9.09,  8.10, 0.765, 0.254],
            # 'Ca': [6.11,   0.70, 3.64,  22.8,  1.44, 0.853, 6.78, 0.757, 1.22],
            # 'Fe': [7.87,   0.91, 3.09,  9.4,   1.21, 1.14,  5.13, 0.725, 1.26]

            # Marsch 1995 + Laming 2004 values + Schwardon turbulence speed
            'H':  [13.60,  65,   None,  0.667, 3, 8.59,  None, None, 0.132],
            'He': [24.59,  227,  1.56,  0.205, 3, 38.3,  89.4, 0.950, 0.169],
            'C':  [11.26,  27,   2.34,  1.76,  3, 8.35,  22.8, 0.836, 0.309],
            'N':  [14.53,  68,   2.11,  1.10,  3, 14.7,  20.1, 0.827, 0.216],
            'O':  [13.62,  62,   2.26,  0.802, 3, 13.0,  17.9, 0.817, 0.210],
            'Ne': [21.56,  81,   1.75,  0.396, 3, 19.2,  14.6, 0.806, 0.237],
            'Mg': [7.65,   0.78, 3.09,  10.6,  3, 1.06,  11.2, 0.794, 1.36],
            'Si': [8.15,   1.1,  2.91,  5.38,  3, 1.34,  9.99, 0.784, 1.22],
            'S':  [10.36,  11.6, 2.58,  2.90,  3, 4.90,  8.97, 0.774, 0.423],
            'Ar': [15.76,  50,   2.07,  1.64,  3, 9.09,  8.10, 0.765, 0.254],
            'Ca': [6.11,   0.70, 3.64,  22.8,  3, 0.853, 6.78, 0.757, 1.22],
            'Fe': [7.87,   0.91, 3.09,  9.4,   3, 1.14,  5.13, 0.725, 1.26]

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
            'vj [cm/s]': data[4]*1e5, # converting to cm
            'lambda_j [km]': data[5]*1e5, # converting to cm
            'Hj [km]': data[6]*1e5, # converting to cm
            'Hj+ [km]': data[7]*1e5, # converting to cm
            'wj [km/s]': data[8]*1e5 # converting to cm
        }
        
        return parameters


    # def process_collision_components(self, s):
    #     v_ion, v_neutral = self._vsisn_calculation(self.loop_properties.T(s), self.loop_properties.nh(s), self.ionisation_fraction)
    #     v_eff = self._veff_calculation(v_ion, v_neutral, self.ionisation_fraction)
    #     v_s = self._vs_getter()    
        # return v_ion, v_neutral, v_eff, self.ionisation_fraction, v_s
        # return CollisionComponents(v_ion, v_neutral, v_eff, self.ionisation_fraction, v_s)
    def get_atomic_data(self):
        element_data = {
            # 'H':  (1, 1),
            # 'He': (2, 4),
            'C':  (6, 12.0096),
            'N':  (7, 14.0064),
            'O':  (8, 15.999),
            'Ne': (10, 20.1797),
            'Mg': (12, 24.304), # true value
            'Si': (14, 28.084),
            'S':  (16, 32.059), # true value
            # 'S': (26, 55.845),
            # 'Ar': (8, 15.999), # changed it to O to test its behaviour
            'Ar': (18, 39.948),
            'Ca': (20, 40.078),
            'Fe': (26, 55.845)
        }
        
        if self.element in element_data:
            atomic_number, mass_number = element_data[self.element]
            print(f"Element: {self.element}, Atomic Number: {atomic_number}, Mass Number: {mass_number}")
            return atomic_number, mass_number
        else:
            print(f"Element '{self.element}' not found in the database.")
            return None, None

    @staticmethod    
    def get_element_properties(element):
        """
        Get ionization properties for a given element.
        
        Parameters:
        element (str): Chemical symbol of the element
        
        Returns:
        dict: Dictionary containing ionization_energy, g_i, and g_j for the element
        """
        atomic_data = {
            # 'H': {'ionization_energy': 13.6, 'g_i': 2, 'g_j': 1},
            # 'He': {'ionization_energy': 24.6, 'g_i': 1, 'g_j': 2},
            'Ca': {'ionization_energy': 6.1132*u.eV, 'g_i': 1, 'g_j': 2},
            'Mg': {'ionization_energy': 7.6462*u.eV, 'g_i': 1, 'g_j': 2},
            'Fe': {'ionization_energy': 7.9024*u.eV, 'g_i': 25, 'g_j': 30},
            'Si': {'ionization_energy': 8.1517*u.eV, 'g_i': 9, 'g_j': 6},
            'S': {'ionization_energy': 10.3600*u.eV, 'g_i': 9, 'g_j': 4},
            'C': {'ionization_energy': 11.2603*u.eV, 'g_i': 9, 'g_j': 6},
            'O': {'ionization_energy': 13.6181*u.eV, 'g_i': 9, 'g_j': 4}, 
            'N':  {'ionization_energy': 14.5341*u.eV, 'g_i': 4, 'g_j': 9},
            'Ar': {'ionization_energy': 15.7596*u.eV, 'g_i': 1, 'g_j': 6},
            'Ne': {'ionization_energy': 21.5645*u.eV, 'g_i': 1, 'g_j': 6},

        }
        
        if element not in atomic_data:
            raise ValueError(f"Properties for element '{element}' not found")
            
        return atomic_data[element]

    def _vsisn_calculation(self, T, n_H, H_i_fraction):
        """
        Calculate collision frequencies for ions and neutrals of a given element.
        Equations 12 and 13 from Schwadron 1999
        Args:
        Z: atomic number of the element
        A: number of nucleons
        n_H: number density of hydrogen (1e10 cm^-3)
        chi: fraction of hydrogen which is ionized (Marsch et al. 1995) - Avrett nH1/nH?
        alpha: 
        r: (Table 1. Marsch et al. 1995)
        """
        Z, A = self.get_atomic_data()
        # data = get_Marsch1995_data(self.element)
        r = self.Marsch1995['rjH [Å]']
        alpha = self.Marsch1995['alpha_j [Å²]']
        # ----------------------------
        # Schwadron 1999 formule
        # v_ion = (1.1e4 * Z**2 / np.sqrt(A*(A+1)) * ((1e4/T) ** (3/2)) * n_H/1e10 * H_i_fraction) + (20.74/np.sqrt(A*(A+1)))*(n_H/1e10)*(1-H_i_fraction)
        # v_neutral = (25.4*np.sqrt(alpha/(A*(A+1)))*(n_H/1e10)*H_i_fraction) + (6.09*(r**2)/np.sqrt(A*(A+1))*(n_H/1e10)*(1-H_i_fraction))
        # ----------------------------
        # Laming 2004 formule
        v_ion_p = (3.1e4 / A) * (1e4/T)**(3/2) * n_H/1e10 * H_i_fraction
        v_ion_H = (9.1 * r)/A * (T/1e4)**(1/2) * n_H/1e10
        v_ion = v_ion_p + v_ion_H
        v_neutral = v_ion_H*(1+H_i_fraction)
        # ----------------------------

        return v_ion, v_neutral, Z, A
        
    def _veff_calculation(self, v_ion, v_neutral, ionisation_fraction):
        """
        Calculate the effective collision frequency for a elemenetal species consisting of ions and neutrals.
        """
        v_eff = (v_ion*v_neutral)/(ionisation_fraction*v_neutral + (1-ionisation_fraction)*v_ion)

        return v_eff
    
    def _vs_getter(self):
        """
        Get mass dependent turbulent speed to avoid gravitational settling
        """
        return self.Marsch1995['vj [cm/s]']
    
    
    # def calculate_collision_components(self, s):
    #     ionisation_fraction = get_ionisation_fraction(
    #         self.element,
    #         self.atomic_data,
    #         self.loop_properties.ne(s),
    #         self.loop_properties.T(s))
        
    #     v_s = self._vs_getter()
    #     v_ion, v_neutral, Z, A = self._vsisn_calculation(
    #         self.loop_properties.T(s).to(u.K).value, 
    #         self.loop_properties.nh(s).value, 
    #         ionisation_fraction)
    #     v_eff = self._veff_calculation(v_ion, v_neutral, ionisation_fraction)
        
    #     return CollisionComponents(
    #         v_ion=v_ion,
    #         v_neutral=v_neutral, 
    #         v_eff=v_eff,
    #         ionisation_fraction=ionisation_fraction,
    #         v_s=v_s,
    #         atomic_number=Z,
    #         mass_number=A
    #     )

    def calculate_collision_components(self, s):
        element_ionisation_fraction = get_ionisation_fraction(
            self.element,
            self.atomic_data,
            self.loop_properties.ne(s),
            self.loop_properties.T(s))

        
        v_s = self._vs_getter()
        v_ion, v_neutral, Z, A = self._vsisn_calculation(
            self.loop_properties.T(s).to(u.K).value,
            self.loop_properties.nh(s).value, # hydrogen number density
            self.loop_properties.h_ionisation(s) #Use hydrogen ionization fraction for background (collisions with H and H+)
        )
        v_eff = self._veff_calculation(v_ion, v_neutral, element_ionisation_fraction)
        
        return CollisionComponents(
            v_ion=v_ion,
            v_neutral=v_neutral, 
            v_eff=v_eff,
            ionisation_fraction=element_ionisation_fraction,
            v_s=v_s,
            atomic_number=Z,
            mass_number=A
        )