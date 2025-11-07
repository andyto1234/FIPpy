import astropy.units as u
import numpy as np

class AlfvenSpeedCalculator:
    """
    Calculate the Alfven speed at a given position s.
    """
    def __init__(self, magnetic_field, density):
        self.magnetic_field = magnetic_field
        self.density = density

    def calculate(self, s):
        """Calculate Alfvén speed at position s."""
        B_s = np.abs(self.magnetic_field(s))
        rho_s = self.density(s)
        # rho_s = np.maximum(rho_s, 1e-30 * u.g/u.cm**3)  # guard against interp overshoots
        # import pdb; pdb.set_trace()
        # Alfvén speed: V_A = B / sqrt(4π * rho)
        # Need to ensure proper unit handling
        B_cgs = B_s.to(u.G)  # Convert to Gauss
        rho_cgs = rho_s.to(u.g / u.cm**3)  # Convert to g/cm^3
        
        # Calculate V_A with explicit unit handling
        V_A_value = B_cgs.value / np.sqrt(4 * np.pi * rho_cgs.value)
        V_A = V_A_value * (u.cm / u.s)

        # return Alfven_speed(B=B_s, density=rho_s).to(u.cm/u.s)
        # print(V_A.value/3e10)
        return V_A