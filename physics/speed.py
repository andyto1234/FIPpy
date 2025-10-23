from plasmapy.formulary.speeds import Alfven_speed
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
        return Alfven_speed(B=B_s, density=rho_s).to(u.cm/u.s)