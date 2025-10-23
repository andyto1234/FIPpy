import astropy.units as u

class ScaleHeightCalculator:
    def __init__(self, property_function):
        self.property_function = property_function

    def calculate(self, s):
        """Calculate scale height at position s."""
        property_s = self.property_function(s)
        delta_s = 1e4 * u.cm
        dproperty_ds = (self.property_function(s + delta_s) - self.property_function(s - delta_s)) / (2 * delta_s)
        return property_s / dproperty_ds
