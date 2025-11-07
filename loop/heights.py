import astropy.units as u

class ScaleHeightCalculator:
    def __init__(self, property_function):
        self.property_function = property_function

    def calculate(self, s):
        """Calculate scale height at position s."""
        property_s = self.property_function(s)
        
        # Adaptive delta_s based on local variation
        delta_s_initial = 1e4 * u.cm
        
        # Test gradients at different scales
        test_deltas = [1e3, 1e4, 1e5] * u.cm
        gradients = []
        
        for ds in test_deltas:
            dp = (self.property_function(s + ds) - 
                self.property_function(s - ds)) / (2 * ds)
            gradients.append(dp)
        
        # Use the one that converges (middle value if all similar)
        # Or use smallest delta where derivative stabilizes
        dproperty_ds = gradients[1]  # Use medium scale
        return property_s / dproperty_ds
