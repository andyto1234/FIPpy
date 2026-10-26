import unittest
import numpy as np
import astropy.units as u
from physics.saha import chianti_ionised_fraction, get_ionisation_fraction
from physics.collision import CollisionCalculator


class TestEquilibrium(unittest.TestCase):
    def test_pipeline_elements(self):
        temperatures = np.geomspace(4000, 1e7, 100)*u.K
        for element in ['Ca','Mg','Fe','Si','S','C','N','O','Ne','Ar']:
            fractions = get_ionisation_fraction(element,
                CollisionCalculator.get_element_properties(element),
                np.ones(100)*1e11/u.cm**3, temperatures)
            self.assertTrue(np.all(np.isfinite(fractions)))
            self.assertTrue(np.all((fractions >= 0)&(fractions <= 1)))

    def test_low_temperature_not_old_plateau(self):
        values = chianti_ionised_fraction('Fe', np.array([4000,6000,10000])*u.K)
        self.assertLess(values[0], 1e-3)
        self.assertTrue(np.all(np.diff(values)>0))

    def test_no_silent_boundary_clamping(self):
        with self.assertRaises(ValueError):
            chianti_ionised_fraction('Fe', np.array([500])*u.K)
