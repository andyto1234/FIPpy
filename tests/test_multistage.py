import unittest
import numpy as np
import astropy.units as u
from physics.ionisation import adjacent_equilibrium

class TestMultistage(unittest.TestCase):
    def test_two_stage(self):
        f=adjacent_equilibrium(np.array([[2.]])/u.s,np.array([[3.]])/u.s)
        np.testing.assert_allclose(f,[[.6,.4]])
    def test_three_stage(self):
        f=adjacent_equilibrium(np.array([[2.,3.]])/u.s,np.ones((1,2))/u.s)
        np.testing.assert_allclose(f,[[1/9,2/9,6/9]])
    def test_zero_and_dynamic_range(self):
        f=adjacent_equilibrium(np.array([[0.,1.],[1e200,1e200]])/u.s,np.array([[1.,1.],[1e-100,1e-100]])/u.s)
        np.testing.assert_allclose(f[0],[1,0,0]);np.testing.assert_allclose(f[1],[0,0,1],atol=1e-100)
    def test_missing_reverse(self):
        with self.assertRaises(ValueError):
            adjacent_equilibrium(np.ones((1,2))/u.s,np.zeros((1,2))/u.s)
