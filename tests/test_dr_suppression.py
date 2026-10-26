"""Nikolić et al. (2018) dielectronic suppression, checked on the CHIANTI 11 file."""
import unittest
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.constants import k_B
import fiasco

from physics.ionisation import multistage_equilibrium, recombination_coefficient

CHIANTI_H5 = Path('/Volumes/Crucial X9/databases/CHIANTI_11.0.2_database.h5')


def _magnesium_ii_factor(temperature, electron_density):
    """Na-like q=1 factor from Nikolić et al. (2018), independent of fiasco.

    A(N) follows their equations 6 and 7 for the third row (N1, N2) = (11, 18)
    at N = 11. The core energy is the Na-like row of their Table 5. There is
    no psi adjustment for this sequence.
    """
    temperature = np.asarray(temperature.to_value(u.K), dtype=float)
    density = np.asarray(electron_density.to_value(u.cm**-3), dtype=float)
    charge = 1.0
    electrons = 11
    row_start, row_end = 11, 18
    a_factor = 12 + 10 * row_start + (
        (10 * row_start - 2 * row_end) / (row_start - row_end) * (electrons - row_start))
    q0 = (1 - np.sqrt(2 / (3 * charge))) * a_factor / np.sqrt(charge)
    activation = 10.1821 + np.log10((charge / q0)**7 * np.sqrt(temperature / (5e4 * q0**2)))
    width = 5.64586
    log_density = np.log10(density)
    high_temperature = np.exp(-(log_density - activation)**2 * np.log(2) / width**2)
    high_temperature = np.where(log_density <= activation, 1.0, high_temperature)
    coefficients = np.array([2.248, 22.28, -1.123, 0.9027, -0.03860, 0.01468])
    excitation = np.sum(coefficients * (charge / 10)**np.arange(6)) * u.eV
    turn_off = np.exp(-(excitation / (10 * k_B * temperature * u.K)).to_value(u.dimensionless_unscaled))
    return 1 - (1 - high_temperature) * turn_off


class TestDielectronicSuppression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHIANTI_H5.exists():
            raise unittest.SkipTest('CHIANTI 11 file is not available.')
        cls.root = CHIANTI_H5

    def test_magnesium_ii_matches_published_factor(self):
        temperature = np.array([1.0e4, 1.5e4, 2.0e4]) * u.K
        density = np.array([1.0e10, 5.0e9, 1.0e11]) / u.cm**3
        ion = fiasco.Ion('Mg II', temperature, hdf5_dbase_root=self.root)
        coefficient, info = recombination_coefficient(ion, density)
        self.assertTrue(info['applied'])
        self.assertEqual(info['treatment'], 'rr + suppressed dr')
        np.testing.assert_allclose(info['factor'], _magnesium_ii_factor(temperature, density), rtol=1e-5)
        zero_density = ion.recombination_rate
        radiative = ion.radiative_recombination_rate
        self.assertTrue(np.all(coefficient < zero_density))
        self.assertTrue(np.all(coefficient > radiative))

    def test_suppression_is_off_below_the_activation_density(self):
        temperature = np.array([1.5e4, 2.0e4]) * u.K
        density = np.full(temperature.shape, 1e-4) / u.cm**3
        ion = fiasco.Ion('Mg II', temperature, hdf5_dbase_root=self.root)
        _, info = recombination_coefficient(ion, density)
        np.testing.assert_allclose(info['factor'], 1.0)

    def test_inseparable_total_is_not_rescaled(self):
        temperature = np.array([1.0e4, 2.0e4]) * u.K
        density = np.full(temperature.shape, 1e12) / u.cm**3
        ion = fiasco.Ion('Fe II', temperature, hdf5_dbase_root=self.root)
        coefficient, info = recombination_coefficient(ion, density)
        self.assertFalse(info['applied'])
        self.assertEqual(info['treatment'], 'trparams')
        self.assertTrue(np.all(info['factor'] < 0.2))
        np.testing.assert_allclose(coefficient.value, ion.recombination_rate.value)

    def test_suppression_reduces_the_magnesium_neutral_fraction(self):
        temperature = np.array([8e3, 1.2e4, 1.6e4, 2.2e4, 4e4]) * u.K
        density = np.full(temperature.shape, 1e10) / u.cm**3
        options = dict(hdf5_dbase_root=self.root)
        plain = multistage_equilibrium('Mg', temperature, density, **options)
        suppressed = multistage_equilibrium('Mg', temperature, density,
                                            suppress_dielectronic=True, **options)
        np.testing.assert_allclose(suppressed['populations'].sum(axis=1), 1, atol=1e-12)
        self.assertTrue(suppressed['dielectronic_suppression_applied'][0])
        self.assertFalse(plain['dielectronic_suppression_applied'].any())
        neutral = plain['populations'][:, 0]
        neutral_suppressed = suppressed['populations'][:, 0]
        self.assertLess(neutral_suppressed[2], 0.5 * neutral[2])
        self.assertTrue(np.all(neutral_suppressed < neutral))
