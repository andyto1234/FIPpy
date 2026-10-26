"""Checks for WHI parsing and boundary-field units/normalisation."""

import tempfile
from pathlib import Path
import unittest

import astropy.units as u
from astropy.constants import au, R_sun
import numpy as np

from physics.radiation import (
    load_whi_spectrum, surface_hemisphere_mean_intensity,
    two_sided_mean_intensity,
)


class RadiationTests(unittest.TestCase):
    def test_reference_spectrum_integrals(self):
        for spectrum, expected in (
            ("quiet", 1309.84), ("sunspot", 1309.70), ("faculae", 1309.94)
        ):
            with self.subTest(spectrum=spectrum):
                wavelength, flux, flags = load_whi_spectrum(spectrum=spectrum)
                self.assertEqual(wavelength.shape, (24000,))
                np.testing.assert_allclose(np.diff(wavelength.value), 0.1)
                total = (flux.sum() * 0.1 * u.nm).to_value(u.W / u.m**2)
                self.assertAlmostEqual(total, expected, delta=0.1)
                self.assertTrue(np.any(flags == -1))

    def test_column_selection_and_filled_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "whi.dat"
            path.write_text("; header\nFORMAT = ...\n5 columns : ...\n"
                            "100.05 1 2 3 -1\n100.15 4 5 6 3\n")
            wavelength, flux, flags = load_whi_spectrum(path, "quiet")
            np.testing.assert_array_equal(flux.value, [3, 6])
            np.testing.assert_array_equal(flags, [-1, 3])

    def test_boundary_normalisation_and_equivalent_units(self):
        flux = np.array([1.0, 2.0]) * u.W / (u.m**2 * u.nm)
        J = surface_hemisphere_mean_intensity(flux)
        # Recover the hemispheric surface flux, then dilute back to 1 au.
        recovered = (2 * np.pi * u.sr * J * (R_sun / au)**2).to(flux.unit)
        np.testing.assert_allclose(recovered.value, flux.value)
        equivalent = flux.to(u.erg / (u.s * u.cm**2 * u.AA))
        np.testing.assert_allclose(
            surface_hemisphere_mean_intensity(equivalent).value, J.value
        )
        with self.assertRaises(u.UnitConversionError):
            surface_hemisphere_mean_intensity(1 * u.W / u.m**2)

    def test_unknown_spectrum(self):
        with self.assertRaises(ValueError):
            load_whi_spectrum(spectrum="flare")

    def test_two_sided_attenuation_and_transparent_band(self):
        unit = u.erg / (u.s * u.cm**2 * u.sr * u.nm)
        bottom = np.array([2., 3.]) * unit
        top = np.array([5., 7.]) * unit
        nb = np.array([0., 1e18]) / u.cm**2
        nt = np.array([1e18, 0.]) / u.cm**2
        sigma = np.array([1e-18, 0.]) * u.cm**2
        J = two_sided_mean_intensity(bottom, top, nb, nt, sigma)
        np.testing.assert_allclose(J.value, [
            [2 + 5 * np.exp(-1), 10],
            [2 * np.exp(-1) + 5, 10],
        ])
        # Swapping boundary labels must preserve the radiation field.
        swapped = two_sided_mean_intensity(top, bottom, nt, nb, sigma)
        np.testing.assert_allclose(J.value, swapped.value)
        with self.assertRaises(u.UnitConversionError):
            two_sided_mean_intensity(bottom, top, nb / u.cm, nt, sigma)
        with self.assertRaises(ValueError):
            two_sided_mean_intensity(bottom[:1], top, nb, nt, sigma)


if __name__ == "__main__":
    unittest.main()
