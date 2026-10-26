"""Regression checks for the public Verner cross-section interface."""

import os
from pathlib import Path
import tempfile
import unittest

import astropy.units as u
import numpy as np

from physics import parse_ion_name, photoionisation_cross_section, verner_parameters


class CrossSectionTests(unittest.TestCase):
    def test_default_path_is_independent_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(verner_parameters("Fe", ion=1)["N"], 26)
            finally:
                os.chdir(previous)

    def test_parameters_are_not_shared_mutable_cache_entries(self):
        row = verner_parameters("Fe", ion=1)
        row["sigma0"] = -1
        self.assertGreater(verner_parameters("Fe", ion=1)["sigma0"], 0)

    def test_stage_selection(self):
        self.assertEqual(parse_ion_name("fe_II"), ("Fe", 2))
        self.assertEqual(verner_parameters("Fe", ion="II"),
                         verner_parameters(26, electrons=25))
        for bad in (1.5, True, 0, 27):
            with self.subTest(ion=bad), self.assertRaises(ValueError):
                verner_parameters("Fe", ion=bad)

    def test_hydrogen_threshold_scale_and_shape(self):
        sigma = photoionisation_cross_section(
            np.array([[10.0, 13.6], [20.0, 40.0]]) * u.eV, "H"
        ).to_value(u.cm**2)
        self.assertEqual(sigma.shape, (2, 2))
        self.assertEqual(sigma[0, 0], 0)
        # Independent physical scale: H I threshold cross-section ~6.3 Mb.
        self.assertAlmostEqual(sigma[0, 1] / 6.3e-18, 1, delta=0.02)
        self.assertGreater(sigma[1, 0], sigma[1, 1])

    def test_units_and_scalar_input(self):
        ev = photoionisation_cross_section(20 * u.eV, "H")
        erg = photoionisation_cross_section((20 * u.eV).to(u.erg), "H")
        self.assertEqual(ev.shape, ())
        np.testing.assert_allclose(ev.value, erg.value, rtol=1e-14, atol=0)
        with self.assertRaises(TypeError):
            photoionisation_cross_section(20, "H")

    def test_energy_domain(self):
        for energy in (0, -1, np.nan, np.inf):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                photoionisation_cross_section(energy * u.eV, "H")
        with self.assertRaisesRegex(ValueError, "inner-shell"):
            photoionisation_cross_section(66 * u.eV, "Fe")
        with self.assertRaisesRegex(ValueError, "nonrelativistic"):
            photoionisation_cross_section(50000 * u.eV, "H")

    def test_malformed_table_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.dat"
            path.write_text("1 1 13.6\n")
            with self.assertRaisesRegex(ValueError, "expected 11 columns"):
                verner_parameters("H", ion=1, path=path)


if __name__ == "__main__":
    unittest.main()
