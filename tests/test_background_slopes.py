import unittest
import numpy as np
import astropy.units as u
from scipy.interpolate import PchipInterpolator
from wave.solver import WaveAnalyzer


class _Loop:
    def __init__(self):
        self.s_array = np.linspace(0.0, 6.0e9, 40) * u.cm
        self.loop_length = self.s_array[-1] - self.s_array[0]
        centres = self.s_array.to_value(u.cm)
        # A smooth density plus one narrow transition-region-like drop.
        self.rho_interp = PchipInterpolator(
            centres,
            1e-12 * np.exp(-((centres - 1.42e8) / 2.5e6) ** 2) + 1e-15,
        )

    def rho(self, s):
        return self.rho_interp(s.to_value(u.cm)) * u.g / u.cm**3

    def ne(self, s):
        return np.ones(np.shape(s.to_value(u.cm))) / u.cm**3

    def velocity(self, s):
        return np.zeros(np.shape(s.to_value(u.cm))) * u.cm / u.s


class BackgroundSlopeTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = WaveAnalyzer(_Loop())

    def test_alfven_slope_follows_the_field_and_density(self):
        s = np.array([1.0e8, 2.5e8, 4.0e8, 5.7e9])
        density_slope, alfven_slope = self.analyzer._log_slopes(s * u.cm)
        field_slope = self.analyzer.B.logarithmic_derivative(s)
        np.testing.assert_allclose(alfven_slope, field_slope - 0.5 * density_slope)

    def test_scale_heights_match_those_slopes(self):
        s = 1.42e8 * u.cm
        density_slope, alfven_slope = self.analyzer._log_slopes(s)
        self.assertAlmostEqual(self.analyzer.HD(s).to_value(u.cm), 1.0 / density_slope)
        self.assertAlmostEqual(self.analyzer.HA(s).to_value(u.cm), 1.0 / alfven_slope)

    def test_field_is_continuous_at_the_coronal_joins(self):
        field = self.analyzer.B
        for join in field.magnetic_joins():
            left = field((join - 1.0) * u.cm).to_value(u.G)
            right = field((join + 1.0) * u.cm).to_value(u.G)
            self.assertAlmostEqual(left, field._CORONAL_FIELD, places=6)
            self.assertAlmostEqual(right, field._CORONAL_FIELD, places=6)

    def test_coronal_field_slope_is_zero_and_footpoint_slope_is_analytic(self):
        field = self.analyzer.B
        self.assertEqual(field.logarithmic_derivative(4.0e8), 0.0)
        polynomial = field._FOOTPOINT_POLYNOMIAL
        offset = field._FOOTPOINT_OFFSET
        s = 1.0e8
        expected = polynomial.deriv()(s) / (polynomial(s) - offset)
        self.assertAlmostEqual(field.logarithmic_derivative(s), expected)
        right = field.loop_properties.loop_length.to_value(u.cm) - s
        self.assertAlmostEqual(field.logarithmic_derivative(right), -expected)

    def test_integration_crosses_both_magnetic_joins(self):
        solution = self.analyzer._solve_ode(
            np.array([1.0e7, 5.9e9]),
            [1.0e5, 0.0, 0.0, 0.0],
            2 * np.pi * 0.206,
        )
        self.assertTrue(solution.success, solution.message)
        self.assertEqual(solution.y.shape[0], 4)
        self.assertEqual(solution.t.shape[0], solution.y.shape[1])
        self.assertTrue(np.all(np.isfinite(solution.y)))
        joins = self.analyzer.B.magnetic_joins()
        self.assertGreater(np.sum(solution.t < joins[0]), 1)
        self.assertGreater(np.sum(solution.t > joins[1]), 1)


if __name__ == "__main__":
    unittest.main()
