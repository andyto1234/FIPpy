import numpy as np
import astropy.units as u

class MagneticField:
    # Footpoint polynomial in Gauss for s in centimetres, joined to a constant
    # corona. The raw polynomial is 107.89 G at 3 Mm, 1.99 G above the coronal
    # value. That mismatch is removed by a constant shift so B is continuous
    # and the corona stays 105.9 G. The derivative of a constant is zero, so
    # the footpoint shape of dB/ds is unchanged.
    _FOOTPOINT_POLYNOMIAL = np.polynomial.Polynomial(
        [1031.61, -1.33014e-5, 8.48378e-14, -3.03276e-22, 5.88384e-31, -4.7167e-40]
    )
    _FOOTPOINT_DEPTH = 3e8  # cm
    _CORONAL_FIELD = 105.9  # G
    _FOOTPOINT_OFFSET = float(
        _FOOTPOINT_POLYNOMIAL(_FOOTPOINT_DEPTH) - _CORONAL_FIELD
    )

    def __init__(self, loop_properties):
        self.loop_properties = loop_properties
        self._precompute_tr_weight()

    def magnetic_joins(self):
        """Positions [cm] where the footpoint polynomial meets the corona."""
        length = self.loop_properties.loop_length.to(u.cm).value
        depth = self._FOOTPOINT_DEPTH
        return (depth, length - depth)

    def logarithmic_derivative(self, s_cm):
        """d ln B/ds on each smooth piece. s_cm is in centimetres."""
        s = np.asarray(s_cm, dtype=float)
        scalar = s.ndim == 0
        s = np.atleast_1d(s)
        length = self.loop_properties.loop_length.to(u.cm).value
        depth = self._FOOTPOINT_DEPTH
        polynomial = self._FOOTPOINT_POLYNOMIAL
        slope = polynomial.deriv()
        offset = self._FOOTPOINT_OFFSET
        deriv = np.zeros(s.shape)
        left = s < depth
        right = s > length - depth
        if np.any(left):
            x = s[left]
            deriv[left] = slope(x) / (polynomial(x) - offset)
        if np.any(right):
            x = length - s[right]
            deriv[right] = -slope(x) / (polynomial(x) - offset)
        return deriv[0] if scalar else deriv

    def __call__(self, s):
        """Calculate magnetic field strength at position s."""
        loop_length = self.loop_properties.loop_length
        s_cm = s.to(u.cm).value
        # return self.pressure_scaled(s)
        # return self.flux_conserving(
        #     s,
        #     B_fp=100*u.G,
        #     f_apex=10.0,
        #     m=2,
        #     B_min=5*u.G,
        #     tr_boost=5.0,        # extra expansion near TR (tune 2–10)
        #     tr_sharpness=2.0,    # how concentrated the boost is
        #     tr_threshold=0.5     # only top 50% of grad(|ln ne|) contributes
        # )

        
        depth = self._FOOTPOINT_DEPTH
        polynomial = self._FOOTPOINT_POLYNOMIAL
        length = loop_length.to(u.cm).value
        offset = self._FOOTPOINT_OFFSET
        B_s = np.where(s_cm < depth,
                    polynomial(s_cm) - offset,
                    np.where(s_cm > length - depth,
                            polynomial(length - s_cm) - offset,
                            self._CORONAL_FIELD))
                            # 7.9))

        return B_s * u.G # artificially decrease the magnetic field strength

    def pressure_scaled(self, s, B_fp=100*u.G, B_min=10*u.G, alpha=0.1,
                    tr_boost=0.0, tr_sharpness=2.0, tr_threshold=0.0):
        """
        Pressure-based, monotonic B(s):
        - B_fp: field at each footpoint
        - B_min: floor in the corona
        - alpha: controls how strongly B falls with P/P0 (larger => more drastic)
        - tr_boost/sharpness/threshold: optional extra expansion near the TR
        """
        # Keep original scalar/array shape
        s_in = s
        s_arr = np.atleast_1d(s_in)

        # Geometry and side selection
        s_apex = (self.loop_properties.s_array[0] + self.loop_properties.s_array[-1]) / 2
        s_vals = s_arr.to(u.cm).value
        left = s_vals <= s_apex.to(u.cm).value

        # Pressures
        P = self.loop_properties.P(s_arr)  # Quantity array
        P0_left  = self.loop_properties.P(self.loop_properties.s_array[0])
        P0_right = self.loop_properties.P(self.loop_properties.s_array[-1])

        # Dimensionless ratios, capped to <= 1 per leg
        r = np.empty(P.shape, dtype=float)
        if np.any(left):
            r[left] = (P[left] / P0_left).to_value(u.dimensionless_unscaled)
        if np.any(~left):
            r[~left] = (P[~left] / P0_right).to_value(u.dimensionless_unscaled)
        r = np.clip(r, 0.0, 1.0)

        # Enforce monotonic decrease toward the apex on each leg
        if r.ndim == 1:
            if np.any(left):
                r[left] = np.minimum.accumulate(r[left])
            if np.any(~left):
                r_right = np.minimum.accumulate(r[~left][::-1])[::-1]
                r[~left] = r_right

        B_fp_val = B_fp.to_value(u.G)
        B_min_val = B_min.to_value(u.G)

        # Stronger, tunable falloff with alpha; bounded by [B_min, B_fp]
        B_s = B_min_val + (B_fp_val - B_min_val) * (r ** alpha)

        # Optional TR-weighted extra expansion (reduces B where TR weight is high)
        if tr_boost and tr_boost != 0.0:
            w = self._tr_weight(s_arr)
            if tr_threshold != 0.0:
                w = np.clip((w - tr_threshold) / (1.0 - tr_threshold), 0.0, 1.0)
            w = w ** tr_sharpness
            B_s = B_s / (1.0 + tr_boost * w)

        B_out = np.maximum(B_s, B_min_val) * u.G
        return B_out[0] if np.ndim(s_in) == 0 else B_out

    def _precompute_tr_weight(self):
        """
        Precompute a 0..1 weight field that highlights the transition region by the
        magnitude of the density scale-change: w(s) ~ |d ln ne / ds| (smoothed, normalized).
        """
        s = self.loop_properties.s_array.to(u.cm).value
        ne = self.loop_properties.ne(self.loop_properties.s_array).to(u.cm**-3).value
        ne = np.maximum(ne, 1e-8 * np.max(ne))  # avoid log underflow
        dlogne_ds = np.gradient(np.log(ne), s)
        g = np.abs(dlogne_ds)

        # Light smoothing (moving-average-like kernel)
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        kernel /= kernel.sum()
        g_smooth = np.convolve(g, kernel, mode='same')

        g_min = g_smooth.min()
        g_max = g_smooth.max()
        if g_max - g_min < 1e-20:
            w = np.zeros_like(g_smooth)
        else:
            w = (g_smooth - g_min) / (g_max - g_min)
        self._s_tr = s
        self._w_tr = w

    def _tr_weight(self, s):
        s_cgs = np.atleast_1d(s.to(u.cm).value)
        return np.interp(s_cgs, self._s_tr, self._w_tr)


    def flux_conserving(self, s, B_fp=100*u.G, f_apex=10.0, m=2, B_min=5*u.G,
                        tr_boost=0.0, tr_sharpness=2.0, tr_threshold=0.0):
        """
        Flux-conserving, smooth B(s) with optional TR-weighted extra expansion.
        - tr_boost: multiplicative expansion added near TR (0 disables)
        - tr_sharpness: power applied to TR weight (larger -> more localized)
        - tr_threshold: ignore low weights; only use top fraction (0..1)
        """
        L = self.loop_properties.loop_length
        xi = (s / L).decompose().value  # 0..1 along the loop

        f_base = 1.0 + (f_apex - 1.0) * np.sin(np.pi * xi)**m
        f_total = f_base

        if tr_boost and tr_boost != 0.0:
            w = self._tr_weight(s)
            if tr_threshold != 0.0:
                w = np.clip((w - tr_threshold) / (1.0 - tr_threshold), 0.0, 1.0)
            w = w**tr_sharpness
            f_total = f_base * (1.0 + tr_boost * w)

        B = (B_fp / f_total)
        return np.maximum(B.to_value(u.G), B_min.to_value(u.G)) * u.G