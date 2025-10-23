import numpy as np
import astropy.units as u

class MagneticField:
    def __init__(self, loop_properties):
        self.loop_properties = loop_properties
        self._precompute_tr_weight()

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

        
        def calculate_field(x):
            return 1031.61 - 1.33014e-5 * x + 8.48378e-14 * x**2 - 3.03276e-22 * x**3 + 5.88384e-31 * x**4 - 4.7167e-40 * x**5

            # Use numpy.where to handle array inputs
        B_s = np.where(s_cm < 3e8, 
                    calculate_field(s_cm),
                    np.where(s_cm > loop_length.to(u.cm).value - 3e8,
                            calculate_field(loop_length.to(u.cm).value - s_cm),
                            105.9))
                            # 7.9))

        return B_s/3 * u.G # artificially decrease the magnetic field strength

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