"""Six-state rates following HYDRAD CRadiativeRates (Bradshaw, 2022).

Read the user's original atomic tables, without copying or modifying them.
Implements GetCollisionalRatesRH and log-linear radiative table interpolation.
Lyman radiative transitions are in detailed balance (zero net radiative rates),
as in HYDRAD's SolveHIIFraction. These are not optically thin CHIANTI rates.
"""
from pathlib import Path
import hashlib
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .profiles import check_populations


class HydrogenRates:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / 'Radiation_Model/atomic_data/OpticallyThick/radiative_rates'
        names = (self.directory / 'rates_files.txt').read_text().split()
        if len(names) != 4:
            raise ValueError('Expected four HYDRAD rate table paths')
        paths = [self.root / n for n in names]
        arrays = [np.fromstring(p.read_text(), sep=' ') for p in paths]
        self.fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
        bb, bf, fb, coll = arrays
        nb, nt = bb[:2].astype(int)
        if nb != 6:
            raise ValueError('Expected six radiative bound-bound transitions')
        b = bb[2:].reshape(nt, 1+2*nb)
        self.bb_t, self.bb_up, self.bb_down = b[:, 0], b[:, 1:7], b[:, 7:]
        nb, nt = bf[:2].astype(int)
        if nb != 4:
            raise ValueError('Expected four radiative bound-free transitions')
        b = bf[2:].reshape(nt, 1+nb)
        self.bf_t, self.bf = b[:, 0], b[:, 1:]
        nb, nr, ne = fb[:3].astype(int)
        if nb != 4:
            raise ValueError('Expected four free-bound transitions')
        self.fb_rt = fb[3:3+nr]
        self.fb_et = fb[3+nr:3+nr+ne]
        b = fb[3+nr+ne:].reshape(nb, ne, nr)
        self.fb_interp = [RegularGridInterpolator((self.fb_et, self.fb_rt), np.log10(x)) for x in b]
        nx, ni, nt = coll[:3].astype(int)
        if (nx, ni) != (10, 5):
            raise ValueError('Expected ten collisional bound-bound and five bound-free transitions')
        offset = 3
        blocks = []
        for count in (nx, nx, ni, ni):
            block = coll[offset:offset+nt*(count+1)].reshape(nt, count+1)
            blocks.append(block)
            offset += block.size
        if offset != coll.size:
            raise ValueError('Unexpected collisional table size')
        self.coll_t = blocks[1][:, 0]
        self.coll_down = blocks[1][:, 1:]
        self.coll_ion = blocks[2][:, 1:]
        for block in blocks:
            if not np.array_equal(block[:, 0], self.coll_t):
                raise ValueError('Collisional temperature grids disagree')
        config = (self.root / 'Radiation_Model/source/config.h').read_text()
        match = re.search(r'^\s*#define\s+MIN_DENSITY_LIMIT\s+([\d.eE+-]+)', config, re.M)
        if not match:
            raise ValueError('Missing HYDRAD MIN_DENSITY_LIMIT; choose an NLTE atomic-data source')
        self.minimum_density = float(match[1])

    @staticmethod
    def interpolate(grid, table, values):
        values = np.asarray(values)
        if values.ndim == 1:
            values = np.broadcast_to(values[:, None], (len(values), table.shape[1]))
        return np.column_stack([10**np.interp(values[:, j], grid, np.log10(table[:, j])) for j in range(table.shape[1])])

    def generator(self, te, ne, radiation_temperature):
        """Return Q with df/dt = Q f; columns sum to zero. Units: s^-1."""
        te, ne, trad = map(np.asarray, (te, ne, radiation_temperature))
        if te.ndim != 1 or ne.shape != te.shape or trad.shape != (len(te), 10):
            raise ValueError('Expected Te/ne (cells,), radiation temperatures (cells, 10)')
        if any(np.any(~np.isfinite(x)) or np.any(x <= 0) for x in (te, ne, trad)):
            raise ValueError('Temperatures and densities must be finite and positive')
        ne = np.maximum(ne, self.minimum_density)
        # HYDRAD clamps temperature before both interpolation and RH factors.
        logt = np.clip(np.log10(te), self.coll_t[0], self.coll_t[-1])
        t = 10**logt
        g = np.array([.25, .11111111, .0625, .04, .44444444, .25, .16, .5625, .36, .64, 2, 8, 18, 32, 50])
        de = np.array([118301.48,140234.04,147864.85,151464.29,21918.168,29587.368,33148.41,7673.9985,11225.443,3551.444,157895.28,39464.221,17536.454,9867.2552,6311.0119])
        down = ne[:, None] * g[:10] * np.sqrt(t[:, None]) * self.interpolate(self.coll_t, self.coll_down, logt) * 1e6
        up = down * np.exp(-de[:10]/t[:, None]) / g[:10]
        base = ne[:, None] * np.sqrt(t[:, None]) * self.interpolate(self.coll_t, self.coll_ion, logt) * 1e6
        ion = base * np.exp(-de[10:]/t[:, None])
        rec = 2.071e-16 * base * ne[:, None] * g[10:] * t[:, None]**-1.5
        logr = np.log10(trad)
        up[:, 4:] += self.interpolate(self.bb_t, self.bb_up, logr[:, :6])
        down[:, 4:] += self.interpolate(self.bb_t, self.bb_down, logr[:, :6])
        ion[:, 1:] += self.interpolate(self.bf_t, self.bf, logr[:, 6:])
        # GetBoundFreeRates mutates its argument before GetFreeBoundRates uses it.
        lr = np.clip(logr[:, 6:], self.bf_t[0], self.bf_t[-1])
        lr = np.clip(lr, self.fb_rt[0], self.fb_rt[-1])
        lt = np.clip(np.log10(te), self.fb_et[0], self.fb_et[-1])
        for j, interp in enumerate(self.fb_interp):
            rec[:, j+1] += ne * 10**interp(np.column_stack([lt, lr[:, j]]))
        q = np.zeros((len(te), 6, 6))
        k = 0
        for low in range(5):
            for high in range(low+1, 5):
                q[:, high, low] = up[:, k]
                q[:, low, high] = down[:, k]
                k += 1
            q[:, 5, low] = ion[:, low]
            q[:, low, 5] = rec[:, low]
        diag = np.arange(6)
        q[:, diag, diag] = -q.sum(axis=1)
        return q


def equilibrium(q):
    a = q.copy()
    # Replacing the ground-state equation follows HYDRAD's initial solve.
    a[:, 0, :] = 1
    b = np.zeros(q.shape[:2])
    b[:, 0] = 1
    return check_populations(np.linalg.solve(a, b[..., None])[..., 0])


def backward_euler(q, populations, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive')
    return check_populations(np.linalg.solve(np.eye(6)[None] - dt*q, populations[..., None])[..., 0])
