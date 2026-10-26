"""Read only the HYDRAD quantities needed for hydrogen reconstruction (CGS)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Profile:
    index: int
    time: float
    length: float
    s: np.ndarray
    widths: np.ndarray
    rho: np.ndarray
    velocity: np.ndarray
    ne: np.ndarray
    nh: np.ndarray
    te: np.ndarray

    @property
    def edges(self):
        return np.r_[self.s - self.widths / 2, self.s[-1] + self.widths[-1] / 2]


def profile_indices(root):
    paths = list((Path(root) / 'Results').glob('profile*.amr'))
    if not paths:
        raise FileNotFoundError(f'No Results/profile*.amr in {root}; supply the raw HYDRAD run, not FIP outputs.')
    return sorted(int(p.stem[7:]) for p in paths)


def read_profile(root, index):
    base = Path(root) / 'Results' / f'profile{index}'
    with base.with_suffix('.amr').open() as f:
        time, _, length, count = [float(f.readline()) for _ in range(4)]
        amr = np.loadtxt(f, ndmin=2)
    phy = np.loadtxt(base.with_suffix('.phy'), ndmin=2)
    if len(amr) != int(count) or len(phy) != len(amr):
        raise ValueError(f'Inconsistent cell counts for {base}')
    if not np.allclose(phy[:, 0], amr[:, 0], rtol=1e-8, atol=1e-3):
        raise ValueError(f'AMR/physical coordinates disagree for {base}')
    p = Profile(index, time, length, amr[:, 0], amr[:, 1], amr[:, 2],
                phy[:, 1], phy[:, 3], phy[:, 4], phy[:, 7])
    for x in (p.widths, p.rho, p.ne, p.nh, p.te):
        if not np.all(np.isfinite(x)) or np.any(x <= 0):
            raise ValueError(f'Invalid physical data for {base}')
    if not np.all(np.isfinite(p.velocity)) or np.any(np.diff(p.s) <= 0):
        raise ValueError(f'Invalid coordinates/velocity for {base}')
    if not np.allclose(p.s[:-1] + p.widths[:-1]/2, p.s[1:] - p.widths[1:]/2, atol=1e-3, rtol=1e-9):
        raise ValueError('Only contiguous one-dimensional grids are supported')
    return p


def read_populations(root, profile):
    data = np.loadtxt(Path(root) / 'Results' / f'profile{profile.index}.Hstate', ndmin=2)
    if data.shape != (len(profile.s), 7) or not np.allclose(data[:, 0], profile.s, rtol=1e-8, atol=1e-3):
        raise ValueError('Hydrogen populations do not match profile coordinates')
    return check_populations(data[:, 1:])


def check_populations(f, tolerance=2e-7):
    f = np.asarray(f, dtype=float)
    if f.ndim != 2 or f.shape[1] != 6 or not np.all(np.isfinite(f)):
        raise ValueError('Expected finite (cells, 6) hydrogen populations')
    if np.min(f) < -1e-12 or np.max(np.abs(f.sum(axis=1)-1)) > tolerance:
        raise ValueError('Hydrogen populations violate positivity or sum-to-one')
    # Remove only floating-point roundoff, never materially invalid solutions.
    f = np.maximum(f, 0)
    return f / f.sum(axis=1, keepdims=True)
