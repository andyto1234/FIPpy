"""Reconstruct HYDRAD's approximate radiation temperatures from snapshots.

Critical columns are initialized from VAL.T and the initial mass distribution.
Bound-bound brightness temperatures scale with the density at 9000 K. Between
saved outputs we cannot recover HYDRAD's internal updates of critical columns.
"""
import hashlib
import numpy as np


class RadiationField:
    def __init__(self, rates, initial):
        directory = rates.directory
        transitions = np.loadtxt(directory / 'transitions.txt', skiprows=3)
        terms = np.loadtxt(directory / 'terms.txt', skiprows=3)
        if transitions.shape != (10, 5) or terms.shape != (10, 3):
            raise ValueError('Unrecognized HYDRAD radiation prescription')
        self.top = transitions[:, 2]
        self.nu = transitions[:, 3]
        self.power = terms[:, 2]
        self.scaling = np.array([.1188, .1116, .1061, .1460, .1402, .1979, 0, 0, 0, 0])
        val_path = directory.parent / 'VAL_atmospheres/VAL.T'
        val = np.loadtxt(val_path, skiprows=1)
        self.fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in (directory/'transitions.txt', directory/'terms.txt', val_path))).hexdigest()
        z = np.array([self.crossing(val[:, 1], val[:, 0], t) for t in transitions[:, 4]])
        self.critical_s = np.vstack([z, initial.length-z])
        self.scaled_top = np.tile(self.top, (2, 1))
        self.critical_column = np.empty((2, 10))
        columns = self.columns(initial)
        for side in range(2):
            self.critical_column[side] = np.interp(self.critical_s[side], initial.s, columns)

    @staticmethod
    def crossing(temperature, values, target):
        found = np.flatnonzero((temperature[:-1]-target)*(temperature[1:]-target) <= 0)
        if not len(found):
            raise ValueError(f'Atmosphere does not cross {target:.1f} K; cannot reconstruct radiation field')
        j = found[0]
        if temperature[j+1] == temperature[j]:
            return values[j]
        w = (target-temperature[j])/(temperature[j+1]-temperature[j])
        return values[j]*(1-w)+values[j+1]*w

    @staticmethod
    def columns(p):
        # HYDRAD includes the central cell in both integrations.
        center = np.searchsorted(p.s, p.length/2)
        center = min(center, len(p.s)-1)
        mass = p.rho*p.widths
        columns = np.empty_like(mass)
        columns[:center+1] = np.cumsum(mass[:center+1][::-1])[::-1]
        columns[center:] = np.cumsum(mass[center:])
        return columns

    def temperatures(self, p):
        column = self.columns(p)
        center = min(np.searchsorted(p.s, p.length/2), len(p.s)-1)
        for side, ids in enumerate((np.arange(center, -1, -1), np.arange(center, len(p.s)))):
            cold = ids[p.te[ids] <= 9000]
            if not len(cold):
                raise ValueError('No <=9000 K footpoint on a loop leg. This radiation prescription is not supported for this atmosphere.')
            foot_density = p.nh[cold[0]]
            top = self.top * max(foot_density/1.9e11, 1)**self.scaling
            for j in range(10):
                if top[j] == self.scaled_top[side, j]:
                    continue
                critical_t = 4.7979e-11*self.nu[j] / (-.19314718 + 4.7979e-11*self.nu[j]/top[j])
                self.critical_s[side, j] = self.crossing(p.te[ids], p.s[ids], critical_t)
                self.critical_column[side, j] = np.interp(self.critical_s[side, j], p.s, column)
                self.scaled_top[side, j] = top[j]
        side = (p.s > p.length/2).astype(int)
        a = 4.7979e-11*self.nu
        intensity = (1+(column[:, None]/self.critical_column[side])**self.power) / np.expm1(a/self.scaled_top[side])
        trad = a / np.log1p(1/intensity)
        thermal = (p.s[:, None] <= self.critical_s[0]) | (p.s[:, None] >= self.critical_s[1])
        return np.where(thermal, p.te[:, None], trad)
