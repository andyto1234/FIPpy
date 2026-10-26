"""Explicit, cached hydrogen reconstruction; never silently replace missing data."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import warnings
import numpy as np
from .profiles import Profile, profile_indices, read_profile, read_populations, check_populations
from .rates import HydrogenRates, equilibrium, backward_euler
from .radiation import RadiationField
from .transport import remap, advect


@dataclass
class HydrogenHistory:
    indices: np.ndarray
    times: np.ndarray
    offsets: np.ndarray
    coordinates: np.ndarray
    populations: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.populations = check_populations(self.populations)
        if len(self.indices) != len(self.times) or len(self.offsets) != len(self.indices)+1:
            raise ValueError('Invalid hydrogen cache dimensions')
        if (self.offsets[0] != 0 or self.offsets[-1] != len(self.coordinates)
                or len(self.populations) != len(self.coordinates) or np.any(np.diff(self.offsets) <= 0)
                or np.any(np.diff(self.indices) <= 0) or np.any(np.diff(self.times) <= 0)
                or not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(self.coordinates))):
            raise ValueError('Invalid hydrogen cache offsets or ordering')

    def fraction(self, index, coordinates=None, *, root=None, time=None):
        found = np.flatnonzero(self.indices == index)
        if len(found) != 1:
            raise KeyError(f'Profile {index} not present in hydrogen cache')
        k = found[0]
        if root is not None and str(Path(root).resolve()) != self.metadata['hydrad_root']:
            raise ValueError('Hydrogen cache belongs to a different HYDRAD run')
        if time is not None and not np.isclose(time, self.times[k], rtol=0, atol=1e-5):
            raise ValueError('Hydrogen cache timestamp does not match profile')
        lo, hi = self.offsets[k:k+2]
        s = self.coordinates[lo:hi]
        if coordinates is not None and (np.shape(coordinates) != s.shape or not np.allclose(coordinates, s, rtol=1e-8, atol=1e-3)):
            raise ValueError('Hydrogen cache coordinates do not match profile; regenerate the cache')
        if root is not None:
            current = profile_digest(root, int(index))
            if current != self.metadata['profile_sha256'][str(int(index))]:
                raise ValueError('HYDRAD profile changed since cache creation; regenerate the cache')
        return self.populations[lo:hi, 5].copy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            np.savez_compressed(f, indices=self.indices, times=self.times, offsets=self.offsets,
                                coordinates=self.coordinates, populations=self.populations,
                                metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            metadata = json.loads(str(f['metadata']))
            if metadata.get('schema_version') != 1:
                raise ValueError('Unsupported hydrogen cache schema')
            return cls(*(f[k].copy() for k in ('indices','times','offsets','coordinates','populations')), metadata)


def profile_digest(root, index):
    base = Path(root)/'Results'/f'profile{index}'
    return hashlib.sha256(base.with_suffix('.amr').read_bytes()+base.with_suffix('.phy').read_bytes()).hexdigest()


def check_supported(root):
    """Reject known incompatible physics instead of silently omitting it."""
    files = ['HYDRAD/source/config.h', 'Heating_Model/source/config.h']
    active = '\n'.join((Path(root)/f).read_text() for f in files)
    for flag in ('OPEN_FIELD', 'BEAM_HEATING', 'NON_UNIFORM_CROSS_SECTION'):
        if re.search(r'^\s*#define\s+'+flag+r'\b', active, re.M):
            raise ValueError(f'{flag} is not supported by this reconstruction')


def reconstruct_hydrogen(root, *, mode='equilibrium', atomic_data_root=None,
                         stop_index=None, output_stride=1, max_step=0.25,
                         initial='equilibrium', cache=None, progress=None, input_stride=1):
    """Reconstruct profiles from physical data, never reading reference Hstate
    unless initial='saved' is explicitly selected for a diagnostic experiment.

    equilibrium: reproduce the bundled HYDRAD statistical-equilibrium closure.
    time_dependent: experimental prescribed-background evolution with first-order
    splitting, backward Euler reactions, and conservative implicit upwind transport.
    Only closed, constant-area, thermally heated loops are supported.
    """
    root = Path(root).resolve()
    if mode not in ('equilibrium', 'time_dependent') or initial not in ('equilibrium', 'saved'):
        raise ValueError('Unknown reconstruction mode or initial condition')
    if not isinstance(output_stride, int) or output_stride < 1:
        raise ValueError('output_stride must be a positive integer')
    if not isinstance(input_stride, int) or input_stride < 1:
        raise ValueError('input_stride must be a positive integer')
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError('max_step must be positive')
    if initial == 'saved' and mode != 'time_dependent':
        raise ValueError('Saved initialization is only meaningful for time_dependent mode')
    check_supported(root)
    indices = profile_indices(root)
    if stop_index is not None:
        if stop_index not in indices:
            raise ValueError(f'Unknown stop_index {stop_index}')
        indices = [i for i in indices if i <= stop_index]
    if indices[0] != 0:
        raise ValueError('Profile 0 is required to initialize the radiation prescription')
    if np.any(np.diff(indices) != 1):
        raise ValueError('Missing intermediate profiles; complete history is required')
    indices = sorted(set(indices[::input_stride] + [indices[-1]]))
    rates = HydrogenRates(atomic_data_root or root)
    first = read_profile(root, indices[0])
    field = RadiationField(rates, first)
    trad = field.temperatures(first)
    q = rates.generator(first.te, first.ne, trad)
    f = read_populations(root, first) if initial == 'saved' else equilibrium(q)
    records, digests, residuals = [], {}, []
    previous, previous_trad = first, trad
    if mode == 'time_dependent':
        warnings.warn('Experimental time-dependent hydrogen on a prescribed background; this is not validated by an equilibrium HYDRAD reference.', UserWarning)
    for step, index in enumerate(indices):
        p = first if step == 0 else read_profile(root, index)
        if step:
            if p.time <= previous.time or not np.isclose(p.length, first.length):
                raise ValueError('Nonincreasing timestamps or changing loop length')
            trad = field.temperatures(p)
            if mode == 'equilibrium':
                f = equilibrium(rates.generator(p.te, p.ne, trad))
            else:
                # Remap species and carrier together to the new AMR grid.
                species = remap(previous.edges, previous.nh[:, None]*f, p.edges)
                carrier0 = species.sum(axis=1)
                f = check_populations(species/carrier0[:, None])
                old_te = np.interp(p.s, previous.s, previous.te)
                old_ne = np.interp(p.s, previous.s, previous.ne)
                old_v = np.interp(p.s, previous.s, previous.velocity)
                old_trad = np.column_stack([np.interp(p.s, previous.s, previous_trad[:, j]) for j in range(10)])
                duration = p.time-previous.time
                substeps = int(np.ceil(duration/max_step))
                dt = duration/substeps
                carrier = carrier0.copy()
                for sub in range(substeps):
                    w = (sub+.5)/substeps
                    te = old_te*(1-w)+p.te*w
                    ne = old_ne*(1-w)+p.ne*w
                    v = old_v*(1-w)+p.velocity*w
                    rt = old_trad*(1-w)+trad*w
                    q = rates.generator(te, ne, rt)
                    # Equilibrium incoming populations are an explicit boundary assumption.
                    boundary = equilibrium(q[[0, -1]])
                    f, carrier = advect(p.edges, carrier, f, v, dt, boundary)
                    f = backward_euler(q, f, dt)
                # Prescribed atmosphere resets the carrier at each saved snapshot.
                # This is a diagnostic of continuity/transport mismatch, not a hidden correction.
                residuals.append(float(np.sum(abs(carrier-p.nh)*p.widths)/np.sum(p.nh*p.widths)))
        if step % output_stride == 0 or step == len(indices)-1:
            records.append((p.index, p.time, p.s.copy(), f.copy()))
            digests[str(index)] = profile_digest(root, index)
        if progress and (step % 100 == 0 or step == len(indices)-1):
            progress(step+1, len(indices), p.time)
        previous, previous_trad = p, trad
    lengths = [len(r[2]) for r in records]
    metadata = dict(schema_version=1, hydrad_root=str(root), atomic_data_root=str(rates.root.resolve()),
                    mode=mode, initial=initial, max_step=max_step, output_stride=output_stride,
                    input_stride=input_stride,
                    rates_sha256=rates.fingerprint, radiation_sha256=field.fingerprint,
                    profile_sha256=digests, background='prescribed Te, ne, nH and velocity',
                    boundary='equilibrium inflow; interior outflow',
                    radiation='HYDRAD density-scaled brightness and VAL initial critical columns; snapshot updates',
                    maximum_carrier_relative_l1=max(residuals, default=0),
                    validation_status='not automatically qualified for scientific use')
    history = HydrogenHistory(np.array([r[0] for r in records]), np.array([r[1] for r in records]),
                              np.r_[0, np.cumsum(lengths)], np.concatenate([r[2] for r in records]),
                              np.concatenate([r[3] for r in records]), metadata)
    if cache is not None:
        history.save(cache)
    return history
