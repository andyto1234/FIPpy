"""Adjacent-charge-state stationary balance; no empirical regime blending."""
import numpy as np
import astropy.units as u
from scipy.special import logsumexp


def adjacent_equilibrium(upward, downward):
    """Rates shaped (positions, Z), connecting q and q+1, in s^-1.

    Returns normalised populations shaped (positions, Z+1). Positive reverse
    rates ensure a unique equilibrium even where upward rates underflow to zero.
    """
    up = np.asarray(upward.to_value(1/u.s))
    down = np.asarray(downward.to_value(1/u.s))
    if up.ndim != 2 or up.shape != down.shape or up.shape[1] == 0:
        raise ValueError('Rates must have matching (positions, Z) shapes.')
    if (np.any(~np.isfinite(up)) or np.any(~np.isfinite(down))
            or np.any(up < 0) or np.any(down <= 0)):
        raise ValueError('Require finite upward >= 0 and downward > 0 rates.')
    with np.errstate(divide='ignore'):
        log_ratios = np.log(up) - np.log(down)
    log_weights = np.concatenate([np.zeros((len(up), 1)),
                                  np.cumsum(log_ratios, axis=1)], axis=1)
    return np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))


def recombination_coefficient(ion, electron_density=None):
    """Total recombination coefficient of one recombining ion.

    With no density, this is the zero-density CHIANTI rate from
    ``ion.recombination_rate``. With a density, dielectronic recombination is
    multiplied by the suppression factor of Nikolić et al. (2018, ApJS, 237,
    41, doi:10.3847/1538-4365/aad3c5). Radiative recombination is unchanged.
    The factor is evaluated with fiasco's implementation, which follows that
    paper and is tested against CHIANTI's ``ch_dr_suppress`` (Young, CHIANTI
    Technical Report No. 17).

    A CHIANTI total-recombination file (``.trparams``) is not a sum of the
    separate radiative and dielectronic files. CHIANTI's ``recomb_rate`` keeps
    that total and does not apply the suppression, and this function does the
    same. Fe II is the case that matters here: its preferred rate is the Nahar,
    Bautista & Pradhan (1997) total, not the separate Mazzotta dielectronic fit.

    The published fits were made to the collisional-radiative calculations of
    Summers (1974, 1979). Those calculations do not include recombination of
    singly charged ions, so q = 1 is an extrapolation (Nikolić et al. 2018,
    section III.2). Sequences with no tabulated core-excitation energy also
    receive no low-temperature turn-off of the suppression.
    """
    total = ion.recombination_rate
    if electron_density is None:
        return total, _recombination_info(np.ones(ion.temperature.shape), False, 'zero-density')
    factor = _nikolic_suppression_factor(ion, electron_density)
    if _has_inseparable_total(ion):
        return total, _recombination_info(factor, False, 'trparams')
    from fiasco.util.exceptions import MissingDatasetException
    try:
        dielectronic = ion.dielectronic_recombination_rate
    except MissingDatasetException:
        return total, _recombination_info(factor, False, 'no dr file')
    try:
        radiative = ion.radiative_recombination_rate
    except MissingDatasetException:
        radiative = np.zeros(ion.temperature.shape) * u.cm**3 / u.s
    return radiative + factor * dielectronic, _recombination_info(factor, True, 'rr + suppressed dr')


def _recombination_info(factor, applied, treatment):
    factor = np.asarray(factor, dtype=float)
    return dict(factor=factor, applied=bool(applied), treatment=treatment)


def _has_inseparable_total(ion):
    from fiasco.util.exceptions import MissingDatasetException
    try:
        ion._total_recombination_rate
    except MissingDatasetException:
        return False
    return True


def _nikolic_suppression_factor(ion, electron_density):
    """S(n_e, T; q, N) from Nikolić et al. (2018), via fiasco."""
    if ion.charge_state < 1:
        raise ValueError('Suppression applies to a recombining ion with q >= 1.')
    factor = np.asarray(ion._dielectronic_recombination_suppression(electron_density), dtype=float)
    # A bare nucleus has no core electron to excite, so fiasco returns a scalar 1.
    if factor.shape != ion.temperature.shape:
        if factor.size == 1:
            factor = np.full(ion.temperature.shape, float(factor))
        else:
            raise ValueError('Suppression factor does not match the ion temperature grid.')
    if np.any(~np.isfinite(factor)) or np.any((factor < 0) | (factor > 1)):
        raise ValueError('Suppression factor must lie in [0, 1].')
    return factor


def multistage_equilibrium(element, temperature, electron_density, *,
                           neutral_photoionisation=0/u.s,
                           neutral_cx_ionisation=0/u.s,
                           singly_ionised_cx_recombination=0/u.s,
                           suppress_dielectronic=False,
                           hdf5_dbase_root=None):
    """All-stage CHIANTI collisions plus supplied rates on the 0 <-> 1 pair.

    Higher-stage photoionisation/CX, excited levels, three-body recombination,
    advection and time dependence are omitted. CX inputs are rates per atom,
    already multiplied by the appropriate collision-partner density.

    ``suppress_dielectronic`` multiplies each stage's dielectronic coefficient
    by the Nikolić et al. (2018) factor at the local electron density. The
    default remains the zero-density rate, which is what fiasco's
    collision-only equilibrium uses. See ``recombination_coefficient``.
    """
    import fiasco
    atom = fiasco.Element(element, temperature, hdf5_dbase_root=hdf5_dbase_root)
    ne = electron_density.to_value(u.cm**-3)
    if np.shape(ne) != atom.temperature.shape or np.any(ne <= 0):
        raise ValueError('Positive electron density must match temperature grid.')
    coefficients = []
    factors = []
    applied = []
    treatment = []
    for q in range(atom.atomic_number):
        recombining = atom[q + 1]
        if suppress_dielectronic:
            coefficient, info = recombination_coefficient(recombining, electron_density)
        else:
            coefficient, info = recombining.recombination_rate, _recombination_info(
                np.ones(atom.temperature.shape), False, 'zero-density')
        values = coefficient.to_value(u.cm**3 / u.s)
        if values.shape != atom.temperature.shape or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f'{recombining.ion_name}: recombination coefficient must be finite and positive.')
        coefficients.append(values)
        factors.append(np.asarray(info['factor'], dtype=float))
        applied.append(info['applied'])
        treatment.append(info['treatment'])
    up = np.column_stack([atom[q].ionization_rate.to_value(u.cm**3/u.s)
                          for q in range(atom.atomic_number)]) * ne[:, None] / u.s
    electron_recombination = np.column_stack(coefficients) * ne[:, None] / u.s
    for rate in (neutral_photoionisation, neutral_cx_ionisation,
                 singly_ionised_cx_recombination):
        values = rate.to_value(1/u.s)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Additional rates must be finite and nonnegative.')
    up = up.copy()
    down = electron_recombination.copy()
    up[:, 0] = up[:, 0] + neutral_photoionisation + neutral_cx_ionisation
    down[:, 0] = down[:, 0] + singly_ionised_cx_recombination
    fractions = adjacent_equilibrium(up, down)
    return dict(populations=fractions, ionised_fraction=fractions[:, 1:].sum(axis=1),
                upward=up, downward=down, electron_recombination=electron_recombination,
                dielectronic_suppression=np.column_stack(factors),
                dielectronic_suppression_applied=np.asarray(applied, dtype=bool),
                recombination_treatment=treatment)
