"""Reference spectra and prescribed radiation boundary fields.

WHI data: Woods et al. (2009), doi:10.1029/2008GL036373.
https://lasp.colorado.edu/lisird/resources/whi_ref_spectra/data/ref_solar_irradiance_whi-2008_ver2.dat

This module reads local data only. It is separate from HYDRAD's hydrogen
radiation-temperature reconstruction in ``hydrogen.radiation``.
"""

from io import StringIO
from pathlib import Path

import astropy.units as u
from astropy.constants import au, R_sun
import numpy as np

__all__ = [
    "load_whi_spectrum", "surface_hemisphere_mean_intensity",
    "two_sided_mean_intensity",
]

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "ref_solar_irradiance_whi-2008_ver2.dat"
)
_SPECTRUM_COLUMNS = {"sunspot": 1, "faculae": 2, "quiet": 3}
_IRRADIANCE_UNIT = u.W / (u.m**2 * u.nm)


def load_whi_spectrum(path=None, spectrum="quiet"):
    """Read a WHI spectrum and return wavelength, irradiance and source flags.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Local WHI .dat file. Defaults to the repository's database directory.
    spectrum : {"quiet", "sunspot", "faculae"}
        Quiet: April 10–16; sunspot: March 25–29; faculae: March 30–April 4,
        all in 2008.

    Returns
    -------
    wavelength : astropy.units.Quantity
        Bin centres in nm. The reference file uses 0.1-nm-wide bins.
    irradiance : astropy.units.Quantity
        Spectral irradiance at 1 au, in W m^-2 nm^-1 (not per-bin energy).
    source_flags : numpy.ndarray
        Integer provenance flags; -1 denotes filled bins, which are retained.

    No geometrical conversion, attenuation or spectral filtering is applied.
    """
    if spectrum not in _SPECTRUM_COLUMNS:
        raise ValueError("spectrum must be 'quiet', 'sunspot' or 'faculae'.")

    path = _DEFAULT_PATH if path is None else Path(path).expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        (i + 1 for i, line in enumerate(lines)
         if line.strip().startswith("5 columns")),
        None,
    )
    if start is None:
        raise ValueError(f"WHI data header not found in {path}.")

    data = np.loadtxt(StringIO("\n".join(lines[start:])), ndmin=2)
    if data.shape[0] == 0 or data.shape[1] != 5:
        raise ValueError("WHI data must contain five numerical columns.")
    if not np.all(np.isfinite(data)):
        raise ValueError("WHI data must be finite.")
    if np.any(data[:, 0] <= 0) or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("WHI wavelengths must be positive and increasing.")
    if np.any(data[:, 1:4] < 0):
        raise ValueError("WHI irradiances must be nonnegative.")
    if np.any(data[:, 4] != np.floor(data[:, 4])):
        raise ValueError("WHI source flags must be integers.")

    return (
        data[:, 0] * u.nm,
        data[:, _SPECTRUM_COLUMNS[spectrum]] * _IRRADIANCE_UNIT,
        data[:, 4].astype(int),
    )


def surface_hemisphere_mean_intensity(irradiance):
    """Convert F_lambda at 1 au to a prescribed near-surface J_lambda.

    Assumes a uniformly bright surface filling one hemisphere, so
    J_lambda = F_lambda * (au / R_sun)^2 / (2*pi).

    ``irradiance`` must be an Astropy Quantity per unit wavelength. Returns
    erg s^-1 cm^-2 sr^-1 nm^-1, preserving the input shape. This is an
    unattenuated boundary-field approximation, not the radiation inside a
    chromosphere. Applying it to downward EUV requires an additional physical
    assumption; it does not infer coronal illumination geometry.
    """
    flux = irradiance.to(_IRRADIANCE_UNIT)
    scale = (au / R_sun).decompose().value**2
    return (flux * scale / (2 * np.pi * u.sr)).to(
        u.erg / (u.s * u.cm**2 * u.sr * u.nm)
    )


def two_sided_mean_intensity(J_bottom, J_top, N_bottom, N_top, sigma):
    """Attenuate and sum two prescribed incident mean-intensity contributions.

    J_bottom, J_top : Quantity, shape (n_wavelengths,)
        Boundary contributions to J_lambda, already including their angular
        normalisation (no extra hemisphere factor is applied here).
    N_bottom, N_top : Quantity, shape (n_positions,)
        Absorber column densities from each boundary to each position [cm^-2].
        For H I opacity, supply neutral-H columns, not total-H columns.
    sigma : Quantity, shape (n_wavelengths,)
        Absorption cross-section [cm^2], on the same grid as both spectra.

    Returns J_lambda with shape (n_positions, n_wavelengths), in
    erg s^-1 cm^-2 sr^-1 nm^-1:
    J_bottom exp(-N_bottom sigma) + J_top exp(-N_top sigma).

    This is effective-path absorption only, without internal emission,
    scattering or angular transfer. The caller defines ray geometry and
    boundary spectra; bottom/top normally refer to one atmospheric column,
    not automatically the two footpoints of a curved magnetic loop.
    Zero cross-sections transmit both incident components unchanged.
    """
    unit = u.erg / (u.s * u.cm**2 * u.sr * u.nm)
    bottom = np.asarray(J_bottom.to_value(unit))
    top = np.asarray(J_top.to_value(unit))
    nb = np.asarray(N_bottom.to_value(u.cm**-2))
    nt = np.asarray(N_top.to_value(u.cm**-2))
    cross_section = np.asarray(sigma.to_value(u.cm**2))

    arrays = (bottom, top, nb, nt, cross_section)
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Spectra, columns and cross-section must be 1D arrays.")
    if bottom.shape != top.shape or bottom.shape != cross_section.shape:
        raise ValueError("Both spectra and cross-section must share a wavelength grid.")
    if nb.shape != nt.shape:
        raise ValueError("Both columns must share a spatial grid.")
    if any(np.any(~np.isfinite(a)) or np.any(a < 0) for a in arrays):
        raise ValueError("Intensities, columns and cross-sections must be finite and nonnegative.")

    # Reuse buffers rather than retaining separate optical-depth arrays.
    result = -nb[:, None] * cross_section[None, :]
    np.exp(result, out=result)
    result *= bottom[None, :]
    incoming_top = -nt[:, None] * cross_section[None, :]
    np.exp(incoming_top, out=incoming_top)
    incoming_top *= top[None, :]
    result += incoming_top
    return u.Quantity(result, unit, copy=False)
