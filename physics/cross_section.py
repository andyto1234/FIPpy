"""Ground-state photoionisation cross-sections from Verner et al. (1996).

Reference: https://doi.org/10.1086/177435
Data: https://www.pa.uky.edu/~verner/photo.html

The bundled table lives in the repository's database directory. An explicit
``path`` can select another copy. These fits are not electron-impact cross-
sections and do not supply a radiation field or a photoionisation rate.
"""

from functools import lru_cache
from pathlib import Path
import numpy as np
import astropy.units as u

__all__ = ["photoionisation_cross_section", "verner_parameters", "parse_ion_name"]

_DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "database" / "photo.dat"

_COLUMNS = ("Z", "N", "E_th", "E_max", "E0", "sigma0", "ya", "P", "yw", "y0", "y1")

_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 16: "S",
    18: "Ar", 20: "Ca", 26: "Fe",
}
_Z_OF = {symbol.lower(): z for z, symbol in _SYMBOLS.items()}

_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19,
    "XX": 20, "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25,
    "XXVI": 26,
}


@lru_cache(maxsize=8)
def _load_photo_dat(path):
    """Cache immutable rows; public access returns independent dictionaries.

    Table files are treated as immutable during a process. Restart after editing
    a table in place.
    """
    rows = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        parts = line.split()
        if len(parts) != 11:
            raise ValueError(f"{path}:{line_number}: expected 11 columns.")
        try:
            values = tuple(float(value) for value in parts)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric data.") from exc
        z, n, eth, emax, e0, sigma0, ya, p, yw, y0, y1 = values
        if (
            not np.all(np.isfinite(values))
            or z != int(z) or n != int(n) or not 1 <= n <= z
            or not 0 < eth < emax or min(e0, sigma0, ya, p) <= 0
            or min(yw, y0, y1) < 0
        ):
            raise ValueError(f"{path}:{line_number}: invalid fit parameters.")
        key = (int(z), int(n))
        if key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate species {key}.")
        seen.add(key)
        rows.append((*key, *values[2:]))
    if not rows:
        raise ValueError(f"No fit parameters found in {path}.")
    return tuple(rows)


def _positive_integer(value, name):
    """Accept integers or decimal integer strings, never truncate floats."""
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return int(value)


def _ion_stage(ion, z):
    if isinstance(ion, str):
        ion = _ROMAN.get(ion.strip().upper(), ion)
    stage = _positive_integer(ion, "ion stage")
    if stage > z:
        raise ValueError(f"Ion stage must be between 1 and {z}; bare nuclei have no bound electrons.")
    return stage


def _atomic_number(element):
    if isinstance(element, (bool, np.bool_)):
        raise ValueError("A boolean is not an atomic number.")
    if isinstance(element, (int, np.integer)):
        z = int(element)
    else:
        text = str(element).strip()
        if text.isdigit():
            z = int(text)
        else:
            z = _Z_OF.get(text.lower())
    if z not in _SYMBOLS:
        known = ", ".join(_SYMBOLS[z] for z in sorted(_SYMBOLS))
        raise ValueError(f"{element!r} is not in photo.dat. Available: {known}")
    return z


def verner_parameters(
    element,
    ion=None,
    electrons=None,
    path=None,
):
    """Verner et al. (1996) fit parameters for one element.

    element: symbol or atomic number, e.g. "Fe" or 26.
    ion: spectroscopic stage, e.g. 1 or "I" for the neutral. Optional.
    electrons: electron count N. Neutral is N = Z. Optional.

    Returns one dict, or a list of dicts (neutral first) if neither
    ion nor electrons is given. Keys are Z, N, E_th, E_max, E0,
    sigma0, ya, P, yw, y0, y1. Energies are eV and sigma0 is Mb.
    """
    if ion is not None and electrons is not None:
        raise ValueError("Pass ion or electrons, not both.")

    z = _atomic_number(element)
    data_path = _DEFAULT_DATA_PATH if path is None else Path(path).expanduser().resolve()
    rows = sorted(
        [dict(zip(_COLUMNS, row)) for row in _load_photo_dat(str(data_path)) if row[0] == z],
        key=lambda row: -row["N"],
    )
    if not rows:
        raise ValueError(f"No parameters for {_SYMBOLS[z]} in {data_path}.")
    if electrons is None and ion is not None:
        electrons = z - _ion_stage(ion, z) + 1
    if electrons is None:
        return rows

    electrons = _positive_integer(electrons, "electrons")
    for row in rows:
        if row["N"] == electrons:
            return row
    raise ValueError(
        f"No photo.dat row for {_SYMBOLS[z]} with N={electrons} electrons "
        f"(ion stage {z - electrons + 1})."
    )


def parse_ion_name(name):
    """Split 'Fe I' or 'Fe 1' into (element, spectroscopic stage)."""
    parts = str(name).replace("_", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Expected 'Fe I' or 'Fe 1', got {name!r}.")
    element, stage = parts
    z = _atomic_number(element)
    return _SYMBOLS[z], _ion_stage(stage, z)


def photoionisation_cross_section(
    energy,
    element,
    ion=1, 
    path=None,
):
    """Photoionisation cross-section from the Verner et al. (1996) fit.
    element is a symbol or atomic number. ion is the spectroscopic stage
    (1 is neutral, so Fe I is element="Fe", ion=1). ``energy`` must be an
    Astropy Quantity convertible to energy; scalars and arrays are supported.
    Returns a Quantity in cm^2, with the input shape, zero below threshold.
    Energies at or above the tabulated E_max are rejected. For species with
    more than two electrons, inner-shell terms are needed there. For H-like
    and He-like species, E_max marks the adopted nonrelativistic range.

    Convert wavelengths explicitly with ``wavelength.to(u.eV,
    equivalencies=u.spectral())``. No out-of-range extrapolation is performed.
    """
    p = verner_parameters(element, ion=ion, path=path)
    E = np.asarray(energy.to_value(u.eV))

    if np.any(E >= p["E_max"]):
        raise ValueError(
            f"Photon energies must be below {p['E_max']:g} eV for this fit."
        )

    E_eval = np.maximum(E, p["E_th"])
    x = E_eval / p["E0"] - p["y0"]
    y = np.sqrt(x**2 + p["y1"]**2)
    sigma = (
        1e-18 * p["sigma0"]
        * ((x - 1) ** 2 + p["yw"] ** 2)
        * y ** (0.5 * p["P"] - 5.5)
        * (1 + np.sqrt(y / p["ya"])) ** (-p["P"])
    )
    return np.where(E >= p["E_th"], sigma, 0.0) * u.cm**2
