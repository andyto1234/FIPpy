"""Rebuild FIPpy's collision-only equilibrium table from CHIANTI 11 rates.

Run with --database /path/to/CHIANTI_11.0.2_database.h5.
"""
import argparse
from pathlib import Path
import numpy as np
import astropy.units as u
import fiasco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    args = parser.parse_args()
    temperature = np.geomspace(1e3, 1e9, 1201) * u.K
    result = dict(temperature=temperature.value, database=str(args.database),
                  fiasco_version=fiasco.__version__, model='CHIANTI 11 CIE; all charge states')
    for symbol in ['Ca', 'Mg', 'Fe', 'Si', 'S', 'C', 'N', 'O', 'Ne', 'Ar']:
        element = fiasco.Element(symbol, temperature, hdf5_dbase_root=args.database)
        fractions = element.equilibrium_ionization.value
        if not np.all(np.isfinite(fractions)):
            raise ValueError(f'Nonfinite populations for {symbol}')
        np.testing.assert_allclose(fractions.sum(axis=1), 1, atol=1e-12)
        result[f'{symbol}_ioneq'] = np.clip(fractions[:, 1:].sum(axis=1), 0, 1)
        print(symbol, 'complete', flush=True)
    output = Path(__file__).resolve().parents[1] / 'chianti/ion_fraction_with_T_CHIANTI_v11.npz'
    np.savez_compressed(output, **result)
    print(output)


if __name__ == '__main__':
    main()
