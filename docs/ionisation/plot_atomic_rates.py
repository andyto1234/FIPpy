"""Plot available fiasco rate coefficients without replacing missing data.

Example:
    python docs/ionisation/plot_atomic_rates.py --ion 'Fe II' --output rates.png
"""
import argparse

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from fiasco import Ion
from fiasco.util.exceptions import MissingDatasetException


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ion', default='Fe II')
    parser.add_argument('--database', help='Path to the CHIANTI HDF5 database')
    parser.add_argument('--output', help='Save figure instead of opening a window')
    args = parser.parse_args()
    options = {} if args.database is None else {'hdf5_dbase_root': args.database}
    ion = Ion(args.ion, np.logspace(4, 8, 100) * u.K, **options)
    components = [
        ('recombination_rate', 'Total recombination', 'C0', '-'),
        ('dielectronic_recombination_rate', 'Dielectronic recombination', 'C0', '--'),
        ('radiative_recombination_rate', 'Radiative recombination', 'C0', ':'),
        ('ionization_rate', 'Total ionisation', 'C1', '-'),
        ('direct_ionization_rate', 'Direct ionisation', 'C1', '--'),
        ('excitation_autoionization_rate', 'Excitation-autoionisation', 'C1', ':'),
    ]
    fig, ax = plt.subplots(tight_layout=True)
    missing = []
    for attribute, label, colour, style in components:
        try:
            coefficient = getattr(ion, attribute).to_value(u.cm**3 / u.s)
        except MissingDatasetException as error:
            print(f'{label}: not available ({error})')
            missing.append(label)
            continue
        values = np.where(coefficient > 0, coefficient, np.nan)
        ax.plot(ion.temperature.to_value(u.K), values,
                label=label, color=colour, ls=style)
    ax.set(xscale='log', yscale='log', xlabel='Electron temperature [K]',
           ylabel=r'Rate coefficient [cm$^3$ s$^{-1}$]', title=args.ion)
    ax.legend(fontsize=8)
    if missing:
        ax.text(.02, .02, 'Not tabulated: ' + ', '.join(missing),
                transform=ax.transAxes, fontsize=8, wrap=True)
    if args.output:
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
