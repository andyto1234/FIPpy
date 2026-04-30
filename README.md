# FIPpy

FIPpy is a research code for calculating solar First Ionisation Potential
(FIP) bias from HYDRAD loop atmospheres. It reads HYDRAD strand output,
solves the Alfven wave transport problem, estimates the ponderomotive
acceleration, and post-processes elemental fractionation along the loop.

This repository contains the bare source code used for the paper
**"Chromospheric dynamics and turbulence regulate the solar FIP effect"** by
Andy S. H. To, J. Martin Laming, Jeffrey Reep, and Adam J. Finley.

## What Is Included

- Core Python modules for loop atmospheres, wave transport, collision rates,
  ionisation fractions, and FIP-bias calculations.
- Driver scripts for running the post-processing pipeline on a HYDRAD output
  directory.
- A clean tutorial notebook, `tutorial_hydrad_output.ipynb`, showing how to
  run the code with your own HYDRAD output and recreate representative
  paper-style plots.

## What Is Not Included

Large or machine-local research products are intentionally excluded:

- HYDRAD simulation output folders.
- Generated `results/`, figures, and pickle files.
- Local CHIANTI database files and derived ion-fraction tables.

You will need to provide your own HYDRAD output directory. The code expects it
to be readable by `pydrad`.

## Installation

The code was developed with Python 3.11. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pydrad` must be able to parse your HYDRAD run directory. If you use a conda
environment for HYDRAD/solar physics packages, installing the packages from
`requirements.txt` into that environment is also fine.

## Required Inputs

At minimum, you need:

- `HYDRAD_DIR`: path to a HYDRAD output directory.
- `STRAND_NUM`: strand index to read from that HYDRAD output.
- Wave parameters such as frequency and amplitude.

Optional CHIANTI ionisation data can be provided at:

```text
chianti/ion_fraction_with_T.npz
```

If that file is not present, `physics/saha.py` falls back to Saha-only
ionisation fractions.

## Basic Usage

From the repository root, run the batch driver with a HYDRAD output directory:

```bash
python main_fip_bias_batch.py /path/to/HYDRAD_output /path/to/output 700
```

The current research driver keeps some author-local defaults and writes derived
products under `results/`. The first argument is the HYDRAD directory and the
third argument is the strand number. The second argument is retained for
compatibility with the original script interface.

For interactive use, the main public entry point is:

```python
import astropy.units as u
from main_fip_bias_batch import process_hydrad_data

output_data = process_hydrad_data(
    input_file="/path/to/HYDRAD_output",
    frequency=0.206 * u.s**-1,
    amplitude=0.97 * u.km / u.s,
    output_dir="/path/to/HYDRAD_output",
    strand_num=700,
    acoustic_flux=1e8 * u.erg / u.cm**2 / u.s,
    v_add=0.0 * u.km / u.s,
)
```

`output_data` is a dictionary containing arrays and derived quantities used by
the analysis notebooks, including:

- `s_values`
- `T`, `ne`, `B`, `rho`
- `ponderomotive_acceleration`
- `ionization_fractions`
- `fip_bias_heights`
- `frequency`, `amplitude`, `acoustic_flux`, `v_add`

See `tutorial_hydrad_output.ipynb` for plotting examples.

## Repository Layout

```text
composition/        FIP-bias and fractionation calculations
loop/               HYDRAD reader, loop atmosphere, magnetic-field helpers
maths/              Numerical interpolation helpers
physics/            Ionisation, collision, speed, and frequency utilities
wave/               Alfven and slow-mode wave solvers/extractors
main_fip_bias.py    Single-run research driver
main_fip_bias_batch.py
                    Batch/post-processing driver and public workflow entry point
optimize_amplitude.py
optimise_vadd.py    Parameter-study scripts used in the paper workflow
```

## Notes For Reuse

This is a research code release rather than a packaged library. Some scripts
still contain author-local paths and paper-specific defaults so that the
original workflow can be rerun unchanged. For new studies, prefer using
`process_hydrad_data(...)` from your own script or notebook and pass explicit
paths to your HYDRAD output.
