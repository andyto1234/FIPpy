# FIPpy

FIPpy is a research code for calculating solar First Ionisation Potential
(FIP) bias from HYDRAD loop atmospheres. It reads HYDRAD strand output,
solves the Alfven wave transport problem, estimates the ponderomotive
acceleration, and post-processes elemental fractionation along the loop. The
repository provides core Python modules (loop atmospheres, wave transport,
collision rates, ionisation fractions, FIP-bias calculations), driver scripts for
batch post-processing, and `tutorial_hydrad_output.ipynb`, which walks through a
minimal public workflow and representative plots.

This repository contains the source code used for the paper
[**"Chromospheric dynamics and turbulence regulate the solar FIP effect"**](https://ui.adsabs.harvard.edu/abs/2026arXiv260413174T/abstract)
by Andy S. H. To, J. Martin Laming, Jeffrey Reep, and Adam J. Finley.

If you use this code, please cite the paper listed in **Citation** below.

## When Should You Use This?

FIPpy is appropriate when you:

- Have HYDRAD loop output on disk that `pydrad` can parse.
- Want to estimate FIP bias and ponderomotive fractionation along a strand for
  elements treated in the collision and composition modules.
- Are reproducing or extending the workflow behind the associated paper.

## Installation

The code was developed with Python 3.11. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pydrad` must be able to parse your HYDRAD run directory. If you use a conda
environment for HYDRAD or solar-physics packages, installing the packages from
`requirements.txt` into that environment is also fine.

## Quick Start

For programmatic use, the main entry point is `process_hydrad_data(...)` in
`main_fip_bias_batch.py`:

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

See `tutorial_hydrad_output.ipynb` for plotting examples built from the returned
dictionary.

## Command-Line Driver

`main_fip_bias_batch.py` is the legacy/research batch driver. From the
repository root, typical usage is:

```bash
python main_fip_bias_batch.py /path/to/HYDRAD_output /path/to/output 700
```

The first argument is the HYDRAD directory, the third is the strand number, and
the second argument is retained for compatibility with the original script
interface. The driver keeps some author-local defaults and research-oriented
behaviour, and writes derived products under `results/` as in the original
workflow.

## Required Inputs

You need:

- A **HYDRAD output directory** path readable by `pydrad` (passed as
  `input_file` in `process_hydrad_data(...)`, or as the first CLI argument).
- A **strand number** (`strand_num` / third CLI argument).
- **Wave frequency and amplitude** (`frequency`, `amplitude` in
  `process_hydrad_data(...)`), in appropriate `astropy.units` form as in the
  example above.
- The **ionisation-equilibrium table**
  `chianti/ion_fraction_with_T_CHIANTI_v10.npz`, loaded by `physics/saha.py`
  from the repository root (see **Included Data**). If that file is absent,
  ionisation modelling falls back to Saha-only behaviour (see **Notes and
  Caveats**).

Common **optional** model parameters in `process_hydrad_data(...)` include
`acoustic_flux` and `v_add` (artificial turbulent velocity used in the paper
workflow). The Quick Start example shows typical values.

## Included Data

The repository bundles CHIANTI v10 ionisation-equilibrium data in:

```text
chianti/ion_fraction_with_T_CHIANTI_v10.npz
```

`physics/saha.py` resolves this path relative to the **repository root**, so
your shell working directory does not affect loading.

The archive contains:

- `temperature`: 1D array of temperature in kelvin (ascending)
- one array per element `{SYMBOL}_ioneq` (for example `Fe_ioneq`, `O_ioneq`) on
  the same grid

The shipped file covers C, N, O, Ne, Mg, Si, S, Ar, Ca, and Fe. To replace it
(for example for more elements or a newer CHIANTI build), extract the
corresponding `ioneq` data with the CHIANTI IDL or Python interfaces and save a
new `.npz` in that format under the same filename. If you still use an older
export named `ion_fraction_with_T.npz`, you can copy it to
`ion_fraction_with_T_CHIANTI_v10.npz` in `chianti/` as long as the key layout
matches.

## What Is Not Included

The following are not part of the repository:

- HYDRAD simulation output folders.
- Generated `results/`, figures, and pickle files from local runs.
- A full CHIANTI database installation (only the derived NumPy archive above is
  provided).

You must supply your own HYDRAD output when running the code.

## Outputs

**`process_hydrad_data(...)`** returns a Python dictionary `output_data` with
arrays and derived quantities used in the tutorial and paper-style workflows,
including:

- `s_values`
- `T`, `ne`, `B`, `rho`
- `ponderomotive_acceleration`
- `ionization_fractions`
- `fip_bias_heights`
- `frequency`, `amplitude`, `acoustic_flux`, `v_add`

The **`main_fip_bias_batch.py`** command-line driver writes batch products under
`results/` (for example images and intermediate files), reflecting author-local
defaults and the original driver design.

## Repository Layout

```text
composition/        FIP-bias and fractionation calculations
chianti/            `ion_fraction_with_T_CHIANTI_v10.npz` (bundled CHIANTI v10 table)
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

## Citation

If you use FIPpy or the methods in the paper, please cite:

```bibtex
@ARTICLE{To2026RSPTA.38450374T,
       author = {{To}, Andy S.~H. and {Laming}, J. Martin and {Reep}, Jeffrey and {Finley}, Adam J.},
        title = "{Chromospheric dynamics and turbulence regulate the solar FIP effect}",
      journal = {Philosophical Transactions of the Royal Society of London Series A},
     keywords = {solar corona, active region, elemental abundance, FIP bias, HYDRAD, Solar and Stellar Astrophysics, Space Physics},
         year = 2026,
        month = jul,
       volume = {384},
       number = {2323},
          eid = {20250374},
        pages = {20250374},
          doi = {10.1098/rsta.2025.0374},
archivePrefix = {arXiv},
       eprint = {2604.13174},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026RSPTA.38450374T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Notes and Caveats

This is a **research-code release**, and hopefully I will transfer this to a
better packaged library later. Some driver scripts retain paper-specific
defaults, historical command-line arguments, and author-local assumptions so
that the original studies can be rerun without modification.

For **new studies**, prefer calling `process_hydrad_data(...)` directly from
your own scripts or notebooks and pass explicit paths and physical parameters.

If the CHIANTI table is missing, `physics/saha.py` falls back to **Saha-only**
ionisation fractions. Paper-style ionisation blending and FIP results should use
the bundled `chianti/ion_fraction_with_T_CHIANTI_v10.npz` or an equivalent
replacement in the same format.
