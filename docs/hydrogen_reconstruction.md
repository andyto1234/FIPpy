# Hydrogen reconstruction on a prescribed HYDRAD atmosphere

This module is an **experimental reconstruction and validation tool**, not a
validated automatic substitute for missing hydrogen populations. Saved HYDRAD
H II populations remain the preferred input. If a profile has no Hstate and its
saved electron density equals its hydrogen density, the loop properties use the
fully ionised quasi-neutral closure (ne = nH; Bradshaw & Cargill 2013) and record
that source. That is the closure those runs already evolved, not an LTE chromosphere.

## What was implemented

- Six hydrogen states (five bound levels and the proton continuum).
- Direct loading of the original HYDRAD atomic tables, RH collisional factors,
  and log-linear/bilinear radiative interpolation. Lyman radiative transitions
  are in detailed balance, as in the supplied HYDRAD implementation.
- A reconstruction of the density-scaled radiation-temperature prescription,
  including VAL initial critical heights/column masses and snapshot updates.
- `equilibrium`: solve statistical equilibrium on the prescribed Te and ne.
- `time_dependent`: experimental backward-Euler reactions, implicit upwind
  finite-volume transport, and conservative remapping across AMR grids.
- Explicit equilibrium inflow boundaries, equilibrium or diagnostic saved
  initialization, and a carrier-density continuity diagnostic.
- Cached populations with source/profile fingerprints, coordinates, timestamps,
  assumptions, and checks against accidentally using another simulation's cache.
- Optional `hydrogen_history` in both FIP entry points. Output includes provenance.
- Native hydrogen now uses `level_population_hydrogen_6`, not `1 - ground_state`.
  Hydrogen interpolation holds endpoint values beyond the cell centers.

The source accompanying the supplied reference solves `Q f = 0` at each physical
update (`CRadiativeRates::SolveHIIFraction`). Non-LTE does **not** by itself mean
time-dependent populations. Therefore that run validates the equilibrium rate
port and approximate radiation reconstruction; it cannot establish accuracy of
the separate time-dependent model.

## Input location

The folder under this repository's `results/` contains FIP outputs, not the raw
simulation. The matching raw simulation found on this machine is:

```text
/Volumes/Crucial X9/Python_Script/fip_bias_hydrad/hydards/HYDRAD_flare_LL6e9_0.009_train_with_H_thermal_heating_2000s_1s_step_4_heatings
```

It contains **1519** profiles, covering **0–1518 seconds**, despite the 2000s name.
Original simulation files are never modified.

## Use in a notebook

```python
from hydrogen import reconstruct_hydrogen, HydrogenHistory
from main_fip_bias_batch import process_hydrad_data
import astropy.units as u

root = "/path/to/raw/HYDRAD/run"

# Explicitly choose the approximation; native Hstate is not read here.
history = reconstruct_hydrogen(
    root,
    mode="equilibrium",
    cache="results/hydrogen_validation/my_history.npz",
)

# In a later session:
# history = HydrogenHistory.load("results/hydrogen_validation/my_history.npz")

output_data = process_hydrad_data(
    input_file=root,
    frequency=0.206 / u.s,
    amplitude=0.97 * u.km / u.s,
    output_dir="results/reconstructed_fip",  # keep distinct from native results
    strand_num=30,
    acoustic_flux=1e8 * u.erg / u.cm**2 / u.s,
    v_add=0 * u.km / u.s,
    hydrogen_history=history,
)
```

Omit `hydrogen_history` to use native H II. Supplying a cache explicitly overrides
native populations, which permits validation with identical backgrounds.
`strand_num` retains the existing API name but selects a time-profile index.
No changes were made to the other elements' ionisation models or to the existing
mass-density convention.

For experiments with non-equilibrium evolution:

```python
history = reconstruct_hydrogen(
    root,
    mode="time_dependent",
    max_step=0.25,  # a starting value, NOT a convergence guarantee
    initial="equilibrium",
    stop_index=100,
)
```

`initial="saved"` uses **only profile 0** Hstate and is a diagnostic sensitivity
experiment, not a reconstruction available for runs without Hstate. Later Hstate
files are read only by the separate validation functions. `input_stride=10`
subsamples the history for cadence tests; `output_stride=10` merely reduces the
cache's saved profiles and still processes the complete input history.

The atomic data can be supplied separately with `atomic_data_root=...`. This
changes which rate tables/VAL atmosphere are used, not the target hydrodynamics.

## Reproduce checks

Use the `hydrad-fip` Python environment, from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/check_hydrad_rates.py "$HYDRAD_RUN" --output results/hydrogen_validation/rate_port.json
python -m hydrogen "$HYDRAD_RUN" --cache results/hydrogen_validation/equilibrium.npz --validate results/hydrogen_validation/equilibrium.json --plot results/hydrogen_validation/equilibrium.png
python -m hydrogen "$HYDRAD_RUN" --mode time_dependent --cache results/hydrogen_validation/time_dependent.npz --validate results/hydrogen_validation/time_dependent.json
python scripts/hydrogen_sensitivity.py "$HYDRAD_RUN" --output results/hydrogen_validation
python scripts/validate_hydrogen_fip.py "$HYDRAD_RUN" results/hydrogen_validation/equilibrium.npz --indices 30 677 --output results/hydrogen_validation/fip
```

`check_hydrad_rates.py` compiles the original C++ rate source into a temporary
executable and compares 54 independently calculated equilibrium populations.
It requires a C++ compiler. No C++ build is required for ordinary reconstruction.
The FIP check actually runs the current batch pipeline twice per requested profile
and writes outputs into separate native/reconstructed folders.

## Validation findings (2026-10-13)

- C++ reference check: 54 cases passed; maximum absolute population difference
  **4.75e-12**.
- Initial atmosphere: maximum H II fraction difference **7.44e-6**.
- Equilibrium reconstruction over all 1519 profiles: chromospheric
  width-weighted mean absolute H II error **3.89e-4**; worst absolute error
  **0.204** at t=677 s. Chromosphere here means electron temperature <=100,000 K.
  The mean is weighted by cell width across saved snapshots, not by the FIP
  integrand, and must not be used to dismiss the local discrepancy.
- Actual FIP pipeline, all ten elements at profiles 30 and 677: maximum relative
  curve difference **36.1%** at profile 30 (S, right leg), and **4.15%** at
  profile 677 (S, left leg). Native comparisons use the corrected continuum
  fraction, not the old ground-state complement.
- Time-dependent mode over all profiles: mean absolute chromospheric error
  **0.0131**, maximum **0.998** against this equilibrium reference. This is a
  different physical model and does not pass as a reconstruction of that run.
- Halving max_step from .25 to .125 s changes local time-dependent fractions
  by up to **0.166**; this default step is **not numerically converged**.
- Using only every tenth input profile changes fractions by up to **0.793**.
- Replacing equilibrium initialization with saved profile-0 populations changes
  fractions by at most **7.44e-6**; initial population error is not the dominant
  issue in this experiment.

Machine-readable reports, caches and plots are in `results/hydrogen_validation/`
(ignored by Git). `screening_passed` uses an explicitly provisional maximum
absolute fraction tolerance of .05; it is **false**, not a certification of the
scientific model. The integration tests prove the code runs; they do not prove
acceptable scientific accuracy.

## Limits and next scientific step

Radiation temperatures and internal-timestep critical-height updates are not
saved in Hstate. Their reconstruction is approximate, even with an exact atomic
rate port. A decisive next experiment is to save those quantities from HYDRAD and
compare the same rate solve using the actual radiation field. A time-dependent
reference is separately needed to validate time-dependent physics.

The experimental transport is first order and diffusive. It transports the six
species conservatively and obtains their fractions from their sum, then conditions
the carrier on saved hydrogen density at each output. Metadata reports the
maximum width-integrated relative L1 carrier mismatch. This conditioning and fixed
Te/ne make it a prescribed-background model, not a hydrodynamically self-consistent
calculation. Step/grid/cadence convergence must be demonstrated for the case of
interest. Adaptive output cannot recover sub-cadence shocks or radiation history.

Only closed, constant-area, thermally heated loops with suitable cool footpoints
are supported. There is no beam ionisation, helium solver, non-local radiative
transfer, or energy feedback. Known incompatible build flags are rejected; users
must also verify their run geometry. Existing metal ionisation/FIP prescriptions
remain independent approximations.

**The original Emax0.5 run is not supported by this radiation prescription:** its
initial atmosphere has no <=9000 K footpoint on a leg. The reconstruction
explicitly rejects it, even when supplied with the reference atomic data. The FIP
pipeline can still read that run: it has ne = nH and no Hstate, so hydrogen is
taken as fully ionised. That does not invent a cool chromosphere.
