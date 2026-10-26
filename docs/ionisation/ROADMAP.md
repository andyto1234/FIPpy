# FIPpy ionisation implementation roadmap

Prepared 20 October 2026. This is an implementation plan, not a claim that the
proposed physics has been implemented or validated.

## Recommended destination

Replace the empirical Saha–CHIANTI bridge with an auditable population-rate
solver. Retain CHIANTI as a source of atomic rates and a coronal benchmark.
First solve metals on a prescribed atmosphere with a specified radiation field;
subsequently add multilevel physics, radiative transfer and time dependence.

The first supported set is the ten elements currently listed in
`physics/collision.py`: Ca, Mg, Fe, Si, S, C, O, N, Ar and Ne. Treat H as a
background with separately validated populations. Add He as a background and
eventually a fractionating species. An extensible interface is preferable to
claiming complete atomic coverage of the periodic table.

The main FIP output is the fraction in **all ionised stages**:

$$
\xi_X=1-f_{X^0}=1-\sum_{\ell\in X^0}f_\ell .
$$

It is not the singly ionised fraction and it is not one minus the neutral
ground-state population. Higher charge states matter when they affect this
fraction, charge-dependent coupling, opacity, electron density or additional
diagnostics. Resolving every highly ionised state is not the first requirement
for an accurate chromospheric neutral fraction.

## Physical priorities: what matters most and why

This is a provisional ranking for chromospheric FIP predictions, based on the
literature and FIPpy's current structure. It is not a measured ranking for the
user's simulation. Contributions vary with element, height and time.
Electron-impact ionisation and recombination are mandatory baseline processes;
their position below does not mean that they can be omitted.

1. **Hydrogen populations and a consistent background electron density.**
   H controls proton/neutral collision partners, charge transfer and important
   continuum opacity. Errors can therefore affect every metal simultaneously.
   Native HYDRAD H populations are preferable to a new equilibrium reconstruction
   of a dynamical atmosphere. Distinguish prescribed `ne` from a charge-conserving
   solve; do not silently change the HYDRAD thermodynamic background.
   Read Carlsson & Stein (2002), then the existing
   [hydrogen reconstruction notes](../hydrogen_reconstruction.md).
   This is a prerequisite, rather than one additional metal reaction.

2. **Photoionisation and the radiation field that drives it.**
   Local temperature does not determine the chromospheric ionisation state.
   Radiation from above and below, internal emission and attenuation can change
   where species become ionised relative to the ponderomotive force.
   Ground-state Lyα ionisation requires a threshold below its photon energy;
   do not apply it indiscriminately to all elements. Excited levels can have
   different thresholds. Start with specified spectra and a documented
   attenuation model; improve the transfer treatment later.
   Read Laming (2015), pp. 26–28; Verner et al. (1996); Dufresne et al. (2025).

3. **Charge transfer with H, followed by He where justified.**
   Include both ionisation and recombination directions. Oxygen is an excellent
   first case because exchange with H can strongly couple their ionisation.
   Sulphur is a critical next case: the 2021 Laming update shows that the adopted
   charge-transfer data need careful selection. A universal Langevin estimate
   is a documented approximation, not a substitute for reliable species-specific
   low-temperature data.
   Read Laming (2012), §3.3; Laming (2021), §4.3;
   Dufresne, Del Zanna & Badnell (2021).

4. **Electron-impact ionisation plus radiative and dielectronic recombination.**
   These establish the baseline balance and the hot-atmosphere limit. Include
   low-temperature DR where relevant and direct ionisation/excitation-autoionisation
   consistently with the selected dataset. Code these first because they provide
   the cleanest test of the solver; do not deploy their isolated result as the
   final chromospheric model.
   Read Dufresne et al. (2024); follow its underlying atomic-data references.

5. **Density-dependent recombination and metastable populations.**
   Density can suppress DR and alter populations of long-lived levels whose
   ionisation rates differ from the ground state. Add these together with a clear
   distinction between effective charge-state rates and explicit level rates.
   Never suppress rates twice. Read Dufresne et al. (2024, 2025) and
   Nikolić et al. (2018). The latter's fitted suppression is an approximation;
   it is not a replacement for a full collisional-radiative calculation.

6. **Time-dependent ionisation and advection.**
   This rises into the highest-priority group for rapid heating, shocks and
   evaporation whenever relaxation is too slow for equilibrium. It follows the
   equilibrium machinery in implementation order, not necessarily in physical
   importance. Requires initial populations, boundary conditions, velocity,
   geometry and adequately sampled atmosphere/radiation histories.
   Read Carlsson & Stein (2002), Osborne & Milić (2021).

7. **Self-consistent multilevel radiation transfer and helium evolution.**
   The radiation input is already essential at rank 2; solving it self-consistently
   is the later refinement here. H/He and metal opacity may couple radiation and
   populations nonlocally. He becomes an early priority if He fractionation or
   EUV shielding is the target. Read Dufresne et al. (2025),
   Osborne & Milić (2021), and Golding et al. (2016).

8. **Specialised extensions: nonthermal electrons, beams and hydrodynamic feedback.**
   Promote these only when the simulation or observation demands them.
   Beam-driven flares require beam excitation/ionisation earlier; very cool
   photospheric layers may require molecules and negative-ion opacity.
   A non-Maxwellian distribution cannot consistently be introduced by modifying
   one ionisation coefficient alone. A fully coupled calculation also feeds
   ionisation energy and radiative losses back into the energy equation.

Ranks 2 and 3 are effectively tied: diagnose their actual competition using
per-process rates and the resulting fractionation changes.

## Sequential implementation tickets

Each ticket should be a separate, reviewable change. The exit checks below are
proposed acceptance criteria, not completed tests or established physical
accuracy guarantees.

### 0. Freeze the reference calculation and define the interfaces

**Deliverable:** a reproducible baseline and an ionisation-provider boundary.

- Save legacy fractions, H fractions, collision rates, the local FIP integrand
  and final fractionation for both loop legs on selected actual profiles.
  Include initial, heating, cooling and steep-gradient cases.
- Make `physics/collision.py` consume an injected ionisation provider. Preserve
  `physics/saha.py` as an explicitly selected legacy backend during development.
- Proposed provider input: element, coordinates, time, Te, ne, total H density,
  H populations, optional He populations and radiation descriptor.
- Proposed result: state populations, total ionised fraction, process-resolved
  rates, relaxation estimates, validity flags and provenance.
- Use a single internal unit convention: K, cm, s, cm^-3, cm^3 s^-1 for
  binary coefficients, s^-1 for actual transition rates. Convert at boundaries.
- Cache by atmosphere, coordinates, radiation, atomic data and solver settings;
  temperature alone is not an adequate cache key.

**Exit check:** selecting the legacy provider reproduces the stored baseline
within numerical tolerance. No physics change in this ticket.

**Suggested location:** new `ionisation/` package, separate from the specialised
existing `hydrogen/` implementation.

### 1. Audit the atmosphere and build the atomic-data registry

**Deliverable:** explicit background assumptions and a coverage report.

- Verify that total H density and H I/H II populations have consistent meanings.
  Excited neutral H is still neutral. Prefer native Hstate where available.
- Report the electron charge-balance residual where the species inventory is
  sufficient. With prescribed ne, report rather than silently correct it.
- Record radiation and H/He sources. The ten hydrogen brightness temperatures
  in `hydrogen/radiation.py` are transition-specific parameters, not a general
  spectral mean intensity usable for every metal.
- For each transition record initial/final state, collider, reaction, data source,
  version, units, valid temperature/density range and extrapolation policy.
- Distinguish unavailable data from a physically zero rate. Unsupported species
  should fail explicitly or carry an explicit approximate-model status.
- Audit each of the ten target elements. CHIANTI v11's advanced low-stage models
  cover C, N, O, Ne, Mg, Si and S; this does not establish equivalent advanced
  coverage for Ca, Fe or Ar.

**Exit check:** one selected atmosphere has complete documented inputs, and
the first element's required rates have verified sources and domain checks.

**Read:** [R1], [R2], [R5], [R9], plus the hydrogen notes.

### 2. Implement and validate the equilibrium rate engine

**Deliverable:** a general rate-matrix builder and constrained equilibrium solver.

Use the convention already employed by `hydrogen/rates.py`:

$$
\frac{d\mathbf f}{dt}=\mathbf R\mathbf f,\qquad
R_{ij}=P_{j\rightarrow i}\;(i\ne j),\qquad
R_{jj}=-\sum_{i\ne j}P_{j\rightarrow i}.
$$

Columns sum to zero. Solve `R f = 0` with `sum(f) = 1`.
Start with electron-impact ionisation, RR and DR, retaining enough charge states
for the selected temperature range.

- Test the analytic two-state result: `f_ion = I/(I+R)`, for positive total
  ionisation and recombination rates I and R.
- Verify finite, nonnegative rates, population normalisation, positivity and
  a scaled equilibrium residual.
- Treat a disconnected network or nonunique equilibrium as a model error.
  Clipping negative populations must not conceal failed solves.
- Recover a coronal-equilibrium reference using identical atomic data and
  assumptions. This tests implementation, not independent physical accuracy.
- Do not demand Saha agreement yet: an optically thin ionisation/RR/DR network
  does not contain all inverse processes required for LTE detailed balance.

**Exit check:** analytic checks and matched-data coronal comparison pass;
charge-stage truncation is tested. This remains a diagnostic backend.

**Read:** [R5].

### 3. Add H charge transfer; use oxygen as the first physical demonstration

**Deliverable:** explicit rates for reactions such as
`X^q + H+ <-> X^(q+1) + H0`.

- Multiply coefficients by the density of the appropriate collision partner.
- Include both directions from consistent data; derive a reverse rate by detailed
  balance only where the state definitions and assumptions support it.
- Use the appropriate collision temperature/distribution for heavy particles;
  do not assume electron temperature is always the correct input.
- Add O first. Then S, with the source choice from the 2021 update documented;
  then the remaining elements as data coverage permits.
- Keep charge-transfer reaction rates distinct from elastic momentum-transfer
  collision frequencies in `physics/collision.py`.

**Exit check:** equilibrium and relaxation tests on a controlled exchange-only
network; demonstrate O–H coupling in the exchange-dominated regime without
forcing exact equality; record FIP changes when exchange is enabled.

**Read:** [R3], [R4], [R7]. Charge transfer and photoionisation are both required
before calling the result a chromospheric replacement.

### 4. Add photoionisation and an explicit radiation model

**Deliverable:** a radiation-to-rates interface and reproducible prescribed-field
chromospheric calculations.

For spectral mean intensity J_nu:

$$
\Gamma_i=4\pi\int_{\nu_{0,i}}^\infty
\frac{J_\nu}{h\nu}\sigma_i(\nu)\,d\nu .
$$

- Distinguish mean intensity, irradiance and photon flux; document the angular
  convention. Verify frequency/wavelength Jacobians and photon-energy units.
- First accept supplied J_nu and implement Verner cross-sections within their
  coverage and validity domains.
- Add a separately documented Laming-style incident-spectrum and trapped-Lyα
  prescription. Define illumination geometry for each leg of the loop.
- Calculate H continuum attenuation from neutral column density. Add He opacity
  once its populations are specified. State where neglected metal opacity,
  scattering or diffuse emission limits this approximation.
- Do not equate large Lyα optical depth with an arbitrary ionising flux.
  Trapped emission requires an explicit source/escape or transfer treatment.
- Keep low-energy photospheric/chromospheric radiation available in the interface;
  an EUV-only spectrum is not universally adequate for low-FIP atoms.

**Exit check:** zero-field limit; linear scaling of rates with imposed intensity;
threshold and quadrature checks; thin/thick limits of the chosen attenuation
approximation; spectral-grid convergence. Compare O, S, Si and Fe where their
data are available, using the same atmosphere for every comparison.

**Read:** [R1], [R2], [R7], [R8].

### 5. Add density effects and complete the first ten-element release

**Deliverable:** the first scientifically useful replacement for the bridge,
with per-element coverage and approximation labels.

- Include density suppression of DR using one coherent approach: already
  density-dependent effective rates, or a documented correction to zero-density
  rates. Do not combine them indiscriminately.
- Introduce metastable-resolved populations or validated effective rates where
  available; do not simultaneously include the same pathways in both forms.
- Separate a Laming-reproduction configuration from a modern-data configuration.
  Compare them on identical atmospheres so atomic-data changes can be identified.
- Add the remaining elements individually with data provenance and diagnostics.
  Do not fill gaps with unlabelled extrapolations from neighbouring elements.
- Keep all supported charge populations; pass `1 - f_neutral` to fractionation.
- Audit charge dependence of the ion momentum-transfer rates before interpreting
  substantially multiply ionised mixtures. A population-weighted single rate
  needs a derivation and validation against the assumed transport equations.

**Exit check:** no empirical Saha blending or spatial smoothing is needed in the
new backend. Test grid convergence of fractions and FIP bias, record process
sensitivities, and confirm unsupported domains are visible to the user.

**Read:** [R4], [R5], [R6], [R12].

### 6. Benchmark and improve the radiation/atomic-level treatment

**Deliverable:** a controlled comparison with a multilevel NLTE calculation,
starting with selected snapshots and elements rather than an entire time series.

- Use Lightweaver as a reference framework with explicitly chosen atomic models.
- Start with H and the C/Si/S models discussed by Dufresne et al. (2025).
- Separate experiments with fixed ne from charge-conserving runs; changing ne
  changes the physical problem.
- Compare level and neutral populations, J_nu and integrated photoionisation
  rates. Include sensitivity to boundary illumination and atomic-model size.
- In level-resolved models, avoid counting RR/DR twice through both explicit
  bound-free transitions and externally added total recombination coefficients.
- Add three-body and inverse radiative processes where needed for an LTE limit,
  with consistent statistical weights and partition functions.
- Add He I/II/III and, as required, excited He states. Validate EUV shielding and
  electron contributions before using an assumed He fraction as universal.

**Exit check:** either agreement with a matched independent calculation or a
quantified, understood discrepancy tied to a specific approximation. Test LTE
only where the implemented physics actually supports LTE.

**Read:** [R6], [R10], [R11].

### 7. Add time-dependent reactions, then transport

**Deliverable:** separately selectable equilibrium and nonequilibrium modes.

- First solve local `df/dt = R(t) f` with a stiff solver. Verify a constant-rate
  solution against a matrix exponential and its long-time equilibrium.
- Estimate slow relaxation times from nonzero rate-matrix eigenvalues; document
  assumptions and supplement with explicit perturbation relaxation tests.
- Compare these times with thermal evolution, advection across relevant gradients
  and changes in irradiation. Promote this ticket earlier for rapidly evolving
  HYDRAD cases where equilibrium fails.
- Add conservative transport of state densities:

  $$
  \partial_t(A n_i)+\partial_s(A u n_i)=A\sum_j R_{ij}n_j .
  $$

  Use the actual cross-sectional area A, velocity and boundary populations.
  This expression assumes a common bulk velocity for those populations;
  differential fractionating transport is a further extension.
- On AMR grids, test conservative remapping, state sums and boundary fluxes.
- Refine solver timesteps and saved atmospheric/radiation cadence independently.
  Interpolation cannot recover unsaved shocks or radiation variations.
- Do not equate time-dependent ionisation with time-dependent abundance
  fractionation: FIPpy's fractionation closure needs a separate validity check
  in rapidly evolving flows.

**Exit check:** reaction, transport and coupled convergence tests pass. Quantify
the change in final FIP predictions and dependence on initial/boundary states.
The existing experimental hydrogen time-dependent mode is not already certified
for this use; its documented convergence issues must first be resolved.

**Read:** [R9], [R10], [R11].

### 8. Couple feedback only when the science requires it

**Deliverable:** a clearly separate extension beyond prescribed-atmosphere
post-processing.

Couple the electron budget, ionisation energy, radiative cooling/heating and
possibly the wave response back to the atmosphere. Add nonthermal electron or
beam processes for compatible simulations, with consistent rates. Revalidate
fractionating transport if composition feeds back on the atmosphere.

**Exit check:** conservation and benchmark checks appropriate to the coupled
model, beyond population normalisation alone.

## How to establish the actual importance ranking in your run

At each physical release, keep the atmosphere and wave solution fixed while
changing one process. Record:

- Each forward and reverse rate in s^-1 and its contribution to the balance.
- Neutral fraction, relaxation time and local FIP integrand on both loop legs.
- Change in log fractionation, `Δ ln F_X`, and in the observable abundance ratio.
  If comparing X/O, also report the change in O; it is not a fixed reference.
- Parameter perturbations within documented uncertainties and alternative
  supported radiation fields.

Disabling a process is a useful attribution experiment, not necessarily a
realistic physical uncertainty. Also vary uncertain rates and radiation inputs
within defensible ranges. A large local rate does not by itself imply a large
change in FIP bias.

Evaluate the errors where the fractionation integral is sensitive, and inspect
local maxima as well as weighted averages. Do not hide a local neutral-fraction
error inside a domain-wide mean. Establish an acceptable FIP-error budget for the
intended scientific comparison before declaring a model adequate.

## First action

Implement **ticket 0**, then **tickets 1–2 for O**. Add O–H charge transfer,
then photoionisation. Use **S next** to exercise sensitive rate selection,
followed by **Si/Mg/Fe**, then **C/N/Ne/Ar/Ca** as datasets are verified.
This order is an engineering recommendation, not a claim that oxygen alone
validates other elements.

Do not switch the production default until tickets 0–5 have passed their checks
for the intended atmosphere and each supported species. Earlier tickets remain
useful selectable research backends.

## Reading pack

Eight open-access PDFs were downloaded; four of the user's Laming PDFs were
copied here for convenience. Downloads include preprints, which can have different
page numbering from journal versions. `references/manifest.json` records URLs,
source paths, page counts, retrieval date and SHA-256 hashes. These are papers,
not downloaded atomic-rate tables; obtaining and verifying the datasets remains
part of ticket 1.

- **[R1] Laming (2015)** — overall FIP/ionisation treatment, especially printed
  pp. 26–28. [Local PDF](references/Laming_2015_LivingRev.pdf).
- **[R2] Laming et al. (2019)** — §2.3: atmosphere, radiation and ionisation
  assumptions. [Local PDF](references/Laming_2019_ApJ.pdf).
- **[R3] Laming (2012)** — §§3.3–3.4: charge exchange and comparison of
  approximations. [Local PDF](references/Laming_2012_ApJ.pdf).
- **[R4] Laming (2021)** — §4.3: revised charge-transfer choices, particularly S.
  [Local PDF](references/Laming_2021_ApJ.pdf).
- **[R5] Dufresne et al. (2024)** — CHIANTI v11 advanced ionisation models.
  [Local PDF](references/Dufresne_2024_CHIANTI_v11.pdf) ·
  [Source](https://arxiv.org/abs/2403.16922).
- **[R6] Dufresne, Del Zanna & Osborne (2025)** — updated atomic models for UV
  radiative-transfer calculations. [Local PDF](references/Dufresne_2025_UV_radiative_transfer.pdf) ·
  [Source](https://arxiv.org/abs/2508.06422).
- **[R7] Dufresne, Del Zanna & Badnell (2021)** — photo-induced processes and
  charge transfer for C/O. Correction to the earlier chat: the third author is
  Badnell, not Storey.
  [Local PDF](references/Dufresne_2021_C_O_photoionisation_charge_transfer.pdf) ·
  [Source](https://arxiv.org/abs/2102.09278).
- **[R8] Verner et al. (1996)** — analytic ground-state photoionisation fits;
  use the coverage specified in the paper.
  [Local PDF](references/Verner_1996_Photoionisation.pdf) ·
  [Source](https://arxiv.org/abs/astro-ph/9601009).
- **[R9] Carlsson & Stein (2002)** — dynamical hydrogen ionisation and relaxation.
  [Local PDF](references/Carlsson_Stein_2002_Dynamic_Hydrogen.pdf) ·
  [Source](https://arxiv.org/abs/astro-ph/0202313).
- **[R10] Osborne & Milić (2021)** — Lightweaver framework, NLTE methods and
  time-dependent populations.
  [Local PDF](references/Osborne_Milic_2021_Lightweaver.pdf) ·
  [Source](https://arxiv.org/abs/2107.00475).
- **[R11] Golding, Leenaarts & Carlsson (2016)** — nonequilibrium H/He in a
  solar-atmosphere simulation.
  [Local PDF](references/Golding_2016_NE_Helium.pdf) ·
  [Source](https://arxiv.org/abs/1512.04738).
- **[R12] Nikolić et al. (2018)** — refined density-suppression fits for DR and
  their limits.
  [Local PDF](references/Nikolic_2018_DR_suppression.pdf) ·
  [Source](https://arxiv.org/abs/1808.07365).

Suggested first reading sequence: **R1 → R4 → R7 → R5 → R6**.
Use R8/R12 while implementing rates and R9–R11 for the dynamical/transfer stages.
