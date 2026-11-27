# What is validated

Every number loadpath reports comes from a model. This page says how far
each model has been checked, and against what. Read it before you trust a
safety factor, a temperature or a warp prediction with something that
matters.

Each claim carries one of these labels, strongest first:

| Label | Meaning |
| --- | --- |
| **Tested** | Compared with a physical part or coupon that was printed and measured for this project. |
| **Matched published experiment** | A published physical test reproduced from its stated geometry and material, with nothing fitted to it; the error against the measurement is reported, good or bad. |
| **Measured data** | Compared with published measurements of a physical part (for example NIST AM-Bench). Says whether a parameter was fitted to the same data. |
| **Analytic** | Matches a closed-form answer (beam theory, Lamé, Poiseuille) in the test suite. |
| **Cross-checked** | Agrees with an independent solver run on the same problem (CalculiX, or loadpath's second, tetrahedral solver). |
| **Benchmarked** | Compared with published optimization codes on the same problem (top88, top3d, Sigmund's 99-line mechanism code). This measures the optimizer, not the physics. |
| **Simulation only** | No outside check. The number is what the model says. |

## Nothing is physically tested yet

No loadpath prediction has been compared with a part printed and measured
for this project (published tests of other people's prints are below, under
[Printed lattices against published tests](#printed-lattices-against-published-tests)
and [Printed plates pulled through a pin](#printed-plates-pulled-through-a-pin)). A calibration kit is ready and waiting to be printed
([CALIBRATION.md](CALIBRATION.md)): stiffness and strength coupons along and
across layers, a gyroid-core beam, TPU creep and crush coupons, PETG warp
bars, and a bench test of the robot-arm wrist yoke with a predicted rotation per
kilogram. Until those numbers come back, treat every safety factor as an
estimate for comparing designs and catching undersized parts, not as a
certified strength.

## Structural analysis

| Claim | Label | Evidence |
| --- | --- | --- |
| Hex8 voxel FEA gives the right deflection | Analytic | Cantilever tip load within 3% and distributed load within 5% of Euler-Bernoulli (`tests/test_loadpath.py`). |
| Peak stress at a stress raiser | Analytic | Plate with a hole lands within about ±13% of Peterson's Kt; the reported safety factor carries a 1.15 allowance for this (`tests/test_loadpath.py`, `verify.STRESS_ALLOWANCE`). |
| Pressure loads | Analytic | Thick sphere reads 5-7% over Lamé's bore stress; thin closed cylinder 8-11% over hoop stress (`tests/test_pressure.py`). |
| Large deflection and buckling | Analytic | Tip-loaded cantilever against the elastica, column against Euler and its post-buckled shape (`tests/test_nonlinear.py`). |
| Contact and tendons in large deflection | Analytic | Tendon through guides curls a cantilever into the constant-curvature arc within 1%; a cantilever landing on a rigid roller, two fingers of one part pressed together (self contact), a ball driven into a cantilever and a cantilever pressed onto a frictionless floor through large deflection (elastica) within 3-5%; with Coulomb friction, drag = mu N within 1% on a rough floor and on the part itself, and the rough-floor elastica (mu 0.1) within 0.2% reaction, 1% friction (`tests/test_contact.py`, docs/MECHANISMS.md). |
| Contact with friction | CalculiX 2.21 | Same voxel meshes, C3D8I, NLGEOM, penalty contact with friction (`bench/contact_ccx.py`): see docs/MECHANISMS.md for the table. |
| Natural frequencies | Analytic | Cantilever f1 within 0.5% of Euler-Bernoulli, f2/f1 = 6.27 (`modal.py`, docs/IMPACT.md). |
| Thermal conduction and expansion | Analytic | Conduction exact to solver tolerance; bimetal curvature within 2.5% of Timoshenko (`tests/test_thermal.py`). |
| The delivered part's safety factor | Cross-checked | Every safety-factor search confirms its pick on a tetrahedral (P2) mesh of the delivered STL. On the same tet mesh, CalculiX and loadpath's native P2 solver agree within 0.1% on the governing peak, and to 3-4 digits for layered (anisotropic) studies. The voxel check reads 24-49% higher peak stress than the tet check on the motor mount and bell crank, so voxel-only numbers err safe. The tet mesh (isosurface stuffing) is deterministic: the same part reads the same every run. |
| Strength at a pinned or bolted hole (a bearing load through a pin or shaft) | Matched published experiment | 16 published tests of printed PLA and ABS plates pulled through a steel pin: the first-failure reading is under the measured break on all 16, 0.28-0.76x on the voxel check and 0.21-0.82x on the tet check, read at the hole's side (where the net-tension specimens broke) or on the face the pin bears on. Before 2026-11-25 the stress at the bore was left out of the reading, and on four specimens that read 1.29-1.42x the measured break: unsafe ([below](#printed-plates-pulled-through-a-pin)). |
| Layered (printed) stiffness and layer bond | Simulation only | Transversely isotropic stiffness and a tension-across-layers check, with factors from datasheets and literature. Interlayer shear is not checked separately. Waiting on the calibration coupons. |
| Material cards | Measured data (datasheets) | Every value cites its datasheet (`loadpath materials`). Datasheet values describe the manufacturer's test bars, not your printer. |
| Fatigue and creep of printed plastics | Simulation only | One published design curve for printed PLA (Ezeh and Susmel), scaled by strength for PETG, ABS and ASA; rough values for TPU, resins and metals. Not tested. |

## Topology optimization

| Claim | Label | Evidence |
| --- | --- | --- |
| The optimizer finds near-optimal layouts | Benchmarked | At equal material, 3-5% less stiff than top88 on the 2D MBB beam and L-bracket, 1.2% less stiff than top3d on its 3D cantilever, and stiffer than top3d's thresholded design on a 3D bridge. Most of the 2D gap is the cost of the minimum-member rule. The references are NumPy ports of the published MATLAB codes, not the originals ([case study](case-studies/benchmarks.md)). |
| Compliant mechanisms | Benchmarked | Within 1-2% of a port of Sigmund's 99-line mechanism code on the force inverter and gripper ([MECHANISMS.md](MECHANISMS.md)). |
| GE bracket challenge: 370 g at SF 1.06 | Simulation only | Level with the winning entries' ~80% reduction, but on an approximate, roomier envelope and checked only by loadpath's own analysis ([BENCHMARKS.md](BENCHMARKS.md)). |
| Eval and robot-arm part masses and safety factors | Simulation only, cross-checked solver | The numbers are model predictions under the stated loads. The solver behind them is cross-checked as above; the loads, material values and print quality are not ([eval case study](case-studies/evals.md), [the robot arm](case-studies/robot-arm.md)). |
| Run-to-run reproducibility | Measured on the model | Mass at the target safety factor moves 3.5-7% when the search starts from a different fraction, and 5% (median) when the grid pitch changes by 3%. Mass differences under about 5% are noise. |

## Fields, lattices and meshing

| Claim | Label | Evidence |
| --- | --- | --- |
| Lattice stiffness and strength fits | Simulation only | Unit-cell FEA (strut and TPMS cells) fitted by power laws. A coarse homogenization lands within 15% of the fits (`tests/test_loadpath.py`). Not tested; the gyroid-core coupon is in the calibration kit. |
| Homogenized lattice parts | Cross-checked (against resolved geometry) | A skin-and-gyroid bar and an octet bar analysed homogenized and fully resolved (`docs/fields.md`). Thin solid walls read about 3% stiffer homogenized. |
| Anisotropic lattice tensors (stiffness and directional wall stress) | Cross-checked (against resolved geometry and CalculiX) | Eleven bars and beams of cubic, BCCz, octet and gyroid cells, straight and turned 45 degrees, with and without a skin, analysed homogenized and resolved on 0.15-0.25 mm voxels; CalculiX C3D8I on the same voxels agrees with the resolved solve to 1e-8. Skinned: stiffness within 5%, peak stress 1.07-1.89x with the skin-junction factor (1.25x where lattice meets skin, fitted to the tall BCC beam). Stretched (aspect) cells, three skinned beams: stiffness within 2%, peak stress 1.01-1.55x, except one pull reading 0.88x at the edge of a 6 mm contact band by the loaded block. Skinless lattices need the free-surface knockdown and one case (turned cubic in twist) still reads 0.64x ([fields.md](fields.md#lattices-are-read-anisotropic), `bench/lattice_aniso_validate.py`). Not tested on prints. |
| Lattice orientation optimization (`orient`, `loadpath lattice-orient`) | Cross-checked (against resolved geometry): **gains refuted without a skin** | Homogenized, skinless lattice bars read 7-23% lighter turned (`bench/lattice_orient_mass.py`). At full resolution (0.15 mm voxels, `bench/lattice_orient_confirm.py`) every one reverses: at its thinner wall the turned bar's peak is 3.2x (BCCz, claimed -23%), 2.2x (cubic in twist, -15%) and 1.3x (octet pull, -7%) the as-laid bar's; the skinless BCCz L-bracket (`bench/lattice_zones_mass.py`, claimed -41% at equal SF, -51% at equal stiffness) comes out 1.97x the peak and 2.26x the compliance (0.2 mm). Cause: the free-surface knockdown reads square-cut cells at a bare face 1.5-2.8x conservative on the peak and bias-cut (turned) cells 0.8-1.1x, so the comparison favours turning. So the optimizers (`lattice-orient`, zones, `lattice-grade --orient`) keep bare-faced cells as laid by default and report `guard`; `--confirm resolved` checks a turn at full resolution (`resolved.resolved_check`) and keeps it only if it holds. One turn has held: a graded skinless cubic deep beam turned 45 degrees, within 4% of resolved (`bench/results/lattice-seams-deep-beam.json`). The bare-face knockdown now follows the cut angle per cell family (`homogenized.FACE_CUT`, fitted on 18 resolved skinless designs, worst peak under-read 0.31x to 0.91x); re-optimized on it, an octet bar's turn (-5%) held at full resolution (peak 0.91x as laid) and the guard is lifted for octet only, while a BCCz L-bracket's (-40%) was again refuted (3.4x weaker). Under a skin the optimizer leaves the cells as laid anyway. |
| Lattice wall grading (`loadpath lattice-grade`, density grid) | Simulation only | Fully stressed sizing to a target SF on the homogenized check: 25% (L-bracket) and 10% (deep beam) lighter than a uniform 2 mm wall under a 1 mm skin; 17-48% skinless (`bench/lattice_grade_mass.py`). Resolved check (0.2 mm voxels) of the graded skinned deep beam: compliance 2-6% above the homogenized reading, peak 12.5 MPa against 18.8-22.0 (conservative); the skinless rows are unconfirmed (see orientation row). Sizing-grid dependence: 8k-cell sizing read SF 0.161 on 8k cells, 0.144 on 30k. |
| Printed lattice stiffness (33 published compression tests) | Matched published experiment (fitted, held out) | Read as designed, the homogenized check is 1.0-4.1x stiffer than the prints (median error 1.68x). The as-printed factor (`lattice_print.py`, default whenever the study names a process) is fitted to the same 33 tests by process group, strut or sheet, and density; held out one specimen at a time the median error is 1.23x (worst 2.04x), and with a whole paper held out its specimens read a median 0.92-1.17x of the measurement ([below](#as-printed-lattice-stiffness)). |
| Printed lattice strength (first yield) | Matched published experiment | With the as-printed factor the first-yield reading is under the measured first peak, plateau or yield on all 33 specimens (0.18-0.92x, factor held out). Read as designed it was under on 32 and over on FDM PLA Schwarz P (1.19x). Sheet prints fail at about the strain the design predicts for first yield, so sheets read wall stress from the strain of the softer lattice; strut lattices read from the stress they carry, a rule chosen and held out a paper at a time ([below](#strength-of-printed-lattices)). |
| Lattice wall and strut buckling | Analytic (unit cell), not tested | Periodic unit-cell eigenvalue buckling (`lattice_buckling.py`): a simple cubic strut cell reads Euler's clamped column within 2% on one cell and the pinned column on a 2-cell block. The homogenized check compares each lattice cell's largest compressive macro stress with half its walls' perfect buckling stress at the as-printed stiffness. On all 33 tests it reads at least 1.1x above first yield (perfect TPMS sheets at rho 0.15-0.3 buckle at 1.5-8x the solid's yield strain), so the knockdown of one half is a margin, not a fit. |
| Lattice crush and impact | Simulation only | Elastic-compensation collapse of a unit cell, fitted within 6% of its own solves (`impact.CRUSH_DATA`); Gibson-Ashby densification. Not tested. |
| Exact bores and interfaces | Analytic | Keep-outs and preserved shapes are cut with exact mesh booleans; intrusion and loss are measured on every delivered part and are zero to rounding. |

## Printed lattices against published tests

Three published studies give everything needed to rebuild their specimens:
cell type and size, wall or strut, specimen size, the printed material's
properties, and the measured modulus and strength of each lattice. Each
specimen is a cube of lattice, cut open at its sides, compressed between
flat platens along the build axis. `bench/published_experiments.py`
rebuilds every one as a field graph (lattice in the cube, 2 mm platens
bonded top and bottom, the top one held sideways and pushed down) and reads
it three ways: the periodic unit-cell tables, the homogenized check of the
specimen (what an agent gets for a lattice part), and a resolved voxel
solve of the designed geometry (0.2 mm voxels, 0.3-2 million cells, matrix-free multigrid).
Refining the resolved grid to 0.15 mm moves the gyroid specimen's modulus 0.1% and a 1 mm-strut diamond lattice's 8% (softer; first yield 13% lower), so the 0.2 mm strut readings carry about that much grid error. The table below fits nothing to these data; the as-printed factor fitted to them afterwards, and held out, is [its own section](#as-printed-lattice-stiffness). The data and their sources are in
`loadpath/data/published_experiments.json`, the results in
`bench/results/published-experiments.json`, and the documented bands are
pinned by `tests/test_published_experiments.py`.

| Source | Specimens | Homogenized modulus / measured | Resolved modulus / measured | Homogenized first yield / measured strength |
| --- | --- | --- | --- | --- |
| Shevchenko et al. 2023, SLS PA12 ([doi](https://doi.org/10.1021/acsomega.3c01631)) | Schwarz P, diamond, gyroid sheets; 10 mm cells, 0.8 mm walls, 30 mm cubes | 0.70, 1.51, 1.30 | 0.72, 1.90, 1.51 | 0.56-0.83 of first peak |
| same, FDM PLA | same | 1.67, 1.73, 1.65 | 1.72, 2.17, 1.91 | 0.66, 0.75, **1.19** (Schwarz P) |
| same, LCD resin | same | 1.43, 1.43, 1.34 | 1.48, 1.80, 1.55 | 0.62-0.82 |
| Guerra Silva et al. 2021, FDM ABS ([doi](https://doi.org/10.3390/ma14061462)) | 8 Kelvin and diamond strut lattices; 5 and 7 mm cells, 1.0 and 1.5 mm struts, 35 mm cubes | 1.68-4.12 (median 2.83) | 1.68-4.60 (median 3.11) | 0.18-0.42 of plateau |
| same, FDM PC | same 8 geometries | 1.33-3.61 (median 2.21) | 1.28-3.90 (median 2.48) | 0.33-0.78 of plateau |
| Abou-Ali et al. 2022, SLS PA11 ([doi](https://doi.org/10.3390/polym14214583)) | gyroid and diamond sheets, octet and Kelvin struts; 8 mm cells, 40 mm cubes | sheets 0.77-1.19, struts 1.28-2.56 | sheets 0.87-1.36, struts 1.49-2.88 | 0.45-0.91 of yield |

What this shows, and what it does not:

- **loadpath agrees with itself on these shapes.** The homogenized check
  reads 0.80-1.04x the resolved solve's modulus on all 33 specimens (the
  free-surface knockdown takes it from 0.96-1.24x without), so the error
  against the tests is not the homogenization.
- **The prints are softer than their design.** Even the resolved designed
  geometry reads 1.25-4.6x stiffer than the measurement on 29 of 33 specimens. The paper's
  material card is part of it for the FDM struts (datasheet modulus of
  injection-moulded filament, not printed struts); so are struts and walls
  thinner and rougher than drawn, layer bonding, and modulus taken from
  crosshead travel without an extensometer, which reads low. Shevchenko et al.
  measured their solid modulus in compression in the same lab, which
  cancels some of the test bias, and still see 1.5-2.2x on diamond and
  gyroid. A thin-feature effect is documented
  elsewhere: Cobian et al. 2024 (PA12 PBF, [data](https://zenodo.org/records/13747315))
  measured single struts of 0.5 mm reaching 20-50% of the 1.5 mm struts'
  tensile strength.
- **What to do with it.** The homogenized check now reads the print, not
  the design, whenever the study names a process (next section). Without a
  process it reads the designed geometry and says so (`latticePrint`).
- **Strength erred safe, with one class of miss.** Read as designed, the
  linear first-yield reading was a lower bound on the measured first peak or
  plateau on 32 of 33 specimens. The exception was Schwarz P in FDM PLA,
  1.19x. It is not wall buckling: a perfect Schwarz P cell at that density
  buckles under compression along z at about 5x its yield strain (`lattice_buckling.py`). The PLA
  print is simply 2.4x softer than the same geometry in SLS PA12, and it
  failed at the same strain. Reading wall stress from the as-printed strain
  fixes it (next section). Resolved first yield is also over the
  measurement on a few strut lattices (octet SLS 1.09-1.15x, thin PC
  diamond 1.24x), so its tighter reading is not a safe substitute.
- **Limits of the data.** Abou-Ali et al. published power-law fits over
  measured density rather than per-specimen values, and their sheets
  weighed 1.4-2.9x the design (powder sintered onto the surface); loadpath
  models the designed walls. Guerra Silva et al. list datasheet properties
  for ABS and PC. None of the papers state where the lattice phase sits at
  the cut faces. No study here tested a part loaded in bending, a skinned
  lattice, or a topology-optimized part.

### As-printed lattice stiffness

`loadpath/lattice_print.py` turns the gap above into a factor the
homogenized check applies to the lattice's stiffness:

    k = 1 / exp(a_process + b [strut cell] + c ln(rho / 0.2)),  capped at 1

fitted by least squares on the log of designed over measured modulus of all
33 specimens (`bench/lattice_print_fit.py`, results in
`bench/results/lattice-print-fit.json`):

| Term | Designed over printed | Fitted on |
| --- | --- | --- |
| FDM | 1.54x | Shevchenko 2023 PLA sheets, Guerra Silva 2021 ABS and PC struts |
| SLS (MJF read the same; no MJF lattice tests found) | 1.08x | Shevchenko 2023 PA12 sheets, Abou-Ali 2022 PA11 sheets and struts |
| Resin (LCD tested; SLA and DLP read the same) | 1.33x | Shevchenko 2023 LCD sheets |
| Strut cells | x1.58 | both strut papers |
| Density | x1.34 per e-fold of rho / 0.2 (rho clipped to the tested 0.1-0.4) | all |

So an FDM gyroid at rho 0.2 is read at 0.65 of its designed stiffness, an
FDM Kelvin lattice at 0.41, an SLS gyroid at 0.93. The density term says the
prints stiffen more slowly with density than the design; the papers' denser
lattices are the most over-read. Terms per cell type (Schwarz P, Kelvin,
octet) did not improve the held-out error and are left out.

| | Modulus over measured (median, range) | Median error | Worst |
| --- | --- | --- | --- |
| Designed geometry | 1.68, 0.70-4.12 | 1.68x | 4.12x |
| As printed, fitted on all 33 | 0.97, 0.64-1.82 | 1.19x | 1.82x |
| As printed, each specimen held out | 0.96, 0.61-2.04 | 1.23x | 2.04x |
| Abou-Ali 2022 held out | 1.00, 0.77-1.50 | | |
| Guerra Silva 2021 held out | 0.92, 0.61-1.84 | | |
| Shevchenko 2023 held out (its resin keeps no process term) | 1.17, 0.68-1.43 | | |

The held-out homogenized checks themselves (`bench/published_experiments.py`,
`asPrinted` rows: the specimen re-checked with the factor fitted without it)
give the same: modulus a median 0.96x of the measurement (0.61-2.04x, median error
1.23x), first yield 0.18-0.92x of the measured strength, and the buckling
check's stress 1.1x or more above first yield on every specimen, so it
never governs on these tests.

#### Strength of printed lattices

A softer print can mean two things for its walls' stress. If the walls are
thinner or defective, each carries more stress than the design says at the
same load, and the right reading is the strain the softer lattice takes
(a pure lattice then reads first yield k times lower). If the softness is in
the joints, layer bonds or the test, the walls carry the stress the design
says. `lattice_print.STRAIN_SHARE` picks per cell family, as the least
share of the strain reading that keeps every specimen's first-yield reading
at or under 0.95 of its measured strength (k held out), and each choice is
held out a paper at a time (`bench/lattice_print_fit.py`, `strainShare` in
its results):

| Family | Shipped | All data chooses | One paper held out: share chosen by the others, what the held-out paper reads |
| --- | --- | --- | --- |
| Strut | stress (0) | 0 | Abou-Ali out: 0, reads 0.65-0.91x. Guerra Silva out: 0, reads 0.18-0.78x |
| Sheet | strain (1) | 0.7 | Abou-Ali out: 0.7, reads 0.45-0.77x. Shevchenko out: 0, and its PLA Schwarz P reads 1.19x (unsafe) |

Struts pass the paper-out test both ways: on the stress reading FDM strut
lattices read 0.18-0.78x of their measured plateau instead of 0.07-0.41x
on the strain reading, 2.5x tighter at the median (0.37 against 0.15), still under every
measurement. The tightest strut is the SLS PA11 octet at 0.91x (its resolved
solve reads 1.09-1.15x, so do not swap in a resolved first yield there).
Sheets do not pass it: only Shevchenko's PLA Schwarz P asks for the strain
reading, so the sheet rule rests on that paper and on the physical reading
(the PLA print was 2.4x softer than the same geometry in SLS PA12 and
failed at the same strain). Sheets ship the full strain reading rather than
the 0.7 the data would allow, so that one specimen reads 0.82x, not 0.93x.
All 33 together, in the re-run checks: sheets 0.37-0.82x, struts 0.18-0.92x.

What the factor is not: a property of your printer. It is the average of
three labs' prints, testing and material cards, with about 1.2x scatter
either way after the fit (2x at worst). For a lattice part that matters,
print the gyroid coupon in the calibration kit: a calibrated card's
measured `lattice.stiffnessFactor` replaces the fit
([CALIBRATION.md](CALIBRATION.md)), and so does a study's
`process.latticeStiffness` (a number, or `"designed"`). lpbf, jetting and
volumetric have no printed-lattice tests here and read as designed, with a
warning.

No open dataset was found with both the geometry of a printed
topology-optimized part and its measured stiffness or strength; published
bracket tests give neither the STL nor enough to rebuild it. The robot-arm wrist
yoke bench test in the calibration kit remains the planned check for a
printed optimized part.

## Printed plates pulled through a pin

A pin or a bolt in a hole is where most printed brackets and lugs break,
and the published tests of it give everything needed to rebuild them: the
plate, the hole, where it sits, the print and the load at which it broke.
`bench/published_joints.py` rebuilds each specimen as a plate with its hole
(an STL, as an agent would hand to `loadpath verify`), held over its grip and
pulled through the hole by a steel pin (`"distribution": "bearing"`, which
the checks solve as a rigid pin that only pushes, `insert.py`), and reads
the load at which the check's stress reaches the strength: 1 kN x strength /
peak, without the 1.15 allowance the reported safety factor adds. The checks
are linear and judge first failure, so a reading under the measured break is
the safe side. Nothing is fitted to these data. Specimens and sources are in
`loadpath/data/published_joints.json`, the readings in
`bench/results/published-joints.json`.

| Source | Specimens | Material | Voxel check / measured | Tet check / measured | As read before 2026-11-25 / measured |
| --- | --- | --- | --- | --- | --- |
| Hertel, Kwon & Sachau 2024, FDM PLA ([doi](https://doi.org/10.3390/ma17225394)) | 140 x 20 x 2 mm strips, 6.35 mm hole for a 6 mm pin at 70, 45, 32.5 and 12.5 mm from the end; lines along the load (0), +-45 and across (90) | measured in the same lab: E 2919 MPa, strength 44 MPa | 0: 0.35-0.39, and 0.56 at 12.5 mm (bearing). +-45: 0.43. 90: 0.73-0.76 | 0: 0.39-0.43, 0.59. +-45: 0.47. 90: 0.78-0.82 | 0: 0.60-0.70, 0.98. +-45: 0.77-0.81. 90: **1.29-1.42** |
| Khosravani et al. 2023, FDM PLA and ABS ([doi](https://doi.org/10.3390/polym15071660)) | ASTM D5961 double-shear plates, 36 x 5 mm, hole 18 mm from the end, D = 12, 6, 4, 3 mm | loadpath's cards (the paper's coupon values are only plotted) | PLA 0.46-0.56, ABS 0.28-0.36 | PLA 0.34-0.59, ABS 0.21-0.38 | PLA 0.81-0.91, ABS 0.49-0.58 |

What this shows, and what it does not:

- **The bore was not read, and that was unsafe.** Every check leaves out
  the stress within a band round the region a load is applied over, because
  an idealized load (a force spread evenly over a face) puts a peak there
  that belongs to the idealization, not to the part. A pin's load was
  treated the same way, so the hole's sides, where a pin-loaded plate breaks
  in net tension, were never judged: the reading came from the edge of the
  band, 2 mm out. On the strips printed with their lines across the load
  (weak between lines, which loadpath's in-plane-isotropic card does not
  know), that read 1.29-1.42x the measured break. The pin's contact is
  solved (a rigid pin that only pushes, `insert.py`), so the stress at the
  bore is the part's: the checks now read it whenever the pin carries a
  case's whole load (`BoundaryConditions.case_ring_nodes`; a thrust on a
  shoulder or a moment on the bore keeps the band). The voxel check's peak
  sits at the hole's side, where the strips broke; the tet check's on the
  wide strips is on the face the pin bears on, within 10% of the same load.
  Reading the bore also showed the voxel check's cut-cell grid giving a
  node just outside a face a whole node's share of a load on half a cell
  of material: a caster fork's axle-bore rim read 18.9 MPa at one grid
  pitch against 5-8 at others and 6.1 on tets. Loads are now shared by the
  material each node stands for (5.2-5.9 MPa on every pitch).
- **It errs safe by 1.2-4.8x.** On the strips with their lines along the
  load the check reads 0.34-0.43x the break. The strips broke at a
  net-section stress (48.6 MPa) above the plain strength (44 MPa): printed
  PLA barely feels a hole. Ahmed & Susmel (2018) measured printed PLA's
  critical distance at 4.6 mm; loadpath averages stress over 0.86 mm (the
  value measured on PA 12) and caps the radius at 4/9 of the thinnest wall
  (0.89 mm on a 2 mm strip), so that a thin wall in bending is not read
  through its thickness. Read over 4.6 mm, uncapped, the net-tension breaks
  come out at 0.92-1.04x (PLA, lines along the load), but a hole 12.5 mm from
  the end, which broke in bearing, at 1.46x, and the double-shear bearing
  plates at 1.49-1.67x: unsafe. So the longer length is not adopted. Doing
  it safely needs a cap that applies across a bending gradient and not
  along an in-plane one, and a bearing reading that knows the crushing
  under a pin is contained.
- **Bearing failure is read as first yield.** The double-shear plates broke
  at a bearing stress (load over D x thickness) of 107-189 MPa, two to
  six times the plastic's tensile strength: the fixture's plates clamp the
  hole, and crushing under a pin is contained. The check reads first yield
  under the pin (0.28-0.56x the break) and does not model crushing.
- **Limits of the data.** No raster direction in a plate is modelled:
  loadpath's card is the same in every in-plane direction, so the strips'
  0, +-45 and 90 rows are one reading against three measurements. Hertel's
  strength is from coupons whose raster is not stated; Khosravani's
  materials are loadpath's datasheet cards, not their prints. Hertel's pin
  had 0.35 mm clearance, which narrows the contact; loadpath's pin fits.
  Neither paper measured the plate's stiffness with an extensometer, so no
  stiffness is compared. No published test of a printed topology-optimized
  bracket with its geometry and a measured stiffness or failure load was
  found (the ones found print an optimizer's output without publishing it,
  or test resin, or low infill).

## Flow and heat

| Claim | Label | Evidence |
| --- | --- | --- |
| Lattice-Boltzmann flow | Analytic | Plane Poiseuille friction exact (fRe 96.0); channel Nusselt 8.26-8.34 against 8.235; voxel pipe pressure gradient within 0.6% of Hagen-Poiseuille (`tests/test_flow.py`). |
| Gyroid cell friction and heat transfer | Cross-checked | Water-side Nusselt within 1-2% of an independent LBM code's results on the same cells; friction about 10% lower. That code is also a simulation. |
| Turbulent flow (Re above about 2000) | Not supported | The heat solve is unstable or model-dominated there. |
| Heat sink and cold plate ratings | Simulation only, grid-checked | Conjugate LBM flow plus finite-volume heat, energy balance closed to 1e-6. Results still move about 8% per grid refinement; differences smaller than that are ties ([cold plate case study](case-studies/cold-plate.md)). |
| Two-fluid heat exchanger effectiveness and pressure drop | Simulation only | Effectiveness-NTU rating from cell correlations; whole-exchanger porous and LBM runs disagree with the rating by up to 2x on header losses. |

## Propellers and rotors

| Claim | Label | Evidence |
| --- | --- | --- |
| Static thrust and power of printed propellers | Published tests | Against the UIUC database's static tests of the 3D-printed NR640 (Clark Y sections; 5 and 9 in; 2, 3 and 4 blades; two pitches), three speeds each: CT +8% mean, 14% RMS; CP -4% mean, 14% RMS. The 2-blade 9 in props are within about 10%; 3 and 4 blades read 15-25% high on thrust, and the 20 deg pitch reads about 20% low on power (`tests/test_propeller.py`, [PROPELLERS.md](PROPELLERS.md)). |
| Rotating blade bending and frequencies | Analytic | Cantilever tip deflection within 2%; first rotating flap frequency within 3% of Southwell's w0^2 + 1.193 Omega^2; torsion against its closed form within 2%. |
| Elastic twist of a spinning printed blade, motor and pack model | Simulation only | Centrifugal twisting moment, pitching moment and lift offset on a strip beam; three-constant motor behind the ESC's and pack's resistance. No thrust-stand check yet. |

## Build simulation (warp and residual stress)

| Claim | Label | Evidence |
| --- | --- | --- |
| LPBF distortion, IN625 | Measured data (fitted) | One inherent-strain factor fitted to the NIST AM-Bench 2018 bridge; then all 11 measured ridges within 7 µm rms. Because the factor was fitted to this data, this shows the model can describe it, not that it predicts blind. |
| LPBF distortion, 15-5 PH | Measured data (blind miss) | The IN625 factor reads 61% high on the same bridge in 15-5 PH; 15-5 PH has its own fitted factor. Ti-6Al-4V, 316L and AlSi10Mg use the IN625 factor and may be off by 2x. |
| FDM warp, ABS | Measured data (fitted) | Fitted to Armillotta et al. 2018 bars: 0.16 mm rms (24%), and misses their warp peak at 2-4 mm thickness. |
| FDM warp, PP | Measured data (blind) | Samy et al. 2022 plate: 0.53 mm predicted against 0.80-1.25 mm measured. A check of magnitude only. |
| FDM warp, PETG, PLA, ASA, nylon | Simulation only | No measured case found. The PETG warp bars in the calibration kit are the planned check. |
| SLS warp | Simulation only | Shrink values from the sintering literature, not fitted to a measured part. |

## What is not modelled

- Contact between parts, friction, bolt preload beyond a simple preload case.
- Print defects: voids, under-extrusion, seams, moisture in nylon.
- Plasticity in the structural check (it is linear elastic up to the
  strength; the safety factor is strength over peak von Mises).
- Turbulent heat transfer.
- Frequency effects in fatigue (self-heating), variable-amplitude loading
  beyond Miner's rule.

If you need any of these for a decision, test the printed part.
