# What is validated

Every number loadpath reports comes from a model. This page says how far
each model has been checked, and against what. Read it before you trust a
safety factor, a temperature or a warp prediction with something that
matters.

Each claim carries one of these labels, strongest first:

| Label | Meaning |
| --- | --- |
| **Tested** | Compared with a physical part or coupon that was printed and measured for this project. |
| **Measured data** | Compared with published measurements of a physical part (for example NIST AM-Bench). Says whether a parameter was fitted to the same data. |
| **Analytic** | Matches a closed-form answer (beam theory, Lamé, Poiseuille) in the test suite. |
| **Cross-checked** | Agrees with an independent solver run on the same problem (CalculiX, or loadpath's second, tetrahedral solver). |
| **Benchmarked** | Compared with published optimization codes on the same problem (top88, top3d, Sigmund's 99-line mechanism code). This measures the optimizer, not the physics. |
| **Simulation only** | No outside check. The number is what the model says. |

## Nothing is physically tested yet

No loadpath prediction has been compared with a part printed and measured
for this project. A calibration kit is ready and waiting to be printed
([CALIBRATION.md](CALIBRATION.md)): stiffness and strength coupons along and
across layers, a gyroid-core beam, TPU creep and crush coupons, PETG warp
bars, and a bench test of the GRIP wrist yoke with a predicted rotation per
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
| Natural frequencies | Analytic | Cantilever f1 within 0.5% of Euler-Bernoulli, f2/f1 = 6.27 (`modal.py`, docs/IMPACT.md). |
| Thermal conduction and expansion | Analytic | Conduction exact to solver tolerance; bimetal curvature within 2.5% of Timoshenko (`tests/test_thermal.py`). |
| The delivered part's safety factor | Cross-checked | Every safety-factor search confirms its pick on a tetrahedral (P2) mesh of the delivered STL. On the same tet mesh, CalculiX and loadpath's native P2 solver agree within 0.1% on the governing peak, and to 3-4 digits for layered (anisotropic) studies. The voxel check reads 24-49% higher peak stress than the tet check on the motor mount and bell crank, so voxel-only numbers err safe. fTetWild meshes are not deterministic, so tet readings move about 3% run to run. |
| Layered (printed) stiffness and layer bond | Simulation only | Transversely isotropic stiffness and a tension-across-layers check, with factors from datasheets and literature. Interlayer shear is not checked separately. Waiting on the calibration coupons. |
| Material cards | Measured data (datasheets) | Every value cites its datasheet (`loadpath materials`). Datasheet values describe the manufacturer's test bars, not your printer. |
| Fatigue and creep of printed plastics | Simulation only | One published design curve for printed PLA (Ezeh and Susmel), scaled by strength for PETG, ABS and ASA; rough values for TPU, resins and metals. Not tested. |

## Topology optimization

| Claim | Label | Evidence |
| --- | --- | --- |
| The optimizer finds near-optimal layouts | Benchmarked | At equal material, 3-5% less stiff than top88 on the 2D MBB beam and L-bracket, 1.2% less stiff than top3d on its 3D cantilever, and stiffer than top3d's thresholded design on a 3D bridge. Most of the 2D gap is the cost of the minimum-member rule. The references are NumPy ports of the published MATLAB codes, not the originals ([case study](case-studies/benchmarks.md)). |
| Compliant mechanisms | Benchmarked | Within 1-2% of a port of Sigmund's 99-line mechanism code on the force inverter and gripper ([MECHANISMS.md](MECHANISMS.md)). |
| GE bracket challenge: 370 g at SF 1.06 | Simulation only | Level with the winning entries' ~80% reduction, but on an approximate, roomier envelope and checked only by loadpath's own analysis ([BENCHMARKS.md](BENCHMARKS.md)). |
| Eval and GRIP part masses and safety factors | Simulation only, cross-checked solver | The numbers are model predictions under the stated loads. The solver behind them is cross-checked as above; the loads, material values and print quality are not ([eval case study](case-studies/evals.md), [GRIP](case-studies/grip.md)). |
| Run-to-run reproducibility | Measured on the model | Mass at the target safety factor moves 3.5-7% when the search starts from a different fraction, and 5% (median) when the grid pitch changes by 3%. Mass differences under about 5% are noise. |

## Fields, lattices and meshing

| Claim | Label | Evidence |
| --- | --- | --- |
| Lattice stiffness and strength fits | Simulation only | Unit-cell FEA (strut and TPMS cells) fitted by power laws. A coarse homogenization lands within 15% of the fits (`tests/test_loadpath.py`). Not tested; the gyroid-core coupon is in the calibration kit. |
| Homogenized lattice parts | Cross-checked (against resolved geometry) | A skin-and-gyroid bar and an octet bar analysed homogenized and fully resolved (`docs/fields.md`). Thin solid walls read about 3% stiffer homogenized. |
| Anisotropic lattice tensors (stiffness and directional wall stress) | Cross-checked (against resolved geometry and CalculiX) | Eleven bars and beams of cubic, BCCz, octet and gyroid cells, straight and turned 45 degrees, with and without a skin, analysed homogenized and resolved on 0.15-0.25 mm voxels; CalculiX C3D8I on the same voxels agrees with the resolved solve to 1e-8. Skinned: stiffness within 5%, wall stress 0.95-1.43x. Stretched (aspect) cells, tall BCC in a skinned beam: stiffness within 2%, wall stress 0.85-1.34x (15% low in a pull). Skinless lattices need the free-surface knockdown and one case (turned cubic in twist) still reads 0.64x ([fields.md](fields.md#lattices-are-read-anisotropic), `bench/lattice_aniso_validate.py`). Not tested on prints. |
| Lattice orientation optimization (`orient`, `loadpath lattice-orient`) | Simulation only | Mass at equal safety factor: 7-23% lighter on lattice bars whose cells lay badly, 0% on one already aligned (`bench/lattice_orient_mass.py`). Zoned orientation (`orient.zones`): 0-5% more on top of the best single direction at a uniform wall (`bench/lattice_zones_mass.py`). Homogenized readings, relying on the tensor validation above; no resolved solve of each optimum. |
| Lattice crush and impact | Simulation only | Elastic-compensation collapse of a unit cell, fitted within 6% of its own solves (`impact.CRUSH_DATA`); Gibson-Ashby densification. Not tested. |
| Exact bores and interfaces | Analytic | Keep-outs and preserved shapes are cut with exact mesh booleans; intrusion and loss are measured on every delivered part and are zero to rounding. |

## Flow and heat

| Claim | Label | Evidence |
| --- | --- | --- |
| Lattice-Boltzmann flow | Analytic | Plane Poiseuille friction exact (fRe 96.0); channel Nusselt 8.26-8.34 against 8.235; voxel pipe pressure gradient within 0.6% of Hagen-Poiseuille (`tests/test_flow.py`). |
| Gyroid cell friction and heat transfer | Cross-checked | Water-side Nusselt within 1-2% of an independent LBM code's results on the same cells; friction about 10% lower. That code is also a simulation. |
| Turbulent flow (Re above about 2000) | Not supported | The heat solve is unstable or model-dominated there. |
| Heat sink and cold plate ratings | Simulation only, grid-checked | Conjugate LBM flow plus finite-volume heat, energy balance closed to 1e-6. Results still move about 8% per grid refinement; differences smaller than that are ties ([cold plate case study](case-studies/cold-plate.md)). |
| Two-fluid heat exchanger effectiveness and pressure drop | Simulation only | Effectiveness-NTU rating from cell correlations; whole-exchanger porous and LBM runs disagree with the rating by up to 2x on header losses. |

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
