# Impact and vibration

Lattices lose to solid topology optimization on stiffness per gram (a graded
diamond/solid hybrid was 56% more compliant than solid TO at equal material on
the arm link; see the moonshots round 1 write-up). They win where a solid can't
compete: stopping an impact at a force you choose. This page covers
`loadpath/impact.py` (crush and impact), `loadpath/modal.py` (natural
frequencies and damping) and the `impact` and `vibration` blocks of a field
graph.

## The case: a corner bumper for a falling robot

`examples/impact/end-stop.json`: a tapered corner bumper for a GRIP-like robot
base. The 5 kg robot falls 0.9 m off a bench (4.2 m/s, 44 J) and the PA12 base
behind the bumper must not take more than 6 kN. The envelope narrows from
40 mm to 22 mm square over 20 mm. Cells are 12 mm so the thinnest SLS wall is
0.8 mm, which sets the lowest density at 0.20.

    python examples/impact/end_stop.py --out impact-out/     # ~10 min on 4 cores

![the three designs](images/impact/end-stop-designs.png)
![force against crush](images/impact/end-stop-crush.png)

| design | mass | peak force | crush | efficiency | 44 J under 6 kN? |
|---|---|---|---|---|---|
| graded gyroid (impact block's grading) | 4.85 g | **5.43 kN** (111 g) | 8.6 mm | **0.94** | **yes** |
| uniform gyroid, same mass (rho 0.243) | 4.85 g | 6.71 kN (137 g) | 10.0 mm | 0.66 | no |
| solid TO, designed for the 6 kN load | 8.25 g | 44 kN if elastic; yields at 4.3 kN | 2.0 mm | n/a | no: holds 0.82 J at 6 kN |

Efficiency is energy / (peak force x crush): 1 is a perfect square pulse.

- **Graded vs uniform, same mass:** 19% lower peak force, and only the graded
  core meets the limit. A uniform core in a taper crushes its narrow top at
  2.7 kN and its wide foot at over 8 kN, so its force climbs all the way; the
  graded one (density 0.20 at the foot to 0.43 at the tip) crushes every layer
  at about 5.3 kN.
- **Graded lattice vs solid TO (1.7x heavier):** the solid is a spring. Within
  the 6 kN limit it can store 0.82 J, 1.8% of the drop; it yields at 4.3 kN
  having stored 0.41 J. The rest of the 44 J has to go into crushing a part
  that was never shaped to crush, and it hands back whatever it stores
  elastically as a rebound. The lattice absorbs the whole drop in 8.6 mm of a
  20 mm pad (64% of its stroke) and keeps it.
- **Vibration:** the robot sitting on the pad (5 kg on the top face, base
  clamped) has its first mode at 101 Hz on the graded pad and 98 Hz on the
  uniform one (`vibration` block, homogenized lattice).

## How the crush model works

1. **Cell strength.** `collapse_strength` crushes one periodic TPMS cell
   (32^3 voxels) by the elastic compensation method (Mackenzie & Boyle 1993):
   linear solves in which every overloaded wall is softened in proportion, so
   load spreads the way a plastic mechanism spreads it. The macro stress at
   which the most loaded wall yields is a lower bound that rises each pass to
   near the limit load. Fits, sigma_pl / sigma_y = D rho^p (within 6% of the
   data, which is in `impact.CRUSH_DATA`):

   | sheet | D | p | sigma_pl / (rho sigma_y) at rho 0.25 |
   |---|---|---|---|
   | gyroid | 0.633 | 1.19 | 0.49 |
   | diamond | 0.675 | 1.12 | **0.57** |
   | Schwarz P | 0.795 | 1.38 | 0.47 |

   Collapse is 1.4-2x first yield. Exponents of 1.1-1.4 mean the sheets carry
   load mostly by stretching, which is why sheet TPMS beat strut lattices
   (bending, p ~ 1.5-2) per gram. Diamond is the strongest per gram, as it is
   the stiffest.
2. **Densification:** eps_D = 1 - 1.4 rho (Gibson & Ashby's foam constant).
3. **Layers in series** (`crush_curve`): each layer's crush force is the sum
   of its cells' plateau stress times area. The weakest layer crushes through
   its densification strain first, then the next. That is the standard model
   for graded foams.
4. **Grading** (`grade_for_impact`): density per layer from
   sigma_pl(rho) A = F, clipped to the printable range. F is the target that
   gives the lowest peak while still holding 1.25x the energy before the core
   densifies. In a field graph, `"grade": true` returns the result as a
   `remap` table node to paste in as the lattice's density.

## What to trust, and what not yet

- **The plateau is a perfect-cell estimate.** No buckling, no print defects,
  quasi-static. Polymers are 10-30% stronger at impact rates, which raises the
  real force. Treat forces as +-25% until a printed crush coupon calibrates D:
  one gyroid cube per density, crushed on a test machine, then scale D.
- **Lateral load spreading.** A 3D check (`limit_load`: elastic compensation
  on the homogenized pad, Deshpande-Fleck crushable-foam yield) finds local
  collapse starting below the layer model's force. Graded: 3.5-4.2 kN against
  5.2 kN. Uniform: about 2.2 kN against 2.7 kN. In a taper the load from a
  narrow layer reaches the middle of the wider layer below before its rim, so
  the middle crushes first. That makes the start of the stroke softer, not
  stiffer, so the peak force estimate is not threatened. But a design graded
  only along z cannot make every cell fully stressed. The next step is 3D
  grading: iterate the density per cell to equalize the 3D utilization, and
  pass the result to the graph as a `grid` node.
- **Few cells.** 12 mm cells in a 40 mm pad are about three cells across, so
  homogenization is approximate. That is the price of 0.8 mm SLS walls at
  rho 0.2. MJF or resin at thinner walls allows finer cells.
- **Materials.** The model is right for ductile polymers (PA12, PA11, PETG)
  and metals. PLA fractures and crushes raggedly. TPU buckles elastically and
  springs back, which this model does not cover.
- **Top and bottom faces.** The example's core is cut open at both faces. A
  real bumper wants thin face sheets so the impactor meets a surface. Add a
  `shell` of the envelope's top and bottom slabs.

## Vibration (`modal.py`)

`modal_analysis(mesh, modulus, density, nu, fixed, n_modes, point_masses)`
solves K phi = w^2 M phi with lumped mass on the voxel mesh. Per-cell modulus
and density mean a homogenized lattice works too (E C rho^n, rho_s rho). It
returns frequencies, mass-normalized modes and the effective mass fraction per
axis. Validated against Euler-Bernoulli: a 100 x 10 x 10 mm cantilever is
within 0.5% for f1, and f2/f1 = 6.27.

Damping: a part made of one linear viscoelastic material has that material's
loss factor in every mode, so **a lattice does not damp better than the solid
it is made of**. `LOSS_FACTOR` holds typical values (PA12 0.04, TPU 0.12, PLA
0.015, metals ~0.002). A lattice changes where the modes are: E/rho falls as
rho^(n-1), so a core lowers frequencies. That suits an isolator and is a
penalty for a stiff bracket, where solid TO wins on first frequency per gram
for the same reason it wins on stiffness. Real lattice damping gains need a
second, lossy material (a TPU core) or unfused powder left in the cells
(particle damping). Neither is modelled yet.

## Field graph blocks

```json
"impact": {"envelope": "envelope", "density": "density", "lattice": "gyroid", "axis": "z",
           "mass": 5.0, "velocity": 4.2, "forceLimit": 6000, "grade": true, "cell": 12, "minWall": 0.8},
"vibration": {"envelope": "envelope", "density": "density", "lattice": "gyroid", "fixed": "base",
              "pointMasses": [{"region": "top", "mass": 5.0}], "modes": 6, "minFrequency": 80,
              "excitation": 50}
```

Both need a `material`. `loadpath field graph.json` reports `report["impact"]`
(force-crush curve, response, pass or fail against the force limit; with
`grade`, the graded table, its response, and a uniform core of the same mass
for comparison) and `report["vibration"]` (frequencies, mass fractions,
damping ratio, worst transmissibility at the excitation).
