# Air that moves by itself: natural convection, solved

`loadpath.convection.natural_h` gives each face a coefficient from plate
correlations: a vertical plate alone in a room, a horizontal one, two
isothermal plates facing each other. That is the right first answer for an
open surface. It is a guess for air that has to find its own way: drawn in
through a slotted grate, up fin channels that are open on one side, along
a gap, out of the next slot up and up the outside wall. There the grate
chokes the flow as much as the fins do, the air warms as it rises, and the
whole column's buoyancy drives it, not each face's.

`loadpath.airflow` solves for that air. It is the steady, laminar
Navier-Stokes equations with Boussinesq buoyancy, and one energy equation
across solid and air together (conjugate), on loadpath's voxel grids.

```python
from loadpath import airflow

r = airflow.solve(solid, pitch=0.5, k=k_cells, heat=watts_cells, ambient=25.0,
                  sides={"x": "periodic", "y-": "open", "y+": "symmetry", "z": "open"})
r.t                 # C per cell, solid and air
r.velocity          # m/s, cell-centred (3, nx, ny, nz)
r.wall_h()          # per wetted face: h to ambient, wall temperature, heat, position
r.balance           # heat in, carried out of the openings, through walls
```

## The model

- **Flow.** Finite volumes on a staggered (MAC) grid: pressure and
  temperature at cell centres, each velocity component on the faces normal
  to it, so there is no checkerboard to suppress. SIMPLEC
  pressure-velocity coupling (Van Doormaal & Raithby 1984). Convection is
  upwind with a deferred correction to central differences, so it is
  second order where the grid resolves the flow and stays bounded where it
  doesn't. Every solid voxel is a no-slip wall.
- **Buoyancy.** rho0 beta (T - T_ambient) g on each face. Air's properties
  are taken at the film temperature (the mean of the room and the wetted
  walls), refreshed as the solution moves.
- **Heat.** Conduction in every cell at its own conductivity (harmonic
  mean across faces), and advection in the air. Heat sources go per cell,
  cells can be held at a temperature, and walls can be adiabatic or
  isothermal. The energy matrix is factorized each pass, with PARDISO
  (`pip install loadpath[fast]`) or SuperLU. Aluminium beside air is a
  conductance contrast of about 8000, which iterative solvers built on the
  symmetric part don't carry.
- **Box sides.** `open` is an opening to the still room: zero total
  pressure where air comes in (at the room's temperature and with no
  velocity along the side), zero static pressure where it leaves. A solid
  cell on an open side is adiabatic there: cut a box at the outside of a
  shell and the shell reaches the room only through the air. The
  others are `wall`, `symmetry`, `periodic` and `inlet` (a fixed velocity
  in, for a duct or a fan).
- **Fans.** `airflow.Fan(cells, axis, sign, dp=...)` adds a fixed pressure
  rise across a zone. `curve=(dp_max, flow_max)` reads a straight fan line
  at the flow the zone passes instead. The fan's head is brought up from
  rest (a tenth of the line at first, then relaxed toward it), because the
  full shut-off pressure on still air throws the first passes far past the
  answer. For a fan through narrow fins, pass `relax=(0.3, 0.5, 1.0)`, and
  `deferred=0` if it still won't settle: first-order upwind, bounded.
- **Not in it.** Turbulence (the flows it is for, a few watts in a
  handheld, run at Rayleigh numbers of 10^3 to 10^5 on the gap or the
  height), radiation (add `convection.radiation_h` by each face's view from
  `convection.openness`, as `wall_h` keeps them apart), and compressibility.

Converged means three things hold: the mass residual is under `tol` of
the largest face flow, the velocities and temperatures have stopped moving,
and the heat balance closes within 0.5%.

## Against textbook cases

`tests/test_airflow.py` runs the starred rows. Each takes seconds.

| Case | Grid | Solver | Reference | Error |
|---|---|---|---|---|
| Heated square cavity, Ra 1e4, mean Nu * | 32 x 32 | 2.261 | 2.243 (de Vahl Davis 1983) | +0.8% |
| same | 48 x 48 | 2.239 | 2.243 | -0.2% |
| Heated square cavity, Ra 1e5 * | 48 x 48 | 4.591 | 4.519 | +1.6% |
| same | 80 x 80 | 4.555 | 4.519 | +0.8% |
| Heated square cavity, Ra 1e6 | 80 x 80 | 8.978 | 8.800 | +2.0% |
| Vertical plate 40 mm, 20 K over the room, mean Nu * | 1 mm | 9.63 | 9.96 (Churchill & Chu, laminar) | -3.3% |
| Plates between plates (forced), f Re * | 16 across | 95.95 | 96 | -0.1% |
| same, Nu developed (isothermal walls) * | 16 across | 7.55 | 7.54 | +0.2% |
| Isothermal plate array, 40 mm, s = 8 mm (El 170) * | 0.5 mm | Nu_s 2.02 | 2.04 (Bar-Cohen & Rohsenow 1984) | -1% |
| same, s = 4 mm (El 10.6) * | 0.5 / 0.25 mm | 0.353 / 0.348 | 0.409 | -14% / -15% |
| same, s = 2 mm (El 0.66) | 0.5 / 0.125 mm | 0.022 / 0.023 | 0.028 | -19% / -15% |
| Fan zone in a channel on a fan line * | 0.5 mm | on the line; dp within 12% of friction + opening losses | | |
| Heated block in open air: heat carried out = heat in * | 1 mm | balance < 0.5% | | |

The plate arrays are the case that matters for fins. Toward the
fully developed end (narrow gaps, low El) the solver reads 15% under
Bar-Cohen & Rohsenow, and the gap holds as the grid is refined, so it is
not resolution. The composite correlation's narrow-gap limit, El / 24,
assumes the air is at the plate temperature from the bottom of the channel
and leaves out the entry and exit losses. At El 0.7 those losses are a
third of the buoyant head. Elenbaas' own plates were also open at the sides.
Where the gaps are wide (El 170), the two agree.

## What it is for

Rating a natural-convection design whose air path is not a textbook case:
a fin stack behind a grate, a chimney through an enclosure, vents high and
low, a fan that may or may not be needed. It solves a box of tens to a few
hundred thousand voxels in minutes to an hour. A design that repeats across
its width (fins, slots) is best run as one periodic period of it.

To use it in a whole-product conduction model (thermal.solve_temperature),
take `wall_h()` from the solved region onto the matching faces of the
product model. Look each face up by its kind (which way it looks, which
material) and its position, keep radiation by view as before, and iterate
until the heat the product model sends into the region matches the
region's input. The Imprint Pro's rear duct is worked this way, in
`pro/cad/skin2_airflow.py` in the imprint repository.
