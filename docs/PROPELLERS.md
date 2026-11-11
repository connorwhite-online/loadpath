# Propellers and rotors (`loadpath.propeller`)

What a drone builder asks of a rotor is thrust per watt at hover, enough thrust at full throttle, a blade that holds
together and holds its shape at speed, and frequencies that stay off the rotor's. `loadpath.propeller` answers each
of those from the physics, on the motor and battery that will turn it, and designs the blade that draws the least
power from the pack at hover while meeting the rest. It writes the printable rotor in both hands of rotation.

```python
from loadpath import propeller as pr
motor = pr.Motor(kv=2450, rm=0.12, i0=0.6)          # an 1804-class outrunner
pack = pr.Battery(cells=4)                          # 4S, 12 mOhm a cell
blade = pr.Blade.uniform(r_tip=44.45, r_hub=7.0, blades=3, chord=9.0, pitch_in=3.0, t_ratio=0.09)
e = pr.evaluate(blade, motor, pack, hover_thrust=0.775)   # N per rotor
e["hover"]["electrical"], e["hover"]["gramsPerWatt"], e["full"]["thrust"]
e["full"]["structure"]["tipTwist_deg"]              # how far the spinning blade untwists

best = pr.design(motor, pack, hover_thrust=0.775, max_thrust=4.5, r_tip=44.45, r_hub=7.0,
                 blades=(2, 3, 4), material="pa12", t_min=0.8)["best"]
pr.rotor_mesh(best["blade"], bore=5.0, rotation="ccw").export("prop-ccw.stl")
pr.rotor_mesh(best["blade"], bore=5.0, rotation="cw").export("prop-cw.stl")
```

Units: lengths in mm, speed in rpm, thrust N, torque N m, power W. Install `loadpath[aero]` for the section polars
(NeuralFoil); without it the module runs on a thin-airfoil estimate and says so in every result (`polars`).

## Method

**Blade-element momentum.** The disk is cut into annuli. In each, the blade section's lift and drag at its angle of
attack balance the momentum the annulus gives the air: axial (thrust) and swirl (torque), with Prandtl's tip and hub
loss factors. The equations are solved on the inflow angle by bracketing (Ning's residual, rewritten so the hover
case, with no free stream, is as well posed as the advancing one), scanning from the lowest angle of attack so the
attached root is the one found. A rotor in a duct (`duct={"exitAreaRatio": ...}`, from `aero.duct_performance`)
carries only the rotor's share of the thrust, and the annulus momentum is the duct's: a pressure jump of
0.5 rho (w / sigma)^2 at disk velocity w, with no tip loss.

**Section polars from the section that's drawn.** Each station's section is the one the mesh will print: a
circular-arc camber line with NACA four-digit thickness, its trailing edge opened to the printer's floor. Its lift,
drag and moment come from NeuralFoil, a network trained on XFOIL's viscous panel method with e^N transition, at that
station's own Reynolds number. On a small rotor that's 2-8 x 10^4, where laminar separation sets lift and drag. The
tables are built at three Reynolds levels across the speed range and read at each station's actual value.

- **Transition**: the critical amplification N comes from the turbulence the rotor breathes (Mack 1977,
  N = -8.43 - 2.4 ln Tu). A rotor hovering in a room ingests its own recirculated wake at about 1%, which gives
  N 2.6. A quiet wind tunnel is N 9.
- **Rotational stall delay** (Snel 1994): Coriolis and centrifugal pumping of the boundary layer keep the inboard
  sections attached past their 2D stall, Cl_3D = Cl_2D + 3 (c/r)^2 (Cl_attached - Cl_2D).

**The motor and the pack.** The motor is modelled by its three constants: Kv, winding resistance and no-load
current. It sits behind the ESC's resistance and the pack's internal resistance, which every motor's current sags
together. Hover is the throttle where the flexing blade gives the hover thrust, and its watts are watts drawn from
the cells. Full throttle is where the motor's torque, at the sagged voltage, meets the blade's. A test stand on a
bench supply reads higher than this, because nothing sags.

**The blade as a rotating beam.**

- **Bending**: flap and lag bending under the thrust and drag loads, with centrifugal stiffening. The stiffness
  about the rotor's axes is the section's own second moments turned by the pitch.
- **Torsion**: three moments twist the blade:
  - the sections' quarter-chord pitching moment
  - the lift's offset from the elastic axis (taken at the section centroid)
  - the centrifugal twisting moment, -rho Omega^2 (I_normal - I_chord) sin(theta) cos(theta), which turns every
    spinning blade toward flat pitch. Its restoring part (the tennis-racket term) stiffens torsion as speed rises.
- **Coupling**: the pitch the air sees is the drawn pitch plus the elastic twist. Performance and structure are
  solved together over a fixed number of relaxed passes, so results are smooth for the optimizer.
- **Root stress**: the centrifugal pull plus both bending moments resolved onto the section's axes, against the
  material's strength.
- **Frequencies**: the first flap and lag frequencies, rotating (centrifugal stiffening, and the lag's -m Omega^2),
  with their margin from 1x, 2x and 3x rotor speed and blade pass.

**Response.** The rotor's inertia (blades from their sections, plus the hub) is added to the motor bell's. The
small-signal time constant against the rotor's aerodynamic damping and the motor's back-EMF damping is
tau = I / (2 Q / omega + 1 / (k^2 R)). The attitude loop can't be faster than its motors, so tau bounds its
bandwidth.

**Design.** The variables are chord at four knots, geometric pitch at three, camber and thickness ratio at root and
tip, for each blade count asked. SLSQP minimizes the hover's electrical power on the flexing blade, subject to:

- full-throttle thrust ≥ `max_thrust`
- root safety factor at full throttle ≥ `sf_min`
- tip Mach number ≤ `tip_mach`
- flap and lag frequency margin at hover ≥ `frequency_margin`
- motor-and-rotor response time ≤ `response_time` (optional)
- thickness ≥ `t_min` everywhere (the process's thinnest reliable wall)

The start is momentum theory's best rotor: even induced velocity, chord from the annulus balance at a working lift
coefficient. Chord is held where blade-element momentum still holds: local solidity under 0.3 from 0.4 R out. Past
that the blades act as a cascade, which this model doesn't represent.

## Root, hub and tip

The beam model reads the root as a section, but the printed root is a joint, and that's where the centrifugal pull and
the bending both peak. Three things there come from the loads:

- **Root fillet** (`root_fillet`): a shoulder fillet of radius r on a root of thickness t raises the bending stress
  by Kt = 1 + 0.2 (r/t)^-0.55 (a fit to Peterson's stepped bar in bending). The radius is the smallest that keeps
  Kt x the full-throttle root stress within strength / sf. It is never under the process's thinnest wall, and never
  over three root thicknesses. `rotor_mesh(..., fillet=r)` grows each root section's thickness and chord by the
  circular fillet's profile over the last r before the hub. Every section lies on a cylinder concentric with the hub,
  so the union meets the hub with that radius all round.
- **Hub height** (`hub_height`): the root section's axial depth, fillet included, plus a wall above and below, so the
  root lands whole on the hub.
- **The motor's hub** (`design(..., hub_height=)`): the hub can only be as tall as the motor takes, which is its shaft
  length or T-mount. Given that height, the root knot's chord is held to what stands inside it, and the blade necks
  to a shank. The chord spline carries the shank out to the working chord, and the root's stress, frequency and
  untwist constraints become the shank's. Without a hub limit, a wide, steep root asks for a hub three or four times
  taller than any motor shaft. On hum's 1804-class motor (6 mm hub), fitting the hub cost 20-50% of hover power on
  the PA12 designs. The narrow shank is soft in torsion, so blades untwist 15-25 deg at full throttle.
- **No overlap**: each section's span round the circle (chord x cos pitch) stays within 80% of its share of the
  circumference, so neighbouring roots don't run into each other on the hub.

The tip closes over its last half chord as a quarter ellipse (`tip_closure`). A square-cut tip at full chord sheds
a vortex off each corner. Closing it brings the chord, and with it the circulation, to zero at the tip, which is
the loading the tip-loss factor already gives there. The aerodynamics sees the closed planform.

**The printed solid** (`solid_check`): quadratic tetrahedra of the rotor as it will print. Every node carries its
centrifugal body force, each blade's surface carries its stations' thrust and drag, and the hub is held at the
bore. Stress is read averaged over the material's own length (PA12's 0.86 mm, held under 4/9 of the thinnest blade
section, as `crosscheck` reads it), not at a node, so a sharp corner's reading is the part's and not the mesh's. It reports the peak on the blades, its radius,
its safety factor, and the hub's own peak. For a 3.5 in two-blade at 25,000 rpm (`tests/test_propeller.py`):

| | Blade peak (root bending) | Junction peak |
|---|---|---|
| Sharp root | 16.4 MPa | 8.2 MPa |
| 2 mm fillet | 16.3 MPa | 2.4 MPa |

What the fillet buys is the junction: it cuts the stress where the blade meets the hub to under a third, which is where a printed
part cracks under cyclic load. The blade's root bending stress outboard of the junction is the section's own, and
the fillet barely changes it. The solve is linear, so it carries no centrifugal stiffening. Its flap deflection and
the bending stress that comes with it read high, which errs safe.

## Validation

Against published static tests of 3D-printed propellers (UIUC propeller database volume 2; Deters, Ananda & Selig,
AIAA 2014-2151). The props are NR640, a printed scale model of the Navy 5868-9 propeller with Clark Y sections of
varying thickness, in eight configurations: 5 and 9 in, 2, 3 and 4 blades, 15 and 20 deg. Three speeds of each run,
default settings (`pr.validate()`, in `tests/test_propeller.py`):

| | mean error | RMS error |
|---|---|---|
| CT (thrust) | +8% | 14% |
| CP (power) | -4% | 14% |

The 2-blade props fit best, within about 10% at the 9 in size, whose Reynolds numbers (3-6 x 10^4) are a small
drone's. The 3- and 4-blade props read high on thrust by 15-25%, and the 20 deg props read low on power by about
20%. Two things the data can't separate may cause this: blade interference, and inboard stall drag at high pitch.
The tested blades were screwed into a hub at a set pitch, and how the 20 deg setting relates to the published
15 deg geometry is assumed here (+5 deg throughout).

## What it is and is not

Blade-element momentum is what propeller designers use for a first and second pass. It treats each annulus on its
own, assumes an axisymmetric wake, and knows nothing of the frame below the rotor (aero.py's `rotors` carry the
wake's blockage onto the frame). Other limits:

- The polars are a trained surrogate of a 2D panel method, with a correction for rotation.
- The beam is a Euler-Bernoulli strip with uncoupled flap, lag and torsion.
- The thin-airfoil fallback is an estimate, not a polar to design on.

Test a printed rotor on a thrust stand before relying on the last 10-15%.

Tests: `tests/test_propeller.py`:

- the rotating beam against the closed forms: cantilever deflection and Southwell's rotating flap frequency
- torsion against its closed form
- momentum bounds and thrust ~ rpm^2
- a spinning plastic blade twisting toward flat and losing thrust
- the motor and pack's energy balance
- watertight meshes in both hands
- the published tests above
