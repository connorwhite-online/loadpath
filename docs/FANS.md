# Axial fan rotors (`loadpath.fan`)

A fan that pushes air through a heat exchanger, a filter or a duct delivers what its pressure-flow curve gives where
it crosses the system's. `loadpath.fan` designs the blades for that crossing, predicts the curve, checks the rotor at
speed and writes a printable rotor: airfoil blades, a tip band that spins with them, and a hub.

```python
from loadpath import fan
air = fan.air(35.0)                                   # deg C: density, viscosity
system = lambda q: 92600 * q * q                      # Pa the system needs at q m^3/s
rotor, rep = fan.optimize(system, r_tip=57.3, r_hub=23.5, power=6.1, rpm=(3500, 6400),
                          axial=14.0, air=air, exit="uniform")
op = rep["operating"]                                 # q, dp_ts, power, eff_ts, per-station flow
fan.strength(rotor, op=op, retention=0.40)            # PA12 at 90 C
fan.rotor_mesh(rotor, z0=4, z1=22).export("rotor.stl")
```

Units: lengths mm, flow m^3/s, pressure Pa, power W, speed rpm.

## Method

The rotor is cut into stream tubes from hub to tip. In each, the blade row is a cascade:

- **Work** by the Euler equation, rho U c_u2, with no swirl coming in.
- **Exit flow**: `exit="sre"` for a free duct downstream (simple radial equilibrium, dp/dr = rho c_u^2 / r, continuity
  holding the flow), `exit="uniform"` for a resistance a few mm behind (a heat exchanger's straight channels hold the
  axial velocity even and carry the radial pressure difference). At a low flow coefficient the two differ a lot: any
  spanwise change in total pressure larger than the axial dynamic pressure (about 10 Pa for a 120 mm fan at 4 m/s)
  moves the flow hard toward one end of the blade in a free duct.
- **Blade angles** from the flow angles: incidence onto the camber line at the leading edge, Carter's rule for the
  deviation (circular-arc camber line), plus up to 1.5 deg at low Reynolds number.
- **Solidity** from Lieblein's diffusion factor (0.45 at design by default, stall flagged past 0.6 on the outer 80 % of
  the span; the hub's own overload is reported apart, as `hub_stall`).
- **Losses**: profile loss from Lieblein's equivalent diffusion ratio with Aungier's momentum-thickness fit, raised by
  (2e5 / Re)^0.4 below Re 2e5; endwall loss from Howell's drag coefficients (annulus 0.02 s/H, secondary 0.018 C_L^2),
  taken off as the passage's flow-weighted average, since it lives in the endwall layers and the secondary flow that
  carries it across the span.
- **Tip**: `band` (no gap; skin friction on its outside as a flat plate), `shroud` with a gap (pressure falls by
  2 gap / span), `open` (Prandtl's tip-loss factor on the work).
- **Sections**: circular-arc camber line, NACA four-digit thickness opened to a printable trailing edge. At the Reynolds
  numbers a small fan runs at (3-8 x 10^4 on the chord) thin cambered plates outperform thick airfoils.

`design` builds blades for one point; `performance` and `curve` evaluate them anywhere; `operating_point` finds the
crossing with a system; `optimize` searches blade count, design point, spanwise work distribution, diffusion and speed
(Nelder-Mead from a start where the system takes about a third of the shaft power as air power, warm-started across
blade counts and polished) for the most flow, or any objective, within a shaft power, speed range and axial depth, with
a stall margin.

## What it is and is not

A correlation model, as a fan designer's first pass is. It is not CFD. Its loss and deviation correlations come from
cascade tests at higher Reynolds number than a small fan sees, and it does not predict the stalled part of the curve
(it flags it). Expect the pressure at a given flow to within 15-25 %; test a printed rotor before relying on the last
few percent. A useful habit the model makes plain: against a system whose pressure rises as q^2, the flow at a fixed
shaft power goes as the cube root of the fan's efficiency, so the motor's power usually matters more than the last
points of blade efficiency.

Tests: `tests/test_fan.py` (design reproduces its point, the curve falls past it and stalls far below, Euler and
efficiency bounds, a free vortex overloading a small hub, open tip < shroud gap < band, the optimizer beating a hand
design at the same power, one watertight body, band hoop stress).
