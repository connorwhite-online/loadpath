# Feel targets: interiors designed for how a product feels

An industrial designer hands over the surfacing. What goes inside is known
(a board, a battery, a microphone capsule) and so is how the product should
feel in the hand: the grip gives a little but is not mushy, a fall onto the
floor does not break what is inside, and rubbing a thumb along the body
does not come through on the recording. Everything inside the exterior is
invisible, so it should be designed for exactly that, with the least
material.

`loadpath feel` does this. As with push pads ([SWITCH.md](SWITCH.md)) the
exterior is kept as a thin cosmetic `skin`, the components are keep-outs
with their **assembly paths** (`insertFrom`), and everything else is design
space. A `feel` block lists the targets:

```json
"feel": {"targets": [
  {"kind": "squeeze", "name": "grip",
   "hold": {"face": "-x", "shape": GRIP}, "press": {"face": "+x", "shape": GRIP},
   "direction": "-x", "force": 20, "travel": [0.3, 1.0]},
  {"kind": "drop", "name": "head-drop", "lands": {"face": "+z"}, "mass": {"region": "mount"},
   "kg": 0.004, "height": 1.0, "maxG": 1500, "maxTravel": 3.0},
  {"kind": "isolate", "name": "handling-noise", "hold": {"region": "skin", "shape": GRIP},
   "mount": {"region": "mount"}, "kg": 0.004, "band": 500, "maxTransmissibility": 0.5,
   "maxTravel": 0.5, "shockG": 20}
]}
```

```
python -m loadpath feel mic.study.json --out mic-out/
```

A study with a `feel` block needs no supports or load cases: every target
says where the product is held and where its load comes in. A `switch`
block in the same study is designed together with the feel targets, so a
push pad's click force and the grip around it come out of one run.

## The three targets

Each target is a stiffness window between where the product is held and
where the load goes in. Where is a `region` (a preserve or keep-out id;
`skin` is the exterior skin), a `shape`, or an exterior `face` (`"-x"`:
the outermost node of every column facing -x); a region or face with a
shape too is only the part of it inside the shape.

**squeeze** — how much a grip gives. Held on one side (`hold`), pressed on
the other (`press`) along `direction` with `force` N, the body should close
by `travel: [min, max]` mm (or give `stiffness: [min, max]` in N/mm).

**drop** — cushioning a part inside (`mass`, `kg`, usually its mount) when
the product falls `height` m and lands on `lands` (a face, or a patch of
one). The structure between the landing patch and the part is a spring k.
With the fall's energy E = m g h, the peak force is sqrt(2 E k) and the
part's travel is sqrt(2 E / k), so

```
2 E / maxTravel^2  <=  k  <=  (maxG m g)^2 / (2 E)
```

`maxG` caps the stiffness and `maxTravel` (the room the part has before it
hits something) bounds it from below. The two together need
maxTravel >= 2 h / maxG whatever is inside: a 1 m fall at under 1500 g
needs at least 1.3 mm. If the numbers ask for less, the study is refused
with that sentence. A linear spring is the worst shape a cushion can have;
a constant-force crush (a lattice, [IMPACT.md](IMPACT.md)) halves the travel.
The shell's own mass is left out: what is asked for is the part's
deceleration.

**isolate** — keeping handling vibration away from a part (`mount`, `kg`)
while the hand (`hold`) shakes the body, along each of `directions`
(default x, y and z). The part on its mount is a suspension of natural
frequency f_n = sqrt(k / m) / 2 pi, and with the material's loss factor eta
(modal.LOSS_FACTOR; TPU 0.12) a shake at f reaches it as

```
T(r) = sqrt((1 + eta^2) / ((1 - r^2)^2 + eta^2)),   r = f / f_n
```

There are two ways to keep a band of handling noise off a part:

- a **soft mount** (`band`): from `band` Hz up, T is at most
  `maxTransmissibility` (default 0.5). That caps f_n, so k.
- a **stiff mount** (`upTo`): the resonance sits over the handling band, so
  nothing up to `upTo` Hz is amplified more than `maxAmplification`
  (default 1.25). That sets a floor on f_n. A part hard-mounted to the
  exterior (a board on rails moulded into the base) can only be this one.

`maxTravel` under a `shockG` knock (default 1 g, its own weight) adds a
floor so a soft mount does not rattle.

## The model

All three are compliance windows. For each target and direction, with the
target's holds fixed and 1 N spread over the loaded nodes (f):

```
c = f . K^-1 f      k = 1 / c      dc/dx_e = -u_e . dK_e u_e
```

so each costs one solve per direction per iteration and its sensitivity is
that solve. Targets that share a hold share a factorization. The
constraints are written on log c, so a factor of two weighs the same at
2 N/mm and at 200 N/mm, and go to the push pad's optimizer (minimum volume,
MMA, thin-skin density floors, coarse to fine).

Three things the first runs got wrong, and what each run does about them
now:

**Grey cells are cheap softness.** A soft window met on the density field
alone was met with cells at 0.3-0.6: under SIMP a grey cell is soft per
gram, which printing does not have. The microphone body's capsule
suspension read 8.7 N/mm on the design and 0.5 N/mm thresholded. Each floor
(k >= min) is now read on the eroded design (projected at 0.7) and each
ceiling (k <= max) on the dilated one (0.3), so the black-and-white part
between them meets both (`"robust": false` turns this off). The
thresholded reading now lands within 5% of the design.

**Members ending on a thin skin stopped short of it.** The delivered mesh
joins the designer's exact skin and is not run on into it (a fillet to a
skin that wraps the whole exterior fills the inside). On a coarse grid the
skin is a row of partly filled cells, and the member's last cell was the
one next to it: the suspension came out as a second, floating body. Every
thin-skin cell beside a solid interior cell is now meshed solid, so members
run into the skin (switch._as_printed, push pads too).

**Smoothing thins one-cell members, and a coarse read-back cannot see
them.** A suspension soft enough for a 4 g capsule is members about one
cell thick. The default smoothing rounded them to a fraction of that, and
the delivered mesh read 2 N/mm against a 50 N/mm design. Feel runs mesh
with reconstruct's thin-member setting (`keepThinMembers`, default true
here) and read the delivered mesh back on a grid of half the design pitch
(`verifyPitch`), where a one-cell member is two cells across.

## Checks on the result

- **Thresholded**: the same readings on the black-and-white design.
- **Delivered**: the STL read back on the half-pitch grid
  (`delivered`, `deliveredOk`, `verifyPitch`).
- **Modes** (isolate targets): the delivered part's modes with the part as
  a point mass on its mount and the structure's own mass, the hand's nodes
  the shaken base; the transmissibility at `band`, or the worst
  amplification up to `upTo`, by modal superposition (`modal`).

## A handheld microphone body

`examples/feel/mic_body.py`: a generic stick microphone, 24 mm across and
60 mm long, in SLS TPU 88A. The exterior is kept as a 1.0 mm skin, open at
the bottom where a plug closes it. A 10 mm capsule sits at the head on a
ring mount; its cable path and a board both slide in from the bottom
(`insertFrom: "-z"`). The targets are the three above: a 20 N grip gives
0.3-1.0 mm, a 1 m fall on the head keeps the 4 g capsule under 1500 g and
3 mm of travel, and handling noise from 500 Hz up reaches the capsule at
under half.

![the microphone body cut along its axis and across the capsule mount](images/feel/mic-body-sections.png)

| design (1 mm cells; delivered read at 0.5 mm) | grip at 20 N | head drop, 4 g capsule | capsule mount, x / y / z |
|---|---|---|---|
| window | 0.3-1.0 mm (20-67 N/mm) | under 1500 g and 3 mm (8.7-44 N/mm) | 1.6-13 N/mm: under half from 500 Hz up |
| bare skin | 1.19 mm: too soft | the capsule is not held at all | not held |
| filled | 0.009 mm | 12,800 g | 1430 Hz: rings in the band |
| **loadpath, design** | **0.96 mm** | 1620 g, 1.2 mm (8% over) | **190 / 185 / 236 Hz; 0.17 / 0.16 / 0.29 at 500 Hz** |
| **loadpath, delivered mesh** | **0.64 mm** | 1690 g, 1.2 mm (13% over) | **226 / 199 / 259 Hz (lumped); modes 164-226 Hz, 0.44 / 0.49 / 0.29 at 500 Hz** |

The interior is 21% of the design space, 6.3 g of TPU. The capsule hangs
from the head on short posts and one side strut, so the fall onto the head
is cushioned by the head's own skin flexing, and the hand's vibration has
to come up the whole tube and through the posts to reach it. Grip and
isolation are met on the delivered mesh, modes included. The drop is not:
the posts that hold the capsule still enough in three directions make the
head 13% too stiff for 1500 g (1690 g at 1.2 mm of travel). On 1 mm cells
the optimizer could not find a hanger that does both; whether a finer
grid (`pitches: [1.0, 0.7]`) can is the next run.

120 iterations, 2085 s on 4 cores shared with another run (about 15 minutes
alone), then 3 minutes of read-back and modes at 0.5 mm.

## On a real product: the touch microphone

The touch microphone's push pad case (`touch-mic-pad`, [SWITCH.md](SWITCH.md))
again, its CAD read from the folder `LOADPATH_TOUCH_MIC_STL` names (it is
not in this repository), with three feel targets added to the same study.
The eval case is `touch-mic-feel`:

- **pinch**: 10 N across the flanks gives 0.15-0.6 mm;
- **face-down drop**: 0.75 m onto the touch face, the 6 g board on its
  rails sees under 2000 g and moves under 1.2 mm;
- **handling noise**: the board sits on rails moulded into the base ring,
  which the study keeps as drawn, so it is hard-mounted. Its bare-skin
  reading is 28-35 N/mm in plane (345-385 Hz) and 7 N/mm (172 Hz) up and
  down. A soft mount is ruled out; the target is the stiff one, the
  board's resonance over a 150 Hz handling band (`upTo`: over 335 Hz).

| design (1 mm cells; delivered read at 0.5 mm) | pinch | face-down drop | board bounce z | presses in window | part |
|---|---|---|---|---|---|
| bare skin, base and rails | 1.26 mm | 1040 g, 1.45 mm | 172 Hz | 9 of 14 | |
| push pad alone (SWITCH.md) | | | | 13 of 14, 11 delivered | 8.2 cm^3 |
| **pad and feel, design** | **0.47 mm** | **1910 g, 0.79 mm** | **338 Hz** | 10 of 14 | |
| **pad and feel, delivered** | **0.40 mm** | 2850 g, 0.53 mm | **351 Hz; modes pass** | 9 of 14 | 10.6 cm^3 |

The feel targets cost the click map. Holding the board's bounce over
335 Hz takes stiff material between the rails and the top skin, and that
stiffens the corners of the touch area (up to 448 gf delivered against a
300 gf ceiling). The drop meets its window on the design and misses on the
delivered mesh: the face-down fall loads the touch face in membrane, which
the 1 mm thin-skin model reads low (it is tuned for bending, membrane
stiffness reads low by phi^2), and the 0.5 mm read-back has the skin
exactly. That is a real limit of the coarse grid for membrane-loaded
skins; the honest reading is the delivered one. Whether to trade some
click-map accuracy for a board that does not ring is the designer's
call; the report gives both. 22 minutes on 4 shared cores.

## Limits

- Linear and quasi-static: the drop is an elastic spring (no crush, no
  rate), the isolation a single-mode estimate in the optimizer (the modal
  check after it has every mode). TPU stiffens with rate
  (viscoelastic.py: about 1.5x at 35 Hz); feel windows are read at the
  material card's modulus.
- A hold is clamped in all three directions; a hand is softer than that.
- The read-back is a voxel model of the mesh at half the pitch, not a tet
  model. For a part that matters, `loadpath verify` on the STL.
