# Designed weak links (`fuses`)

Some crashes bring more energy than a light part can hold in one piece. A small drone that falls a metre
carries about 3 J, and its PA12 duct rings can't take that whole however they're shaped: they fail where the
hit lands. Other examples are a robot's bumper cover, a shelf bracket a child hangs from, and a tool handle
that gets levered too hard. In each case the right design is a link that lets go first, at a force chosen
so the expensive part behind it survives, and that is cheap to reprint afterwards. Aircraft hang their
engines on fuse pins for this reason, and roadside signposts stand on slip bases.

A fuse has to do two jobs, and they pull against each other:

- **Hold every service load**, with the service safety factor, even on its weakest print.
- **Break below what the protected part can carry**, even on its strongest print, loaded at impact rate.
  Polymers get stronger the faster they're loaded.

The gap between the weakest and the strongest print is the fuse's **window**. loadpath sizes the link to
the smallest section that still holds service, which puts the window as low as it can go. Every crash case
that runs through the link is then capped at the top of the window: the hit can't put more on the part
behind than the link passes before it breaks. The same study then designs the protected part for that
capped force.

```json
"fuses": [{"id": "duct1-pins", "type": "shear-pin", "region": "duct1-seat", "axis": [0, 0, 1],
           "count": 2, "planes": 1}]
```

- `region` is the preserved region where the breakaway part's loads act on this part. Every load that
  names it runs through the fuse, so the force through the link comes from statics, not from a guess.
- `holds` lists the cases the link must survive. By default it's every case without an `impact` block. The
  link is sized to the largest of force × safety factor over them, each case at its own `safetyFactor`
  (`serviceSafetyFactor` on the fuse overrides them all). The force is the shear across a pin, or the pull
  along a tension link.
- `breaks` lists the crash cases. By default it's every impact case with a load on the region. Each gets
  `impact.maxForce` set to the top of the window, so the impact model's peak force (spring in series with
  the crusher) is held to it. A plain force case that would exceed the window is scaled down to it.
- `report.fuses` gives, for each link, its size, its break force (nominal, low and high), the window's
  width, what sized it (service or the printable minimum), and which cases it capped. When an impact
  case is capped, its `report.cases[].impact` shows `cappedAt` and the `uncappedForce` it would have been.

## Where the energy goes

Breaking a pin stops almost nothing: a brittle shear takes a few millijoules. A link that lets go at force F
has only absorbed what the crusher in front of it and the part's own give hold up to F, and the rest of the
hit is still moving. That rest arrives at the next contact. So the report gives, for each capped case, the
hit's energy, `releaseEnergy` (what was taken before the link went) and `energyPastRelease` (what is left),
and it warns when the hit is not stopped. The verify report gives the same numbers with the part's
compliance included.

That leaves two ways to use a fuse:

- **Release low.** The link is sized just above service and lets go early, and the next contact gets its
  own crusher and its own case. This fits a part that is in the way rather than one that absorbs, such as
  a bracket a child hangs from, a tool handle, or a camera that should come off before its mount breaks.
- **Ride out the rating, release above it.** Put the rated crash cases in `holds` and give an overload
  case (the same hit, harder) in `breaks`. The crusher stops the rated hit with the link intact, each held
  case at its own `safetyFactor`. A harder hit lets go at the window's top, and the part behind is
  designed for that force. This fits a bumper or a duct: the energy is absorbed where it was meant to be,
  and the link keeps an overload from reaching the core.

`impact.maxForce` can also be set by hand, for anything else in the load path that lets go: a slip
clutch, a magnetic mount, a friction joint.

## The links

| type | breaks at | notes |
|---|---|---|
| `shear-pin` | planes × (σ_u / √3) × π d² / 4 per pin | Von Mises shear: ductile polymers and metals shear at 0.55-0.6 of their tensile strength. The axial share of the load must go to the pin's head or a shoulder, and the report warns when it's over 30%. `fuse_pin_mesh` writes a printable headed pin. |
| `tension-link` | σ_u × width × thickness | A waisted tab pulled along `axis`. A push goes through in bearing and doesn't work the link. |

The strength is the material card's. With `acrossLayers` (the break runs along a layer plane, as in a pin
printed upright and sheared across), it is the layer bond's strength. That is the card's measured
across-layer value when the card was measured on this process, and otherwise the process's factor.

## The window

- **Scatter**: the coefficient of variation of a printed coupon's strength. Defaults by process are in
  `fuse.STRENGTH_SCATTER`: SLS 5%, MJF 4%, FDM 10%, resin 6%. These are assumed, not measured, and a
  measured value (`scatter`) overrides them. The window spans ±2 standard deviations, so about 95% of
  prints break inside it.
- **Rate**: the top of the window is read at impact rate. `fuse.RATE_FACTOR` gives 1.3x for nylons and
  glassy polymers (10-30% stronger at impact rates, using the high end), 1.4x for elastomers and 1.1x for
  metals. The bottom of the window is read at the datasheet rate, because service loads are slow.

On SLS PA12 the window is 1.1 × 1.3 / 0.9 = **1.59x** wide. So the protected part has to carry 1.59 × the
service safety factor × the service load. When it can't, no fuse made by this process protects it. Then
the options are to cut the service load through the link, to tighten the window with a measured scatter
or a material that's less rate-sensitive, or to strengthen the protected part. Given `protectForce`, the
report says whether the link protects that part and, if it doesn't, the largest service load it could
carry and still protect it.

## Limits

- **The second hit.** Once the link breaks, the breakaway part leaves, but the rest of the body still
  arrives at whatever it was hitting. That is a second contact, and the study has to give it as its own
  case: on a drone, the motor bell or the core reaching the wall.
- **Fatigue.** The sizing is static. A link that carries cyclic service loads, such as rotor vibration,
  should be given those cases' cycles (the `durability` block) and a service factor that covers them.
- **Loads off the axis.** A pin is assumed to break in clean shear. Bending of a long pin between lugs
  with a gap is not modelled, so keep the shear planes at the lug faces.
- **Calibration.** The defaults for scatter and rate are assumptions. Until a coupon is broken at impact
  rate, treat the window as ±15%.

Tests: `tests/test_fuse.py` covers the break forces against their closed forms, sizing that holds service
on the weakest print, the window's width by process, the capping of crash cases through a study, the energy
left moving when a link lets go, held cases at their own factors, and a watertight pin.
