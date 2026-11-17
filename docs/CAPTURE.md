# Capture: what holds a socket when a plug pushes on it

A USB-C plug goes in at up to 20 N, comes out at up to 20 N and gets levered
sideways (connectors.py has the numbers). Each of those forces reaches the
socket and then goes one of two places. It can go into **a face**, meaning
material of some part standing right against the socket on the side the force
pushes toward. That path is plastic in compression, it is stiff, and the
optimizer can design it. Otherwise it goes into **the socket's legs**: the
solder joints and shell tabs, then the board, then whatever holds the board.
Pads lifting off a board under repeated plug forces are the classic USB-C
failure.

Whether a part *can* have a face on a side of the socket is decided by how the
product goes together, not by taste. Take the parts apart in reverse order.
When part X leaves the socket moving along `u`, a face of X on the side `d`
that a force pushes the socket toward is:

- **seated** if `u . d > 0`. The face moves away as X comes off, so it can be
  drawn line-to-line: an end stop, or a lid that closes down onto the socket;
- **press** if `u . d` is about 0. The face slides along the socket on the way
  in, so it needs a fit with no play (crush ribs, a transition fit). A
  clamshell's front and rear shoulders are this kind;
- **impossible** if `u . d < 0`. The face would stand in the socket's way in.

For the part the socket goes into, `u` is minus the socket's own way out. So a
socket slid in along its own axis, from behind toward its mouth, can never have
a face of that part behind it. The push-in drives it back the way it came, and
by construction nothing is there. Only a part that goes on later (a cover, a
flap, the other half) can put a face there, and that part has to be held too.
It can bear on the ground part by the same rule, or be held by its joints:
screws hold with no play, and snaps hold with their clearance.

That is why a designer clamshells a USB-C socket instead of sliding it in.
Dropped in across its axis, the half it lands in can stand right behind it and
right in front of it (press faces), and the other half closes on top (seated).
Every plug force then has a face.

A force counts as captured when the play along the chain (socket to part, part
to ground) is no more than the connector's `maxShift` (0.1 mm by default). Then
the face takes the force before the legs do. Pulling the plug out has no face
in most products, because the socket's whole front is the plug's way. The plan
reports that force as the legs' to carry in any assembly (`ok: null`) and asks
for the datasheet's rating (`retention`).

## Where it is used

- **split** (docs/SHELLS.md): a keep-out with `connector: {type, out, wide?}`
  is a socket. The search adds a cut through each socket along its plug axis
  (a clamshell) and always checks it. A cut that leaves a plug force on the
  solder joints ranks behind every cut that does not, whatever its seam. Each
  face the plan gives to a half is drawn as a slab that follows the socket on
  that side (`capture.bearing_pad`). It is preserved in that half, loaded with
  the plug force, and its travel is held to `maxShift`, so the optimizer ties it
  into the shell. The report's `capture` gives every force: who bears it, the
  kind of face, the play, and why. `split.capture: false` turns it off;
  `pressPlay` charges each press fit with expected play.
- **enclosure**: `loads.connector` marks the socket's keep-out, so a split
  enclosure gets all of the above. `retention` passes the datasheet rating
  through.
- **Any study or agent**: `capture.plan(loads, socket_leave, parts, ground)`
  for one way of assembling, `choose_way_in` to rank the socket's ways in, and
  `measure` for the real gaps on drawn or delivered meshes, with the play
  chain and the bearing stress.

## Touch-Mic

The rebuilt Touch-Mic body (`/mnt/project-files/capture/touch-mic/`):

- **As built**, the board slides forward into the port. The body cannot stand
  behind the socket, so the push-in can only be held by a block on the flap
  behind the socket, with two press fits in series. Measured, nothing is within
  2 mm behind the socket, and the body is 0.17 to 0.22 mm off its other sides.
  Every plug force goes to the solder joints first.
- **Dropped in from below with the flap closing under it**, the body itself
  takes the push-in (press), the top (seated) and both sides. The flap takes
  the downward lever, provided the flap is itself held on with no play (screws
  or glue). With nothing holding the flap, that 8 N is back on the legs.
