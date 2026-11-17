"""The drawn solids of the pod, in build123d. Millimetres; z up; the USB-C
port leaves through the +y end. Every number the spec also needs (the
board's place, the port's axis) is read from here, not typed twice."""
from build123d import (Align, Axis, Box, BuildPart, BuildSketch, Cylinder, Locations, Plane, RectangleRounded,
                       SlotOverall, extrude, fillet)

L, W, H = 52.0, 40.0, 14.0          # the pod's outside
BOARD = (17.8, 21.0, 1.2)           # a XIAO-sized board
BOARD_Z = 4.0                       # its underside
BOARD_Y = L / 2 - 2.6 - BOARD[1]    # its near edge: the receptacle's mouth sits 2.6 mm inside the end
PORT_Z = BOARD_Z + BOARD[2] + 1.6   # the receptacle's axis
MOUTH_R = 1.2                       # the port mouth's round
SWITCH = (6.0, 6.0, 5.0)
SWITCH_Y = -8.0


def exterior():
    """A soft-edged puck: the designer's form."""
    with BuildPart() as p:
        Box(W, L, H, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), 12.0)
        fillet(p.edges().group_by(Axis.Z)[-1], 5.0)
        fillet(p.edges().group_by(Axis.Z)[0], 2.5)
    return p.part


def board():
    """The board with its USB-C receptacle (the part bought, as a solid)."""
    with BuildPart() as p:
        with Locations((0, BOARD_Y, BOARD_Z)):
            Box(*BOARD, align=(Align.CENTER, Align.MIN, Align.MIN))
        with Locations((0, BOARD_Y + BOARD[1] - 7.3, BOARD_Z + BOARD[2])):
            Box(8.94, 7.35, 3.2, align=(Align.CENTER, Align.MIN, Align.MIN))
    return p.part


def switch():
    with BuildPart() as p:
        with Locations((0, SWITCH_Y, BOARD_Z)):
            Box(SWITCH[0], SWITCH[1], 3.6, align=(Align.CENTER, Align.CENTER, Align.MIN))
            Cylinder(1.75, SWITCH[2], align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def port():
    """The USB-C plug's way in: a 9.4 x 3.8 stadium through the end wall
    to the receptacle's mouth, rounded at MOUTH_R where it meets the
    outside. The round is drawn against a face 0.1 mm outside the end
    wall, so it leaves the wall at a shallow angle instead of a step."""
    face = L / 2 + 0.1
    with BuildPart() as p:
        with BuildSketch(Plane.XZ.offset(-face)):
            with Locations((0, PORT_Z)):
                SlotOverall(9.4, 3.8)
        extrude(amount=face - (BOARD_Y + BOARD[1]) + 0.3)     # the tunnel, in to the receptacle
        with BuildSketch(Plane.XZ.offset(-face)):
            with Locations((0, PORT_Z)):
                SlotOverall(9.4 + 8, 3.8 + 8)
        extrude(amount=-4.0)                                  # a block outside, to round the mouth against
        mouth = p.edges().filter_by(lambda e: abs(e.center().Y - face) < 1e-6 and e.length < 40)
        fillet(mouth, MOUTH_R)
    return p.part


def plunger():
    """The boss under the top that presses the switch: 0.4 mm over its cap."""
    top = SWITCH[2] + BOARD_Z + 0.4
    with BuildPart() as p:
        with Locations((0, SWITCH_Y, top)):
            Cylinder(1.6, H - top - 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().group_by(Axis.Z)[0], 0.3)
    return p.part


def posts():
    """Four posts down from the roof: the board is pushed up against them
    from below and the lid holds it there."""
    top = BOARD_Z + BOARD[2]
    with BuildPart() as p:
        for x in (-BOARD[0] / 2 + 1.6, BOARD[0] / 2 - 1.6):
            for y in (BOARD_Y + 1.8, BOARD_Y + 12.5):
                with Locations((x, y, top)):
                    Cylinder(1.3, H - top - 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().group_by(Axis.Z)[0], 0.3)
    return p.part


def lid_opening():
    """The underside opens so the board and the switch go in from below: a
    rounded window a separate lid closes."""
    with BuildPart() as p:
        with BuildSketch(Plane.XY.offset(-1.0)):
            with Locations((0, 2.0)):
                RectangleRounded(BOARD[0] + 6.0, 44.0, 4.0)
        extrude(amount=1.0 + BOARD_Z + 0.2)
    return p.part
