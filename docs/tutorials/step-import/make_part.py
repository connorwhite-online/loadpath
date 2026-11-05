"""
How elbow.step was made (once; the STEP is committed, this is not part of
the tutorial). Needs OpenCascade: pip install "loadpath[step]".

A shoulder bracket of the kind the robot arm's arm is built from: a 56 x 36 x 6 mm
base held by four counterbored M3 screws, two each side of an 8 mm
upright carrying a 16 mm bearing seat (a 688 bearing) whose axis runs
along x, 36 mm up.
"""
import os

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

BOLTS = [(8, 8), (8, 28), (48, 8), (48, 28)]   # x, y of the M3 screws (down z)
SEAT = (18.0, 36.0, 8.0)                       # y, z, r of the bearing seat (along x)


def box(a, b):
    return BRepPrimAPI_MakeBox(gp_Pnt(*a), gp_Pnt(*b)).Shape()


def cyl(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*d)), r, h).Shape()


s = BRepAlgoAPI_Fuse(box((0, 0, 0), (56, 36, 6)), box((24, 0, 0), (32, 36, 46))).Shape()
for x, y in BOLTS:
    s = BRepAlgoAPI_Cut(s, cyl((x, y, -1), (0, 0, 1), 1.7, 8)).Shape()     # M3 clearance
    s = BRepAlgoAPI_Cut(s, cyl((x, y, 3), (0, 0, 1), 3.0, 4)).Shape()      # counterbore, 3 mm deep
y, z, r = SEAT
s = BRepAlgoAPI_Cut(s, cyl((23, y, z), (1, 0, 0), r, 10)).Shape()
u = ShapeUpgrade_UnifySameDomain(s)   # one face per plane, as a CAD tool writes it
u.Build()
s = u.Shape()
w = STEPControl_Writer()
w.Transfer(s, STEPControl_AsIs)
assert w.Write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "elbow.step")) == IFSelect_RetDone
