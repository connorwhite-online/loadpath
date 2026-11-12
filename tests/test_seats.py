"""finish_seats on a plate with a block standing on it and a box seated in
a pocket: the block's root gets a fillet, the seat clears the box by its
clearance, the outside of the region is left exact, the mesh is one solid."""
import numpy as np
import trimesh

from loadpath.seats import finish_seats


def test_fillet_seat_and_exact_outside():
    plate = trimesh.creation.box([30, 20, 2]); plate.apply_translation([0, 0, 1])
    block = trimesh.creation.box([10, 12, 4]); block.apply_translation([-6, 0, 4])
    part = trimesh.boolean.union([plate, block], engine="manifold")
    box = trimesh.creation.box([4, 4, 3]); box.apply_translation([-6, 0, 4.5])   # sits in the block
    out, rep = finish_seats(part, [{"mesh": box, "clearance": 0.3, "noOverhang": True}],
                            {"min": [-14, -9, 2.02], "max": [14, 9, 8]}, fillet=0.8, round_=0.3, pitch=0.1)
    assert rep["watertight"] and rep["pieces"] == 1
    # the seat: nothing within the clearance of the box, and its plan clear above
    pq = trimesh.proximity.ProximityQuery(out)
    probe = box.vertices * 1.0
    assert (pq.signed_distance(probe) < -0.25).all()
    assert not out.contains([[-6, 0, 7.5]]).any()
    # the fillet: material in the inside corner where the block meets the plate
    assert out.contains([[-0.85, 0, 2.15]]).all() and not part.contains([[-0.85, 0, 2.15]]).any()
    # below the region the plate is as drawn
    s = out.section([0, 0, 1], [0, 0, 1]).to_2D()[0]
    assert abs(s.area - 600) < 1.0
