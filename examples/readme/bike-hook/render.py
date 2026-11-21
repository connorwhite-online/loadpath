import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]) + "/docs"); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import numpy as np, trimesh
from render3d import Scene
part = trimesh.load("out/bike-hook-30.stl")
tube = trimesh.creation.cylinder(radius=25, height=170, sections=96)
tube.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])); tube.apply_translation([165, 0, 0])
wall = trimesh.creation.box((4, 140, 190)); wall.apply_translation([-2, 0, -15])
s = Scene(panel_size=(760, 560))
p = s.panel(None, view=(18, -62))
p.mesh(wall, color=(0.86, 0.86, 0.84))
p.mesh(part)
p.mesh(tube, color=(0.2, 0.45, 0.85), opacity=0.22)
s.render("bike-hook.png")
