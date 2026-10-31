"""Profile sharpen on one case (debug helper)."""
import cProfile
import pstats
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "moonshots/round2")
from crease_bench import FieldGraph, cases  # noqa: E402

from loadpath.crease import CreaseTree, sharpen  # noqa: E402

name, p = sys.argv[1], float(sys.argv[2])
doc, meta, _ = cases()[name]
g = FieldGraph(doc)
tree = CreaseTree.from_graph(g)
m = g.mesh(pitch=p, sharp=False)
st = {}
pr = cProfile.Profile()
pr.enable()
sharpen(np.asarray(m.vertices), np.asarray(m.faces), tree, p, field=g.field(), stats=st)
pr.disable()
print(st)
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
