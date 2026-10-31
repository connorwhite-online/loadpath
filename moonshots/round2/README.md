# Moonshots round 2 (2026-10-31)

Three stuck limits, three prototypes, cloud box (4 cores, one thread each). One landed as an option, two are honest negatives or partials with useful spin-offs.

## 1. Sharp creases between two organic field surfaces (landed, opt-in)

`loadpath/crease.py`, field graphs `"mesh": {"creases": true}`. After marching cubes, edges whose ends sit on different branches of a hard min/max are split on the true intersection curve (Gauss-Newton onto both zero levels), then slivers are collapsed and any crease vertex that would fold the mesh is put back.

| case (0.4 mm pitch) | crease error before (max / RMS mm) | after |
|---|---|---|
| two spheres, 90 deg rim | 0.258 / 0.149 | 0.004 / 0.001 |
| 30 deg wedge | 0.956 / 0.749 | 0.008 / 0.003 |
| body minus torus | 0.805 / 0.361 | 0.255 / 0.014 |
| gyroid clipped by organic body (0.25) | 0.634 / 0.134 | 0.445 / 0.026 |

Smooth bodies: creases recovered to microns, dihedral angles right, meshes stay watertight. Lattices: RMS drops 4-5x but the worst point only ~30%, because the existing relax step already folds the lattice mesh where walls meet the body, and the guard leaves those spots rounded. Cost: under 1 s on smooth parts, 2-11x the meshing time on lattices. To make it the default: a relax that won't move vertices across a crease. Scripts: `crease_bench.py`, `closeup.py`, results in `results.json`.

## 2. Cut-cell voxel stress (prototype, not wired)

Tried fill-scaled stiffness, finite-cell subcell quadrature, incompatible-mode hex8 and stress read at the true surface. Beam benchmark (`bench_beams.py`) against beam theory, and real parts against P2 tets (`realpart.py`).

- Fill-scaled stiffness is wrong (1-cell strip 3.5x too stiff).
- Subcell + incompatible modes + surface read: 3-5 cell members within 0-12%, real parts within 1-5% of tet (today's verify reads +4 to +13%).
- But 1-2 cell members at 30-45 deg under-read by ~40%, the unsafe direction, and wiring it needs per-cell matrices that the multigrid path can't take. Not landed.
- **Spin-off worth doing:** plain hex8 under-reads bending stress by 38/16/9/4% at 1/2/3/5 cells thick (unsafe). Incompatible-mode hex8 fixes it with one constant 24x24 matrix that also works in mg.py.

## 3. Global layout search at a fixed fraction (negative)

Deflation (Papadopoulos/Farrell) in `simp.py` (internal `SimpOptions` fields, off) plus noise restarts, every candidate verified (`basins.py`).

- Deflation does find distinct layouts, and verified SF swings 3-5x between layouts at the same fraction, uncorrelated with compliance.
- But extra layouts are mostly weaker than the current single run. Arm link: one win (vf 0.25 passes, ~11% lighter than main's pick). GRIP wrist yoke: no wins; only the existing warm chain helped.
- Cost 5-15x per fraction. Not worth it across the search; the cheap form (a few restarts just below the confirmed pick) is what main's "layout search below a tet-confirmed pick" (200ecc4) already does.

## Not prototyped this round

- Turbulent flow without a GPU: the porous whole-exchanger model (porous.py) already covers the cheap path; a RANS closure fit to the cell LBM data is the next step if needed.
- Lattices vs solid on stiffness per gram: settled negative in round 1.
