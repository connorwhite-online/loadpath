"""The study's load cases as a stress limit on a design made for something
else (cooling, feel, a switch's click).

Those designs minimize material subject to their own limits (switch.
optimize_switch with `providers`), and until this read no load at all: a
speaker designed for its heat alone came out with its driver's drawn cradle
a loose piece, nothing tying it to the shell, and its delivered part read SF
0.04 in the drop its study asks it to survive. Under a stricter skin limit
the heat-spreading material had happened to tie it in.

The limit is the one the constrained method holds (constrained.py): per
load case, the relaxed von Mises stress of each cell (sigma_e = x_e^q
vm(C B u_e), q = 1) against the allowable stress (strength over the safety
factor and verify's allowance), with the layer bond beside it where the part
is built in layers, aggregated with a P-mean scaled to the peak (Le et al.
2010). Cells within the contact band of a load or a support are left out,
as the checks leave them out. One adjoint solve per case, on the same
reduced system as the solve. Read at each cell's centre: the design grid's
corner spikes are the mesh's, and the delivered part is checked on its own
mesh after (enclosure.check_delivered).
"""
from __future__ import annotations

import numpy as np

__all__ = ["StressLimit", "stress_limit_applies"]

_Q = 1.0
_P = 8.0


def stress_limit_applies(study) -> bool:
    """Whether a study's loads ask for a stress limit: load cases beside a
    safety factor (the stiffness-only inertial cases are not counted)."""
    goal = getattr(study, "goal", None)
    return bool(getattr(goal, "safety_factor", None)) and bool(study.raw.get("loadCases"))


def _every_cell_solves():
    from .switch import _Solves

    class _EveryCellSolves(_Solves):
        """switch._Solves on every cell. Its pruning keeps only the cells
        joined to a support, which is right for a click's force but blind
        here: a loaded drawn cradle cut loose drops out of the system with
        its load, reads no stress and steers nothing (the speaker's ran
        to 1e8 times its limit and back between iterations). Kept at the
        void modulus, the loose piece's load strains the near-void cells
        that would hold it, and the limit asks for them."""

        def _alive(self, xp):
            return np.ones(self.domain.mesh.n_elem, bool)

    return _EveryCellSolves


def _EveryCell(domain, nu):
    return _every_cell_solves()(domain, nu)


class StressLimit:
    """An optimize_switch provider: one constraint per load case,
    kappa * P-mean(sigma / allowable) - 1 <= 0."""

    def __init__(self, study, domain, nu, modulus):
        from .constrained import _bond_form
        from .fem import elastic_c, hex8_b
        from .study import _layer_bond_limit, contact_band
        from .verify import STRESS_ALLOWANCE

        self.domain = domain
        self.nu = nu
        mesh = domain.mesh
        mat = study.material
        self.allow = float(mat.strength / (study.goal.safety_factor * STRESS_ALLOWANCE))
        inert = set(getattr(domain, "inertial_cases", []) or [])
        self.cases = [c for c in range(len(domain.loads)) if c not in inert]
        self.m = len(self.cases)
        self.kappa = [1.0] * len(self.cases)
        self.it = 0
        self.freeze = False     # keep kappa as it is (a gradient check)
        self.solves = _EveryCell(domain, nu)
        self.cb = elastic_c(1.0, nu) @ hex8_b(0.0, 0.0, 0.0, nu) / mesh.pitch
        from .constrained import _VM
        self.vm_form = _VM
        bond = _layer_bond_limit(study)
        if bond is not None:
            self.bond_ratio = float(bond[0])
            self.bond_m, self.bond_q = _bond_form(bond[1])
            self.bond_k2 = float(bond[2]) ** 2
        else:
            self.bond_ratio = None
        held = domain.support_nodes | domain.load_nodes
        counted = ~held[mesh.enode].any(axis=1)
        band = contact_band(study, domain)
        if band and held.any():
            from scipy.spatial import cKDTree
            centre = (mesh.elem_ijk + 0.5) * mesh.pitch
            near = cKDTree(mesh.node_ijk[held] * mesh.pitch).query(
                centre, k=1, distance_upper_bound=float(band))[0] < np.inf
            counted &= ~near
        self.counted = counted
        self.last = {}

    def evaluate(self, xp, ctx=None):
        from .switch import _dpair

        self.it += 1
        d = self.domain
        mesh = d.mesh
        self.solves.factor(xp)
        us = self.solves.solve([d.loads[c] for c in self.cases], "stress")
        nc = max(int(self.counted.sum()), 1) * (2 if self.bond_ratio is not None else 1)
        act_all = np.flatnonzero(self.counted)
        rhs_all, partial = [], []
        for k, (c, u) in enumerate(zip(self.cases, us)):
            u = np.asarray(u)
            moved = u.reshape(-1, 3).any(axis=1)[mesh.enode[act_all]].any(axis=1)
            act = act_all[moved]
            edof = mesh.edof[act]
            sig = u[edof] @ self.cb.T                         # (cells, 6) MPa (unit-modulus solve)
            xa = xp[act]
            parts = []
            sv = sig @ self.vm_form
            vm = np.sqrt(np.maximum(np.einsum("ij,ij->i", sv, sig), 1e-30))
            parts.append((sv, vm, self.allow, xa ** _Q * vm / self.allow))
            if self.bond_ratio is not None:
                lim = self.allow * self.bond_ratio
                sn = np.maximum(sig @ self.bond_m, 0.0)
                sb = sn[:, None] * self.bond_m + self.bond_k2 * (sig @ self.bond_q)
                eq = np.sqrt(np.maximum(np.einsum("ij,ij->i", sb, sig), 1e-30))
                parts.append((sb, eq, lim, xa ** _Q * eq / lim))
            smax = max((float(sr.max()) for *_, sr in parts if len(sr)), default=0.0)
            sc = max(smax, 1e-30)
            pm = sc * (sum(float(np.sum((sr / sc) ** _P)) for *_, sr in parts) / nc) ** (1.0 / _P)
            if not self.freeze:
                kk = smax / max(pm, 1e-30)
                self.kappa[k] = 0.5 * (self.kappa[k] + kk) if self.it > 1 else kk
            g = self.kappa[k] * pm
            g_x = np.zeros(mesh.n_elem)
            rhs = np.zeros(mesh.n_dof)
            for s, eqv, lim, srel in parts:
                dg_ds = self.kappa[k] * (srel / max(pm, 1e-30)) ** (_P - 1) / nc
                g_x[act] += dg_ds * (_Q * np.maximum(xa, 1e-12) ** (_Q - 1) * eqv / lim)
                coef = dg_ds * (xa ** _Q / (eqv * lim))
                rhs += np.bincount(edof.ravel(), weights=(coef[:, None] * (s @ self.cb)).ravel(),
                                   minlength=mesh.n_dof)
            rhs[d.fixed] = 0.0
            rhs_all.append(rhs)
            partial.append((g, g_x, u, smax))
        lams = self.solves.solve(rhs_all, "stress-adj")
        dpair = _dpair(d, self.nu, xp, 1.0, self.solves)
        out = []
        for c, (g, g_x, u, smax), lam in zip(self.cases, partial, lams):
            g_x = g_x + dpair(np.asarray(lam), u)
            self.last[d.case_names[c]] = smax * self.allow
            out.append((float(g - 1.0), g_x, f"stress.{d.case_names[c]}"))
        self.solves.release()
        return out
