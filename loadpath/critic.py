"""
Vision critic: a model looks at the interface close-ups (render.render_interfaces)
and files what it sees as findings, so the defects Connor used to find by
zooming in become numbers an eval can gate on.

Why: GRIP's wrist yoke passed every metric with sawn-off faces, hard creases
at its bosses and material over its bolt seats (QA, 10/28). The crease and
boundary-cut metrics came out of that, one bug class at a time. A critic
reads the picture the way a person does and catches the next class before
anyone writes a metric for it.

Two backends:

    anthropic  calls the Claude API with the close-ups and a rubric and gets
               structured findings back. Needs the `anthropic` package and a
               credential (ANTHROPIC_API_KEY or an `ant auth login` profile).
               Model: LOADPATH_CRITIC_MODEL (default claude-opus-5-5).
    agent      no API: writes <name>.critic-request.md next to the images
               (rubric, region roles, output schema) for the agent running
               the eval to look at and answer in <name>.critic.json. Claude
               Code sessions read PNGs natively, so this is the free path.
               `python -m loadpath.critic score <out>` folds answers into
               the scorecard.

A finding has a kind (KINDS), a severity and the panel it is in. The score is
SEVERITY-weighted: minor 1, major 3, blocker 10. Evals gate on
`maxCriticBlockers` and `maxCriticScore`.
"""
from __future__ import annotations

import base64
import json
import os
import sys

KINDS = {
    "crease": "a sharp concave edge or notch where two surfaces meet (a stress raiser; should be a fillet)",
    "sawn-face": "organic material cut dead flat where nothing mates, as if sawn off on the design-space box",
    "seat-blocked": "material over a bolt head, nut, washer or bearing seat, or a bore closed over",
    "thin-member": "a strut or wall visibly thinner than its neighbours, a knife edge, or a hairline bridge",
    "floating": "a piece that does not connect, or hangs by a sliver",
    "rough-surface": "stair steps, voxel terraces, holes or tearing in the surface",
    "missing-preserve": "a boss, face or ring that should be kept is partly eaten away",
    "other": "anything else a careful engineer would send back",
}
SEVERITY = {"minor": 1, "major": 3, "blocker": 10}

SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "panel": {"type": "string", "description": "the panel title the finding is in"},
                "kind": {"type": "string", "enum": list(KINDS)},
                "severity": {"type": "string", "enum": list(SEVERITY)},
                "note": {"type": "string", "description": "what and where, one sentence"},
            },
            "required": ["panel", "kind", "severity", "note"],
            "additionalProperties": False}},
        "summary": {"type": "string", "description": "one or two sentences on the finish overall"},
    },
    "required": ["findings", "summary"],
    "additionalProperties": False,
}

RUBRIC = """You are reviewing close-up renders of a part designed by topology
optimization, before it is sent to print. Each panel is one place the part
meets the outside world, titled by region id. Grey shaded surface, lit from
above; the crop box edges cut the mesh, so straight cuts AT THE PANEL EDGE
are the crop, not the part.

Regions:
{regions}

Look for these defects, and only report what you can actually see:
{kinds}

Severity: blocker = the part cannot be assembled or will fail there (a seat
covered, a bore closed, a member nearly cut through); major = would be sent
back for rework (a sharp crease at a loaded boss, a sawn face, a knife edge);
minor = cosmetic or slight. A clean panel gets no finding. Organic, smoothly
blended shapes are intended; do not report them. Do not invent findings to
seem thorough: an empty list is a good answer for a good part."""


def _regions(study) -> str:
    if study is None:
        return "(not given)"
    lines = []
    for r in getattr(study, "preserve", []) or []:
        lines.append(f"- {r.id}: preserved geometry (boss, ring or mounting face kept exactly; should be "
                     f"intact and blend smoothly into the part)")
    for r in getattr(study, "keep_out", []) or []:
        lines.append(f"- {r.id}: keep-out (must stay empty: bore, bolt or tool access, clearance)")
    return "\n".join(lines) or "(none)"


def prompt(study=None) -> str:
    kinds = "\n".join(f"- {k}: {v}" for k, v in KINDS.items())
    return RUBRIC.format(regions=_regions(study), kinds=kinds)


def score(findings) -> dict:
    """Totals from a findings list (critic JSON 'findings')."""
    fs = [f for f in findings or [] if f.get("severity") in SEVERITY]
    by_kind = {}
    for f in fs:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    return {"criticScore": sum(SEVERITY[f["severity"]] for f in fs),
            "criticBlockers": sum(f["severity"] == "blocker" for f in fs),
            "criticMajors": sum(f["severity"] == "major" for f in fs),
            "criticFindings": len(fs), "criticByKind": by_kind}


def critique_anthropic(images, study=None, model=None, effort="medium") -> dict:
    import anthropic

    client = anthropic.Anthropic()
    model = model or os.environ.get("LOADPATH_CRITIC_MODEL", "claude-opus-5-5")
    content = []
    for p in images:
        with open(p, "rb") as fh:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.standard_b64encode(fh.read()).decode()}})
    content.append({"type": "text", "text": prompt(study)})
    resp = client.beta.messages.create(
        model=model, max_tokens=16000,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        # a declined request is re-run on the fallback model in the same call
        betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"},
        messages=[{"role": "user", "content": content}])
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"critic declined: {getattr(resp.stop_details, 'category', None)}")
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    out["model"] = resp.model
    return out


def request_agent(images, study, request_path, answer_path) -> str:
    """Write the critic request for an agent to answer by hand."""
    imgs = "\n".join(f"- {os.path.abspath(p)}" for p in images)
    with open(request_path, "w") as fh:
        fh.write(f"# Critic request\n\nLook at:\n{imgs}\n\n{prompt(study)}\n\n"
                 f"Write the answer as JSON to `{os.path.abspath(answer_path)}`, matching this schema:\n\n"
                 f"```json\n{json.dumps(SCHEMA, indent=1)}\n```\n")
    return request_path


def critique(images, study=None, backend="anthropic", out_dir=None, name="part") -> dict | None:
    """Findings + score for close-up images, or None when the agent backend
    has written a request that nobody has answered yet."""
    images = [p for p in images if p and os.path.exists(p)]
    if not images:
        return None
    if backend == "anthropic":
        res = critique_anthropic(images, study)
    elif backend == "agent":
        out_dir = out_dir or os.path.dirname(images[0])
        answer = os.path.join(out_dir, f"{name}.critic.json")
        if not os.path.exists(answer):
            request_agent(images, study, os.path.join(out_dir, f"{name}.critic-request.md"), answer)
            return None
        with open(answer) as fh:
            res = json.load(fh)
    else:
        raise ValueError(f"unknown critic backend {backend!r} (anthropic, agent)")
    res["images"] = [os.path.basename(p) for p in images]
    return res | score(res.get("findings"))


def fold_into_scorecard(out_dir) -> list:
    """Read every <case>.critic.json in an eval output folder into
    scorecard.json's rows (and their gates), rewrite scorecard.md."""
    from .evals.run import _md, gate_critic

    path = os.path.join(out_dir, "scorecard.json")
    with open(path) as fh:
        rows = json.load(fh)
    for r in rows:
        f = os.path.join(out_dir, f"{r['case']}.critic.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            res = json.load(fh)
        r["critic"] = res | score(res.get("findings"))
        gate_critic(r)
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(out_dir, "scorecard.md"), "w") as fh:
        fh.write(_md(rows))
    return rows


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="loadpath.critic")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="fold <case>.critic.json answers into an eval scorecard")
    s.add_argument("out")
    a = p.parse_args(argv)
    if a.cmd == "score":
        rows = fold_into_scorecard(a.out)
        print(open(os.path.join(a.out, "scorecard.md")).read())
        return sum(0 if r.get("pass") else 1 for r in rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
