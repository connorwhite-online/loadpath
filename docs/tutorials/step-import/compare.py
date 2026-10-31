"""
What study.json changes in the draft `loadpath import` wrote, key by key:

    python compare.py draft/elbow.study.json study.json

Prints the keys that differ and writes them to compare.json. Exits 1 if
anything but the description and the load cases differs: the draft's
geometry (design space, preserved interfaces, keep-outs, support) is
taken as imported, so if a new loadpath drafts it differently, the
tutorial says so instead of running a stale study.
"""
import json
import os
import sys

EDITED = {"description", "loadCases"}

draft_path, final_path = sys.argv[1], sys.argv[2]
with open(draft_path) as fh:
    draft = json.load(fh)
with open(final_path) as fh:
    final = json.load(fh)

# the draft's mesh paths are relative to draft/, study.json's to this folder
rel = os.path.relpath(os.path.dirname(os.path.abspath(draft_path)), os.path.dirname(os.path.abspath(final_path)))


def rebase(x):
    if isinstance(x, dict):
        if x.get("type") == "mesh" and "path" in x:
            return {**x, "path": os.path.normpath(os.path.join(rel, x["path"]))}
        return {k: rebase(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rebase(v) for v in x]
    return x


def same(a, b):
    """Equal, numbers to 1e-3 mm (another OpenCascade may round differently)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-3
    return a == b


draft = rebase(draft)
keys = sorted(set(draft) | set(final))
changed = [k for k in keys if not same(draft.get(k), final.get(k))]
out = {"changed": changed, "same": [k for k in keys if k not in changed],
       "unexpected": [k for k in changed if k not in EDITED]}
with open("compare.json", "w") as fh:
    json.dump(out, fh, indent=1)
for k in changed:
    print(f"{k}:\n  draft: {json.dumps(draft.get(k))[:200]}\n  study: {json.dumps(final.get(k))[:200]}")
if out["unexpected"]:
    print(f"the draft differs from study.json in {out['unexpected']}: re-derive study.json from the draft")
    sys.exit(1)
