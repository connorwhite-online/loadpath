"""
A handheld's cables in the layout search: the sensor puck of
examples/enclosure, now with the two leads that join its parts.

The battery comes with a pigtail (two 26 AWG silicone wires to a plug on
the board) and the haptic motor with its two fine leads. Each component
says where its connector is and which way the cable leaves it (`ports`),
and the brief says what joins what (`wires`). The placement search then
lays every cable, as an elastic rod, for each position it considers: a
position whose cable would be too short, or bent past its radius, is not a
position. The chosen layout's cables are followed as each part goes in
along its assembly path (can the plug be mated with the part still in the
hand?), and the space each cable needs is kept clear of the interior that
is designed next.

    python examples/wires/handheld.py [--out handheld-out] [--lead 45]
    python examples/wires/handheld.py --compare    # free, a 45 mm pigtail, a 14 mm one

--lead: the battery pigtail's length (mm); without it the cable is cut to
the layout. --compare shows the layout move as the pigtail gets shorter,
and the search saying so when no layout can be wired.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "enclosure"))


def write(out, lead=None, hot=True):
    import sensor_puck
    path = sensor_puck.write(out)
    brief = json.load(open(path))
    by = {c["id"]: c for c in brief["components"]}
    # where each connector is, in the component's own frame, and which way its cable leaves
    by["board"]["ports"] = [{"id": "batt", "at": [17.0, 9.0, 1.8], "direction": "+x"},
                            {"id": "haptic", "at": [-15.0, 9.0, 0.8], "direction": "+z"}]
    by["battery"]["ports"] = [{"id": "lead", "at": [22.5, 9.0, 0.0], "direction": "+x", "relief": 2.0}]
    by["motor"]["ports"] = [{"id": "leads", "at": [5.0, 0.0, 0.0], "direction": "+x"}]
    by["motor"]["rotate"] = "z"          # its leads can point any way round
    if hot:
        # a charger on the board, and a cell that should stay under 45 C
        by["board"]["watts"] = 1.2
        by["battery"]["maxTemp"] = 45.0
    batt = {"id": "battery-lead", "from": "battery.lead", "to": "board.batt",
            "cable": {"type": "bundle", "wires": 2, "awg": 26, "insulation": "silicone",
                      "name": "2 x 26 AWG silicone"}, "current": 1.0}
    if lead:
        batt["length"] = float(lead)
    brief["wires"] = [batt,
                      {"id": "motor-leads", "from": "motor.leads", "to": "board.haptic",
                       "cable": {"type": "bundle", "wires": 2, "awg": 30, "insulation": "pvc",
                                 "name": "2 x 30 AWG"}}]
    brief["name"] = "handheld"
    path = os.path.join(out, "handheld.brief.json")
    json.dump(brief, open(path, "w"), indent=1)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="handheld-out")
    ap.add_argument("--lead", type=float)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    runs = {"free": None, "pigtail-45": 45.0, "pigtail-14": 14.0} if a.compare else {"": a.lead}
    rows = []
    for name, lead in runs.items():
        out = os.path.join(a.out, name) if name else a.out
        brief = write(out, lead)
        rc = subprocess.call([sys.executable, "-m", "loadpath", "enclosure", brief, "--out",
                              os.path.join(out, "enclosure")])
        rep = os.path.join(out, "enclosure", "enclosure.json")
        if os.path.exists(rep):
            r = json.load(open(rep))
            for wid, w in (r.get("wires") or {}).items():
                rows.append((name or "run", wid, w.get("length"), w.get("minBendRadius"), w.get("allowedBendRadius"),
                             "ok" if w.get("ok") else "; ".join(w.get("problems", []))))
            for w in r.get("warnings") or []:
                if "no place found" in w:
                    rows.append((name or "run", w.split(":")[0], None, None, None, w.split(":", 1)[1].strip()))
    print(f"{'run':11} {'cable':13} {'length':>7} {'bend':>6} {'allowed':>8}  verdict")
    for n, w, L, b, al, v in rows:
        print(f"{n:11} {w:13} {L or 0:7.1f} {b or 0:6.1f} {al or 0:8.1f}  {v}")
