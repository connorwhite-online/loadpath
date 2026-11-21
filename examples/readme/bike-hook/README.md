# Bike hook (README quick start)

The study an agent wrote for "I need a hook to hang my bike on the garage wall."
The agent's inferred loads, fit and mount are in the study's `description`.

```bash
loadpath run bike-hook.json --out out
python render.py          # out/bike-hook-30.stl -> bike-hook.png
```

Result on 2026-11-21: PA 12 (MJF), 293 g, safety factor 2.8 under the lift-on
load, durability factor 1.55 for ten years of daily lifting. PETG was tried
first and failed on layer-bond fatigue next to the screws.
