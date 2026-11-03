# Case studies

Each case study states the problem, the setup, the result against a named
baseline, and what the result does not show. All results are simulated;
none has been checked against a printed and tested part yet. See
[VALIDATION.md](../VALIDATION.md) for what each model has been checked
against.

| Case study | Result | Baseline | Evidence |
| --- | --- | --- | --- |
| [Eval suite](evals.md) | 15 of 15 parts meet their requirements on a fresh analysis of the delivered mesh | Each part's own requirements | Simulation, solver cross-checked |
| [GRIP robot arm](grip.md) | Load sweep found 3 of 5 committed links below SF 1 under loads the arm can apply; regrown set meets SF 2 and is 18% smaller | GRIP's committed v0.3 parts (made by an earlier loadpath) | Simulation |
| [Arm link cores](arm-link-cores.md) | 42.2 g FDM box section and 36.8 g SLS hollow link, each meeting 0.5 mm and SF 3 | 54.9 g solid web, 54.2 g free-form topology optimization | Simulation |
| [Liquid cold plate](cold-plate.md) | Die-graded gyroid ties streamline channels; channels with a graded gyroid inside are 7% cooler | Streamline channels and pin fins at equal metal and pressure drop | Simulation, grid-checked |
| [TPU finger](tpu-finger.md) | up to 8 mm of tip curl for a 5 mm pull; tendon finger grips 8x harder | A linear-model design (2.5 mm, then the solve fails); a plain bar of equal volume | Simulation |
| [Benchmarks](benchmarks.md) | 1-5% less stiff than top88/top3d at equal material, in 57-60 iterations instead of 130-410 | Ports of top88 and top3d | Benchmarked |
