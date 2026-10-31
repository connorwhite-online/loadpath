# Tutorials

Six workflows end to end, each in its own folder with the input files, the
commands, the numbers you should get and renders of the result. Run the
commands from inside the tutorial's folder.

| tutorial | what you make | process |
|---|---|---|
| [FDM bracket](fdm-bracket/) | topology-optimized wall bracket, oriented and sliced for printing | PETG, FDM |
| [Graded lattice with drains](sls-lattice-drains/) | stress-graded gyroid part with powder drain holes | PA12, SLS |
| [Heat exchanger](heat-exchanger/) | two-fluid gyroid exchanger, sized, designed and simulated whole | |
| [Impact bumper](impact-bumper/) | graded gyroid that stops a falling robot under a force limit | PA12, SLS |
| [Field editor](field-editor/) | an implicit part designed in the browser editor, then tuned | |
| [Arm link from a URDF](urdf-arm-link/) | robot link whose loads and keep-outs come from the robot's URDF | |

## Quick mode

Every command takes `--quick` (or set `LOADPATH_QUICK=1`): a coarse grid and
fewer iterations, so a tutorial runs in a few minutes instead of tens. Masses
and safety factors move by a few to tens of percent; use it to check a setup,
then run without it for the part you print.

## The tutorials are tests

CI runs every command in each tutorial's ```` ```bash ```` blocks in quick
mode and holds the outputs to the folder's `expect.json`
([tests/test_tutorials.py](../../tests/test_tutorials.py)), so a change that
breaks a tutorial fails the build. To run them yourself:

```text
python tests/test_tutorials.py                   # all, quick (a few minutes each)
python tests/test_tutorials.py fdm-bracket       # one
python tests/test_tutorials.py --full fdm-bracket   # full grid: the numbers the README quotes
```

Commands shown in ```` ```sh ```` blocks (a browser session, a long sweep) are
not run by the tests. Each folder's `render.py` made its images; it is not
part of the tutorial.
