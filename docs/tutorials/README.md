# Tutorials

Six workflows end to end, each in its own folder with the input files, the
commands, the numbers you should get and renders of the result. Run the
commands from inside the tutorial's folder.

| tutorial | what you make | headline result (full grid) |
|---|---|---|
| [FDM bracket](fdm-bracket/) | PETG wall bracket: topology optimization to an oriented 3MF | 48.6 g, SF 5.4 against a goal of 3 |
| [Graded lattice with drains](sls-lattice-drains/) | PA12 SLS wrist link: stress-graded skin and gyroid, powder drain holes | 23% lighter than solid, 0 sealed voids after 4 drains |
| [Heat exchanger](heat-exchanger/) | AlSi10Mg LPBF oil cooler: sized, designed, simulated whole, then fixed to its pressure budget | 1.56 kW with oil at 3.9 kPa and water at 2.0 kPa against 5 kPa budgets (simulated); fluids isolated |
| [Impact bumper](impact-bumper/) | PETG graded gyroid that stops a 4 kg rover falling 1 m | 3.8 kN peak against a 4 kN limit; a uniform core of the same mass hits 4.6 kN |
| [Field editor](field-editor/) | PA12 link drawn in the browser editor, then tuned | 11% lighter after `tune`, still SF 3 and 0.8 mm |
| [Arm link from a URDF](urdf-arm-link/) | SLS forearm whose loads and keep-outs come from the robot's URDF | 66 g, SF 3.1; the hand-loaded design reads SF 0.89 under the derived cases |

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
