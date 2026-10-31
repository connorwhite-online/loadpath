#!/bin/bash
# The CI test steps (.github/workflows) plus the suites the element touches,
# one at a time. Usage: fast_tests.sh PYTHON
PY=${1:-python}
LOG=${2:-.}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/../.."
for t in tests/run_tests.py tests/test_loadpath.py tests/test_nonlinear.py tests/test_interface.py tests/test_dfm.py \
         tests/test_voxels.py tests/test_pressure.py tests/test_durability.py tests/test_calibration.py tests/test_impact.py; do
  s=$(date +%s)
  $PY "$t" > "$LOG/fast_$(basename "$t").log" 2>&1
  echo "$t exit=$? $(( $(date +%s) - s ))s"
done
s=$(date +%s)
$PY -m pytest -q tests/test_materials.py tests/test_interfaces.py tests/test_multimat.py tests/test_threemf_multi.py \
  tests/test_voxelprint.py tests/test_field_materials.py tests/test_hex8_im.py tests/test_mg.py tests/test_thermal.py \
  tests/test_struts.py tests/test_buildsim.py > "$LOG/fast_pytest.log" 2>&1
echo "pytest exit=$? $(( $(date +%s) - s ))s"
