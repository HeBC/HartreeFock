# Hybrid HF validation — 2026-10-08

The latest orbital-gap preconditioner passed **30/30 tests**, **40/40 random
starts plus four references**, and **9/9 distinct example scan points**.
The hardest saved O16 start fell from 863 to 71 iterations, with unchanged
tolerances; the random O16 range is now 32–117. See
[PRECONDITIONING.md](PRECONDITIONING.md) and
[preconditioner_validation.json](preconditioner_validation.json). The
earlier measurements below are historical results from before preconditioning.

The readable-input follow-up passed **26 tests** and converged all **9 points**
across four `.inp` examples, including a paired GCM path. The three original
examples reproduce the earlier JSON-run energies within 1e-10 MeV, and resume
does not duplicate attempts. See [input_validation.json](input_validation.json)
and the [input guide](INPUT_GUIDE.md). The original physics/performance benchmark
below is preserved with its original source hashes and 18-test count.

Validation used a clean source copy containing only tracked project files and
the newly included source, documentation and small benchmark inputs. Native
objects and executables were rebuilt with `make -j2 pyHFAndHFB.so HartreeFock.exe`.
No prebuilt objects, personal research output directories or external IMSRG build
were needed to run the tests/examples. The build completed with existing compiler
warnings (visibility and a boolean expression), without compilation/link errors.

Environment: Linux/WSL, Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3,
native MKL/GSL/MPI libraries, one BLAS/OpenMP thread. Timings are one local run,
exclude interaction loading for the same-seed benchmark, and are not universal
performance guarantees. Process RSS in the JSON is a process-wide high-water
mark, not the allocation of an individual solve.

## Regression and numerical correctness

**18 tests passed, zero skipped.** The suite checks analytic energy/constraint
derivatives and Hessians against finite differences; Hessian symmetry; pp, nn
and pn contractions; tensor pair normalization/exchange; an independent rank-2
matrix element; the analytic d5/2 stretched-state quadrupole; agreement with a
bare IMSRG mass quadrupole; reference-density shifts and NPZ round trips;
conservation and state preservation; memory rejection; saddle escape; multi-shell
operators and GCM export ordering.

## Same-seed convergence comparison

The baseline is the corrected projected-gradient solver in `hybrid_hf.py`,
not a historical Broyden or penalty solver. Both methods use the same initial
determinant, targets, tolerances and constrained-curvature acceptance check.
Maximum iterations: 1200. Targets are native oscillator moments:
USDA Mg24 (Q20,Q22sum,Q21real)=(1,0.5,0), USDB Mg24=(2,0.5,0),
O17 emax=3=(1,0.5,0). Q22sum is twice real Q22.

| Case | Method | Outcome | Iterations | Solve seconds | Fock + Hessian responses | Final gradient |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Mg24 / USDA | gradient | iteration_limit | 1200 | 2.141 | 2057 + 31 | 5.715e-05 |
| Mg24 / USDA | hybrid | accepted | 45 | 0.147 | 71 + 182 | 9.044e-07 |
| Mg24 / USDB | gradient | accepted | 325 | 0.650 | 549 + 62 | 7.780e-07 |
| Mg24 / USDB | hybrid | accepted | 38 | 0.145 | 59 + 152 | 1.428e-07 |
| O17 / emax=3 | gradient | iteration_limit | 1200 | 14.698 | 2536 + 0 | 4.225e-06 |
| O17 / emax=3 | hybrid | accepted | 17 | 1.125 | 26 + 151 | 2.696e-08 |

Every hybrid result passed the gradient, moment, energy-change and constrained
curvature checks. Gradient-only USDA and O17 results reached the iteration limit
and were not accepted, so their timings are times to failure, not converged-solve
speedups. The converged USDB comparison provides a direct timing comparison.
The methods can find different local branches; neither these cases nor a
curvature check prove a global minimum. Raw energies, errors, curvature and work
counts are included in [convergence.json](convergence.json).

## Named constraints, evolved operators and continuation

| Example | Accepted distinct points | Forward/reverse points completed | Selected iterations | Worst gradient | Worst moment error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `usda_mg24.json` | 2/2 | 4 | 14–63 | 3.995e-07 | 2.665e-15 |
| `evolved_ne20.json` | 4/4 | 8 | 9–18 | 2.464e-07 | 4.500e-12 |
| `multishell_o16.json` | 1/1 | 2 | 85–85 | 7.793e-07 | 5.437e-13 |

All seven points converged; all 14 directional points completed. Each example's
history contains Newton-CG steps, confirming that the hybrid path was exercised.
Resuming each completed scan left the attempt journal unchanged. The O16 example
has all six requested Q20, Q22, Q10, Q30, Jx and Jz targets nonzero, with Q21=0.
Only accepted determinants were exported to the legacy GCM basis format.

The evolved Ne20 example uses genuine induced two-body mass-quadrupole matrix
elements. Its small fixtures use emax=2, hw=16 MeV, Minnesota plus intrinsic
kinetic energy, and a short IMSRG(2) flow to s=0.2. They validate the interface;
they are not production Ge76/Se76 interactions or a completed IMSRG decoupling.

The bare quadrupole driver also converged **9/9 triaxial USDA Mg24 points** at
beta={0.04,0.08,0.12}, gamma={10,30,50} degrees. That tested grid is now the default
with the corrected standard quadrupole normalization. Both the explicit README
command and the default-grid command were exercised and gave the same minimum.

## Reproduce

From `MyHF/`, after building the native module:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tests/benchmark_hybrid.py
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --check
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --output Output/check_mg24
python3 PythonScript/scan_gcm_hf.py examples/gcm/evolved_ne20.inp --output Output/check_ne20
python3 PythonScript/scan_gcm_hf.py examples/gcm/multishell_o16.inp --output Output/check_o16
python3 PythonScript/scan_hybrid_pes.py --output Output/check_triaxial
```

Use fresh output directories, or `--resume` for identical named-driver inputs
and code. The read-only `--check` option does not need an output directory;
a real scan needs `output` in the file or a CLI `--output` override. Exact source hashes are stored in
`convergence.json`. The main README describes limits of the real-HF space, dense
Hamiltonian memory, operator normal ordering and the earlier beta-axis correction.

## Independent random starts

A subsequent test converged 40/40 fully randomized determinants at four selected
targets. Some starts found lower local minima than the default initialization;
the O16 iteration count originally ranged from 33 to 863 (now 32–117 with
preconditioning, as documented above). See
[RANDOM_STARTS.md](RANDOM_STARTS.md) for the procedure, per-case results and
commands to reproduce the experiment. The production initialization is unchanged.
