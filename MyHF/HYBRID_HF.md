# Constrained hybrid HF and triaxial PES scans

The new production entry point is `PythonScript/scan_hybrid_pes.py`. It uses the
existing C++ Hamiltonian through a checked NumPy interface in `pyHFAndHFB.so`.
`PythonScript/hybrid_hf.py` also provides `Solver` and `Options` for other scripts.
The default case is **USDA Mg24**, beta = 0.04, 0.08, 0.12 and gamma = 10, 30, 50 degrees, a
nine-point grid tested with the corrected quadrupole normalization. Settings at the top of the scan script are editable.

## Build and run

Use Linux or WSL with the existing MPI compiler, MKL, GSL, Python development
headers, and bundled pybind11 headers. Python requires NumPy and SciPy.

```sh
cd '/mnt/d/Code/HF and HFB/MyHF'
make -j2 pyHFAndHFB.so HartreeFock.exe
python3 PythonScript/scan_hybrid_pes.py --check
python3 PythonScript/scan_hybrid_pes.py --output Output/mg24_triaxial
```

Output directories must be new, preventing accidental overwrites. To scan native
quadrupole targets or change the beta/gamma grid:

```sh
python3 PythonScript/scan_hybrid_pes.py --native-grid --q0 1 2 3 --q2 .25 .5 1 --output Output/mg24_q
python3 PythonScript/scan_hybrid_pes.py --beta .2 .3 .4 --gamma 10 30 50 --threads 2 --memory-mb 2048 --output Output/mg24_custom
```

Each run writes `settings.json`, `grid.csv`, a durable `points.jsonl` journal,
`surface.csv`, `summary.json`, and per-point iteration histories. Failed points
keep their diagnostics but have blank accepted and relative energies. Exit status
2 means at least one point failed. Interruptions can be recovered without rerunning
the solver; this regenerates summaries, not wavefunctions or unfinished points:

```sh
python3 PythonScript/scan_hybrid_pes.py --recover-surface Output/mg24_triaxial
```

The Hamiltonian is loaded once. A serpentine grid uses only the last *converged*
state as the next seed. A failed point is retried from the initial state, and never
replaces a good continuation state. Wavefunctions are kept only in memory.
Run one process: MPI process replication is rejected; `--threads` controls BLAS
and OpenMP. This is not a distributed-memory solver.

## Algorithm and acceptance

1. Restore the requested moments using the **joint proton-neutron** Jacobian and
   a rank-revealing SVD. Empty/full species and dependent constraints are allowed.
2. Try a few safeguarded diagonalizations, then projected gradient steps with
   spectral step-length estimates.
3. Use truncated Newton CG in the feasible tangent space, with a positive
   semicanonical orbital-gap preconditioner adapted from CC `hf_real`. Occupied
   and virtual constrained-Fock blocks supply the gaps; the exact Hessian
   response stays in the CG action. Negative curvature
   truncates the step at a radius bound; backtracking and nonlinear constraint
   restoration safeguard the energy descent. This is not a verbatim TRAH/Davidson
   implementation from the paper.
4. Require a small projected gradient, small energy change, and satisfied moments.
   Then estimate the lowest constrained Hessian curvature with matrix-free
   Lanczos. Follow detected negative curvature and continue optimizing before
   accepting the point. Failure of this check is reported as failure.

For occupied orbitals C, rho = C C^T, so particle number and idempotency are
preserved by construction. Polar/SVD retraction preserves orbital orthogonality.
All fields remain in the original oscillator basis. The physical energy contains
no constraint penalty. The reported gradient norm is the Euclidean/Frobenius norm
of the constrained orbital gradient; it includes the factor of two appropriate
to real occupied orbitals.

For L = E + sum(lambda_k Q_k), the covariant Hessian action is

```
delta_rho = V C^T + C V^T
H_L[V] = 2 P_C { Gamma[delta_rho] C + F_L V - V (C^T F_L C) }
```

This is projected onto the moment-constraint tangent space on both sides. The
interaction response Gamma is linear for this two-body Hamiltonian. The CG and
Lanczos steps therefore use the exact analytic Hessian action, without storing
the particle-hole Hessian. `hessian_evaluations` counts these interaction responses
separately from `fock_evaluations`; compare their sum when assessing work.

Preconditioning defaults to `precondition = yes`, with a positive gap floor
`precondition_floor = 0.1` MeV and `max_cg = 35`. The residual stopping test
uses its unpreconditioned norm. `cg_iterations`, `cg_limit_hits` and
`preconditioner_evaluations` expose the inner work. The additional storage
is O(dp² + dn²). Set `precondition = no` only for an unpreconditioned
comparison. See [the implementation and measured acceleration](docs/PRECONDITIONING.md):
the hardest saved O16 start dropped from 863 to 71 iterations at unchanged
tolerances, with all 40 random starts and 30 regression tests passing.

Defaults: gradient 1e-6, moment error 1e-8, energy change 1e-8 MeV, negative
curvature threshold -1e-5. Internal moment restoration is tighter (up to 1e-11).
Stability is a numerical local check in the **real HF manifold and active
constraints**; it is not proof of a global minimum or stability against complex
orbital variations. Comparing different seeds and scan directions remains useful.

## Units and triaxiality

Native operators are Q0 = r^2 Y20 / b^2 and Q2 = r^2 (Y22 + Y2,-2) / b^2.
The second is **twice real Q22**, not the single Q22 used by the CC scan.
`b^2 = 41.47106 / hw` fm^2. With R = 1.2 A^(1/3) fm:

```
scale = 3 A R^2 / (4 pi b^2)
target_Q0 = scale beta cos(gamma)
target_Q2 = sqrt(2) scale beta sin(gamma)
```

A is the full isotope mass. Moments are computed in the active space; an inert
spherical core contributes no quadrupole moment. This convention is explicitly
saved in `settings.json` and need not match the legacy shape-printing routine.
Real Q21 = 0 fixes the remaining real principal-axis orientation by default;
`--free-axes` omits it. `--jx` and `--jz` are direct expectation values in hbar,
not the legacy J(J+1) input used for Jx. A semiclassical spin label J can
be mapped to a target sqrt(J(J+1)); pass that numerical value directly.
The constraint does not make the intrinsic determinant an exact spin-J state.

## Correctness and memory changes

* The old constrained routines used separate proton and neutron response solves
  for a total moment, applying the same full target error to both species. The
  new solver uses a joint Jacobian. The legacy experimental
  `Solve_hybrid_Constraint`, `Solve_gradient_Constraint` and finite-penalty
  `Solve_broyden_Constraint` are retained; use the new Python solver for this PES
  workflow. Their convergence messages are not the acceptance test used here.
* Density construction no longer launches all OpenMP threads writing the same
  temporary arrays. It needs one occupied-orbital buffer per species, not two.
* The proton-neutron contraction uses MKL GEMV and transpose GEMV, eliminating
  strided neutron dot products without a second stored interaction tensor.
* Multi-shell quadrupole radial matrix elements now use Gauss-Laguerre
  integration instead of a single-shell-only formula. The old generic radial
  routine also had shadowed, uninitialized quantum numbers and hardcoded proton
  indices in neutron callers. The radial integration preserves sd-shell radial values; the separate
  angular normalization correction below changes the old Q2 matrices by 1/sqrt(5).
* Dense interaction sizes use checked size arithmetic, fail before unsupported
  legacy indexing or budget overflow, and check allocation failure. Previously
  uninitialized owning pointers are null-initialized for safe error cleanup.
* Python objects keep their ModelSpace/Hamiltonian owners alive. Arrays crossing
  the boundary own their memory and are checked for shape, symmetry and finiteness.
* The Makefile no longer depends on a missing Broyden constraint source, links
  libraries after objects, and rebuilds objects when native headers change.

`--memory-mb` is an **admission budget**, not an operating-system RSS limit. The
driver reserves 64 MiB plus a conservative O(d_p^2+d_n^2) solver allowance, and the
native allocator enforces the remaining dense-interaction budget. The unchanged
interaction representation still costs
`8 * (d_p^4 + d_n^4 + d_p^2*d_n^2)` bytes. Parsed interaction metadata, Python,
MPI/BLAS libraries and allocator caches can add process memory. Very large spaces
still require a sparse/block-distributed interaction backend. No dense Hessian or
unbounded Broyden history is stored.

## Validation and references

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tests/benchmark_hybrid.py
```

Tests compare the native contraction to the original scalar/dot-product layout,
energy gradients, moment Jacobians and Hessian actions to finite differences,
Hessian symmetry, feasible continuation, failed-solve state preservation, empty
and full species, memory rejection, saddle escape, invalid isotopes, grid units,
and recovery of interrupted journals. The benchmark compares the new hybrid with
the same corrected projected-gradient solver, not the old penalty Broyden solver.
Timing is machine-dependent; peak RSS is process-wide high water.

Design references: the CC code's `hf_hybrid_bridge.f90` and `scan_hybrid_pes.py`;
[Baran et al., PRC 78, 014318](https://arxiv.org/abs/0805.4446);
[Helmich-Paris, JCP 154, 164104](https://arxiv.org/abs/2012.08306);
Yamaguchi et al., *Chemical Physics* 147 (1990), 309-326 (orbital-Hessian stability).
The supplied [PRC 95, 064307](https://doi.org/10.1103/PhysRevC.95.064307) concerns
the Hessian of a **projected** VAP energy. This solver optimizes unprojected HF,
so that projected-energy Hessian is not interchangeable with the action above.


## Ge76 and Se76: IMSRG2 and IMSRG3f2

The Ge76/Se76 workflow is installed in `PythonScript/run_ge76_se76_pes.py`.
Both supplied SNT files and the validated calculation outputs are in
`Output/ge76_se76_pes/outputs`; iteration logs and orbital checkpoints are
in `Output/ge76_se76_pes/work` in the original research workspace. These
external production inputs and outputs are not included in a fresh clone.
The IMSRG2 results and checkpoints occupy a
separate `IMSRG2` subdirectory. Original research inputs remain unchanged.

From MyHF in WSL, run each requested nucleus/method combination:

```sh
python3 PythonScript/run_ge76_se76_pes.py Ge76 --method IMSRG2
python3 PythonScript/run_ge76_se76_pes.py Se76 --method IMSRG2
python3 PythonScript/run_ge76_se76_pes.py Ge76 --method IMSRG3f2
python3 PythonScript/run_ge76_se76_pes.py Se76 --method IMSRG3f2
python3 PythonScript/validate_ge76_se76_pes.py
python3 PythonScript/plot_ge76_se76_pes.py
```

The driver runs both scan directions by default and keeps the lower accepted
energy. Use `--pass-name forward` or `--pass-name reverse` for a single pass.
The fixed grid is beta=0..0.16 by 0.01 and gamma=0..60 degrees by 5, with one
spherical point: 209 points per nucleus and method. The input hash and isotope
are checked before reusing results. A previously accepted point is preserved if
a later solve fails. Repeating a pass recalculates it using saved states; it does
not simply skip existing points. Calls use one BLAS/OpenMP thread and a default
512 MiB interaction-plus-solver admission budget (`--memory-mb`), with the same
RSS limitations described above. MPI process replication is rejected.

The user-specified title is **1.8/2.0 (EM), emax=12**, with hw=12 MeV, jj44,
and the IMSRG method. Both SNT headers record NN-only N3LO EM500, SRG1.8,
and `input 3N: none`; the requested display label is recorded separately from
those unchanged source headers. Each Ge76-derived interaction is used for both
Ge76 and Se76. Absolute energies include each SNT zero-body term exactly once;
CSV files retain the unshifted valence energy.

`plot_ge76_se76_pes.py` produces individual and side-by-side vector PDFs/PNGs.
Default plotting uses a common color scale for all four surfaces and writes a
separate IMSRG2 PDF under `outputs/IMSRG2/Ge76_Se76_PES.pdf`; IMSRG3f2 is at
`outputs/Ge76_Se76_PES.pdf`. Labels mark sampled-grid minima. Invalid vertices
and all adjacent triangles are masked. Use `--method IMSRG2` or `--method
IMSRG3f2` to plot only one method. Other result locations are supported through
`--job-root` in the runner/validator and `--output-dir` in the plotter.

Numerical validation checks both complete passes, lower-branch selection,
particle counts, energy offsets, gradient, energy change, quadrupole constraints,
orthogonality, idempotency and constrained curvature. Details and sampled minima
are saved in `outputs/comparison_validation.json` and `outputs/README.txt`.


## Evolved operators and normalization update (2026-10-08)

See [EVOLVED_GCM.md](EVOLVED_GCM.md) for the named GCM interface, tensor-SNT
1b+2b operators, normal-ordering conventions, examples and tests.
The bare Q2 matrix elements now use standard spherical-harmonic normalization:
the older implementation was larger by sqrt(5). Old PES files are preserved;
their standard beta values are beta_old/sqrt(5), with unchanged energies/gamma.
New scans use the corrected operator, and old checkpoints are rejected when
the solver/native-module hashes differ.
