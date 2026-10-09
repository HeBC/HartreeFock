# Random-start hybrid HF convergence — 2026-10-08

**Update:** the default solver now uses an orbital-gap preconditioner.
Replaying these exact starting states converged all 40 random starts again;
the O16 range improved from 33–863 to **32–117** iterations and its hardest
start from 863 to **71**. See [PRECONDITIONING.md](PRECONDITIONING.md).
The original results below are preserved; reproduction commands now use
the improved default. Set `precondition = no` for the earlier CG method.

**All 40 independently randomized starting determinants converged**, together
with all four deterministic reference starts, at the selected targets below.
The solver and production scan defaults were unchanged for this experiment.

## What was randomized

For each seed 1001–1010, independent Gaussian matrices were drawn for the proton
and neutron occupied orbitals and orthonormalized with QR. These are full random
occupied subspaces, not small perturbations of the usual determinant. Particle
numbers, orthogonality and density idempotency were checked before solving.
Saved density projectors were checked to be distinct from one another and far
from the reference state. The method operates in the same real, charge-conserving
HF space as the normal calculations.

Every solve started independently, with no continuation or best-state reuse.
The hybrid solver first restored the imposed moments and then optimized the
energy. Its internal rescue/Lanczos seed was held fixed at 520; only the initial
Slater determinant changed. Convergence required gradient <=1e-6, moment error
<=1e-8, energy change <=1e-8 MeV, and a constrained-curvature check. All accepted
results passed these checks and exercised Newton-CG steps.

## Results

| Case | Converged random starts | Reference iterations | Random iterations | Median | Worst gradient | Worst moment error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| USDA Mg24 | 10/10 | 63 | 19–52 | 29 | 4.610e-07 | 2.230e-12 |
| Ne20, evolved 1b+2b Q | 10/10 | 20 | 23–35 | 27.5 | 5.479e-07 | 8.177e-13 |
| O16, six nonzero moments | 10/10 | 85 | 33–863 | 245.5 | 9.933e-07 | 7.440e-12 |
| O17, emax=3 | 10/10 | 17 | 19–25 | 21 | 7.579e-07 | 6.706e-12 |

The iteration caps were 700 for Mg24/Ne20, 1000 for O16 and 1200 for O17.
The O16 runs show that a fully random start can be substantially slower than
the default start even when it eventually converges.

These are **one selected target per case**, not random restarts over every point
of a complete PES. Mg24 and Ne20 use the first point of their `.inp` examples:
Q20=1 fm², Q22sum=0.5 fm², and Q21=Q10=Q30=Jx=Jz=0. O16 uses its six nonzero
moment example, with Q21=0. O17 repeats the older emax=3 regression target:
Q20=1, Q22sum=0.5, Q21=0 in oscillator units, with Jx/Jz unconstrained.
The small Ne20/O16 Minnesota/IMSRG inputs remain short-flow interface benchmarks.

## Different local minima

The values below are unshifted valence energies, in MeV. Adding a common SNT
zero-body constant does not change any energy difference.

* **Mg24:** the lowest random-start energy agrees with the reference,
  -72.375980280841 MeV. Other accepted starts ended near -72.356425117721 MeV
  and -69.429815223335 MeV. Convergence alone does not select the lowest branch.
* **Ne20:** the reference gave -76.411125252228 MeV; random starts reached
  -77.255887406164 MeV, **0.844762153936 MeV lower**, as well as several
  intermediate local minima.
* **O16:** the reference gave -186.444523701169 MeV; the lowest sampled result
  was -186.532698805623 MeV, **0.088175104454 MeV lower**. Several closely
  spaced lower branches required hundreds of iterations.
* **O17:** all random starts agree with the reference energy to about
  5e-12 MeV at this target.

For a production PES, compare multiple accepted random starts and continuation
states at each target and retain the lowest energy found. This experiment
demonstrates why that comparison matters; neither ten starts nor a local Hessian
check establishes the global minimum. Previous production/example PES files
were not overwritten by this experiment.

## Reproduce

From `MyHF/`, choose a new benchmark output directory:

```sh
python3 tests/benchmark_random_starts.py --input examples/gcm/usda_mg24.inp --output Output/random_mg24
python3 tests/benchmark_random_starts.py --input examples/gcm/evolved_ne20.inp --output Output/random_ne20
python3 tests/benchmark_random_starts.py --input examples/gcm/multishell_o16.inp --output Output/random_o16
python3 tests/benchmark_random_starts.py --input examples/gcm/o17_random.inp --output Output/random_o17
```

The defaults test target index 0 with seeds 1001–1010. Use `--point 1` for another
target in the input grid or `--seeds 41 42 43` for another fixed seed list. Each
benchmark also runs the ordinary deterministic reference. The output contains
starting/final orbitals, iteration histories, source hashes, a JSONL result
journal and `summary.json`. Exit status 2 reports an unconverged reference or
random start; failed results remain in the journal and summary.

Detailed original results, targets, solver options and source hashes are in
[random_start_validation.json](random_start_validation.json).
