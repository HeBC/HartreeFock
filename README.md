# MyHF: nuclear Hartree–Fock

MyHF is a C++/Python nuclear-structure code for real, unrestricted Hartree–Fock
(HF) calculations with proton and neutron orbitals. It reads spherical
J-coupled shell-model interactions in SNT format, builds an m-scheme Hamiltonian,
and optimizes occupied orbitals at fixed proton and neutron numbers. It supports
valence-space and multi-shell calculations within the available memory.

The constrained Python workflow generates triaxial potential-energy surfaces
(PES) and Slater determinants for subsequent angular-momentum-projected
generator-coordinate-method (GCM) calculations. It supports bare multipoles and
IMSRG-evolved operators with one- and two-body terms. The repository also contains
legacy HF, projection/GCM, and experimental HFB sources; the documented hybrid
workflow is real HF, with no anomalous pairing density.

## Acknowledgments

**This code has benefited greatly from Ragnar Stroberg's
[IMSRG code](https://github.com/ragnarstroberg/imsrg) and from Gaute and Thomas's
coupled-cluster (CC) code.** Their codes have provided valuable foundations and
references for this implementation. The IMSRG operator interface and benchmark
generation use conventions from Ragnar's code; the hybrid HF and PES workflow
also benefited from the HF implementation and scan workflow in the CC code.
Please acknowledge these contributions when describing work based on MyHF and
cite the relevant upstream methods/software when using them.

## What the hybrid solver does

The default `method: hybrid` combines safeguarded diagonalization, projected
gradient steps, and matrix-free truncated Newton conjugate gradients with a
positive orbital-gap preconditioner adapted from the CC `hf_real` code. It restores
all requested moments with a joint proton–neutron Jacobian and checks constrained
Hessian curvature before accepting a stationary state. Negative curvature can
trigger further optimization to escape a saddle. Two-body constraint operators
contribute to the expectation value, density-dependent field, and Hessian action.

The physical energy excludes constraint penalties. Occupied-orbital rotations
preserve particle numbers and density idempotency. Default acceptance tolerances
are a projected gradient below `1e-6`, constraint errors below `1e-8`, energy
change below `1e-8 MeV`, and no detected constrained curvature below `-1e-5`.
This is a numerical local stability check in the real HF manifold, not a proof
of the global minimum. Use different seeds and scan directions to compare branches.

## Build on Linux or WSL

Required native dependencies are a C++ compiler with OpenMP, an MPI C++ wrapper
(`mpicxx`), Intel MKL, GSL, GNU Make, and Python development headers.
The pybind11 headers are bundled. The verified Python environment is **Python
3.10.12, NumPy 1.26.4, SciPy 1.15.3**. The requirements file pins the tested NumPy
and SciPy versions, matching the bundled pybind11 headers.

From a clone of this repository:

```sh
cd MyHF
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
make -j2 pyHFAndHFB.so HartreeFock.exe
python3 -c 'import pyHFAndHFB; print("MyHF native module loaded")'
```

The default makefile expects MKL headers in `/usr/include/mkl` and libraries on
the linker search path. If your MKL installation differs, set the include/library
paths in `makefile` or override `CFLAGS`/`LIBS`. For example, after configuring
your Intel environment:

```sh
make -j2 pyHFAndHFB.so HartreeFock.exe \
  CFLAGS="-O3 -fPIC -fopenmp -I$MKLROOT/include" \
  LIBS="-L$MKLROOT/lib/intel64 -Wl,-rpath,$MKLROOT/lib/intel64 -lmkl_rt -lm -lgsl -lgslcblas"
```

Use a Python interpreter matching the development headers used to build the
extension. The `.so` must be rebuilt on the target machine. Optional PDF/PNG PES
plotting scripts require `matplotlib`. The legacy projection executable can be
built with `make HF_Projection.exe`; its input and basis directory are configured
separately in the projection workflow.

All commands below run from **`MyHF/`** in a single process. Do not use `mpirun`
for the Python HF scans: it would replicate the interaction in memory.

## Quick start: triaxial USDA Mg24

The small named-constraint example uses nonzero Q20 and Q22 and the hybrid solver:

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --check
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --resume
```

The first solve needs a new output directory. `--resume` requires identical
configuration, operator/interaction files, and solver/native-module hashes.
Forward and reverse passes run by default, retaining the lower accepted energy.
Use `--passes forward` for one direction. A failed point is reported rather than
being silently included in the accepted PES or GCM basis.

For a bare mass-quadrupole beta/gamma scan, the separate grid driver provides:

```sh
python3 PythonScript/scan_hybrid_pes.py --beta .04 .08 .12 --gamma 10 30 50 --threads 1 --memory-mb 512 --check
python3 PythonScript/scan_hybrid_pes.py --beta .04 .08 .12 --gamma 10 30 50 --threads 1 --memory-mb 512 --output Output/mg24_beta_gamma
```

This driver defaults to USDA Mg24; use `--help` to change the nucleus, interaction,
oscillator frequency, or native quadrupole grid. It keeps continuation states in
memory. Use the named-constraint driver when you need saved orbitals/GCM exports.

## Configure multipole and cranking constraints

Use a commented plain-text **`.inp`** file. For example,
[`examples/gcm/usda_mg24.inp`](MyHF/examples/gcm/usda_mg24.inp) contains:

```ini
[calculation]
nucleus     = Mg24
interaction = ../../Interaction/usda.snt
hw          = 16                   # MeV
basis       = HO
memory_mb   = 512
output      = ../../Output/mg24_gcm
passes      = forward reverse

[constraints]
Q20 = 1.0 1.5                     # fm^2: two values to scan
Q22 = 0.5                         # fm^2: fixed value
Q21 = 0                           # fix orientation
Q10 = 0
Q30 = 0
Jx  = 0                           # <Jx>/hbar
Jz  = 0

[solver]
method         = hybrid
max_iterations = 700
```

One value fixes a moment. Several values scan it. Ranges use
`start:stop:step`, e.g. `Q20 = 1.0:2.0:0.5`. Set `Jx = off` to leave it
unconstrained; this differs from `Jx = 0`. Multiple lists form a Cartesian grid.
For paired targets, use the simple `[path]` table illustrated in
[`mg24_path.inp`](MyHF/examples/gcm/mg24_path.inp).

Paths inside the file are relative to its directory. Windows paths with spaces
are accepted under WSL. The `output` setting makes the filename sufficient to
run; `--output` overrides it. `--check` prints a readable target table without
writing results. See the [input guide](MyHF/docs/INPUT_GUIDE.md) for all settings,
operator files, comments, paths and examples. Legacy JSON inputs still work.

| Name | Quantity constrained | Default units |
| --- | --- | --- |
| `Q20` | sum of r²Y20 over active nucleons | fm² |
| `Q22` | sum of r²(Y22 + Y2,-2), twice real Q22 | fm² |
| `Q10` | sum of rY10; usually set to zero to control displacement | fm |
| `Q30` | sum of r³Y30, axial octupole | fm³ |
| `Jx`, `Jz` | direct component expectations, ⟨Jx⟩/ℏ and ⟨Jz⟩/ℏ | dimensionless |
| `Q21` | real Q21 = (Q21 − Q2,-1)/2; often set to zero to fix orientation | fm² |
| `Q40`, `Q32`, `R2` | additional shape/octupole coordinates and sum of r² | fm⁴, fm³, fm² |

Bare multipoles use proton/neutron weights `(1,1)`: they are **mass** operators.
To constrain only protons or neutrons, add an `[operator NAME]` section with
`name = Q20` (or another built-in) and `weights = 1 0` or `weights = 0 1`.
Use `units = oscillator` in that section for bare radial moments in oscillator units.
The full operator conventions and a separate-species example are in
[EVOLVED_GCM.md](MyHF/EVOLVED_GCM.md).

For semiclassical cranking, a spin label J can be mapped to a target
`Jx = sqrt(J*(J+1))`; enter the numerical value directly (e.g. `2.449489743` for
J=2). This mapping is optional: the intrinsic expectation does not fix an exact
total spin. Angular-momentum projection supplies the final GCM spin quantum number.
Neither the operator nor its output is silently converted to J(J+1).

Odd multipoles can vanish in a restricted space: Q10 and Q30 vanish in the pure
sd shell, and Q10 vanishes in jj44. Unsupported nonzero targets are rejected.
Q10=0 in an inert-core calculation is not a complete center-of-mass projection.
Pairing amplitudes and particle-number fluctuations require an HFB extension.

## IMSRG-evolved one- and two-body operators

The included small examples can be run without installing IMSRG:

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/evolved_ne20.inp
python3 PythonScript/scan_gcm_hf.py examples/gcm/multishell_o16.inp
```

The first uses an evolved mass Q20/Q22 operator. The second exercises nonzero
Q20, Q22, Q10, Q30, Jx and Jz simultaneously. The fixture Hamiltonians/operators
were generated at **emax=2**, hw=16 MeV, using Minnesota plus intrinsic kinetic
energy and a short IMSRG(2) flow to s=0.2. They test the interface and induced
two-body terms; they are not fully decoupled production interactions.

For an evolved Q20/Q22 constraint, add one shared quadrupole section to the
input. It reads the IMSRG `WriteTensorTokyo` tensor-SNT format:

```ini
[quadrupole]
file            = Qmass.snt
units           = fm^2
normal_ordering = core
```

The reader selects rank 2, even parity and components 0/2, and inherits the
basis from `[calculation]`. General tensors use `[operator NAME]` with explicit
rank/component/parity; see the [input guide](MyHF/docs/INPUT_GUIDE.md).
The Hamiltonian and operator
must use the same basis and compatible normal ordering. An evolved electric E2
operator is not automatically a mass quadrupole: the benchmark starts from
`E2 + nE2`, with unit proton and neutron weights. HF/NAT input requires operators
transformed to that same basis; bare radial HO operators cannot be substituted.

Reference-density normal ordering, custom/cached NPZ operators, the Python API,
and benchmark regeneration are documented in [EVOLVED_GCM.md](MyHF/EVOLVED_GCM.md).

## Outputs and memory

The named driver writes `surface.csv` with target/actual moments and diagnostics,
`states/*.npz` with occupied orbitals, and `gcm_basis/*.dat` with accepted states
in the existing `Read_GCM_HF_points` format. Supply the latter directory, including
a trailing slash, to `GCM_Projection.ReadBasis`. Histories, an attempt journal,
restart state, resource estimates and input/code hashes are also saved.

`surface.csv` records both valence and total energies; the total includes the
SNT zero-body term once. GCM basis headers use the valence energy to match the
existing projection Hamiltonian. Basis generation does not perform projection
or solve the Hill–Wheeler equation.

Hamiltonian contractions use native MKL routines. The orbital Hessian is applied
without allocating a dense particle-hole Hessian. Evolved-operator response
kernels are sparse during solving; their conversion currently uses bounded dense
work arrays. `memory_mb` is a conservative admission budget, not an operating
system RSS limit. The Hamiltonian still uses dense m-scheme storage scaling as
8(dp⁴ + dn⁴ + dp²dn²) bytes; large spaces need a different interaction backend.
The named driver uses one BLAS/OpenMP thread; the bare grid driver exposes
`--threads`.

