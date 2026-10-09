# MyHF: Nuclear Hartree–Fock

MyHF is a C++/Python nuclear-structure code for real, unrestricted Hartree–Fock (HF) calculations with proton and neutron orbitals. It reads spherical J-coupled shell-model interactions in SNT format, constructs an m-scheme Hamiltonian, and optimizes occupied orbitals at fixed proton and neutron numbers.

The Python workflow supports constrained triaxial potential-energy-surface (PES) calculations and exports Slater determinants for angular-momentum-projected generator-coordinate-method (GCM) calculations. Both bare multipole operators and IMSRG-evolved one- and two-body operators are supported.

> **Scope:** The documented workflow performs real HF calculations without anomalous pairing density. Pairing and particle-number fluctuations require an HFB extension.

## Features

- Proton–neutron unrestricted Hartree–Fock calculations
- Valence-space and multi-shell SNT interactions
- Triaxial PES scans with continuation between neighboring points
- Constraints on multipole moments and angular-momentum components
- Bare and IMSRG-evolved operators with one- and two-body terms
- Forward and reverse scans for comparing local HF branches
- Export of accepted Slater determinants for projection and GCM calculations
- Restartable scans with saved orbitals and diagnostics

The physical energy reported by MyHF excludes constraint penalties. Converged solutions are local stationary states within the real HF manifold; users should compare different initial seeds and scan directions when studying competing branches.

## Acknowledgments

This code has benefited greatly from Ragnar Stroberg's [IMSRG code](https://github.com/ragnarstroberg/imsrg) and from the coupled-cluster code developed by Gaute and Thomas. These projects provided important foundations and reference implementations for the IMSRG operator interface, benchmark generation, HF methods, and scan workflow.

Please acknowledge these contributions and cite the relevant upstream methods and software when publishing work based on MyHF.

## Installation

### Requirements

MyHF is intended for Linux or Windows Subsystem for Linux (WSL). The native build requires:

- A C++ compiler with OpenMP support
- An MPI C++ wrapper (`mpicxx`)
- Intel MKL
- GSL
- GNU Make
- Python development headers

The tested Python environment is:

- Python 3.10.12
- NumPy 1.26.4
- SciPy 1.15.3

The pybind11 headers are included in the repository.

### Build

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
make -j2 pyHFAndHFB.so HartreeFock.exe
python3 -c 'import pyHFAndHFB; print("MyHF native module loaded")'
```

The default makefile expects MKL headers in `/usr/include/mkl` and MKL libraries on the linker search path. For a different MKL installation, update the makefile or override the build flags. For example:

```sh
make -j2 pyHFAndHFB.so HartreeFock.exe \
  CFLAGS="-O3 -fPIC -fopenmp -I$MKLROOT/include" \
  LIBS="-L$MKLROOT/lib/intel64 -Wl,-rpath,$MKLROOT/lib/intel64 -lmkl_rt -lm -lgsl -lgslcblas"
```

Use the same Python installation for the interpreter and development headers. The native extension must be rebuilt on each target machine.

Optional PES plotting scripts require `matplotlib`.

> Run the Python HF scans as a single process. Do not use `mpirun`, because it will duplicate the interaction in memory.

## Quick Start

### Named-constraint scan

The included USDA calculation for magnesium-24 (`24Mg`) uses nonzero `Q20` and `Q22` constraints:

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --check
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp
```

Use `--check` to inspect the scan points without running the calculation. To continue an interrupted scan, use:

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --resume
```

Forward and reverse passes are performed by default, and the lower-energy accepted solution is retained. Use `--passes forward` to run only one direction.

### Beta–gamma scan

For a bare mass-quadrupole beta–gamma surface:

```sh
python3 PythonScript/scan_hybrid_pes.py \
  --beta .04 .08 .12 \
  --gamma 10 30 50 \
  --threads 1 \
  --memory-mb 512 \
  --check

python3 PythonScript/scan_hybrid_pes.py \
  --beta .04 .08 .12 \
  --gamma 10 30 50 \
  --threads 1 \
  --memory-mb 512 \
  --output Output/mg24_beta_gamma
```

This driver defaults to USDA `24Mg`. Run it with `--help` to view options for the nucleus, interaction, oscillator frequency, and scan grid.

## Input Files

Calculations are configured with commented plain-text `.inp` files. For example:

```ini
[calculation]
nucleus     = Mg24
interaction = ../../Interaction/usda.snt
hw          = 16
basis       = HO
memory_mb   = 512
output      = ../../Output/mg24_gcm
passes      = forward reverse

[constraints]
Q20 = 1.0 1.5
Q22 = 0.5
Q21 = 0
Q10 = 0
Q30 = 0
Jx  = 0
Jz  = 0

[solver]
method         = hybrid
max_iterations = 700
```

Paths are resolved relative to the input file. The output directory can be set in the file or overridden with `--output`.

Constraint values may be specified as:

- A single fixed value: `Q22 = 0.5`
- A list of scan values: `Q20 = 1.0 1.5 2.0`
- A range: `Q20 = 1.0:2.0:0.5`
- Disabled: `Jx = off`

Multiple lists form a Cartesian grid. Paired targets can be defined with a `[path]` table; see [`examples/gcm/mg24_path.inp`](examples/gcm/mg24_path.inp).

For all input settings and examples, see the [input guide](docs/INPUT_GUIDE.md). Legacy JSON input files are also supported.

## Available Constraints

| Name | Quantity | Default units |
| --- | --- | --- |
| `Q20` | Mass axial quadrupole moment | fm² |
| `Q22` | Real triaxial quadrupole combination | fm² |
| `Q21` | Real off-diagonal quadrupole component | fm² |
| `Q10` | Axial dipole moment | fm |
| `Q30` | Axial octupole moment | fm³ |
| `Q40` | Axial hexadecapole moment | fm⁴ |
| `Q32` | Non-axial octupole component | fm³ |
| `R2` | Sum of squared radii | fm² |
| `Jx`, `Jz` | Intrinsic angular-momentum components divided by ℏ | dimensionless |

Bare multipoles use equal proton and neutron weights and are therefore mass operators. To constrain protons or neutrons separately, define a custom operator section:

```ini
[operator proton_Q20]
name    = Q20
weights = 1 0
```

Use `weights = 0 1` for neutrons. Additional operator conventions are documented in [EVOLVED_GCM.md](EVOLVED_GCM.md).

For semiclassical cranking, a spin label `J` may be mapped to the numerical target

```text
Jx = sqrt(J*(J+1))
```

For example, use `Jx = 2.449489743` for `J = 2`. This is an optional intrinsic constraint and does not assign an exact total angular momentum; the final spin is obtained through angular-momentum projection.

Some operators vanish in restricted model spaces. For example, `Q10` and `Q30` vanish in the pure sd shell. Unsupported nonzero targets are rejected.

## IMSRG-Evolved Operators

Included examples can be run without a separate IMSRG installation:

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/evolved_ne20.inp
python3 PythonScript/scan_gcm_hf.py examples/gcm/multishell_o16.inp
```

To use an evolved quadrupole operator in tensor-SNT format:

```ini
[quadrupole]
file            = Qmass.snt
units           = fm^2
normal_ordering = core
```

The Hamiltonian and operator must use compatible bases and normal-ordering conventions. General tensor operators can be configured with `[operator NAME]` sections.

See [EVOLVED_GCM.md](EVOLVED_GCM.md) for operator formats, custom NPZ operators, reference-density normal ordering, the Python API, and benchmark information.

## Output

The named scan driver writes:

- `surface.csv` — target and calculated moments, energies, and convergence diagnostics
- `states/*.npz` — occupied proton and neutron orbitals
- `gcm_basis/*.dat` — accepted states in the projection/GCM input format
- Restart information and scan histories

To use the exported basis with the legacy projection workflow, pass the `gcm_basis/` directory to `GCM_Projection.ReadBasis`.

MyHF generates the intrinsic basis states only. It does not perform angular-momentum projection or solve the Hill–Wheeler equation as part of the PES scan.

## Practical Notes

- The calculations may converge to different local HF branches. Compare multiple seeds and scan directions when necessary.
- A failed scan point is reported and is not silently added to the accepted PES or GCM basis.
- `memory_mb` is an admission estimate rather than a strict operating-system memory limit.
- Dense m-scheme Hamiltonian storage can limit calculations in large model spaces.
- Bare radial harmonic-oscillator operators should not be mixed with Hamiltonians or operators represented in an incompatible transformed basis.

## Documentation

- [Input guide](docs/INPUT_GUIDE.md)
- [Evolved operators and GCM workflow](EVOLVED_GCM.md)
- [Example input files](examples/gcm/)
