# Calculation input

Write the calculation in a plain-text **`.inp`** file. Start by copying one of
the commented files in [`examples/gcm/`](../examples/gcm/). From `MyHF/`:

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --check
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp
```

`--check` prints the nucleus, interaction, solver, memory estimate, output path,
constraint units and target table without creating a result directory. The run
uses the output directory written in the input file. If that directory already
contains a compatible run, add `--resume`; otherwise choose a new output path.
`--output Output/another_run` overrides the file's output setting.

## A complete example

The paths below assume the input is saved in `examples/gcm/`:

```ini
[calculation]
nucleus     = Mg24
interaction = ../../Interaction/usda.snt
hw          = 16                   # oscillator energy, MeV
basis       = HO
memory_mb   = 512
output      = ../../Output/mg24_gcm
passes      = forward reverse

[constraints]
Q20 = 1.0 1.5                     # fm^2; two target values
Q22 = 0.5                         # fm^2; Q22 + Q2,-2
Q21 = 0                           # principal-axis orientation
Q10 = 0                           # fm
Q30 = 0                           # fm^3
Jx  = 0                           # <Jx>/hbar
Jz  = 0                           # <Jz>/hbar

[solver]
method         = hybrid
max_iterations = 700
```

Settings use `name = value`. Blank lines and full-line `#`, `;` or `!` comments
are allowed. Inline comments start with `#` or `;` after whitespace. Spaces do
not need escaping in file paths. Optional surrounding quotes are accepted;
backslashes in Windows paths are literal, not escape sequences. Avoid inline
comment markers preceded by whitespace inside a path. Section and constraint
names are case sensitive: use `[calculation]`, `Q20`, `Jx`, etc.

Every relative path **inside the file** is resolved relative to that file's
directory. If you move a file to `MyHF/`, change `../../Interaction/usda.snt` to
`Interaction/usda.snt`. Absolute Windows paths also work under WSL, e.g.
`D:\Research\my calculation\interaction.snt`. A CLI `--output` path is relative
to the shell's current directory.

## Values, scans and unconstrained moments

```ini
[constraints]
Q20 = 1.0                         # fixed at 1.0
Q22 = 0.0 0.5 1.0                 # scan these three values
Jx  = off                         # unconstrained
Jz  = 0                           # constrained to zero
```

`off` and `free` omit a constraint. They differ physically from a zero target.
At least one constraint must remain active. Removing a constraint line also
leaves that moment free.

Use an inclusive **start:stop:step** range for evenly spaced targets:

```ini
Q20 = 1.0:2.0:0.5                 # 1.0, 1.5, 2.0
Q22 = 1.0:0.0:-0.5                # 1.0, 0.5, 0.0
```

The step must reach the stop exactly. Otherwise give an explicit list. Lists
may use spaces or commas. Scientific notation and Fortran `D` exponents are
accepted. Expressions are not evaluated: for a semiclassical spin J=2, enter
`Jx = 2.449489743` for sqrt(J(J+1)).

Varying several constraints creates their Cartesian product. Three Q20 values
and two Q22 values give six points. `max_points = 10000` in `[calculation]` is
the default cap; oversized ranges/grids are rejected before solving.

## A path of paired targets

For specific pairs rather than every combination, add a `[path]` table:

```ini
[constraints]
Q20 = 0                           # table below supplies the targets
Q22 = 0
Q21 = 0                           # fixed on every row
Jx  = off
Jz  = off

[path]
columns = Q20 Q22
points =
    1.0   0.5
    1.5   1.0
```

Indent each row, with one number per column and no blank lines within the table.
This produces exactly two points. Every column must name an active constraint.
Constraints omitted from the table must have one fixed target. Values in table
columns override the corresponding `[constraints]` values. The complete runnable
example is [`mg24_path.inp`](../examples/gcm/mg24_path.inp).

For a beta/gamma trajectory, use the mass-quadrupole formulas in
[EVOLVED_GCM.md](../EVOLVED_GCM.md) to put the paired Q20/Q22 targets in this table.
The separate bare-quadrupole driver also accepts beta/gamma directly on its CLI.

## An IMSRG-evolved quadrupole

Use `[quadrupole]` to supply one tensor-SNT file for both Q20 and Q22:

```ini
[constraints]
Q20 = 1.0 1.5
Q22 = 0.5 1.0
Q21 = 0

[quadrupole]
file            = ../../Interaction/hybrid_evolved_e2/Qmass_IMSRG2_HO.snt
units           = fm^2
normal_ordering = core
```

The parser selects rank 2, even parity and component 0 or 2 automatically. It
inherits the declared basis from `[calculation]`. It does **not** infer mass
versus electric character: you must provide the desired mass/electric operator
and consistent targets. Q21 remains bare unless you give it a separate operator
section. The complete evolved example is
[`evolved_ne20.inp`](../examples/gcm/evolved_ne20.inp).

`normal_ordering` accepts `core`, `valence_vacuum` or `reference`.
The last also requires `reference_file = reference_density.npz` containing
active-space proton/neutron density arrays `p,n`. See the
[operator guide](../EVOLVED_GCM.md) before using ensemble-normal-ordered inputs.
The Hamiltonian and operator must have compatible bases and normal ordering.

## Individual operators and proton/neutron weights

Give an operator a constraint name and add `[operator NAME]`:

```ini
[constraints]
Q20p = 1.0 1.5
Q20n = 1.0

[operator Q20p]
name    = Q20
weights = 1 0                     # proton, neutron

[operator Q20n]
name    = Q20
weights = 0 1
```

The default is a built-in mass operator with weights `1 1`, physical fm^L
units and a name matching the constraint. For bare radial operators,
`units = oscillator` switches to b^L units. Jx/Jz are direct expectations in
hbar. An individual operator section takes precedence over `[quadrupole]` for
that constraint. Sections for constraints marked `off` are inactive.

For a general evolved tensor:

```ini
[operator Q30]
type            = tensor_snt
file            = Q30_evolved.snt
rank            = 3
component       = 0
parity          = odd
units           = fm^3
normal_ordering = core
basis           = HO
```

A `file` without a `type` implies `tensor_snt`. `parity` can be `even`/`odd`
or `0`/`1`. For a cached component operator saved by the Python API:

```ini
[operator Q20]
type = npz
file = Q20_cached.npz
```

This NPZ must contain the operator metadata, basis and matrices expected by
`hf_operators`, not an arbitrary NumPy archive.

## Solver and run settings

`[solver]` is optional. The defaults are:

```ini
[solver]
method               = hybrid
max_iterations       = 500
gradient_tolerance   = 1e-6
constraint_tolerance = 1e-8
energy_tolerance     = 1e-8
check_stability      = yes
curvature_tolerance  = 1e-5
seed                 = 520
diagonalization_steps = 6
gradient_steps        = 6
trust_radius          = 0.3
max_cg                = 35
precondition          = yes
precondition_floor    = 0.1       # minimum absolute orbital gap, MeV
```

The scan driver uses six diagonalization steps for a fresh start and zero when
continuing an accepted determinant, and adjusts the seed on retries/reverse
passes. These per-attempt choices take precedence over those two input settings.
Use `method = gradient` for the comparison solver. Stability checks should
normally stay enabled. The orbital-gap preconditioner accelerates the inner
Newton-CG solve without changing acceptance tolerances. Its floor must be
finite and positive; use `precondition = no` for diagnostic comparisons.
See [PRECONDITIONING.md](PRECONDITIONING.md) for measured results and
[HYBRID_HF.md](../HYBRID_HF.md) for numerical details. Use a fresh output
directory after changing the solver: source hashes are part of the resume check.

`[calculation]` requires `nucleus`, `interaction` and `hw`. Optional settings
default to `basis = HO`, `memory_mb = 512`, `max_points = 10000`, and
`passes = forward reverse`. Specify `output` there or on the command line for
a solve. No output directory is needed for a preview.

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --resume
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --passes forward --output Output/new_run
```

Old `.json` inputs and the `--config` spelling still work; `--input` is an alias.
Normal previews are text tables; `--check-json` requests a machine-readable
preview. Results/restart metadata continue to use CSV, JSON and NPZ internally;
there is no need to edit those files to define a calculation.
