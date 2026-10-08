# Evolved operators and named HF constraints

`PythonScript/scan_gcm_hf.py` generates constrained real Slater determinants for
PES and GCM calculations. It uses the same safeguarded hybrid/analytic Newton-CG
solver as the earlier quadrupole driver. Constraints can contain one- and
two-body terms. The implementation includes the full density-dependent field
and its response in the orbital Hessian, as well as the expectation value.

## Run the examples

From MyHF in WSL:

```sh
python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp --output Output/my_mg24_gcm
python3 PythonScript/scan_gcm_hf.py examples/gcm/evolved_ne20.inp --output Output/my_evolved_gcm
python3 PythonScript/scan_gcm_hf.py examples/gcm/multishell_o16.inp --output Output/my_octupole_gcm
```

In the original validation workspace, the supplied examples were run into `Output/gcm_usda_mg24_20261008`,
`Output/gcm_evolved_ne20_20261008`, and `Output/gcm_multishell_o16_20261008`. Use a new output
directory, or `--resume` with the identical configuration, input files and code.
`--check` loads/validates the operators, reports memory estimates and prints the
targets without solving. Both forward and reverse passes run by default; the
lower accepted energy is retained. `--passes forward` selects a single pass.

The recommended input is a commented plain-text `.inp` file, with
`[calculation]`, `[constraints]` and optional `[solver]` sections. Write
`Q20 = 1.0 1.5` for a scan, `Q22 = 0.5` for a fixed moment, or `Jx = off`
to leave a moment free. A range such as `1.0:2.0:0.5` includes both endpoints.
Multiple varying constraints form a Cartesian grid; a `[path]` table supplies
specific paired targets instead. The [input guide](docs/INPUT_GUIDE.md) contains
complete examples and syntax. The output directory can be set inside the file.

Relative paths are resolved against the input file's directory. Windows drive
paths are mapped to /mnt/<drive>/ under WSL. Old JSON files remain supported.

To build a beta/gamma grid for a mass quadrupole in fm², use
Q20 = [3 A R²/(4 pi)] beta cos(gamma) and
Q22sum = sqrt(2) [3 A R²/(4 pi)] beta sin(gamma), with R=1.2 A^(1/3) fm.
Put these paired targets in a `[path]` table; do not take their
Cartesian product. An electric E2 operator does not use this mass normalization.

The output contains:

* `surface.csv`: target/actual moments, energies, convergence and curvature checks.
* `states/*.npz`: occupied proton/neutron orbitals for restart or analysis.
* `gcm_basis/*.dat`: converged states in the existing `Read_GCM_HF_points` format;
  this directory contains only basis files. Supply its path, with a trailing
  slash, to the existing `GCM_Projection.ReadBasis` interface.
* `history/`, `attempts.jsonl`, `progress.json`: numerical histories and restart data.
* `settings.json`: input/code hashes, conventions and memory estimates.

The GCM `.dat` header uses the unshifted valence Hamiltonian energy, matching the
existing projection Hamiltonian. `surface.csv` also records total energy with
the SNT zero-body term added once. Constraints generate the basis; they are not
added as penalties to the physical Hamiltonian or the projected kernels.
Projection/Hill-Wheeler solving remains in the existing GCM code.

## Built-in operators and units

Bare operators default to proton and neutron weights `(1,1)`. They are mass
multipoles. Override `weights` with `[1,0]`, `[0,1]`, or another explicit pair for
proton, neutron, or isovector coordinates. For example:

```ini
[constraints]
Q20p = 1.0 2.0

[operator Q20p]
name    = Q20
weights = 1 0
```

* `Q20`: sum of r²Y20, in fm².
* `Q22`: sum of r²(Y22+Y2,-2), in fm², i.e. twice real Q22.
* `Q21`: real Q21 = (Q21-Q2,-1)/2, useful for fixing the remaining real principal axis.
* `Q10`: sum of rY10, in fm; for a mass operator it controls the active-space
  dipole/displacement. Usually constrain it to zero when exploring octupole
  shapes. In an inert-core calculation, zero valence dipole is not a complete
  intrinsic center-of-mass projection.
* `Q30`: sum of r³Y30, in fm³.
* `Jx`, `Jz`: direct expectations divided by hbar. You may map a spin
  label J to the semiclassical target sqrt(J(J+1)) and enter that number
  directly (J=2 gives 2.449489743). This does not impose exact total spin;
  angular-momentum projection is a separate step.
* `Q40`, `Q32`, and other real multipoles through rank 8 are also available.
  Positive-mu components use Q_lmu+(-1)^mu Q_l,-mu, except the Q21 half-sum above.
* `R2`: sum of r², in fm²; it is not divided by particle number or square-rooted.

`units = oscillator` in an operator section uses b^L units for bare radial operators instead of
fm^L. The oscillator length is b²=41.47106/hw fm². Built-in radial operators
require an HO representation. For HF/NAT Hamiltonians, supply operators
transformed to precisely the same basis; orbit labels alone do not certify
that the radial basis transformation agrees. Jx/Jz remain usable within the
spherical j blocks. This solver is real and charge conserving: complex,
charge-changing, and anomalous/pairing fields are not implemented.

Odd multipoles can vanish in a restricted model space. Both Q10 and Q30 vanish
in the pure sd shell. Nonzero targets for identically zero operators are rejected
as infeasible. In jj44, dipole matrix elements vanish by its orbital/j selection
rules even though both parities occur; octupole components can be nonzero.

## IMSRG tensor-SNT input

The reader supports your `ReadWrite::WriteTensorTokyo` export:

1. Orbit header/table with n, l, 2j, tz (protons -1, neutrons +1).
2. One-body header followed by `a b reduced_value` records.
3. Two-body header followed by `a b c d Jbra Jket reduced_value` records.

Pair matrix elements are between normalized antisymmetrized coupled kets.
The conversion uses

```
<J M|T_kq|J' M'> = CG(J' M', k q | J M) <J||T_k||J'> / sqrt(2J+1).
```

It handles pp, nn and pn channels, same-orbit normalization, reordered pairs,
and Hermitian partner reconstruction without double counting. Invalid angular
momentum, parity, particle-species, basis, and duplicate-record conventions are
checked. The older six-decimal exports are allowed a 3e-6 Hermiticity tolerance;
roundoff-level differences are symmetrized.

The format does not encode all required conventions. Declare them explicitly:

```ini
[operator Q20]
file            = Qmass_IMSRG2_HO.snt
rank            = 2
component       = 0
parity          = even
units           = fm^2
normal_ordering = core
basis           = HO
```

Use `component = 2` with the same file for the real Q22 sum. A shared
`[quadrupole]` section can supply both components without duplicating settings;
see [evolved_ne20.inp](examples/gcm/evolved_ne20.inp). `parity` accepts
`even`/`odd` or `0`/`1`. Scalar ordinary Hamiltonian SNT records with six two-body fields
are not tensor-SNT records and are rejected by this reader.

Normal-ordering choices:

* `core`: the operator was re-normal-ordered to the inert core before export.
  The active-space reference is vacuum. This matches your standard valence-space
  IMSRG export after UndoNormalOrdering followed by DoNormalOrderingCore.
* `valence_vacuum`: already expressed against vacuum for the active states.
* `reference`: retain the supplied normal-ordered coefficients and give a
  `reference_file` containing real symmetric active-space density matrices `p,n`.
  The expectation is evaluated with delta-rho=rho-rho_ref. For J-coupled tensor
  input the reference should be spherical; general component matrices can use
  the NPZ interface below. Reference occupations must lie in [0,1].

Never use an ensemble-normal-ordered one-body term as a vacuum one-body term
without accounting for its reference: that double counts contractions. The
zero-body term is included where appropriate. Rank>0 tensor SNT requires zero
scalar zero-body term. An evolved electric E2 operator and an evolved mass
quadrupole are distinct; the reader does not relabel or rescale one into the
other. For mass quadrupole, evolve proton+neutron unit-weight Q consistently
with the Hamiltonian. The laboratory mass quadrupole is not automatically the
intrinsic center-of-mass-subtracted quadrupole, which has additional terms.

## Python API and a cached operator format

```python
from hf_operators import builtin_operator, tensor_snt_operator
from hybrid_hf import Solver, Options

q20 = tensor_snt_operator("Qmass.snt", hf, name="Q20", rank=2, mu=0,
    parity=0, normal_ordering="core", units="fm^2", representation="HO")
q22 = tensor_snt_operator("Qmass.snt", hf, name="Q22", rank=2, mu=2,
    parity=0, normal_ordering="core", units="fm^2", representation="HO")
jx = builtin_operator(hf, "Jx", hw=16)
solver = Solver(hf, constraints=[q20, q22, jx], options=Options(max_iterations=700))
result = solver.solve({"Q20":1.0, "Q22":0.5, "Jx":0.0})
q20.save("Q20_cached.npz")
```

`HFOperator` also accepts custom real one-body matrices and a sparse linear
density-response kernel K, plus a zero-body term and optional reference density:

```
O[rho] = O0 + f : delta-rho + 1/2 delta-rho : K : delta-rho
F_O[rho] = f + K delta-rho
delta F_O = K delta-rho_variation
```

Packing is row-major proton density followed by row-major neutron density. For
antisymmetrized two-body matrix elements V_ij,kl, Gamma_ik=sum_jl V_ij,kl rho_lj.
The sparse kernel is the restriction of this contraction to real symmetric,
species-diagonal densities. It is not an orbital Hessian. The Newton Hessian
uses the Hamiltonian response plus sum(lambda_a delta F_Oa), including the
constraint's density dependence and occupied-space geometric terms.

NPZ stores format metadata, the explicit n/l/2j/2m/tz basis, one-body matrices,
reference densities, and CSR kernel arrays. Loading uses `allow_pickle=False`
and rejects basis mismatches. An `[operator NAME]` section with `type = npz` and `file = ...` loads this cache directly.

No particle-hole Hessian is stored. Sparse kernels are used at solve time;
tensor conversion currently uses bounded dense work arrays, with conservative
peak-memory admission before allocation. The global budget includes existing
Hamiltonian/solver allowance and each resident constraint. It is an admission
budget, not an OS RSS limit; arbitrary large-space sparse tensor conversion is
not yet implemented.

## Validation input and checks

`Interaction/hybrid_evolved_e2` was generated with your installed IMSRG code
at emax=2, hw=16 MeV, an O16 reference, Minnesota plus intrinsic kinetic energy,
and Qmass=E2+nE2. Only a short s=0.2 IMSRG(2) flow is used: these are interface
benchmarks with nonzero induced two-body operators, not production interactions
or a claim of completed decoupling. Both H and Q are transformed back to HO;
sd-shell exports are core normal ordered, while full-space exports use vacuum.
The generator script records these conventions and can regenerate the files.

```sh
python3 examples/gcm/generate_imsrg_benchmark.py --imsrg-build /mnt/d/Code/SRG/IMSRG/imsrg/build --output Interaction/hybrid_evolved_e2
```

This benchmark generator fixes emax=2, respecting the requested emax<4 limit.

Tests cover the analytic 0d5/2 stretched-state quadrupole, bare IMSRG/native
agreement, pp/nn/pn pair normalization, a rank-2 pn tensor, finite-difference
constraint gradients and Lagrangian Hessians with 2b terms, Hessian symmetry,
reference-density shifts, serialization, parity-odd multipoles, memory rejection,
GCM export ordering, and a stable evolved-operator solve.

## Quadrupole normalization correction, 2026-10-08

The prior bare-Q2 implementation expanded reduced matrix elements with a CG
coefficient without dividing by sqrt(2*2+1). This made both Q20 and Q22 too large
by sqrt(5). `Hamiltonian::Calculate_Q2` now supplies that division, and the named
operators independently use the standard spherical-harmonic/Wigner-Eckart
normalization. The existing integer-index Solver API remains supported.

Old Ge76/Se76 data/PDFs have not been changed. Their stored energies and gamma
values remain associated with the same saved determinants, but their standard
mass-deformation beta is beta_old/sqrt(5). The earlier Ge76 grid minimum at
beta=0.07 therefore corresponds to 0.031305; Se76 beta=0.04 corresponds to
0.017889. The scanned upper beta=0.16 corresponds to 0.071554. These are bare
active-space mass moments with full A in the deformation denominator, not an
evolved-operator deformation correction. New scans use the corrected convention.
The old Ge76/Se76 runner refuses to mix checkpoints when its solver or native
module hash changes. Use a fresh job directory for recomputation.

## Further GCM coordinates

For low-lying spectroscopy, quadrupole shape and cranking are a useful starting
set. Add octupole coordinates for parity/asymmetry correlations. Q21=0 helps fix
orientation, and Q10=0 helps suppress displacement when constraining Q30; these
are often auxiliary constraints rather than independent collective coordinates.
Q40/R2, Q32, and separate proton/neutron quadrupoles are available for studying
additional shape, radial, and isovector variations.

Pairing amplitudes/fluctuations are an important further direction, but require
an HFB/Bogoliubov extension: number-conserving Slater determinants have zero
anomalous density and zero variance of each conserved total particle number.
They cannot gain an independent pairing-gap coordinate from this HF interface.

References: [Egido, Phys. Scr. 91 (2016) 073003](https://arxiv.org/abs/1606.00407)
on quadrupole, pairing and cranking coordinates;
[Zhou and Yao, octupole GCM review](https://arxiv.org/abs/2309.09488).
