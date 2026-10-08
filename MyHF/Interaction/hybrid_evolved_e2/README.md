# Small IMSRG operator fixtures

These files exercise the MyHF tensor-SNT reader and constrained hybrid solver.
They were generated with the local build of Ragnar Stroberg's IMSRG code at
**emax=2**, hw=16 MeV, O16 reference, Minnesota plus intrinsic kinetic energy,
and mass quadrupole `E2+nE2`. The short IMSRG(2) Magnus flow ends at **s=0.2**.
It induces nonzero two-body quadrupole terms but does not constitute a fully
decoupled production interaction. See `provenance.json` for the recorded norms.

* `H_bare_HO.snt`, `Qmass_bare_HO.snt`: bare comparison files.
* `H_IMSRG2_HO.snt`, `Qmass_IMSRG2_HO.snt`: sd-shell, inert O16 core.
* `H_IMSRG2_full_HO.snt`, `Qmass_IMSRG2_full_HO.snt`: full emax=2 space.

All files use the HO representation. The evolved sd-shell exports are core
normal ordered; full-space exports are vacuum normal ordered. The bare Q file
has no two-body terms. Tensor files use reduced, normalized J-coupled pair
matrix elements, not the scalar Hamiltonian SNT record layout.

Regenerate from `MyHF/`, using a compatible built IMSRG Python module:

```sh
python3 examples/gcm/generate_imsrg_benchmark.py --imsrg-build /path/to/imsrg/build --output Interaction/hybrid_evolved_e2
```

The generator fixes emax=2. Existing example/test files do not require IMSRG at
runtime. `../FCI_HF_FCI_O17_e3_hw16_E39.snt` is an additional emax=3 regression
Hamiltonian; its original embedded interaction/provenance header is preserved.
