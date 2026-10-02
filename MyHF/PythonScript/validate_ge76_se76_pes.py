#!/usr/bin/env python3
"""Validate both completed forward/reverse PES scans and record provenance."""
from pathlib import Path
import argparse, csv, hashlib, json, math, re

JOB = Path(__file__).resolve().parents[1] / 'Output/ge76_se76_pes'
OUT = JOB / 'outputs'


def validate(method):
    subdir = Path('IMSRG2') if method == 'IMSRG2' else Path('.')
    root = OUT / subdir
    input_name = 'IMSRG2_jj44_Ge76_e12_hw12_E328.snt' if method == 'IMSRG2' else 'interaction.snt'
    snt = OUT / 'input' / input_name
    header = snt.read_text()
    digest = hashlib.sha256(snt.read_bytes()).hexdigest()
    emax = int(re.search(r'e1max:\s*(\d+)', header).group(1))
    offset = float(re.search(r'Zero body term:\s*([+-]?[\d.]+)', header).group(1))
    stats = []
    for nucleus, particles in [('Ge76', (4, 16)), ('Se76', (6, 14))]:
        dest = root / nucleus
        settings = json.loads((dest / 'settings.json').read_text())
        assert settings['snt_sha256'] == digest
        assert settings['zero_body_MeV'] == offset
        assert json.loads((dest / 'summary.json').read_text())['completed_pass'] == 'reverse'
        with (dest / 'surface.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 209 and {int(r['point_id']) for r in rows} == set(range(209))
        assert len({(r['beta'], r['gamma_deg']) for r in rows}) == 209
        assert all(r['converged'] == 'True' and r['stability_checked'] == 'True' for r in rows)
        assert all(math.isfinite(float(r['accepted_energy_MeV'])) for r in rows)
        assert all(float(r['gradient_norm']) <= 1e-6 and float(r['max_constraint_error']) <= 1e-8 for r in rows)
        assert all(abs(float(r['energy_change_MeV'])) <= 1e-8 for r in rows)
        assert all(float(r['smallest_curvature']) >= -1e-5 for r in rows)
        assert all(float(r['orthogonality_error']) < 1e-10 and float(r['idempotency_error']) < 1e-10 for r in rows)
        assert all(abs(float(r['energy_MeV']) - float(r['energy_valence_MeV']) - offset) < 1e-10 for r in rows)
        assert all(abs(float(r['protons']) - particles[0]) < 1e-10 and abs(float(r['neutrons']) - particles[1]) < 1e-10 for r in rows)
        assert all(abs(float(r['q0']) - float(r['target_q0'])) < 1e-8 and abs(float(r['q2']) - float(r['target_q2'])) < 1e-8 and abs(float(r['q21_real'])) < 1e-8 for r in rows)
        phases = {}
        for phase in ('forward', 'reverse'):
            records = [json.loads(line) for line in (JOB / 'work' / subdir / nucleus / f'{phase}_attempts.jsonl').read_text().splitlines()]
            assert {r['point_id'] for r in records} == set(range(209))
            phases[phase] = {}
            for r in records:
                if r['converged']:
                    phases[phase][r['point_id']] = min(r['energy_MeV'], phases[phase].get(r['point_id'], math.inf))
        for row in rows:
            pid = int(row['point_id'])
            expected = min(p.get(pid, math.inf) for p in phases.values())
            assert abs(float(row['accepted_energy_MeV']) - expected) < 1e-10
        differences = [phases['forward'][i] - phases['reverse'][i] for i in phases['forward'] if i in phases['reverse']]
        minimum = min(rows, key=lambda r: float(r['energy_MeV']))
        assert all(abs(float(r['relative_energy_MeV']) - (float(r['energy_MeV']) - float(minimum['energy_MeV']))) < 1e-10 for r in rows)
        stats.append(dict(method=method, nucleus=nucleus, accepted=len(rows),
                          minimum_MeV=float(minimum['energy_MeV']), beta=float(minimum['beta']), gamma_deg=float(minimum['gamma_deg']),
                          max_gradient_norm=max(float(r['gradient_norm']) for r in rows),
                          max_constraint_error=max(float(r['max_constraint_error']) for r in rows),
                          max_idempotency_error=max(float(r['idempotency_error']) for r in rows),
                          max_orthogonality_error=max(float(r['orthogonality_error']) for r in rows),
                          reverse_improved_by_more_than_1e_6_MeV=sum(d > 1e-6 for d in differences),
                          max_forward_reverse_difference_MeV=max(abs(d) for d in differences)))
        settings.update(method=method, emax=emax, interaction_display_name='1.8/2.0 (EM)',
                        interaction_display_name_source='user-specified',
                        input_2N_header=re.search(r'input 2N:\s*(.+)', header).group(1),
                        input_3N_header=re.search(r'input 3N:\s*(.+)', header).group(1))
        (dest / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    (root / 'validation.json').write_text(json.dumps(stats, indent=2) + '\n')
    return stats


def main():
    global JOB, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', type=Path, default=JOB)
    args = parser.parse_args()
    JOB = args.job_root.expanduser().resolve()
    OUT = JOB / 'outputs'
    stats = validate('IMSRG3f2') + validate('IMSRG2')
    (OUT / 'comparison_validation.json').write_text(json.dumps(stats, indent=2) + '\n')
    readme = '''Ge76 and Se76 Hartree-Fock potential-energy surfaces
==================================================

Two separately calculated surfaces: IMSRG3f2 and IMSRG2.
Plot titles use the user-specified interaction name 1.8/2.0 (EM), emax=12,
jj44, and hw=12 MeV. The supplied headers instead record the input 2N file
TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw12_emax16_e2max32.me2j.gz and input 3N:
none. The display name follows the user's explicit choice; the SNT headers
and input files have not been altered. The title emax=12 is the IMSRG e1max,
not the emax16 embedded in the input NN filename.

Within each method, the supplied Ge76-derived SNT is used for BOTH nuclei.
This does not constitute a separately Se76-targeted IMSRG calculation.
Source paths, input hashes, solver hashes and parameters are in settings.json
under each nucleus. Exact input copies are retained in input/.

Calculation: real unrestricted two-body HF, using MyHF's hybrid solver.
Constraints: total proton+neutron Q20, Q22+Q2,-2, and Re Q21=0.
The lower accepted energy from forward and reverse scans is retained.
Grid: beta=0..0.16 by 0.01; gamma=0..60 degrees by 5; beta=0 occurs once.
Each nucleus and method has 209 distinct points. Checks require projected
gradient <=1e-6, constraint error <=1e-8, energy change <=1e-8 MeV, and a
numerical constrained stability check (curvature >=-1e-5).
Runs use one BLAS/OpenMP thread and a 512 MB interaction-memory limit each.

Model space: 0f5/2, 1p3/2, 1p1/2, 0g9/2, 22 m-scheme states per species,
above Z=N=28. Ge76: 4 valence protons, 16 valence neutrons; Se76: 6 and 14.
Quadrupole moments are bare active-space mass moments. A spherical inert
core contributes no Q. No evolved quadrupole operator or effective charges
were supplied. With R=1.2 A^(1/3) fm and b^2=41.47106/hw fm^2:
 Q0/b^2 = [3 A R^2/(4 pi b^2)] beta cos(gamma)
 (Q22+Q2,-2)/b^2 = sqrt(2) [3 A R^2/(4 pi b^2)] beta sin(gamma)
These active-space beta values need not match full-space deformations.

Energies include the respective zero-body term exactly once:
 IMSRG3f2: -495.192562 MeV; IMSRG2: -481.828426 MeV.
CSV files retain the unshifted valence energy as well.
Plots show E-Emin for each surface, with a common color range across all
four surfaces when plot_pes.py runs with its default --method all.
Stars mark sampled-grid minima, not proven continuous or global minima.
Contours use linear triangulation without extrapolation. Failed points
and every triangle touching a failed point are masked.

Results
-------
'''
    for r in stats:
        readme += (f"{r['method']} {r['nucleus']}: {r['accepted']}/209 accepted; "
                   f"E_min={r['minimum_MeV']:.9f} MeV; beta={r['beta']:.2f}; gamma={r['gamma_deg']:.0f} deg.\n"
                   f"  max gradient={r['max_gradient_norm']:.3e}; max moment error={r['max_constraint_error']:.3e}.\n"
                   f"  Reverse scan lowered {r['reverse_improved_by_more_than_1e_6_MeV']} points by >1e-6 MeV.\n")
    readme += '''
Files and reproduction
----------------------
Ge76_Se76_PES.pdf/.png: IMSRG3f2 combined plot, updated titles.
IMSRG2/Ge76_Se76_PES.pdf/.png: separate IMSRG2 combined plot.
Each method also has Ge76_PES.pdf/.png and Se76_PES.pdf/.png.
Ge76/surface.csv and Se76/surface.csv: IMSRG3f2 data.
IMSRG2/Ge76/surface.csv and IMSRG2/Se76/surface.csv: IMSRG2 data.
comparison_validation.json: numerical validation of all four surfaces.

From this outputs folder in WSL, for each nucleus Ge76 and Se76:
 python3 run_pes.py Ge76 --method IMSRG2
 python3 run_pes.py Ge76 --method IMSRG2 --pass-name reverse
Replace IMSRG2 by IMSRG3f2 to reproduce the earlier calculation.
 python3 validate_pes.py
 python3 plot_pes.py
To plot one method only: python3 plot_pes.py --method IMSRG2
Requires the installed MyHF module (path in run_pes.py), NumPy, SciPy,
pandas and Matplotlib. Checkpoints and logs remain in ../work, with
IMSRG2 checkpoints isolated under ../work/IMSRG2.

Plot style adapted from the user-supplied script:
D:\\Code\\CCSD\\Data\\M0v\\Ge76\\Triaxial\\PES\\Plot_v1.py
The original plotting script and SNT source files have not been changed.
'''
    (OUT / 'README.txt').write_text(readme, encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
