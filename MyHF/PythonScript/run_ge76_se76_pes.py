#!/usr/bin/env python3
"""Reproduce the Ge76/Se76 jj44 PES using the installed MyHF hybrid solver.

Linux/WSL: python3 PythonScript/run_ge76_se76_pes.py Ge76 --method IMSRG2
Default: run both forward and reverse passes, keeping the lower accepted energy.
The same supplied Ge76-derived interaction is intentionally used for both nuclei.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,csv,hashlib,io,json,math,re,sys,time
from pathlib import Path
import numpy as np
from scipy import linalg

MYHF=Path(__file__).resolve().parents[1]
JOB=MYHF/'Output/ge76_se76_pes'
INPUTS={
    'IMSRG3f2': ('interaction.snt', r'D:\Research\Research data\IMSRG\IMSRG Snts\n0vv_NME\Ge76_jj44\IMSRG3f2_closureE (wrong)\IMSRG3f2_jj44_Ge76_e12_hw12_E328_HG.snt'),
    'IMSRG2': ('IMSRG2_jj44_Ge76_e12_hw12_E328.snt', r'D:\Research\Research data\IMSRG\IMSRG Snts\n0vv_NME\Ge76_jj44\IMSRG2\IMSRG2_jj44_Ge76_e12_hw12_E328.snt'),
}
sys.path[:0]=[str(MYHF),str(MYHF/'PythonScript')]
import pyHFAndHFB as native
from hybrid_hf import Solver,Options
from scan_hybrid_pes import atomic_write,csv_text,inspect_snt,memory_estimate


def run(nucleus,pass_name,method='IMSRG3f2',memory_mb=512.):
    if not math.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError('--memory-mb must be finite and positive')
    if any(int(os.environ.get(k,'1')) > 1 for k in ('OMPI_COMM_WORLD_SIZE','PMI_SIZE','SLURM_NTASKS')):
        raise ValueError('run one process, not mpirun')
    subdir=Path('IMSRG2') if method=='IMSRG2' else Path('.')
    dest=JOB/'outputs'/subdir/nucleus
    dest.mkdir(parents=True,exist_ok=True)
    scratch=JOB/'work'/subdir/nucleus
    scratch.mkdir(parents=True,exist_ok=True)
    SNT=JOB/'outputs/input'/INPUTS[method][0]
    dims,core=inspect_snt(SNT)
    estimate=memory_estimate(dims)
    if estimate['estimated_mb'] > memory_mb:
        raise MemoryError(f"estimated {estimate['estimated_mb']:.1f} MiB exceeds --memory-mb {memory_mb:g}")
    digest=hashlib.sha256(SNT.read_bytes()).hexdigest()
    if (dest/'selected.json').exists():
        if not (dest/'settings.json').exists():
            raise ValueError('Existing results have no interaction provenance')
        previous=json.loads((dest/'settings.json').read_text())
        if previous['snt_sha256']!=digest or previous['nucleus']!=nucleus:
            raise ValueError('Existing results belong to a different interaction or nucleus')
        for key,source in (('solver_sha256',MYHF/'PythonScript/hybrid_hf.py'),
                           ('native_module_sha256',MYHF/'pyHFAndHFB.so')):
            if previous.get(key)!=hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError('Existing PES uses a different solver/operator convention; use a new --job-root with copied input/ files')
    native.set_hybrid_threads(1)
    text=SNT.read_text()
    offset=float(re.search(r'Zero body term:\s*([+-]?[\d.]+)',text).group(1))
    ms=native.ModelSpace();ms.Set_hw(12.)
    h=native.Hamiltonian(ms);h.SetMemoryLimitMB(memory_mb-estimate['solver_allowance_mb'])
    rw=native.ReadWriteFiles();rw.ReadTokyo(str(SNT),ms,h)
    native.set_hybrid_nucleus(ms,nucleus);ms.InitialModelSpace_HF();h.Prepare_MschemeH_Unrestricted()
    hf=native.HartreeFock(h)
    solver=Solver(hf,active=(0,1,4),options=Options(max_iterations=700,seed=520))
    scale=3*76*(1.2*76**(1/3))**2/(4*math.pi*(41.47106/12))
    points=[]
    for ib in range(17):
        beta=ib*.01
        angles=[0] if ib==0 else list(range(0,61,5))
        if ib%2==0: angles.reverse()
        for gamma in angles:
            a=math.radians(gamma)
            points.append(dict(point_id=len(points),beta=round(beta,2),gamma_deg=gamma,
                               target_q0=scale*beta*math.cos(a),target_q2=math.sqrt(2)*scale*beta*math.sin(a)))
    settings=dict(nucleus=nucleus,method=method,interaction_source=INPUTS[method][1],
                  interaction_display_name='1.8/2.0 (EM)',interaction_display_name_source='user-specified',
                  input_2N_header=re.search(r'input 2N:\s*(.+)',text).group(1),
                  input_3N_header=re.search(r'input 3N:\s*(.+)',text).group(1),
                  emax=int(re.search(r'e1max:\s*(\d+)',text).group(1)),
                  snt_sha256=digest,hw_MeV=12.,zero_body_MeV=offset,
                  memory_budget_MiB=memory_mb,memory_estimate=estimate,threads=1,
                  mass=76,core_protons=28,core_neutrons=28,valence_protons=solver.shapes[0][1],valence_neutrons=solver.shapes[1][1],
                  beta_grid='0:0.16:0.01',gamma_grid='0:60:5; beta=0 occurs once',points=len(points),
                  beta_definition='4 pi sqrt(Q20^2+2 Re(Q22)^2)/(3 A R^2); R=1.2 A^(1/3) fm; b^2=41.47106/hw fm^2',
                  moments='bare oscillator quadrupole of active nucleons; spherical inert core contributes zero',
                  constraints='Q20, Q22+Q2,-2, Re Q21=0; real unrestricted HF',solver_options=vars(solver.options),
                  solver_sha256=hashlib.sha256((MYHF/'PythonScript/hybrid_hf.py').read_bytes()).hexdigest(),
                  native_module_sha256=hashlib.sha256((MYHF/'pyHFAndHFB.so').read_bytes()).hexdigest())
    atomic_write(dest/'settings.json',json.dumps(settings,indent=2)+'\n')
    atomic_write(dest/'grid.csv',csv_text(points))
    selected={}
    if (dest/'selected.json').exists():
        selected={int(k):v for k,v in json.loads((dest/'selected.json').read_text()).items()}
    sequence=points if pass_name=='forward' else points[::-1]
    last_good=None
    def checkpoint(pid):
        file=scratch/f'state_{pid:03d}.npz'
        if not file.exists(): return None
        with np.load(file,allow_pickle=False) as data: return (data['p'].copy(),data['n'].copy())
    with (scratch/f'{pass_name}_attempts.jsonl').open('a',buffering=1) as attempts:
        for point in sequence:
            pid=point['point_id'];target=[point['target_q0'],point['target_q2'],0.]
            old=selected.get(pid)
            candidates=[last_good,checkpoint(pid),None]
            winner=None
            begin=time.perf_counter()
            for attempt,start in enumerate(candidates,1):
                if attempt==2 and start is None: continue
                solver.options.diagonalization_steps=6 if start is None else 0
                solver.options.seed=520+attempt+(100 if pass_name=='reverse' else 0)
                history=scratch/f'{pass_name}_{pid:03d}_{attempt}.csv'
                with history.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['iteration','phase','energy_MeV','gradient_norm','constraint_error','energy_change_MeV'])
                    writer.writeheader()
                    def record(row): writer.writerow(row);stream.flush()
                    r=solver.solve(target,start=start,callback=record)
                state=r.pop('occupied');q=r.pop('moments')
                row=dict(point,**r,energy_valence_MeV=r['energy_MeV'],zero_body_MeV=offset,
                         q0=q[0],q2=q[1],q21_real=q[4],selected_pass=pass_name,attempt=attempt,
                         elapsed_seconds=time.perf_counter()-begin)
                row['energy_MeV']=None if r['energy_MeV'] is None else r['energy_MeV']+offset
                row={k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in row.items()}
                attempts.write(json.dumps(row,allow_nan=False)+'\n')
                if row['converged']:
                    winner=(row,state)
                    break
                if winner is None: winner=(row,state)
            if winner[0]['converged']:
                if old is None or not old['converged'] or winner[0]['energy_MeV']<old['energy_MeV']:
                    selected[pid]=winner[0]
                    np.savez(scratch/f'state_{pid:03d}.npz',p=winner[1][0],n=winner[1][1])
                last_good=checkpoint(pid)
            elif old is None or not old['converged']:
                selected[pid]=winner[0]
            else:
                last_good=checkpoint(pid)
            rows=[selected[k] for k in sorted(selected)]
            accepted=[row['energy_MeV'] for row in rows if row['converged']]
            minimum=min(accepted) if accepted else None
            surface=[dict(row,accepted_energy_MeV=row['energy_MeV'] if row['converged'] else None,
                          relative_energy_MeV=row['energy_MeV']-minimum if row['converged'] else None) for row in rows]
            atomic_write(dest/'surface.csv',csv_text(surface))
            atomic_write(dest/'selected.json',json.dumps(selected,indent=2,allow_nan=False)+'\n')
            summary=dict(nucleus=nucleus,points=len(rows),planned=len(points),accepted=len(accepted),failed=len(rows)-len(accepted),
                         minimum_MeV=minimum,completed_pass=pass_name if pid==sequence[-1]['point_id'] else None)
            atomic_write(dest/'summary.json',json.dumps(summary,indent=2)+'\n')
            print(f"{nucleus} {pass_name} {pid:3d}/{len(points)-1} beta={point['beta']:.2f} gamma={point['gamma_deg']:2d}: {selected[pid]['status']} E={selected[pid]['energy_MeV']} iterations={selected[pid]['iterations']}",flush=True)

    return 0 if all(row['converged'] for row in selected.values()) else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nucleus',choices=['Ge76','Se76'])
    parser.add_argument('--pass-name',choices=['forward','reverse','both'],default='both')
    parser.add_argument('--job-root',type=Path,default=JOB,help='parent of outputs/ and work/')
    parser.add_argument('--memory-mb',type=float,default=512.,help='interaction + solver admission budget in MiB, not an OS RSS limit')
    parser.add_argument('--method',choices=list(INPUTS),default='IMSRG3f2')
    args=parser.parse_args();JOB=args.job_root.expanduser().resolve()
    passes=['forward','reverse'] if args.pass_name=='both' else [args.pass_name]
    statuses=[run(args.nucleus,phase,args.method,args.memory_mb) for phase in passes]
    raise SystemExit(max(statuses))
