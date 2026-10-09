#!/usr/bin/env python3
"""Test hybrid HF from independent random Slater determinants at one target.

From MyHF:
python3 tests/benchmark_random_starts.py --input examples/gcm/usda_mg24.inp --output Output/random_mg24
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import linalg

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'PythonScript')]
from hf_input import read_config
from scan_gcm_hf import load_problem,target_points


def random_determinant(shapes,seed):
    """Independent real Gaussian/QR occupied subspaces for each species."""
    rng=np.random.default_rng(seed)
    state=[]
    for d,n in shapes:
        if not n:
            state.append(np.empty((d,0)))
            continue
        q,r=linalg.qr(rng.normal(size=(d,n)),mode='economic')
        q*=np.where(np.diag(r)<0,-1.,1.)
        state.append(q)
    return tuple(state)


def finite_json(value):
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input','--config',dest='config',type=Path,required=True)
    parser.add_argument('--point',type=int,default=0,help='zero-based target index in the input grid')
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1001,1011)))
    parser.add_argument('--output',type=Path,required=True,help='new directory for this benchmark')
    args=parser.parse_args(argv)
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        parser.error('random seeds must be unique nonnegative integers')
    config=read_config(args.config)
    points=target_points(config)
    if not 0<=args.point<len(points): parser.error('point index is outside the input grid')
    solver,resources=load_problem(config)
    if solver.options.method!='hybrid': parser.error('this benchmark requires method = hybrid')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ('history','starts','states'): (out/name).mkdir()
    target=points[args.point]
    reference=tuple(x.copy() for x in solver.initial)
    metadata=dict(config=config,point_index=args.point,targets=target,resources=resources,
        randomization='independent Gaussian/QR real occupied subspaces for proton and neutron species',
        seeds=args.seeds,solver_options=vars(solver.options).copy(),continuation=False,
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (ROOT/'pyHFAndHFB.so',ROOT/'PythonScript/hybrid_hf.py',ROOT/'PythonScript/hf_operators.py',
             ROOT/'PythonScript/hf_input.py',ROOT/'PythonScript/scan_gcm_hf.py',Path(__file__))})
    (out/'settings.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[]
    for seed in (None,*args.seeds):
        label='reference' if seed is None else f'random_{seed}'
        start=reference if seed is None else random_determinant(solver.shapes,seed)
        rho=tuple(c@c.T for c in start)
        orth=max(np.max(np.abs(c.T@c-np.eye(c.shape[1])),initial=0.) for c in start)
        idem=max(np.max(np.abs(r@r-r),initial=0.) for r in rho)
        assert orth<1e-12 and idem<1e-12
        assert all(abs(np.trace(r)-c.shape[1])<1e-11 for r,c in zip(rho,start))
        distance=float(np.sqrt(sum(linalg.norm(r-c@c.T)**2 for r,c in zip(rho,reference))))
        np.savez(out/'starts'/(label+'.npz'),p=start[0],n=start[1])
        initial_energy=float(solver.evaluate(start)[0])
        initial_moments=dict(zip(solver.constraint_names,solver.moments(start).tolist()))
        phases=Counter()
        with (out/'history'/(label+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['iteration','phase','energy_MeV','gradient_norm','constraint_error','energy_change_MeV'])
            writer.writeheader()
            def callback(row):
                phases[row['phase']]+=1
                writer.writerow(row);stream.flush()
            begin=time.perf_counter()
            result=solver.solve(target,start=start,callback=callback)
            seconds=time.perf_counter()-begin
        state=result.pop('occupied')
        np.savez(out/'states'/(label+'.npz'),p=state[0],n=state[1])
        result.update(label=label,random_seed=seed,seconds=seconds,initial_energy_MeV=initial_energy,
            initial_moments=initial_moments,initial_projector_distance=distance,
            initial_orthogonality_error=float(orth),initial_idempotency_error=float(idem),phases=dict(phases))
        if result['converged']:
            assert result['stability_checked'] and result['smallest_curvature']>=-solver.options.curvature_tolerance
            assert result['gradient_norm']<=solver.options.gradient_tolerance
            assert result['max_constraint_error']<=solver.options.constraint_tolerance
            assert abs(result['energy_change_MeV'])<=solver.options.energy_tolerance
        result=finite_json(result)
        rows.append(result)
        with (out/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        print(f"{config['nucleus']} {label}: {result['status']}, iterations={result['iterations']}, E={result['energy_MeV']}, seconds={seconds:.2f}",flush=True)
    random=rows[1:];good=[r for r in random if r['converged']]
    summary=dict(nucleus=config['nucleus'],targets=target,reference=rows[0],
        random_attempts=len(random),random_converged=len(good),failures=[r for r in random if not r['converged']],
        iterations=[min(r['iterations'] for r in good),max(r['iterations'] for r in good)] if good else None,
        median_iterations=float(np.median([r['iterations'] for r in good])) if good else None,
        accepted_energy_range_MeV=[min(r['energy_MeV'] for r in good),max(r['energy_MeV'] for r in good)] if good else None,
        max_gradient=max((r['gradient_norm'] for r in good),default=None),
        max_constraint_error=max((r['max_constraint_error'] for r in good),default=None),
        minimum_curvature=min((r['smallest_curvature'] for r in good),default=None),
        source_sha256=metadata['source_sha256'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return 0 if len(good)==len(random) and rows[0]['converged'] else 2


if __name__=='__main__': raise SystemExit(main())
