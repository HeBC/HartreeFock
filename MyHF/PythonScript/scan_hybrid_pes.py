#!/usr/bin/env python3
"""Memory-bounded MyHF PES scan with converged-state continuation.

Edit USER SETTINGS or use --help. Run under Linux/WSL, not mpirun.
Native Q2 means Q22+Q2,-2 (twice the real Q22); Q0/Q2 are in b^2 units.
"""
import os
from pathlib import Path

# ------------------------- USER SETTINGS -------------------------
ROOT = Path(__file__).resolve().parents[1]
INTERACTION = ROOT / "Interaction/usda.snt"
NUCLEUS = "Mg24"
HW_MEV = 16.
Q0_VALUES = (0., 1., 2.)
Q2_VALUES = (0., .5)
BETA_VALUES = (.1, .2, .3, .4)  # use --native-grid for the Q0/Q2 grid
GAMMA_DEGREES = (10., 20., 30., 40., 50.)
MEMORY_MB = 2048.  # estimated tensor + solver working-set admission budget
THREADS = 1
MAX_ITERATIONS = 500
RETRY_FAILED = True
SAVE_ITERATION_HISTORY = True
# ----------------------------------------------------------------

import argparse
import csv
import io
import json
import math
import re
import sys
import tempfile
import time


def atomic_write(path, text):
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
            tmp = Path(f.name); f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists(): tmp.unlink()


def csv_text(rows):
    if not rows: return ""
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return f.getvalue()


def write_surface(output, rows, planned):
    good = [r["energy_MeV"] for r in rows if r["converged"]]
    if any(e is None or not math.isfinite(e) for e in good):
        raise ValueError('a converged point must have a finite energy')
    minimum = min(good) if good else None
    surface = [dict(r, accepted_energy_MeV=r["energy_MeV"] if r["converged"] else None,
                    relative_energy_MeV=r["energy_MeV"]-minimum if r["converged"] else None) for r in rows]
    atomic_write(output/"surface.csv", csv_text(surface))
    summary = dict(points=len(rows), planned_points=planned, converged=len(good), failed=len(rows)-len(good),
                   scan_complete=len(rows)==planned, minimum_converged_energy_MeV=minimum,
                   fock_evaluations=sum(r["fock_evaluations"] for r in rows),
                   hessian_evaluations=sum(r["hessian_evaluations"] for r in rows))
    atomic_write(output/"summary.json", json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def recover_surface(directory):
    output = Path(directory).resolve()
    lines=(output/"points.jsonl").read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"): lines.pop()
    rows=[json.loads(line) for line in lines]
    planned=json.loads((output/"settings.json").read_text())["planned_points"]
    if len({r['point_id'] for r in rows}) != len(rows): raise ValueError("duplicate point IDs")
    with (output/'grid.csv').open() as f:
        ids={int(row['point_id']) for row in csv.DictReader(f)}
    if not {r['point_id'] for r in rows}.issubset(ids): raise ValueError('unknown point ID')
    if any(type(r['converged']) is not bool for r in rows): raise ValueError('invalid convergence flag')
    return write_surface(output, rows, planned)


def inspect_snt(path):
    # Admission check before native interaction allocation. SNT shell degeneracies.
    with path.open(encoding="utf-8-sig") as f:
        lines=(line.split('!')[0].split('#')[0].strip() for line in f)
        lines=(line for line in lines if line)
        header=next(lines).split()
        np_shell, nn_shell, core_p, core_n=map(int,header[:4])
        dims=[0,0]
        for i in range(np_shell+nn_shell):
            fields=next(lines).split()
            dims[i >= np_shell] += int(fields[3])+1
    return dims, (core_p,core_n)


def memory_estimate(dims):
    p,n=dims
    # Conservative solver working-set estimate, no n_ph^2 Hessian storage.
    tensor=8*(p**4+n**4+p*p*n*n)
    workspace=8*100*(p*p+n*n)+64*1048576
    return dict(interaction_mb=tensor/1048576, solver_allowance_mb=workspace/1048576,
                estimated_mb=(tensor+workspace)/1048576)


def grid(args):
    if args.beta is not None:
        match=re.fullmatch(r"(?:[A-Za-z]+(\d+)|(\d+)[A-Za-z]+)",args.nucleus)
        if not match: raise ValueError("nucleus must be an isotope such as Mg24 or 24Mg")
        mass=int(next(x for x in match.groups() if x))
        b2=41.47106/args.hw
        scale=3*mass*(1.2*mass**(1/3))**2/(4*math.pi*b2)
        pairs=[]
        if any(b<0 for b in args.beta) or any(not 0<=g<=60 for g in args.gamma):
            raise ValueError("require beta >= 0 and gamma in [0,60]")
        for row,b in enumerate(sorted(set(args.beta))):
            angles=[0.] if b==0 else sorted(set(args.gamma),reverse=bool(row%2))
            for g in angles:
                pairs.append(dict(beta=b,gamma_deg=g,q0=scale*b*math.cos(math.radians(g)),
                                  q2=math.sqrt(2)*scale*b*math.sin(math.radians(g))))
    else:
        pairs=[dict(beta=None,gamma_deg=None,q0=q0,q2=q2) for row,q0 in enumerate(sorted(set(args.q0)))
               for q2 in sorted(set(args.q2),reverse=bool(row%2))]
    result=[dict(point_id=i,**p) for i,p in enumerate(pairs)]
    return result[::-1] if args.reverse else result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interaction',type=Path,default=INTERACTION)
    p.add_argument('--nucleus',default=NUCLEUS)
    p.add_argument('--hw',type=float,default=HW_MEV)
    p.add_argument('--q0',nargs='+',type=float,default=Q0_VALUES)
    p.add_argument('--q2',nargs='+',type=float,default=Q2_VALUES)
    p.add_argument('--beta',nargs='+',type=float,default=BETA_VALUES)
    p.add_argument('--native-grid',action='store_true',help='use native Q0/Q2 instead of beta/gamma')
    p.add_argument('--gamma',nargs='+',type=float,default=GAMMA_DEGREES)
    p.add_argument('--jx',type=float,help='target <Jx> in hbar (not J(J+1))')
    p.add_argument('--jz',type=float,help='target <Jz> in hbar')
    p.add_argument('--free-axes',action='store_true',help='omit the real Q21=0 principal-axis constraint')
    p.add_argument('--memory-mb',type=float,default=MEMORY_MB)
    p.add_argument('--threads',type=int,default=THREADS)
    p.add_argument('--max-iterations',type=int,default=MAX_ITERATIONS)
    p.add_argument('--method',choices=('hybrid','gradient'),default='hybrid')
    p.add_argument('--seed',type=int,default=520)
    p.add_argument('--reverse',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'Output/hybrid_pes')
    p.add_argument('--check',action='store_true')
    p.add_argument('--recover-surface',type=Path)
    args=p.parse_args(argv)
    if args.recover_surface:
        print(json.dumps(recover_surface(args.recover_surface),indent=2)); return 0
    if args.native_grid: args.beta=None
    values=[args.hw,args.memory_mb,*args.q0,*args.q2,*args.gamma,*(args.beta or []),
            *([] if args.jx is None else [args.jx]),*([] if args.jz is None else [args.jz])]
    if not all(math.isfinite(v) for v in values) or min(args.hw,args.memory_mb,args.threads,args.max_iterations)<=0:
        raise ValueError('finite settings and positive hw, memory, threads and iterations required')
    if any(int(os.environ.get(k,'1'))>1 for k in ('OMPI_COMM_WORLD_SIZE','PMI_SIZE','SLURM_NTASKS')):
        raise ValueError('run one process; use --threads for BLAS parallelism')
    args.interaction=args.interaction.expanduser().resolve()
    dims,core=inspect_snt(args.interaction)
    estimate=memory_estimate(dims)
    if estimate['estimated_mb']>args.memory_mb:
        raise MemoryError(f"estimated {estimate['estimated_mb']:.1f} MiB exceeds --memory-mb {args.memory_mb:g}")
    points=grid(args)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(estimate, dimensions=dims, inert_core=core, planned_points=len(points),
                    units='Q0=r^2 Y20/b^2; Q2=r^2 (Y22+Y2,-2)/b^2; b^2=41.47106/hw fm^2',
                    beta_convention='full isotope A and R0=1.2 A^(1/3); active-space mass moments only')
    if args.check:
        print(json.dumps(dict(settings=settings,grid=points),indent=2)); return 0
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]=str(args.threads)
    sys.path.insert(0,str(ROOT))
    import numpy as np
    import pyHFAndHFB as native
    from hybrid_hf import Solver, Options
    native.set_hybrid_threads(args.threads)
    ms=native.ModelSpace(); ms.Set_hw(args.hw); ms.Set_RefString(args.nucleus)
    h=native.Hamiltonian(ms); h.SetMemoryLimitMB(args.memory_mb-estimate['solver_allowance_mb'])
    rw=native.ReadWriteFiles()
    # Preserve paths containing spaces: the legacy combined reader strips them.
    rw.ReadTokyo(str(args.interaction),ms,h)
    # GetAZfromString is internal; expose a checked constructor below via the module.
    native.set_hybrid_nucleus(ms,args.nucleus)
    ms.InitialModelSpace_HF(); h.Prepare_MschemeH_Unrestricted()
    hf=native.HartreeFock(h)
    active=[0,1]+([] if args.jx is None else [2])+([] if args.jz is None else [3])+([] if args.free_axes else [4])
    opt=Options(max_iterations=args.max_iterations,method=args.method,seed=args.seed)
    solver=Solver(hf,active=active,options=opt)
    output=args.output.expanduser().resolve(); output.mkdir(parents=True,exist_ok=False)
    atomic_write(output/'settings.json',json.dumps(settings,indent=2)+'\n')
    atomic_write(output/'grid.csv',csv_text(points))
    rows=[]; last_good=None; seed_point=None
    with (output/'points.jsonl').open('x',encoding='utf-8') as journal:
        for point in points:
            target=[point['q0'],point['q2']]+([] if args.jx is None else [args.jx])+([] if args.jz is None else [args.jz])+([] if args.free_axes else [0.])
            history=None; writer=None
            if SAVE_ITERATION_HISTORY:
                history=(output/f"history_{point['point_id']:04d}.csv").open('x',newline='')
                writer=csv.DictWriter(history,fieldnames=['attempt','iteration','phase','energy_MeV','gradient_norm','constraint_error','energy_change_MeV']); writer.writeheader()
            begin=time.perf_counter(); calls=responses=0
            try:
                for attempt in range(1,3 if RETRY_FAILED else 2):
                    solver.options.diagonalization_steps=6 if last_good is None or attempt>1 else 0
                    solver.options.seed=args.seed+attempt-1
                    def record(row):
                        if writer: writer.writerow(dict(attempt=attempt,**row)); history.flush()
                    result=solver.solve(target,start=last_good if attempt==1 else None,callback=record)
                    calls+=result['fock_evaluations']; responses+=result['hessian_evaluations']
                    if result['converged'] or result['status']=='infeasible_target': break
            finally:
                if history: history.close()
            state=result.pop('occupied'); moments=result.pop('moments')
            row=dict(point,**result,seconds=time.perf_counter()-begin,attempts=attempt,seed_point_id=seed_point if attempt==1 else None,
                     q0_actual=moments[0],q2_actual=moments[1],jx=moments[2],jz=moments[3],q21_real=moments[4])
            row['fock_evaluations']=calls; row['hessian_evaluations']=responses
            row={k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in row.items()}
            journal.write(json.dumps(row,allow_nan=False)+'\n'); journal.flush(); os.fsync(journal.fileno())
            rows.append(row); write_surface(output,rows,len(points))
            if result['converged']: last_good=state; seed_point=point['point_id']
            print(f"point {point['point_id']}: {result['status']}, E={result['energy_MeV']}, iterations={result['iterations']}",flush=True)
    return 0 if all(r['converged'] for r in rows) else 2


if __name__=='__main__':
    raise SystemExit(main())
