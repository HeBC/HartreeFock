#!/usr/bin/env python3
"""Named multipole/evolved-operator HF generator states for PES and GCM.

python3 PythonScript/scan_gcm_hf.py examples/gcm/usda_mg24.inp
Targets have each operator's declared units; electric E2 is never mapped to mass beta automatically.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,csv,hashlib,itertools,json,math,re,sys,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pyHFAndHFB as native
from hybrid_hf import Solver,Options
from hf_operators import builtin_operator,load_operator,tensor_snt_operator
from scan_hybrid_pes import inspect_snt,memory_estimate,atomic_write,csv_text


from hf_input import read_config, read_job


def target_points(config):
    specs=config['constraints'];names=[s['name'] for s in specs]
    if len(set(names))!=len(names) or not names: raise ValueError('need unique nonempty constraint names')
    if 'points' in config:
        points=config['points']
        if any(set(p)!=set(names) for p in points): raise ValueError('explicit points must specify every constraint')
    else:
        values=[s.get('values',[0.]) for s in specs]
        count=math.prod(len(v) for v in values)
        if count>config.get('max_points',10000): raise ValueError('Cartesian grid exceeds max_points; provide an explicit path')
        points=[dict(zip(names,x)) for x in itertools.product(*values)]
    if not points or len(points)>config.get('max_points',10000): raise ValueError('empty or oversized grid')
    if any(not math.isfinite(float(v)) for p in points for v in p.values()): raise ValueError('nonfinite target')
    if len({tuple(p[n] for n in names) for p in points})!=len(points): raise ValueError('duplicate target point')
    return points


def load_problem(config):
    if any(int(os.environ.get(k,'1'))>1 for k in ('OMPI_COMM_WORLD_SIZE','PMI_SIZE','SLURM_NTASKS')):
        raise ValueError('run one process, not mpirun')
    hw=float(config['hw_MeV']);budget=float(config.get('memory_mb',512.))
    if not math.isfinite(hw) or not math.isfinite(budget) or min(hw,budget)<=0: raise ValueError('invalid hw or memory budget')
    dims,core=inspect_snt(Path(config['interaction']));estimate=memory_estimate(dims)
    if estimate['estimated_mb']>=budget: raise MemoryError('Hamiltonian and solver exceed memory budget')
    native.set_hybrid_threads(1)
    ms=native.ModelSpace();ms.Set_hw(hw)
    ham=native.Hamiltonian(ms);ham.SetMemoryLimitMB(budget-estimate['solver_allowance_mb'])
    rw=native.ReadWriteFiles();rw.ReadTokyo(config['interaction'],ms,ham)
    native.set_hybrid_nucleus(ms,config['nucleus']);ms.InitialModelSpace_HF();ham.Prepare_MschemeH_Unrestricted()
    hf=native.HartreeFock(ham)
    operators=[];resident=estimate['estimated_mb'];representation=config.get('basis_representation','HO')
    for spec in config['constraints']:
        opname=spec['name'];definition=dict(spec.get('operator',{}));kind=definition.pop('type','builtin')
        if kind=='builtin':
            builtin=definition.pop('name',opname)
            if representation!='HO' and builtin not in ('Jx','Jz'):
                raise ValueError('bare radial multipoles need HO basis; supply transformed operators for HF/NAT Hamiltonians')
            operator=builtin_operator(hf,builtin,hw=hw,**definition);operator.name=opname;operator.metadata['name']=opname
        elif kind=='tensor_snt':
            op_rep=definition.pop('basis_representation',None)
            if op_rep!=representation: raise ValueError('declare matching operator/Hamiltonian basis representations')
            reference=None
            if 'reference_npz' in definition:
                reference_path=definition.pop('reference_npz')
                with zipfile.ZipFile(reference_path) as archive:
                    if 4*sum(item.file_size for item in archive.infolist())>(budget-resident)*1048576:
                        raise MemoryError('reference-density archive exceeds memory budget')
                with np.load(reference_path,allow_pickle=False) as data:
                    reference=(data['p'],data['n'])
            operator=tensor_snt_operator(definition.pop('path'),hf,name=opname,representation=op_rep,
                                         reference=reference,memory_mb=budget-resident,**definition)
            if (operator.metadata['core_protons'],operator.metadata['core_neutrons'])!=tuple(core):
                raise ValueError('operator/Hamiltonian inert cores differ')
        elif kind=='npz':
            operator=load_operator(definition.pop('path'),hf,name=opname,memory_mb=budget-resident)
            if definition: raise ValueError(f'unknown NPZ settings: {definition}')
            if operator.metadata.get('basis_representation')!=representation:
                raise ValueError('operator/Hamiltonian basis representations differ')
        else: raise ValueError(f'unknown operator type {kind}')
        operators.append(operator);resident+=operator.storage_bytes/1048576
        if resident>budget: raise MemoryError('constraints exceed aggregate memory budget')
    options=Options(**config.get('solver',{}));solver=Solver(hf,options=options,constraints=operators)
    import re
    match=re.search(r'Zero body term:\s*([-+\d.eEdD]+)',Path(config['interaction']).read_text())
    offset=float(match.group(1).replace('D','E')) if match else 0.
    return solver,dict(estimated_resident_MiB=resident,budget_MiB=budget,threads=1,zero_body_MeV=offset,
                       operator_metadata=[op.metadata for op in operators])


def export_gcm(path,state,energy):
    """Legacy Read_GCM_HF_points format: occupied-orbital-major, p then n."""
    p,n=state
    values=np.concatenate((p.T.ravel(),n.T.ravel()))
    text=f'{p.shape[1]} {p.shape[0]} {n.shape[1]} {n.shape[0]} {energy:.16g}\n'
    text+=''.join(f'{i} {v:.17g}\n' for i,v in enumerate(values))
    atomic_write(path,text)


def run(config,output,passes=('forward','reverse'),resume=False):
    points=target_points(config)
    solver,resources=load_problem(config)
    hashes={config['interaction']:hashlib.sha256(Path(config['interaction']).read_bytes()).hexdigest()}
    for spec in config['constraints']:
        for key in ('path','reference_npz'):
            path=spec.get('operator',{}).get(key)
            if path: hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for relative in ('pyHFAndHFB.so','PythonScript/hybrid_hf.py','PythonScript/hf_operators.py','PythonScript/scan_gcm_hf.py','PythonScript/hf_input.py'):
        code=ROOT/relative
        hashes[str(code)]=hashlib.sha256(code.read_bytes()).hexdigest()
    signature=hashlib.sha256(json.dumps(dict(config=config,hashes=hashes),sort_keys=True).encode()).hexdigest()
    output=Path(output).resolve()
    if output.exists():
        if not resume: raise FileExistsError('output exists; use --resume only for matching inputs')
        settings=json.loads((output/'settings.json').read_text())
        if settings['signature']!=signature: raise ValueError('resume inputs/grid/operator hashes differ')
    else:
        output.mkdir(parents=True)
        settings=dict(config=config,resources=resources,hashes=hashes,signature=signature,
                      solver_sha256=hashlib.sha256((ROOT/'PythonScript/hybrid_hf.py').read_bytes()).hexdigest())
        atomic_write(output/'settings.json',json.dumps(settings,indent=2)+'\n')
    for folder in ('states','gcm_basis','history'): (output/folder).mkdir(exist_ok=True)
    selected={};done=set()
    if (output/'progress.json').exists():
        saved=json.loads((output/'progress.json').read_text());done=set(saved['done'])
        selected={int(k):v for k,v in saved['selected'].items()}
    def state(pid):
        with np.load(output/'states'/f'{pid:05d}.npz',allow_pickle=False) as f: return f['p'].copy(),f['n'].copy()
    for phase in passes:
        sequence=list(range(len(points)))
        if phase=='reverse': sequence.reverse()
        last=None
        for pid in sequence:
            if f'{phase}:{pid}' in done:
                if selected.get(pid,{}).get('converged'): last=state(pid)
                continue
            best=None;begin=time.perf_counter()
            candidates=[last]
            if selected.get(pid,{}).get('converged'): candidates.append(state(pid))
            candidates.append(None)
            for attempt,start in enumerate(candidates):
                if attempt and start is None and candidates[0] is None: continue
                solver.options.diagonalization_steps=6 if start is None else 0
                solver.options.seed=int(config.get('solver',{}).get('seed',520))+attempt+(100 if phase=='reverse' else 0)
                with (output/'history'/f'{phase}_{pid:05d}_{attempt}.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['iteration','phase','energy_MeV','gradient_norm','constraint_error','energy_change_MeV'])
                    writer.writeheader()
                    def callback(row): writer.writerow(row);stream.flush()
                    result=solver.solve(points[pid],start=start,callback=callback)
                occupied=result.pop('occupied');result.pop('moments')
                result={k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in result.items()}
                result.update(point_id=pid,targets=points[pid],pass_name=phase,attempt=attempt,
                              elapsed_seconds=time.perf_counter()-begin,
                              energy_total_MeV=None if result['energy_MeV'] is None else result['energy_MeV']+resources['zero_body_MeV'])
                with (output/'attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                best=(result,occupied)
                if result['converged']: break
            previous=selected.get(pid)
            if best[0]['converged'] and (previous is None or not previous['converged'] or best[0]['energy_MeV']<previous['energy_MeV']):
                selected[pid]=best[0]
                temp=output/'states'/f'{pid:05d}.tmp.npz'
                np.savez(temp,p=best[1][0],n=best[1][1]);os.replace(temp,output/'states'/f'{pid:05d}.npz')
                export_gcm(output/'gcm_basis'/f'{pid:05d}.dat',best[1],best[0]['energy_MeV'])
            elif previous is None or not previous['converged']: selected[pid]=best[0]
            if selected[pid]['converged']: last=state(pid)
            done.add(f'{phase}:{pid}')
            atomic_write(output/'progress.json',json.dumps(dict(done=sorted(done),selected=selected),indent=2,allow_nan=False)+'\n')
            accepted=[r['energy_total_MeV'] for r in selected.values() if r['converged']]
            minimum=min(accepted) if accepted else None
            rows=[]
            for i,row in sorted(selected.items()):
                flat={k:v for k,v in row.items() if k not in ('targets','constraint_values')}
                flat.update({f'target_{k}':v for k,v in row['targets'].items()})
                flat.update({f'actual_{k}':v for k,v in row['constraint_values'].items()})
                flat['accepted_energy_MeV']=row['energy_total_MeV'] if row['converged'] else None
                flat['relative_energy_MeV']=row['energy_total_MeV']-minimum if row['converged'] else None
                rows.append(flat)
            atomic_write(output/'surface.csv',csv_text(rows))
            atomic_write(output/'summary.json',json.dumps(dict(points=len(rows),planned=len(points),accepted=len(accepted),
                         failed=len(rows)-len(accepted),minimum_MeV=minimum,completed_attempts=len(done)),indent=2)+'\n')
            print(f'{phase} {pid+1}/{len(points)}: {selected[pid]["status"]}',flush=True)
    return 0 if len(selected)==len(points) and all(r['converged'] for r in selected.values()) else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file',nargs='?',type=Path,help='readable .inp calculation file')
    parser.add_argument('--input','--config',dest='config',type=Path,help='input file (legacy .json also accepted)')
    parser.add_argument('--output',type=Path,help='override the input file output directory; relative to the current directory')
    parser.add_argument('--passes',nargs='+',choices=['forward','reverse'],help='override the input file scan directions')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check',action='store_true',help='show the calculation, targets and memory without solving')
    parser.add_argument('--check-json',action='store_true',help='machine-readable version of --check')
    args=parser.parse_args(argv)
    if bool(args.input_file)==bool(args.config):
        parser.error('supply one input file, either positional or with --input/--config')
    path=args.input_file or args.config
    try:
        config,execution=read_job(path)
        output=args.output or execution.get('output')
        passes=args.passes or execution.get('passes',('forward','reverse'))
        if not args.check and not args.check_json and output is None:
            parser.error('set output in [calculation] or supply --output')
        points=target_points(config)
        if args.check or args.check_json:
            solver,resources=load_problem(config)
            if args.check_json:
                print(json.dumps(dict(points=points,resources=resources),indent=2))
            else:
                print(f"Calculation: {config['nucleus']}   hw = {config['hw_MeV']:g} MeV   basis = {config.get('basis_representation','HO')}")
                print(f"Interaction: {config['interaction']}")
                print(f"Solver: {solver.options.method}   maximum iterations: {solver.options.max_iterations}")
                print(f"Memory: estimated resident {resources['estimated_resident_MiB']:.1f} MiB / budget {resources['budget_MiB']:g} MiB")
                print(f"Output: {output or '(not set)'}")
                print(f"Passes: {' '.join(passes)}   distinct points: {len(points)}")
                print('Constraints:')
                for op in solver.constraints:
                    print(f"  {op.name}: {op.metadata.get('units','declared operator units')}")
                names=solver.constraint_names
                print('Point  '+'  '.join(f'{name:>12}' for name in names))
                for i,point in enumerate(points[:20]):
                    print(f"{i+1:5d}  "+'  '.join(f'{point[name]:12.6g}' for name in names))
                if len(points)>20: print(f'... {len(points)-20} further points')
            return 0
        print(f"{config['nucleus']}: {len(points)} points, {' '.join(passes)} -> {output}",flush=True)
        return run(config,output,passes,args.resume)
    except (ValueError,OSError,MemoryError) as error:
        parser.exit(2,f'Input/calculation error: {error}\n')


if __name__=='__main__':
    raise SystemExit(main())
