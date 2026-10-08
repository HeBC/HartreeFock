"""Small genuine IMSRG(2) evolution; every model space has emax=2 (<4)."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,argparse,hashlib
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--imsrg-build',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'Interaction/hybrid_evolved_e2')
args=parser.parse_args()
sys.path.insert(0,str(args.imsrg_build.resolve()))
from pyIMSRG import ModelSpace, Operator, OperatorFromString, HartreeFock, IMSRGSolver, ReadWrite, Commutator

out=args.output.resolve()
out.mkdir(parents=True,exist_ok=True)
ms=ModelSpace(2,'O16','sd-shell');ms.SetHbarOmega(16.);ms.SetE3max(6)
rw=ReadWrite()
h=OperatorFromString(ms,'VMinnesota')+OperatorFromString(ms,'Trel')
q=OperatorFromString(ms,'E2')+OperatorFromString(ms,'nE2')
rw.WriteTensorTokyo(str(out/'Qmass_bare_HO.snt'),q)
rw.WriteTokyo(h.DoNormalOrderingCore(),str(out/'H_bare_HO.snt'),'')
hf=HartreeFock(h);hf.Solve()
hn=hf.GetNormalOrderedH(2)
qn=hf.TransformToHFBasis(q).DoNormalOrdering()
Commutator.SetUseIMSRG3(False)
flow=IMSRGSolver(hn);flow.SetMethod('magnus');flow.SetGenerator('atan')
flow.SetSmax(.2);flow.SetDs(.01);flow.SetDsmax(.05);flow.SetEtaCriterion(1e-6)
flow.Solve()
# Export matched H/Q in HO: full-space vacuum and sd-shell core normal ordering.
he_full=hf.TransformToHOBasis(flow.GetH_s().UndoNormalOrdering())
qe_full=hf.TransformToHOBasis(flow.Transform(qn).UndoNormalOrdering())
he=he_full.DoNormalOrderingCore()
qe=qe_full.DoNormalOrderingCore()
rw.WriteTokyo(he,str(out/'H_IMSRG2_HO.snt'),'')
rw.WriteTensorTokyo(str(out/'Qmass_IMSRG2_HO.snt'),qe)
fullspace=ModelSpace(2,'O16','FCI');fullspace.SetHbarOmega(16.);fullspace.SetE3max(6)
he_full.SetModelSpace(fullspace);qe_full.SetModelSpace(fullspace)
rw.WriteTokyo(he_full,str(out/'H_IMSRG2_full_HO.snt'),'')
rw.WriteTensorTokyo(str(out/'Qmass_IMSRG2_full_HO.snt'),qe_full)
info=dict(emax=2,reference='O16',valence='sd-shell',hw_MeV=16.,interaction='Minnesota + intrinsic kinetic energy',
          operator='E2 + nE2 = mass quadrupole, unit proton and neutron weights',basis='HO',normal_ordering='core',
          purpose='short-flow interface/derivative benchmark; not a fully decoupled production interaction',smax=.2,
          two_body_norm=float(qe.TwoBodyNorm()),
          imsrg_module_sha256=hashlib.sha256((args.imsrg_build/'pyIMSRG.so').read_bytes()).hexdigest())
assert info['two_body_norm']>1e-8, 'Need a nonzero induced two-body operator for validation'
(out/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
