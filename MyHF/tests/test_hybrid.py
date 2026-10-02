"""Run from MyHF: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -v."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy import linalg
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'PythonScript')]
import pyHFAndHFB as native
from hybrid_hf import Solver, Options

def load(nucleus='Mg24', memory=128., interaction='usda.snt'):
    native.set_hybrid_threads(1)
    ms=native.ModelSpace(); ms.Set_hw(16.)
    h=native.Hamiltonian(ms); h.SetMemoryLimitMB(memory)
    rw=native.ReadWriteFiles()
    rw.Read_KShell_HF_input(str(ROOT/'Interaction'/interaction),ms,h,nucleus)
    return native.HartreeFock(h)

class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.hf=load()
        self.s=Solver(self.hf,options=Options(max_iterations=300))
        rng=np.random.default_rng(17)
        self.c=self.s.retract(self.s.initial,self.s.tangent(self.s.initial,rng.normal(size=self.s.size))*.1)

    def test_native_contraction_and_variational_derivatives(self):
        s,c=self.s,self.c
        e,f=s.evaluate(c)
        ref=self.hf.hybrid_reference(*(x@x.T for x in c))
        for a,b in zip(f,ref): np.testing.assert_allclose(a,b,atol=2e-12)
        rng=np.random.default_rng(82)
        v=s.tangent(c,rng.normal(size=s.size)); v/=linalg.norm(v)
        eps=1e-5
        plus,minus=s.retract(c,eps*v),s.retract(c,-eps*v)
        ep,fp=s.evaluate(plus); em,fm=s.evaluate(minus)
        grad=s.tangent(c,s.pack([2*a@x for a,x in zip(f,c)]))
        self.assertAlmostEqual((ep-em)/(2*eps),np.dot(grad,v),places=7)
        for k in range(len(s.active)):
            jac=s.tangent(c,s.pack([2*q[k]@x for q,x in zip(s.ops,c)]))
            self.assertAlmostEqual((s.moments(plus)[k]-s.moments(minus)[k])/(2*eps),np.dot(jac,v),places=8)
        lam=np.array([.3,-.2])
        def gradient(x,fields):
            return s.tangent(x,s.pack([2*(a+np.einsum('k,kab->ab',lam,q))@cc for a,q,cc in zip(fields,s.ops,x)]))
        numerical=s.tangent(c,(gradient(plus,fp)-gradient(minus,fm))/(2*eps))
        np.testing.assert_allclose(s.hessian(c,f,lam,v),numerical,atol=3e-7,rtol=3e-7)
        w=s.tangent(c,rng.normal(size=s.size)); w/=linalg.norm(w)
        self.assertAlmostEqual(np.dot(w,s.hessian(c,f,lam,v)),np.dot(v,s.hessian(c,f,lam,w)),places=9)

    def test_feasible_hybrid_and_continuation(self):
        result=self.s.solve([1.,.5],start=self.c)
        self.assertTrue(result['converged'],str({k:v for k,v in result.items() if k!='occupied'}))
        self.assertLess(result['max_constraint_error'],1e-8)
        self.assertLess(result['gradient_norm'],1e-6)
        self.assertLess(result['orthogonality_error'],1e-12)
        self.assertLess(result['idempotency_error'],1e-12)
        self.assertAlmostEqual(result['protons'],4.,places=11)
        self.assertAlmostEqual(result['neutrons'],4.,places=11)
        again=self.s.solve([1.1,.5],start=result['occupied'])
        self.assertTrue(again['converged'],again['status'])
        state=self.hf.hybrid_state()
        for a,b in zip(state,again['occupied']): np.testing.assert_allclose(a@a.T,b@b.T,atol=1e-12)

    def test_failure_is_not_convergence(self):
        impossible=self.s.solve([1e6,0.])
        self.assertEqual(impossible['status'],'infeasible_target')
        before=self.hf.hybrid_state()
        short=Solver(self.hf, options=Options(max_iterations=1)).solve([1.,.5],start=self.c)
        self.assertFalse(short['converged'])
        for a,b in zip(before,self.hf.hybrid_state()): np.testing.assert_array_equal(a,b)

    def test_budget_and_input_validation(self):
        with self.assertRaisesRegex(RuntimeError,'memory_limit_mb'): load(memory=.001)
        with self.assertRaises(ValueError): self.hf.hybrid_response(np.zeros((2,2)),np.zeros((2,2)))
        p,n=(x@x.T for x in self.c); p[0,1]+=1
        with self.assertRaises(ValueError): self.hf.hybrid_evaluate(p,n)

    def test_empty_and_full_species(self):
        for nucleus in ('O18','S36'):
            solver=Solver(load(nucleus),active=(),options=Options(max_iterations=300))
            result=solver.solve([])
            self.assertTrue(result['converged'],str((nucleus,result['status'],result['gradient_norm'])))
            self.assertLess(result['idempotency_error'],1e-12)

    def test_multishell_quadrupole_and_hessian(self):
        self.hf=load('O17',512.,'FCI_HF_FCI_O17_e3_hw16_E39.snt')
        self.s=Solver(self.hf)
        rng=np.random.default_rng(17)
        self.c=self.s.retract(self.s.initial,self.s.tangent(self.s.initial,rng.normal(size=self.s.size))*.01)
        for op in self.hf.hybrid_operators():
            np.testing.assert_allclose(op,op.transpose(0,2,1),atol=1e-12)
        self.test_native_contraction_and_variational_derivatives()

    def test_invalid_isotope_raises_without_terminating(self):
        for name in ('Mg','Foo24','Mg0','Mg99999'):
            with self.assertRaises(ValueError): load(name)

    def test_hessian_escapes_mg24_saddle(self):
        first=Solver(self.hf,active=(0,1,4),options=Options(check_stability=False)).solve([0.,0.,0.])
        phases=[]
        stable=Solver(load(),active=(0,1,4)).solve([0.,0.,0.],callback=lambda row:phases.append(row['phase']))
        self.assertTrue(stable['converged'],stable['status'])
        self.assertTrue(stable['stability_checked'])
        self.assertGreaterEqual(stable['smallest_curvature'],-1e-5)
        self.assertLess(stable['energy_MeV'],first['energy_MeV']-.1)
        self.assertIn('negative_curvature',phases)

if __name__=='__main__': unittest.main()
