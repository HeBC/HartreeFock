"""Constraint-safe preconditioning and the previously slow random O16 case."""
import unittest
import numpy as np
from scipy import linalg
from test_hybrid import load,ROOT,Solver,Options
from hf_input import parse_text
from scan_gcm_hf import load_problem,read_config,target_points
from benchmark_random_starts import random_determinant


class PreconditionerTests(unittest.TestCase):
    def test_projected_metric_is_positive_symmetric_and_feasible(self):
        s=Solver(load(),active=(0,1,4));rng=np.random.default_rng(81)
        c=s.retract(s.initial,.1*s.tangent(s.initial,rng.normal(size=s.size)))
        _,f=s.evaluate(c);_,lam,u=s.stationarity(c,f)
        pre=s.prepare_preconditioner(c,f,lam)
        def project(x):
            x=s.tangent(c,x)
            return x-u@(u.T@x)
        a,b=(project(rng.normal(size=s.size)) for _ in range(2))
        ma=s.precondition_residual(c,a,u,pre);mb=s.precondition_residual(c,b,u,pre)
        self.assertGreater(np.dot(a,ma),0.)
        self.assertAlmostEqual(np.dot(a,mb),np.dot(b,ma),places=12)
        np.testing.assert_allclose(ma,project(ma),atol=2e-13)
        np.testing.assert_allclose(u.T@ma,0,atol=2e-13)

    def test_empty_and_full_species_have_no_singular_gap_blocks(self):
        for nucleus in ('O16','Ca40','O18','S36'):
            s=Solver(load(nucleus),active=());c=s.initial
            _,f=s.evaluate(c);g,lam,u=s.stationarity(c,f)
            pre=s.prepare_preconditioner(c,f,lam)
            r=s.tangent(c,np.random.default_rng(2).normal(size=s.size))
            z=s.precondition_residual(c,r,u,pre)
            self.assertTrue(np.isfinite(z).all())
            for x,w,p in zip(c,s.unpack(z),pre):
                if x.shape[1] in (0,x.shape[0]):
                    self.assertIsNone(p);np.testing.assert_allclose(w,0,atol=1e-14)

    def test_input_controls_and_invalid_floor(self):
        config,_=parse_text('''[calculation]
nucleus=Mg24
interaction=usda.snt
hw=16
[constraints]
Q20=1
[solver]
precondition=no
precondition_floor=0.2
''')
        self.assertFalse(config['solver']['precondition'])
        self.assertEqual(config['solver']['precondition_floor'],.2)
        for floor in (0,-1,float('nan')):
            with self.assertRaises(ValueError): Options(precondition_floor=floor).validate()
        with self.assertRaises(ValueError): Options(precondition='yes').validate()

    def test_hard_random_o16_reaches_low_branch_without_long_tail(self):
        config=read_config(ROOT/'examples/gcm/multishell_o16.inp')
        config['solver']['max_iterations']=150
        solver,_=load_problem(config)
        start=random_determinant(solver.shapes,1009)
        result=solver.solve(target_points(config)[0],start=start)
        self.assertTrue(result['converged'],result['status'])
        self.assertTrue(result['stability_checked'])
        self.assertLess(result['energy_MeV'],-186.53)
        self.assertLess(result['gradient_norm'],1e-6)
        self.assertLess(result['max_constraint_error'],1e-8)
        self.assertLess(result['hessian_evaluations'],3000)
        self.assertGreater(result['preconditioner_evaluations'],0)
        self.assertGreater(result['cg_iterations'],0)


if __name__=='__main__': unittest.main()
