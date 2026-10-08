"""Physics and format checks for density-dependent multipole constraints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
from scipy import linalg,sparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'PythonScript')]
import pyHFAndHFB as native
from hf_operators import HFOperator,builtin_operator,tensor_snt_operator,load_operator
from hybrid_hf import Solver,Options
from scan_gcm_hf import export_gcm,target_points


def load(nucleus='Mg24',interaction='usda.snt',memory=512.):
    native.set_hybrid_threads(1)
    ms=native.ModelSpace();ms.Set_hw(16.)
    h=native.Hamiltonian(ms);h.SetMemoryLimitMB(memory)
    native.ReadWriteFiles().ReadTokyo(str(ROOT/'Interaction'/interaction),ms,h)
    native.set_hybrid_nucleus(ms,nucleus);ms.InitialModelSpace_HF();h.Prepare_MschemeH_Unrestricted()
    return native.HartreeFock(h)


class SmallBasis:
    def hybrid_basis(self):
        return tuple(np.array([[0,0,1,-1,tz],[0,0,1,1,tz]]) for tz in (-1,1))


class EvolvedConstraints(unittest.TestCase):
    def test_tensor_rank_two_pair_component(self):
        # A pure pn spin-triplet rank-2 tensor. Standard WE coefficient gives
        # <up,up|T22|down,down> = RME/sqrt(5). Product x-spinors therefore
        # have <T22+T2,-2> = RME/(2 sqrt(5)), independently of the converter.
        text='''! Zero body term: 0
1 1 0 0
1 0 0 1 -1
2 0 0 1 1
0 0 16
1 0 16
1 2 1 2 1 1 1.0
'''
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rank2.snt';path.write_text(text)
            op=tensor_snt_operator(path,SmallBasis(),name='Q22',rank=2,mu=2,parity=0,normal_ordering='valence_vacuum',units='1')
        rho=np.full((2,2),.5)
        self.assertAlmostEqual(op.evaluate((rho,rho))[0],1/(2*np.sqrt(5)),places=12)

    def test_pair_normalization_exchange_and_pn(self):
        # V = g N(N-1)/2 on four single-particle states, independent of spin/isospin.
        # Normalized pair RMEs are sqrt(2J+1)*g; pp and nn J=0, pn J=0,1.
        g=1.7
        text=f'''! Zero body term: 0
1 1 0 0
1 0 0 1 -1
2 0 0 1 1
0 0 16
4 0 16
1 1 1 1 0 0 {g}
2 2 2 2 0 0 {g}
1 2 1 2 0 0 {g}
1 2 1 2 1 1 {np.sqrt(3)*g}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'number_pairs.snt';path.write_text(text)
            op=tensor_snt_operator(path,SmallBasis(),name='pairs',rank=0,mu=0,parity=0,normal_ordering='valence_vacuum',units='1')
        p=np.array([[.2,.12],[.12,.7]]);n=np.array([[.4,-.1],[-.1,.3]])
        value,fields=op.evaluate((p,n))
        expected=g*(np.linalg.det(p)+np.linalg.det(n)+np.trace(p)*np.trace(n))
        self.assertAlmostEqual(value,expected,places=12)
        total=np.trace(p)+np.trace(n)
        for field,rho in zip(fields,(p,n)): np.testing.assert_allclose(field,g*(total*np.eye(2)-rho),atol=1e-12)
        # A determinant with one particle has identically zero two-body expectation.
        self.assertAlmostEqual(op.evaluate((np.diag([1.,0.]),np.zeros((2,2))))[0],0.,places=12)

    def test_standard_quadrupole_normalization(self):
        hf=load();q=builtin_operator(hf,'Q20',hw=16.,units='oscillator')
        for basis,op,native_ops in zip(hf.hybrid_basis(),q.one_body,hf.hybrid_operators()):
            # |0d5/2,m=5/2>: <r²Y20>/b² = -sqrt(5)/(2sqrt(pi)).
            i=next(i for i,row in enumerate(basis) if tuple(row[:4])==(0,2,5,5))
            self.assertAlmostEqual(op[i,i],-np.sqrt(5)/(2*np.sqrt(np.pi)),places=12)
            np.testing.assert_allclose(op,native_ops[0],atol=2e-12)
        q22=builtin_operator(hf,'Q22',hw=16.,units='oscillator')
        for q,n in zip(q22.one_body,hf.hybrid_operators()): np.testing.assert_allclose(q,n[1],atol=2e-12)

    def test_parity_odd_and_cranking_operators(self):
        hf=load('O17','FCI_HF_FCI_O17_e3_hw16_E39.snt')
        operators=[builtin_operator(hf,name,hw=16.) for name in ('Q20','Q22','Q10','Q30','Jx','Jz')]
        for op in operators:
            self.assertGreater(sum(np.linalg.norm(x) for x in op.one_body),1e-4)
            for matrix in op.one_body: np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
        for name in ('Q10','Q30'):
            odd=builtin_operator(load(),name,hw=16.)
            self.assertEqual(sum(np.linalg.norm(x) for x in odd.one_body),0.)
            r=Solver(load(),constraints=[odd]).solve({name:1.})
            self.assertEqual(r['status'],'infeasible_target')

    def test_shifted_reference_value_gradient_and_npz(self):
        hf=load();bare=builtin_operator(hf,'Q20',hw=16.)
        rng=np.random.default_rng(81);dims=bare.dims;size=bare.size
        b=rng.normal(size=(size,3))*.003;k=sparse.csr_matrix(b@b.T)
        # An arbitrary self-adjoint kernel must be restricted to symmetric
        # fields by the public API, just as the orbital derivative requires.
        ref=tuple(np.eye(d)*.25 for d in dims)
        op=HFOperator('custom',bare.one_body,k,2.3,ref,hf.hybrid_basis(),dict(basis_representation='HO'))
        np.testing.assert_allclose(op.evaluate(ref)[0],2.3,atol=1e-14)
        for a,b in zip(op.evaluate(ref)[1],bare.one_body): np.testing.assert_allclose(a,b,atol=1e-14)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'op.npz';op.save(path);again=load_operator(path,hf)
            for a,b in zip(op.evaluate(ref)[1],again.evaluate(ref)[1]): np.testing.assert_array_equal(a,b)
            with self.assertRaises(MemoryError): load_operator(path,hf,memory_mb=.001)
        self.check_derivatives(hf,[op])

    def check_derivatives(self,hf,operators):
        s=Solver(hf,constraints=operators);rng=np.random.default_rng(67)
        c=s.retract(s.initial,s.tangent(s.initial,rng.normal(size=s.size))*.07)
        v=s.tangent(c,rng.normal(size=s.size));v/=linalg.norm(v)
        eps=2e-6;plus=s.retract(c,eps*v);minus=s.retract(c,-eps*v)
        values,fields=s.constraint_data(c)
        for k in range(len(operators)):
            gradient=s.tangent(c,s.pack([2*f[k]@x for x,f in zip(c,fields)]))
            derivative=(s.moments(plus)[k]-s.moments(minus)[k])/(2*eps)
            self.assertAlmostEqual(derivative,np.dot(gradient,v),places=7)
        energy,f=s.evaluate(c);_,fp=s.evaluate(plus);_,fm=s.evaluate(minus)
        lam=np.linspace(.17,-.23,len(operators))
        def gradient(x,fields):
            cq=s.constraint_data(x)[1]
            return s.tangent(x,s.pack([2*(a+np.einsum('k,kab->ab',lam,q))@cc for a,q,cc in zip(fields,cq,x)]))
        numerical=s.tangent(c,(gradient(plus,fp)-gradient(minus,fm))/(2*eps))
        np.testing.assert_allclose(s.hessian(c,f,lam,v),numerical,atol=5e-7,rtol=5e-7)
        w=s.tangent(c,rng.normal(size=s.size));w/=linalg.norm(w)
        self.assertAlmostEqual(np.dot(w,s.hessian(c,f,lam,v)),np.dot(v,s.hessian(c,f,lam,w)),places=9)

    def test_imsrg_evolved_two_body_constraints(self):
        directory=ROOT/'Interaction/hybrid_evolved_e2'
        if not directory.exists(): self.skipTest('generate the emax=2 IMSRG benchmark inputs first')
        hf=load('Ne20','hybrid_evolved_e2/H_IMSRG2_HO.snt')
        ops=[tensor_snt_operator(directory/'Qmass_IMSRG2_HO.snt',hf,name=name,rank=2,mu=mu,parity=0,
                                 normal_ordering='core',units='fm^2') for name,mu in [('Q20',0),('Q22',2)]]
        self.assertTrue(all(op.kernel.nnz>0 for op in ops));self.check_derivatives(hf,ops)
        bare=tensor_snt_operator(directory/'Qmass_bare_HO.snt',hf,name='bare',rank=2,mu=0,parity=0,normal_ordering='core',units='fm^2')
        builtin=builtin_operator(hf,'Q20',hw=16.)
        for a,b in zip(bare.one_body,builtin.one_body): np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)
        result=Solver(hf,constraints=ops,options=Options(max_iterations=600)).solve({'Q20':1.,'Q22':.5})
        self.assertTrue(result['converged'],result['status']);self.assertTrue(result['stability_checked'])
        self.assertLess(result['max_constraint_error'],1e-8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.dat';export_gcm(path,result['occupied'],result['energy_MeV'])
            lines=path.read_text().splitlines();self.assertEqual(tuple(map(int,lines[0].split()[:4])),(2,12,2,12))
            numbers=np.array([float(line.split()[1]) for line in lines[1:]])
            np.testing.assert_allclose(numbers,np.concatenate([x.T.ravel() for x in result['occupied']]),atol=1e-15)

    def test_grid_validation(self):
        with self.assertRaises(ValueError): target_points({'constraints':[dict(name='Q20',values=[0,0])]})
        with self.assertRaises(ValueError): target_points({'constraints':[dict(name='Q20',values=list(range(200)))], 'max_points':10})
        with self.assertRaises(ValueError): target_points({'constraints':[dict(name='Q20')], 'points':[dict(Q22=1)]})


if __name__=='__main__': unittest.main()
