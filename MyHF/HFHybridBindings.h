#pragma once
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

// Copies across the Python boundary own their memory. No exposed native pointers.
struct HFHybridAccess {
    using Array = pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;
    static pybind11::tuple basis(HartreeFock &h) {
        pybind11::list result;
        for (int s=0;s<2;++s) {
            int d=s?h.dim_n:h.dim_p, iso=s?Neutron:Proton;
            pybind11::array_t<int> rows({d,5});
            for (int i=0;i<d;++i) {
                int o=s?h.modelspace->Get_NeutronOrbitIndexInMscheme(i):h.modelspace->Get_ProtonOrbitIndexInMscheme(i);
                const auto &orb=s?h.modelspace->Orbits_n[o]:h.modelspace->Orbits_p[o];
                rows.mutable_at(i,0)=orb.n; rows.mutable_at(i,1)=orb.l;
                rows.mutable_at(i,2)=orb.j2;
                rows.mutable_at(i,3)=h.modelspace->Get_MSmatrix_2m(iso,i);
                rows.mutable_at(i,4)=s?1:-1;
            }
            result.append(rows);
        }
        return pybind11::make_tuple(result[0],result[1]);
    }
    // Standard spherical harmonics and Wigner-Eckart convention, in b^power.
    // Positive mu returns Q_lmu + (-1)^mu Q_l,-mu (a real Hermitian component).
    static pybind11::tuple multipole(HartreeFock &h,int rank,int mu,int power) {
        if(rank<0 || rank>8 || mu<0 || mu>rank || power<0 || power>12)
            throw std::invalid_argument("invalid multipole rank, component or radial power");
        pybind11::list result;
        auto phase=[](int k){return k%2 ? -1.:1.;};
        for(int s=0;s<2;++s) {
            int d=s?h.dim_n:h.dim_p, iso=s?Neutron:Proton;
            const auto &orbits=s?h.modelspace->Orbits_n:h.modelspace->Orbits_p;
            Array matrix({d,d}); std::fill(matrix.mutable_data(),matrix.mutable_data()+size_t(d)*d,0.);
            for(int i=0;i<d;++i) for(int j=0;j<d;++j) {
                int a=s?h.modelspace->Get_NeutronOrbitIndexInMscheme(i):h.modelspace->Get_ProtonOrbitIndexInMscheme(i);
                int b=s?h.modelspace->Get_NeutronOrbitIndexInMscheme(j):h.modelspace->Get_ProtonOrbitIndexInMscheme(j);
                const auto &oa=orbits[a], &ob=orbits[b];
                int ma=h.modelspace->Get_MSmatrix_2m(iso,i), mb=h.modelspace->Get_MSmatrix_2m(iso,j);
                int dm=ma-mb;
                if ((oa.l+ob.l+rank)%2 || std::abs(dm)!=2*mu ||
                    std::abs(oa.j2-ob.j2)>2*rank || oa.j2+ob.j2<2*rank) continue;
                double reduced=phase((ob.j2-1)/2+rank)*std::sqrt((oa.j2+1.)*(ob.j2+1.)*(2*rank+1.)/(4.*M_PI))
                    *AngMom::threej(.5*oa.j2,.5*ob.j2,rank,.5,-.5,0.)
                    *h.Ham->HarmonicRadialIntegral(iso,power,a,b);
                double value=phase((oa.j2-ma)/2)*AngMom::threej(.5*oa.j2,rank,.5*ob.j2,-.5*ma,.5*dm,.5*mb)*reduced;
                if(mu>0 && dm<0) value*=phase(mu);
                matrix.mutable_at(i,j)=value;
            }
            result.append(matrix);
        }
        return pybind11::make_tuple(result[0],result[1]);
    }
    static Array copy(const double *p, int d, int n) {
        Array a({d,n});
        if (d*n) std::copy(p,p+size_t(d)*n,a.mutable_data());
        return a;
    }
    static void check(const Array &a,int d,int n,bool symmetric=false) {
        if (a.ndim()!=2 || a.shape(0)!=d || a.shape(1)!=n)
            throw std::invalid_argument("incorrect hybrid matrix shape");
        auto v=a.unchecked<2>();
        for (int i=0;i<d;++i) for (int j=0;j<n;++j) {
            if (!std::isfinite(v(i,j))) throw std::invalid_argument("nonfinite hybrid matrix");
            if (symmetric && std::abs(v(i,j)-v(j,i))>1.e-10)
                throw std::invalid_argument("density/response must be symmetric");
        }
    }
    static pybind11::tuple state(HartreeFock &h) {
        Array p({h.dim_p,h.N_p}), n({h.dim_n,h.N_n});
        for(int i=0;i<h.dim_p;++i) for(int j=0;j<h.N_p;++j)
            p.mutable_at(i,j)=h.U_p[i*h.dim_p+h.holeorbs_p[j]];
        for(int i=0;i<h.dim_n;++i) for(int j=0;j<h.N_n;++j)
            n.mutable_at(i,j)=h.U_n[i*h.dim_n+h.holeorbs_n[j]];
        return pybind11::make_tuple(p,n);
    }
    static pybind11::tuple response(HartreeFock &h,Array p,Array n) {
        check(p,h.dim_p,h.dim_p,true); check(n,h.dim_n,h.dim_n,true);
        Array fp({h.dim_p,h.dim_p}),fn({h.dim_n,h.dim_n});
        h.ContractDensity(p.data(),n.data(),fp.mutable_data(),fn.mutable_data());
        return pybind11::make_tuple(fp,fn);
    }
    static pybind11::tuple evaluate(HartreeFock &h,Array p,Array n) {
        auto fields=response(h,p,n);
        auto fp=fields[0].cast<Array>(),fn=fields[1].cast<Array>();
        double energy=0;
        for(int i=0;i<h.dim_p*h.dim_p;++i) {
            energy+=p.data()[i]*(h.T_term_p[i]+0.5*fp.data()[i]);
            fp.mutable_data()[i]+=h.T_term_p[i];
        }
        for(int i=0;i<h.dim_n*h.dim_n;++i) {
            energy+=n.data()[i]*(h.T_term_n[i]+0.5*fn.data()[i]);
            fn.mutable_data()[i]+=h.T_term_n[i];
        }
        return pybind11::make_tuple(energy,fp,fn);
    }
    // Independent original contraction for regression and benchmark checks.
    static pybind11::tuple reference(HartreeFock &h,Array p,Array n) {
        check(p,h.dim_p,h.dim_p,true); check(n,h.dim_n,h.dim_n,true);
        int dp=h.dim_p,dn=h.dim_n,pp=dp*dp,nn=dn*dn;
        Array fp({dp,dp}),fn({dn,dn});
        auto vp=h.Ham->MSMEs.GetVppPrt(),vn=h.Ham->MSMEs.GetVnnPrt(),vpn=h.Ham->MSMEs.GetVpnPrt();
        for(int i=0;i<dp;++i) for(int j=i;j<dp;++j) {
            double f=h.T_term_p[i*dp+j]+cblas_ddot(pp,p.data(),1,vp+size_t(i*dp+j)*pp,1);
            if(nn) f+=cblas_ddot(nn,n.data(),1,vpn+size_t(i*dp+j)*nn,1);
            fp.mutable_at(i,j)=fp.mutable_at(j,i)=f;
        }
        for(int i=0;i<dn;++i) for(int j=i;j<dn;++j) {
            double f=h.T_term_n[i*dn+j]+cblas_ddot(nn,n.data(),1,vn+size_t(i*dn+j)*nn,1);
            if(pp) f+=cblas_ddot(pp,p.data(),1,vpn+i*dn+j,nn);
            fn.mutable_at(i,j)=fn.mutable_at(j,i)=f;
        }
        return pybind11::make_tuple(fp,fn);
    }
    static pybind11::tuple operators(HartreeFock &h) {
        pybind11::list out;
        for(int species=0;species<2;++species) {
            int d=species?h.dim_n:h.dim_p;
            auto &q=species?h.Ham->Q2MEs_n:h.Ham->Q2MEs_p;
            auto &ob=species?h.Ham->MSMEs.OB_n:h.Ham->MSMEs.OB_p;
            Array ops({4,d,d}); std::fill(ops.mutable_data(),ops.mutable_data()+size_t(4)*d*d,0.);
            auto add=[&](int slot,const std::vector<int>&ids,const std::vector<double>&v) {
                for(size_t k=0;k<ids.size();++k)
                    ops.mutable_at(slot,ob[ids[k]].GetIndex_a(),ob[ids[k]].GetIndex_b())+=v[k];
            };
            add(0,q.Q0_list,q.Q0_MSMEs);
            add(1,q.Q2_list,q.Q2_MSMEs); add(1,q.Q_2_list,q.Q_2_MSMEs);
            int iso=species?Neutron:Proton;
            for(int i=0;i<d;++i) {
                ops.mutable_at(3,i,i)=0.5*h.modelspace->Get_MSmatrix_2m(iso,i);
                for(int j=0;j<d;++j) {
                    int oi=species?h.modelspace->Get_NeutronOrbitIndexInMscheme(i):h.modelspace->Get_ProtonOrbitIndexInMscheme(i);
                    int oj=species?h.modelspace->Get_NeutronOrbitIndexInMscheme(j):h.modelspace->Get_ProtonOrbitIndexInMscheme(j);
                    if(oi!=oj) continue;
                    double jj=h.modelspace->Get_MSmatrix_2j(iso,j),mj=h.modelspace->Get_MSmatrix_2m(iso,j);
                    int mi=h.modelspace->Get_MSmatrix_2m(iso,i);
                    if(mi==mj+2) ops.mutable_at(2,i,j)=0.25*std::sqrt((jj-mj)*(jj+mj+2));
                    if(mi==mj-2) ops.mutable_at(2,i,j)=0.25*std::sqrt((jj+mj)*(jj-mj+2));
                }
            }
            out.append(ops);
        }
        return pybind11::make_tuple(out[0],out[1]);
    }
    static void accept(HartreeFock &h,Array p,Array n) {
        check(p,h.dim_p,h.dim_p); check(n,h.dim_n,h.dim_n);
        auto orthogonal=[](const Array&a,int d) {
            for(int i=0;i<d;++i) for(int j=0;j<d;++j) {
                double g=0; for(int k=0;k<d;++k) g+=a.at(k,i)*a.at(k,j);
                if(std::abs(g-(i==j))>1.e-8) throw std::invalid_argument("hybrid basis must be orthogonal");
            }
        };
        orthogonal(p,h.dim_p); orthogonal(n,h.dim_n);
        std::copy(p.data(),p.data()+p.size(),h.U_p); std::copy(n.data(),n.data()+n.size(),h.U_n);
        std::iota(h.holeorbs_p,h.holeorbs_p+h.N_p,0); std::iota(h.holeorbs_n,h.holeorbs_n+h.N_n,0);
        h.UpdateDensityMatrix(); h.UpdateF(); h.CalcEHF();
    }
};

inline void bind_hybrid_backend(pybind11::module_ &m) {
    m.def("set_hybrid_nucleus",[](ModelSpace &ms,const std::string &name) {
        double a=0,z=0; ms.GetAZfromString(name,a,z);
        int p=static_cast<int>(z)-ms.GetCoreProtonNum();
        int n=static_cast<int>(a-z)-ms.GetCoreNeutronNum();
        int dp=0,dn=0;
        for(const auto &o:ms.Orbits_p) dp+=o.j2+1;
        for(const auto &o:ms.Orbits_n) dn+=o.j2+1;
        if(p<0 || n<0 || p>dp || n>dn || a<=0)
            throw std::invalid_argument("isotope is outside the interaction valence space");
        ms.Set_RefString(name); ms.SetProtonNum(p); ms.SetNeutronNum(n);
    });
    m.def("set_hybrid_threads",[](int n) {
        if(n<1) throw std::invalid_argument("threads must be positive");
        omp_set_num_threads(n); mkl_set_dynamic(0); mkl_set_num_threads(n);
    });
}
