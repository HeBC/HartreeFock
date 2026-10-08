"""Real charge-conserving 0b+1b+2b constraints for Slater determinants.

O[rho] = O0 + f:(rho-rho_ref) + (rho-rho_ref):K:(rho-rho_ref)/2.
K contracts antisymmetrized two-body matrix elements, not an orbital Hessian.
Tensor SNT input uses normalized antisymmetric J-coupled kets and the standard
Wigner-Eckart convention <J M|T_kq|J' M'> = CG(J'M',kq|JM)*RME/sqrt(2J+1).
"""
from functools import lru_cache
from pathlib import Path
import hashlib,json,math,re,zipfile
import numpy as np
from scipy import sparse,linalg


def _real(a):
    a=np.asarray(a)
    if np.iscomplexobj(a):
        raise ValueError('complex operators require a complex HF solver')
    a=np.asarray(a,dtype=float)
    if not np.isfinite(a).all(): raise ValueError('nonfinite operator data')
    return a


class HFOperator:
    def __init__(self,name,one_body,kernel=None,zero_body=0.,reference=None,basis=None,metadata=None):
        self.name=str(name)
        self.one_body=tuple(_real(x).copy() for x in one_body)
        if len(self.one_body)!=2 or any(x.ndim!=2 or x.shape[0]!=x.shape[1] for x in self.one_body):
            raise ValueError('need square proton and neutron one-body matrices')
        if any(not np.allclose(x,x.T,atol=2e-10,rtol=2e-10) for x in self.one_body):
            raise ValueError('constraint one-body part must be real Hermitian')
        self.dims=tuple(x.shape[0] for x in self.one_body)
        self.size=sum(d*d for d in self.dims)
        if kernel is not None and np.iscomplexobj(kernel.data if sparse.issparse(kernel) else kernel):
            raise ValueError('complex two-body constraints require a complex solver')
        self.kernel=sparse.csr_matrix((self.size,self.size)) if kernel is None else sparse.csr_matrix(kernel,dtype=float)
        if self.kernel.shape!=(self.size,self.size) or not np.isfinite(self.kernel.data).all():
            raise ValueError('invalid two-body contraction kernel')
        delta=self.kernel-self.kernel.T
        if np.max(np.abs(delta.data),initial=0.)>2e-9:
            raise ValueError('two-body kernel must be self-adjoint')
        self.zero_body=float(zero_body)
        if not np.isfinite(self.zero_body): raise ValueError('nonfinite zero-body term')
        self.reference=tuple(np.zeros_like(x) for x in self.one_body) if reference is None else tuple(_real(x).copy() for x in reference)
        if len(self.reference)!=2 or any(r.shape!=f.shape or not np.allclose(r,r.T,atol=1e-10) for r,f in zip(self.reference,self.one_body)):
            raise ValueError('invalid reference density')
        if any(np.min(linalg.eigvalsh(r),initial=0.) < -1e-9 or np.max(linalg.eigvalsh(r),initial=0.) > 1+1e-9 for r in self.reference):
            raise ValueError('reference occupations must lie in [0,1]')
        self.basis=None if basis is None else tuple(np.asarray(x,dtype=np.int64).copy() for x in basis)
        self.metadata=dict(metadata or {})
        self.metadata.update(name=self.name,zero_body=self.zero_body,two_body_nnz=int(self.kernel.nnz))
        self.linear=not self.kernel.nnz
        # A constant scale for the joint constraint solve; no density-dependent scaling.
        row_norm=np.max(np.asarray(abs(self.kernel).sum(axis=1)),initial=0.)
        self.scale=max(1.,*(linalg.norm(x,2) if x.size else 0. for x in self.one_body),float(row_norm))
        for array in (*self.one_body,*self.reference): array.setflags(write=False)

    def pack(self,matrices): return np.concatenate([x.ravel() for x in matrices])

    def unpack(self,v):
        p,n=self.dims
        return v[:p*p].reshape(p,p),v[p*p:].reshape(n,n)

    def symmetric_fields(self,v):
        return tuple(.5*(x+x.T) for x in self.unpack(v))

    def evaluate(self,rho):
        dr=self.pack([r-ref for r,ref in zip(rho,self.reference)])
        response=self.kernel@dr
        f=self.pack(self.one_body)
        return float(self.zero_body+np.dot(f,dr)+.5*np.dot(dr,response)),self.symmetric_fields(f+response)

    def response(self,delta_rho): return self.symmetric_fields(self.kernel@self.pack(delta_rho))

    def bounds(self,occupations):
        if not self.linear: return -math.inf,math.inf
        lo=hi=self.zero_body-sum(np.sum(f*r) for f,r in zip(self.one_body,self.reference))
        for f,n in zip(self.one_body,occupations):
            eigen=linalg.eigvalsh(f)
            lo+=eigen[:n].sum();hi+=eigen[-n:].sum() if n else 0.
        return float(lo),float(hi)

    @property
    def storage_bytes(self):
        return sum(x.nbytes for x in (*self.one_body,*self.reference))+self.kernel.data.nbytes+self.kernel.indices.nbytes+self.kernel.indptr.nbytes

    def save(self,path):
        if self.basis is None: raise ValueError('operator export requires explicit basis metadata')
        meta=dict(self.metadata,format='myhf.operator.v1',zero_body=self.zero_body)
        np.savez_compressed(path,metadata=np.asarray(json.dumps(meta)),basis_p=self.basis[0],basis_n=self.basis[1],
                            one_p=self.one_body[0],one_n=self.one_body[1],ref_p=self.reference[0],ref_n=self.reference[1],
                            kernel_data=self.kernel.data,kernel_indices=self.kernel.indices,kernel_indptr=self.kernel.indptr)


def load_operator(path,hf,*,name=None,memory_mb=256.):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        if 4*sum(item.file_size for item in archive.infolist())>memory_mb*1048576:
            raise MemoryError('operator archive exceeds conversion memory budget')
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if meta.get('format')!='myhf.operator.v1': raise ValueError('unknown operator format')
        basis=tuple(data[k] for k in ('basis_p','basis_n'))
        if any(not np.array_equal(a,b) for a,b in zip(basis,hf.hybrid_basis())):
            raise ValueError('operator and Hamiltonian m-scheme bases differ')
        one=tuple(data[k] for k in ('one_p','one_n'));size=sum(x.size for x in one)
        kernel=sparse.csr_matrix((data['kernel_data'],data['kernel_indices'],data['kernel_indptr']),shape=(size,size))
        kernel.check_format(full_check=True)
        return HFOperator(name or meta['name'],one,kernel,meta['zero_body'],tuple(data[k] for k in ('ref_p','ref_n')),basis,meta)


def builtin_operator(hf,name,*,hw,units='physical',weights=(1.,1.)):
    """Q22 is the cosine sum; Q21 is its half-sum principal-axis convention."""
    basis=hf.hybrid_basis()
    if name in ('Jx','Jz'):
        matrices=tuple(x[2 if name=='Jx' else 3] for x in hf.hybrid_operators())
        unit='hbar'
    else:
        if name=='R2': rank,mu,power,factor=0,0,2,math.sqrt(4*math.pi)
        else:
            match=re.fullmatch(r'Q([0-8])([0-8])',name)
            if not match: raise ValueError(f'unknown built-in constraint {name}')
            rank,mu=map(int,match.groups());power=rank;factor=.5 if name=='Q21' else 1.
            if mu>rank: raise ValueError('multipole component exceeds rank')
        if units not in ('physical','oscillator') or not np.isfinite(hw) or hw<=0:
            raise ValueError('positive hw and physical/oscillator units required')
        if units=='physical': factor*=(41.47106/hw)**(.5*power)
        matrices=tuple(factor*x for x in hf.hybrid_multipole(rank,mu,power))
        unit=f'fm^{power}' if units=='physical' else f'b^{power}'
    weights=_real(weights)
    if weights.shape!=(2,): raise ValueError('need proton and neutron weights')
    return HFOperator(name,tuple(w*x for w,x in zip(weights,matrices)),basis=basis,
                      metadata=dict(kind='bare',units=unit,weights=weights.tolist(),basis_representation='HO',
                                    component='cosine_sum' if name=='Q22' else 'real',normal_ordering='valence_vacuum'))


def _records(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number,line in enumerate(stream,1):
            line=line.split('!')[0].split('#')[0].strip()
            if line: yield number,line.replace('D','E').replace('d','e').split()


def tensor_snt_operator(path,hf,*,name,rank,mu,parity,normal_ordering,units,
                        representation='HO',reference=None,memory_mb=256.,hermitian_tolerance=3e-6):
    """Read IMSRG WriteTensorTokyo output (3-field 1b, 7-field 2b records).

    rank/parity/normal_ordering/units are explicit: tensor SNT does not encode them.
    core means normal ordered against the inert core, i.e. vacuum for active states.
    reference requires explicit active-space Gaussian reference density matrices.
    """
    import pyHFAndHFB as native
    if not isinstance(rank,int) or not 0<=mu<=rank<=8 or parity not in (0,1):
        raise ValueError('invalid tensor rank/component/parity')
    if normal_ordering not in ('core','valence_vacuum','reference'):
        raise ValueError('specify core, valence_vacuum or reference normal ordering')
    if (reference is not None)!=(normal_ordering=='reference'):
        raise ValueError('reference normal ordering requires exactly one reference density')
    if representation not in ('HO','HF','NAT'): raise ValueError('unknown basis representation')
    path=Path(path);basis=tuple(np.asarray(x) for x in hf.hybrid_basis());dp,dn=map(len,basis)
    size=dp*dp+dn*dn
    # Conservative conversion peak: dense work bounds plus sparse accumulators.
    if not np.isfinite(memory_mb) or memory_mb<=0 or (64*size*size+16*path.stat().st_size)>memory_mb*1048576:
        raise MemoryError('tensor conversion exceeds memory admission budget')
    text=path.read_text(encoding='utf-8-sig')
    match=re.search(r'Zero body term:\s*([-+\d.eEdD]+)',text)
    zero=float(match.group(1).replace('D','E')) if match else 0.
    if rank and abs(zero)>1e-12: raise ValueError('a nonscalar tensor cannot have a scalar zero-body term')
    records=iter(_records(path))
    def take(count):
        try: number,fields=next(records)
        except StopIteration: raise ValueError('truncated tensor SNT') from None
        if len(fields)!=count: raise ValueError(f'line {number}: expected {count} fields, got {len(fields)}')
        return fields
    np_shell,nn_shell,corep,coren=map(int,take(4))
    orbits={}
    for _ in range(np_shell+nn_shell):
        idx,n,l,j,tz=map(int,take(5))
        if idx in orbits or n<0 or l<0 or j not in (2*l-1,2*l+1) or tz not in (-1,1):
            raise ValueError('invalid or duplicated SNT orbit')
        orbits[idx]=(n,l,j,tz)
    if sum(x[-1]==-1 for x in orbits.values())!=np_shell or sum(x[-1]==1 for x in orbits.values())!=nn_shell:
        raise ValueError('SNT species counts disagree with orbit table')
    combined=np.concatenate(basis)
    lookup={tuple(row):i for i,row in enumerate(combined)}
    expected={(n,l,j,m,tz) for n,l,j,tz in orbits.values() for m in range(-j,j+1,2)}
    if set(lookup)!=expected: raise ValueError('tensor SNT orbit space differs from Hamiltonian')
    states={idx:[(lookup[(n,l,j,m,tz)],m) for m in range(-j,j+1,2)] for idx,(n,l,j,tz) in orbits.items()}
    cg=lru_cache(maxsize=200000)(native.hybrid_cg)
    def we(jb,mb,jk,mk):
        dm=mb-mk
        if dm==2*mu: return cg(jk,mk,2*rank,2*mu,jb,mb)/math.sqrt(jb+1)
        if mu and dm==-2*mu: return (-1)**mu*cg(jk,mk,2*rank,-2*mu,jb,mb)/math.sqrt(jb+1)
        return 0.
    one=np.zeros((dp+dn,dp+dn))
    header=take(3);count=int(header[0])
    if count<0: raise ValueError('negative one-body record count')
    if int(header[1])!=0: raise ValueError('mass-dependent tensor SNT is unsupported')
    def number(s):
        v=float(s)
        if not np.isfinite(v): raise ValueError('nonfinite SNT matrix element')
        return v
    ob={}
    for _ in range(count):
        aa,bb,value=take(3);a,b=int(aa),int(bb);value=number(value)
        if a not in orbits or b not in orbits: raise ValueError('unknown orbit index')
        if (a,b) in ob: raise ValueError('duplicate one-body record')
        oa,ob_=orbits[a],orbits[b]
        if oa[-1]!=ob_[-1]: raise ValueError('charge-changing constraints are unsupported')
        if (oa[1]+ob_[1])%2!=parity and value: raise ValueError('one-body parity mismatch')
        if not abs(oa[2]-ob_[2])<=2*rank<=oa[2]+ob_[2] and value: raise ValueError('one-body rank mismatch')
        ob[a,b]=value
    for (a,b),v in list(ob.items()):
        partner=(-1)**((orbits[a][2]-orbits[b][2])//2)*v
        if (b,a) in ob and abs(ob[b,a]-partner)>hermitian_tolerance:
            raise ValueError('inconsistent reduced one-body Hermitian partners')
        ob.setdefault((b,a),partner)
    for (a,b),v in ob.items():
        for i,ma in states[a]:
            for j,mb in states[b]: one[i,j]+=v*we(orbits[a][2],ma,orbits[b][2],mb)
    # Couple normalized antisymmetric pair kets directly into m-scheme pairs.
    @lru_cache(maxsize=20000)
    def pair(a,b,j,m):
        values={};ja,jb=orbits[a][2],orbits[b][2]
        for i,ma in states[a]:
            for k,mb in states[b]:
                if ma+mb!=2*m or i==k: continue
                value=cg(ja,ma,jb,mb,2*j,2*m)/math.sqrt(1+(a==b))
                key=(min(i,k),max(i,k))
                values[key]=values.get(key,0.)+(value if i<k else -value)
        return tuple((i,k,v) for (i,k),v in values.items() if abs(v)>1e-14)
    def canonical(a,b,j):
        if a<=b: return (a,b,j),1
        return (b,a,j),(-1)**((orbits[a][2]+orbits[b][2])//2-j+1)
    tb={};header=take(3);count=int(header[0])
    if count<0: raise ValueError('negative two-body record count')
    if int(header[1])!=0: raise ValueError('mass-dependent tensor SNT is unsupported')
    for _ in range(count):
        fields=take(7);a,b,c,d,jb,jk=map(int,fields[:6]);v=number(fields[6])
        if any(i not in orbits for i in (a,b,c,d)): raise ValueError('unknown two-body orbit')
        for x,y,j in ((a,b,jb),(c,d,jk)):
            if j<0 or not abs(orbits[x][2]-orbits[y][2])<=2*j<=orbits[x][2]+orbits[y][2] or (x==y and j%2):
                raise ValueError('invalid or Pauli-forbidden coupled pair')
        if orbits[a][3]+orbits[b][3]!=orbits[c][3]+orbits[d][3]:
            raise ValueError('charge-changing two-body constraint')
        if (sum(orbits[i][1] for i in (a,b,c,d))%2!=parity or not abs(jb-jk)<=rank<=jb+jk) and v:
            raise ValueError('two-body tensor selection rule mismatch')
        bra,sb=canonical(a,b,jb);ket,sk=canonical(c,d,jk);v*=sb*sk
        key=bra+ket
        if key in tb and abs(tb[key]-v)>hermitian_tolerance: raise ValueError('inconsistent duplicate two-body record')
        tb[key]=v
    try: next(records)
    except StopIteration: pass
    else: raise ValueError('unexpected records after two-body block')
    for key,v in list(tb.items()):
        a,b,jb,c,d,jk=key;other=(c,d,jk,a,b,jb);partner=(-1)**(jb-jk)*v
        if other in tb and abs(tb[other]-partner)>hermitian_tolerance:
            raise ValueError('inconsistent reduced two-body Hermitian partners')
        tb.setdefault(other,partner)
    # Bounded dense conversion work, then sparse storage; no four-index tensor.
    kernel=np.zeros((size,size))
    def index(i,k):
        if i<dp and k<dp: return i*dp+k
        if i>=dp and k>=dp: return dp*dp+(i-dp)*dn+(k-dp)
        return None
    for (a,b,jb,c,d,jk),v in tb.items():
        for mb in range(-jb,jb+1):
            for mk in ({mb-mu,mb+mu} if mu else {mb}):
                if abs(mk)>jk: continue
                angular=we(2*jb,2*mb,2*jk,2*mk)*v
                if abs(angular)<1e-16: continue
                for i,j,x in pair(a,b,jb,mb):
                    for k,l,y in pair(c,d,jk,mk):
                        value=angular*x*y
                        for u,w,z,t,sgn in ((i,k,j,l,1),(j,k,i,l,-1),(i,l,j,k,-1),(j,l,i,k,1)):
                            row,col=index(u,w),index(z,t)
                            if row is not None and col is not None: kernel[row,col]+=sgn*value
    # Roundoff in old 6-digit exports is checked, then projected onto Hermiticity.
    one_error=np.max(np.abs(one-one.T),initial=0.)
    if one_error>hermitian_tolerance: raise ValueError('non-Hermitian m-scheme one-body operator')
    one=(one+one.T)*.5
    kernel=(kernel+kernel.T)*.5
    # Symmetrize field/density index pairs: HF here only uses real symmetric rho.
    trans=np.concatenate([np.arange(dp*dp).reshape(dp,dp).T.ravel(),dp*dp+np.arange(dn*dn).reshape(dn,dn).T.ravel()])
    kernel=(kernel+kernel[trans,:]+kernel[:,trans]+kernel[trans,:][:,trans])*.25
    kernel[np.abs(kernel)<1e-14]=0.
    return HFOperator(name,(one[:dp,:dp],one[dp:,dp:]),sparse.csr_matrix(kernel),zero,reference,basis,
                      dict(kind='tensor_snt',rank=rank,mu=mu,parity=parity,units=units,
                           normal_ordering=normal_ordering,basis_representation=representation,
                           core_protons=corep,core_neutrons=coren,source=str(path.resolve()),
                           source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),component='cosine_sum' if mu else 'mu0'))
