"""Equality-constrained real HF on occupied orbital subspaces.

Hybrid: safeguarded diagonalization, projected descent, then matrix-free
truncated Newton CG. The full orbital Hessian is never allocated.
Q0 and Q2 here use the native r^2 Y20 / b^2 and (Y22+Y2,-2) r^2 / b^2.
"""
from dataclasses import dataclass
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
from hf_operators import HFOperator


@dataclass
class Options:
    max_iterations: int = 500
    diagonalization_steps: int = 6
    gradient_steps: int = 6
    gradient_tolerance: float = 1e-6
    constraint_tolerance: float = 1e-8
    energy_tolerance: float = 1e-8
    trust_radius: float = 0.3
    max_cg: int = 35
    precondition: bool = True
    precondition_floor: float = 0.1
    method: str = "hybrid"
    seed: int = 520
    check_stability: bool = True
    curvature_tolerance: float = 1e-5

    def validate(self):
        for k in ("max_iterations", "max_cg"):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError(f"{k} must be a positive integer")
        for k in ("diagonalization_steps", "gradient_steps"):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 0:
                raise ValueError(f"{k} must be a nonnegative integer")
        for k in ("gradient_tolerance", "constraint_tolerance", "energy_tolerance", "trust_radius", "curvature_tolerance", "precondition_floor"):
            if not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f"{k} must be finite and positive")
        if not isinstance(self.precondition, bool):
            raise ValueError("precondition must be true or false")
        if self.method not in ("hybrid", "gradient"):
            raise ValueError("method must be hybrid or gradient")


class Solver:
    def __init__(self, hf, active=(0, 1), options=None, constraints=None):
        self.hf, self.options = hf, options or Options()
        self.options.validate()
        self.initial = tuple(np.array(c) for c in hf.hybrid_state())
        self.shapes = [c.shape for c in self.initial]
        self.sizes = [c.size for c in self.initial]
        self.size = sum(self.sizes)
        native_ops = hf.hybrid_operators()
        for op in native_ops:
            if not np.isfinite(op).all() or not np.allclose(op, op.transpose(0,2,1), atol=1e-11, rtol=1e-11):
                raise ValueError("constraint operators must be finite and symmetric; rebuild MyHF")
        operators = []
        for op in native_ops:
            # Real Q21 fixes principal-axis orientation for beta/gamma scans.
            jx, jz = op[2:4]
            lower = jx - (jz @ jx - jx @ jz)
            m = np.diag(jz)
            q22 = np.where(m[:, None] > m[None, :], op[1], 0.)
            q21 = 0.5 * (lower @ q22 - q22 @ lower)
            operators.append(np.concatenate((op, ((q21 + q21.T)*0.5)[None])))
        self.all_ops = tuple(operators)
        self.named_constraints = constraints is not None
        self.active = tuple(active) if constraints is None else tuple(range(len(constraints)))
        if constraints is None and (len(set(active)) != len(active) or any(i not in range(5) for i in active)):
            raise ValueError("invalid or repeated constraint operator")
        if constraints is None:
            names=('legacy_Q20','legacy_Q22sum','Jx','Jz','legacy_Q21real')
            self.constraints=tuple(HFOperator(names[k],tuple(op[k] for op in self.all_ops)) for k in active)
        else:
            self.constraints=tuple(constraints)
        if any(op.dims!=tuple(x.shape[0] for x in self.initial) for op in self.constraints):
            raise ValueError('constraint dimensions differ from Hamiltonian')
        self.constraint_names=tuple(op.name for op in self.constraints)
        if len(set(self.constraint_names))!=len(self.constraint_names): raise ValueError('duplicate constraint names')
        self.ops=tuple(np.stack([op.one_body[s] for op in self.constraints]) if self.constraints else np.zeros((0,x.shape[0],x.shape[0]))
                       for s,x in enumerate(self.initial))
        self.scales=np.array([op.scale for op in self.constraints])
        self._constraint_cache=None
        self.fock_evaluations = self.hessian_evaluations = 0
        self.cg_iterations = self.cg_limit_hits = self.preconditioner_evaluations = 0

    def pack(self, matrices):
        return np.concatenate([m.ravel() for m in matrices])

    def unpack(self, vector):
        return (vector[:self.sizes[0]].reshape(self.shapes[0]),
                vector[self.sizes[0]:].reshape(self.shapes[1]))

    def tangent(self, c, v):
        return self.pack([w - x @ (x.T @ w) for x, w in zip(c, self.unpack(v))])

    def retract(self, c, v):
        out = []
        for x, w in zip(c, self.unpack(v)):
            if not x.shape[1]:
                out.append(x.copy())
            else:
                u, _, vt = linalg.svd(x+w, full_matrices=False, check_finite=False)
                out.append(u @ vt)
        return tuple(out)

    def moments(self, c, all_ops=False):
        if all_ops and not self.named_constraints:
            return sum(np.einsum("ai,kab,bi->k", x, q, x, optimize=True) for x, q in zip(c, self.all_ops))
        return self.constraint_data(c)[0]

    def constraint_data(self,c):
        cached=self._constraint_cache
        if cached is not None and all(np.array_equal(a,b) for a,b in zip(c,cached[0])):
            return cached[1],cached[2]
        rho=tuple(x@x.T for x in c)
        results=[op.evaluate(rho) for op in self.constraints]
        values=np.array([v for v,f in results])
        fields=tuple(np.stack([f[s] for v,f in results]) if results else np.zeros((0,x.shape[0],x.shape[0])) for s,x in enumerate(c))
        self._constraint_cache=(tuple(x.copy() for x in c),values,fields)
        return values,fields

    def jacobian(self, c):
        if not self.active or not self.size:
            return np.empty((self.size, 0)), np.empty(0), np.empty((0, len(self.active)))
        fields=self.constraint_data(c)[1]
        jac = np.column_stack([self.tangent(c, self.pack([2*q[k] @ x for q, x in zip(fields, c)]))
                               / self.scales[k] for k in range(len(self.active))])
        u, s, vt = linalg.svd(jac, full_matrices=False, check_finite=False)
        keep = s > max(1e-13, 1e-11*s.max(initial=0.))
        return u[:, keep], s[keep], vt[keep]

    def restore(self, c, target):
        for _ in range(80):
            r = self.moments(c)-target
            # Restoration should be tighter than the final acceptance threshold;
            # otherwise constraint noise can dominate tiny late energy steps.
            restore_tol = min(self.options.constraint_tolerance*.01, 1e-11)
            if np.max(np.abs(r), initial=0.) <= restore_tol:
                return c, True
            u, s, vt = self.jacobian(c)
            v = -u @ ((vt @ (r/self.scales))/s)
            norm = linalg.norm(v)
            if not np.isfinite(norm) or norm < 1e-14:
                break
            v *= min(1., .4/norm)
            error = linalg.norm(r/self.scales)
            for back in range(20):
                trial = self.retract(c, v * .5**back)
                if linalg.norm((self.moments(trial)-target)/self.scales) < error:
                    c = trial
                    break
            else:
                break
        return c, np.max(np.abs(self.moments(c)-target), initial=0.) <= self.options.constraint_tolerance

    def evaluate(self, c):
        self.fock_evaluations += 1
        e, fp, fn = self.hf.hybrid_evaluate(*(x @ x.T for x in c))
        if not np.isfinite(e) or any(not np.isfinite(f).all() for f in (fp, fn)):
            raise FloatingPointError("nonfinite energy/Fock matrix")
        return e, (fp, fn)

    def stationarity(self, c, f):
        g = self.tangent(c, self.pack([2*a @ x for a, x in zip(f, c)]))
        u, s, vt = self.jacobian(c)
        lambdas = -(vt.T @ ((u.T @ g)/s))/self.scales
        return g-u @ (u.T @ g), lambdas, u

    def hessian(self, c, f, lambdas, v):
        """Exact covariant Hessian of E + lambda.Q for a two-body HF functional."""
        self.hessian_evaluations += 1
        v = self.tangent(c, v)
        matrices = self.unpack(v)
        delta_rho = [w @ x.T + x @ w.T for x, w in zip(c, matrices)]
        response = [np.array(x,copy=True) for x in self.hf.hybrid_response(*delta_rho)]
        for weight,operator in zip(lambdas,self.constraints):
            if weight and not operator.linear:
                for df,dq in zip(response,operator.response(delta_rho)): df+=weight*dq
        fields=self.constraint_data(c)[1]
        out = []
        for x, w, field, df, op in zip(c, matrices, f, response, fields):
            effective = field + np.einsum("k,kab->ab", lambdas, op)
            out.append(2*(df @ x + effective @ w - w @ (x.T @ effective @ x)))
        return self.tangent(c, self.pack(out))

    @staticmethod
    def boundary(z, d, radius):
        zd = np.dot(z, d)
        dd = np.dot(d, d)
        return (-zd + np.sqrt(max(0., zd*zd + dd*(radius*radius-np.dot(z,z)))))/dd

    def prepare_preconditioner(self, c, f, lambdas):
        """Positive orbital-gap inverse, following the CC hf_real implementation.

        Only the occupied/virtual Fock blocks are diagonalized. The exact
        Hamiltonian and nonlinear-constraint response stays in the Hessian.
        Storage is O(dp**2 + dn**2); no particle-hole Hessian is constructed.
        """
        self.preconditioner_evaluations += 1
        pre = []
        fields = self.constraint_data(c)[1]
        for x, field, ops in zip(c, f, fields):
            dim, occupied = x.shape
            if occupied == 0 or occupied == dim:
                pre.append(None)
                continue
            effective = field + np.einsum("k,kab->ab", lambdas, ops)
            virtual = linalg.qr(x, mode="full", check_finite=False)[0][:, occupied:]
            eo, uo = linalg.eigh(x.T @ effective @ x, check_finite=False)
            ev, uv = linalg.eigh(virtual.T @ effective @ virtual, check_finite=False)
            inverse = 0.5 / np.maximum(self.options.precondition_floor,
                                       np.abs(ev[:, None] - eo[None, :]))
            pre.append((virtual @ uv, uo, inverse))
        return pre

    def precondition_residual(self, c, r, u, pre):
        """Apply P M P; r is already in the joint constraint tangent space."""
        matrices = []
        for w, block in zip(self.unpack(r), pre):
            if block is None:
                matrices.append(np.zeros_like(w))
                continue
            virtual, occupied, inverse = block
            small = (virtual.T @ w @ occupied) * inverse
            matrices.append(virtual @ small @ occupied.T)
        z = self.tangent(c, self.pack(matrices))
        return z - u @ (u.T @ z)

    def newton_step(self, c, f, lambdas, u, g, radius):
        def project(v):
            v = self.tangent(c, v)
            return v-u @ (u.T @ v)
        def action(v):
            return project(self.hessian(c, f, lambdas, project(v)))
        pre = self.prepare_preconditioner(c, f, lambdas) if self.options.precondition else None
        def metric(r):
            if pre is None:
                return r.copy()
            z = self.precondition_residual(c, r, u, pre)
            # The positive floor keeps M well-defined at indefinite points.
            # Fall back to the identity if roundoff destroys a positive norm.
            return z if np.dot(r, z) > np.finfo(float).tiny else r.copy()
        step, r = np.zeros_like(g), -g.copy()
        z = metric(r)
        d, rz = z.copy(), np.dot(r, z)
        gnorm = linalg.norm(g)
        tol = max(1e-12, min(.3, np.sqrt(gnorm))*gnorm)
        for _ in range(self.options.max_cg):
            hd = action(d)
            self.cg_iterations += 1
            curvature = np.dot(d, hd)
            if curvature <= 1e-14*np.dot(d, d):
                return step+self.boundary(step, d, radius)*d
            alpha = rz/curvature
            if linalg.norm(step+alpha*d) >= radius:
                return step+self.boundary(step, d, radius)*d
            step += alpha*d
            r = project(r-alpha*hd)
            # Stop on the physical residual, not its preconditioned norm.
            if linalg.norm(r) <= tol:
                return step
            z = metric(r)
            new_rz = np.dot(r, z)
            d = z+(new_rz/rz)*d
            rz = new_rz
        self.cg_limit_hits += 1
        return step

    def lowest_curvature(self, c, f, lambdas, u):
        """Lanczos in the feasible tangent space; O(n*ncv), not O(n^2) memory."""
        def project(v):
            v=self.tangent(c,v)
            return v-u@(u.T@v)
        rng=np.random.default_rng(self.options.seed)
        start=project(rng.normal(size=self.size))
        if linalg.norm(start)<1e-12:
            return 0.,np.zeros(self.size)
        def action(v):
            return project(self.hessian(c,f,lambdas,project(v)))
        operator=LinearOperator((self.size,self.size),matvec=action,dtype=float)
        values,vectors=eigsh(operator,k=1,which='SA',v0=start/linalg.norm(start),
                             tol=1e-6,ncv=min(20,self.size),maxiter=300)
        vector=project(vectors[:,0])
        return float(values[0]),vector/max(1e-30,linalg.norm(vector))

    def solve(self, target, start=None, callback=None):
        opt = self.options
        if isinstance(target,dict):
            if set(target)!=set(self.constraint_names): raise ValueError('target names must match the configured constraints')
            target=[target[name] for name in self.constraint_names]
        target = np.asarray(target, dtype=float)
        if target.shape != (len(self.active),) or not np.isfinite(target).all():
            raise ValueError("invalid constraint targets")
        c = tuple(x.copy() for x in (self.initial if start is None else start))
        if [x.shape for x in c] != self.shapes or any(not np.isfinite(x).all() for x in c):
            raise ValueError("invalid starting orbitals")
        c = self.retract(c, np.zeros(self.size))
        self.fock_evaluations = self.hessian_evaluations = 0
        self.cg_iterations = self.cg_limit_hits = self.preconditioner_evaluations = 0
        self.smallest_curvature = None
        self.stability_checked = False
        # Necessary spectral feasibility bounds, including empty/full species.
        for operator, t in zip(self.constraints,target):
            low,high=operator.bounds([x.shape[1] for x in c])
            if t < low-opt.constraint_tolerance or t > high+opt.constraint_tolerance:
                return self.result(c, target, "infeasible_target", 0, None, np.inf, 0.)
        c, ok = self.restore(c, target)
        rng = np.random.default_rng(opt.seed)
        for attempt in range(6):
            if ok:
                break
            perturb = self.tangent(c, rng.normal(size=self.size))
            trial = self.retract(c, perturb*(.08*(attempt+1)/max(1., linalg.norm(perturb))))
            trial, good = self.restore(trial, target)
            if good or linalg.norm((self.moments(trial)-target)/self.scales) < linalg.norm((self.moments(c)-target)/self.scales):
                c = trial
            ok = good
        if not ok:
            return self.result(c, target, "constraint_restoration_failed", 0, None, np.inf, 0.)
        e, f = self.evaluate(c)
        radius, step, change = opt.trust_radius, .05, 0.
        status, phase = "iteration_limit", "initial"
        for iteration in range(opt.max_iterations+1):
            g, lambdas, u = self.stationarity(c, f)
            norm = linalg.norm(g)
            error = np.max(np.abs(self.moments(c)-target), initial=0.)
            if callback:
                callback(dict(iteration=iteration, phase=phase, energy_MeV=e, gradient_norm=norm,
                              constraint_error=error, energy_change_MeV=change))
            if norm <= opt.gradient_tolerance and error <= opt.constraint_tolerance and abs(change) <= opt.energy_tolerance:
                if opt.check_stability:
                    try:
                        curvature,escape=self.lowest_curvature(c,f,lambdas,u)
                    except ArpackNoConvergence:
                        status="stability_check_failed"
                        break
                    self.smallest_curvature=curvature
                    self.stability_checked=True
                    if curvature < -opt.curvature_tolerance:
                        if iteration==opt.max_iterations:
                            break
                        best=None
                        for back in range(12):
                            for sign in (-1.,1.):
                                trial=self.retract(c,sign*.5**back*min(.2,radius)*escape)
                                trial,feasible=self.restore(trial,target)
                                if not feasible: continue
                                te,tf=self.evaluate(trial)
                                if te<e-1e-13*max(1.,abs(e)) and (best is None or te<best[1]):
                                    best=(trial,te,tf)
                            if best is not None: break
                        if best is None:
                            status="saddle_escape_failed"
                            break
                        change=best[1]-e
                        c,e,f=best
                        phase="negative_curvature"
                        self.stability_checked=False
                        continue
                status = "converged"
                break
            if iteration == opt.max_iterations:
                break
            # A stationary point reached after a large step needs no extra move;
            # verify it again with zero energy change, not a failed line search.
            if norm <= opt.gradient_tolerance:
                change = 0.
                continue
            phase = "gradient"
            direction = -g*min(step, radius/max(norm, 1e-30))
            if iteration < opt.diagonalization_steps:
                trial = []
                for x, field, op in zip(c, f, self.constraint_data(c)[1]):
                    effective = field+np.einsum("k,kab->ab", lambdas, op)
                    _, full = linalg.eigh(effective)
                    candidate = full[:, :x.shape[1]]
                    a, _, bt = linalg.svd(candidate.T @ x, full_matrices=False)
                    trial.append(candidate @ a @ bt)
                direction = self.pack([a-b for a,b in zip(trial,c)])
                phase = "diagonalization"
            elif opt.method == "hybrid" and iteration >= opt.diagonalization_steps+opt.gradient_steps:
                direction = self.newton_step(c, f, lambdas, u, g, radius)
                phase = "newton_cg"
            accepted = False
            for fallback in range(2):
                slope = np.dot(g, direction)
                if slope < 0:
                    for back in range(22):
                        alpha = .5**back
                        trial = self.retract(c, alpha*direction)
                        trial, feasible = self.restore(trial, target)
                        if not feasible:
                            continue
                        te, tf = self.evaluate(trial)
                        if te <= e+1e-4*alpha*slope+2e-14*max(1.,abs(e)):
                            accepted = True
                            break
                if accepted:
                    break
                direction = -g*min(step, radius/max(norm,1e-30))
                phase = "gradient_fallback"
            if not accepted:
                status = "line_search_failed"
                break
            old_g = self.tangent(trial, g)
            new_g, _, _ = self.stationarity(trial, tf)
            displacement = self.tangent(trial, self.pack([a-b for a,b in zip(trial,c)]))
            y = new_g-old_g
            curvature = np.dot(displacement, y)
            if curvature > 1e-16:
                step = np.clip(np.dot(displacement,displacement)/curvature, 1e-5, 2.)
            if phase == "newton_cg":
                radius = min(1., radius*1.5) if back == 0 else max(1e-5, radius*.5)
            change = te-e
            c,e,f = trial,te,tf
        result = self.result(c, target, status, iteration, e, norm, change)
        if result["converged"]:
            full = [linalg.qr(x, mode="full")[0] if x.shape[1] else np.eye(x.shape[0]) for x in c]
            self.hf.hybrid_accept(*full)
        return result

    def result(self, c, target, status, iteration, energy, norm, change):
        orth = max(np.max(np.abs(x.T @ x-np.eye(x.shape[1])),initial=0.) for x in c)
        idem = max(np.max(np.abs((x@x.T)@(x@x.T)-x@x.T),initial=0.) for x in c)
        return dict(converged=status=="converged", status=status, energy_MeV=energy,
                    iterations=iteration, gradient_norm=float(norm), energy_change_MeV=float(change),
                    max_constraint_error=float(np.max(np.abs(self.moments(c)-target),initial=0.)),
                    orthogonality_error=float(orth), idempotency_error=float(idem),
                    protons=float(np.sum(c[0]**2)), neutrons=float(np.sum(c[1]**2)),
                    fock_evaluations=self.fock_evaluations, hessian_evaluations=self.hessian_evaluations,
                    cg_iterations=self.cg_iterations, cg_limit_hits=self.cg_limit_hits,
                    preconditioner_evaluations=self.preconditioner_evaluations,
                    stability_checked=self.stability_checked, smallest_curvature=self.smallest_curvature,
                    moments=self.moments(c, True).tolist(),
                    constraint_values=dict(zip(self.constraint_names,self.moments(c).tolist())),occupied=c)
