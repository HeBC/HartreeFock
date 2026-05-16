#include "HartreeFock.h"

#include <limits>

namespace
{
    struct ConstraintData
    {
        std::vector<std::string> names;
        std::vector<double> targets;
        std::vector<std::vector<double>> Qp;
        std::vector<std::vector<double>> Qn;
    };

    double dot_product(const std::vector<double>& a, const std::vector<double>& b)
    {
        double v = 0.0;
        for (size_t i = 0; i < a.size(); ++i) v += a[i] * b[i];
        return v;
    }

    double vector_norm(const std::vector<double>& a)
    {
        return std::sqrt(std::max(0.0, dot_product(a, a)));
    }

    double l1_norm(const std::vector<double>& a)
    {
        double v = 0.0;
        for (double x : a) v += std::fabs(x);
        return v;
    }

    bool all_finite(const std::vector<double>& a)
    {
        for (double x : a)
        {
            if (!std::isfinite(x)) return false;
        }
        return true;
    }

    void zero_hh_pp_block(std::vector<double>& z, size_t offset, int dim, int n_holes)
    {
        for (int i = 0; i < dim; ++i)
        {
            const bool occ_i = (i < n_holes);
            for (int j = 0; j < dim; ++j)
            {
                const bool occ_j = (j < n_holes);
                if (occ_i == occ_j)
                    z[offset + static_cast<size_t>(i) * dim + j] = 0.0;
            }
        }
    }

    void enforce_antisymmetric_block(std::vector<double>& z, size_t offset, int dim)
    {
        for (int i = 0; i < dim; ++i)
        {
            z[offset + static_cast<size_t>(i) * dim + i] = 0.0;
            for (int j = i + 1; j < dim; ++j)
            {
                const size_t ij = offset + static_cast<size_t>(i) * dim + j;
                const size_t ji = offset + static_cast<size_t>(j) * dim + i;
                const double a = 0.5 * (z[ij] - z[ji]);
                z[ij] = a;
                z[ji] = -a;
            }
        }
    }

    void enforce_thouless_generator(std::vector<double>& z, int dim_p, int dim_n, int N_p, int N_n)
    {
        const size_t np = static_cast<size_t>(dim_p) * dim_p;
        zero_hh_pp_block(z, 0, dim_p, N_p);
        zero_hh_pp_block(z, np, dim_n, N_n);
        enforce_antisymmetric_block(z, 0, dim_p);
        enforce_antisymmetric_block(z, np, dim_n);
    }

    void cap_generator_element(std::vector<double>& z, double max_abs_allowed)
    {
        double max_abs = 0.0;
        for (double x : z) max_abs = std::max(max_abs, std::fabs(x));
        if (max_abs > max_abs_allowed && max_abs > 0.0)
        {
            const double scale = max_abs_allowed / max_abs;
            for (double& x : z) x *= scale;
        }
    }

    bool solve_linear_system(std::vector<double> A, std::vector<double>& b, int n)
    {
        const double tol = 1.0e-14;
        for (int k = 0; k < n; ++k)
        {
            int piv = k;
            double mx = std::fabs(A[static_cast<size_t>(k) * n + k]);
            for (int i = k + 1; i < n; ++i)
            {
                const double v = std::fabs(A[static_cast<size_t>(i) * n + k]);
                if (v > mx)
                {
                    mx = v;
                    piv = i;
                }
            }
            if (mx < tol) return false;
            if (piv != k)
            {
                for (int j = k; j < n; ++j)
                    std::swap(A[static_cast<size_t>(k) * n + j], A[static_cast<size_t>(piv) * n + j]);
                std::swap(b[k], b[piv]);
            }
            const double diag = A[static_cast<size_t>(k) * n + k];
            for (int i = k + 1; i < n; ++i)
            {
                const double f = A[static_cast<size_t>(i) * n + k] / diag;
                A[static_cast<size_t>(i) * n + k] = 0.0;
                for (int j = k + 1; j < n; ++j)
                    A[static_cast<size_t>(i) * n + j] -= f * A[static_cast<size_t>(k) * n + j];
                b[i] -= f * b[k];
            }
        }
        for (int i = n - 1; i >= 0; --i)
        {
            double rhs = b[i];
            for (int j = i + 1; j < n; ++j)
                rhs -= A[static_cast<size_t>(i) * n + j] * b[j];
            const double diag = A[static_cast<size_t>(i) * n + i];
            if (std::fabs(diag) < tol) return false;
            b[i] = rhs / diag;
        }
        return true;
    }
}

void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    Solve_broyden_impl(history_size, alpha, w0, false, 0.0);
}

void HartreeFock::Solve_broyden_Constraint(int history_size, double alpha, double w0, double constraint_strength)
{
    Solve_broyden_impl(history_size, alpha, w0, true, constraint_strength);
}

//*********************************************************************
// Modified Broyden acceleration in Thouless particle-hole space.
//
// The Broyden vector is the Thouless displacement that would reduce the
// particle-hole Fock residual.  The sign is chosen so the linear step is the
// same descent direction as the working gradient solver.
void HartreeFock::Solve_broyden_impl(int history_size, double alpha, double w0,
                                     bool use_constraints, double constraint_strength)
{
    if (history_size < 1) history_size = 1;
    if (alpha <= 0.0) alpha = use_constraints ? 0.20 : 0.25;
    if (alpha > 1.0) alpha = 1.0;
    if (w0 <= 0.0) w0 = 0.01;
    if (constraint_strength < 0.0) constraint_strength = 0.0;

    const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
    const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
    const size_t nvec = np + nn;
    if (nvec == 0) return;

    const double inv_sqrt_nvec = 1.0 / std::sqrt(static_cast<double>(nvec));
    const double max_thouless_element = use_constraints ? 3.0e-2 : 5.0e-2;

    ConstraintData constraints;
    if (use_constraints)
    {
        if (modelspace->GetIsShapeConstrained())
        {
            constraints.names.push_back("QuadrupoleQ0");
            constraints.targets.push_back(modelspace->GetShapeQ0());
            constraints.names.push_back("QuadrupoleQ2");
            constraints.targets.push_back(modelspace->GetShapeQ2());
        }
        if (modelspace->Get_Jx_constraint())
        {
            constraints.names.push_back("Jx");
            constraints.targets.push_back(std::sqrt(modelspace->GetTargetJx()));
        }
        if (modelspace->Get_Jz_constraint())
        {
            constraints.names.push_back("Jz");
            constraints.targets.push_back(modelspace->GetTargetJz());
        }

        if (constraints.names.empty())
        {
            std::cout << "  No active constraints. Falling back to unconstrained Broyden." << std::endl;
            use_constraints = false;
        }
        else
        {
            constraints.Qp.assign(constraints.names.size(), std::vector<double>(np, 0.0));
            constraints.Qn.assign(constraints.names.size(), std::vector<double>(nn, 0.0));

            for (size_t iq = 0; iq < constraints.names.size(); ++iq)
            {
                if (constraints.names[iq] == "QuadrupoleQ0")
                {
                    for (size_t k = 0; k < Ham->Q2MEs_p.Q0_list.size(); ++k)
                    {
                        const int qidx = Ham->Q2MEs_p.Q0_list[k];
                        const int ia = Ham->MSMEs.OB_p[qidx].GetIndex_a();
                        const int ib = Ham->MSMEs.OB_p[qidx].GetIndex_b();
                        constraints.Qp[iq][ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[k];
                    }
                    for (size_t k = 0; k < Ham->Q2MEs_n.Q0_list.size(); ++k)
                    {
                        const int qidx = Ham->Q2MEs_n.Q0_list[k];
                        const int ia = Ham->MSMEs.OB_n[qidx].GetIndex_a();
                        const int ib = Ham->MSMEs.OB_n[qidx].GetIndex_b();
                        constraints.Qn[iq][ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[k];
                    }
                }
                else if (constraints.names[iq] == "QuadrupoleQ2")
                {
                    for (size_t k = 0; k < Ham->Q2MEs_p.Q2_list.size(); ++k)
                    {
                        const int qidx = Ham->Q2MEs_p.Q2_list[k];
                        const int ia = Ham->MSMEs.OB_p[qidx].GetIndex_a();
                        const int ib = Ham->MSMEs.OB_p[qidx].GetIndex_b();
                        constraints.Qp[iq][ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[k];
                    }
                    for (size_t k = 0; k < Ham->Q2MEs_p.Q_2_list.size(); ++k)
                    {
                        const int qidx = Ham->Q2MEs_p.Q_2_list[k];
                        const int ia = Ham->MSMEs.OB_p[qidx].GetIndex_a();
                        const int ib = Ham->MSMEs.OB_p[qidx].GetIndex_b();
                        constraints.Qp[iq][ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[k];
                    }
                    for (size_t k = 0; k < Ham->Q2MEs_n.Q2_list.size(); ++k)
                    {
                        const int qidx = Ham->Q2MEs_n.Q2_list[k];
                        const int ia = Ham->MSMEs.OB_n[qidx].GetIndex_a();
                        const int ib = Ham->MSMEs.OB_n[qidx].GetIndex_b();
                        constraints.Qn[iq][ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[k];
                    }
                    for (size_t k = 0; k < Ham->Q2MEs_n.Q_2_list.size(); ++k)
                    {
                        const int qidx = Ham->Q2MEs_n.Q_2_list[k];
                        const int ia = Ham->MSMEs.OB_n[qidx].GetIndex_a();
                        const int ib = Ham->MSMEs.OB_n[qidx].GetIndex_b();
                        constraints.Qn[iq][ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[k];
                    }
                }
                else if (constraints.names[iq] == "Jx")
                {
                    for (int i = 0; i < dim_p; ++i)
                    {
                        for (int j = 0; j < dim_p; ++j)
                        {
                            const int oi = modelspace->Get_ProtonOrbitIndexInMscheme(i);
                            const int oj = modelspace->Get_ProtonOrbitIndexInMscheme(j);
                            if (modelspace->Orbits_p[oi].l != modelspace->Orbits_p[oj].l) continue;
                            if (modelspace->Orbits_p[oi].n != modelspace->Orbits_p[oj].n) continue;
                            const int ji = modelspace->Get_MSmatrix_2j(Proton, i);
                            const int jj = modelspace->Get_MSmatrix_2j(Proton, j);
                            if (ji != jj) continue;
                            const int mi = modelspace->Get_MSmatrix_2m(Proton, i);
                            const int mj = modelspace->Get_MSmatrix_2m(Proton, j);
                            if (mi == mj + 2)
                                constraints.Qp[iq][i * dim_p + j] = 0.5 * std::sqrt((jj - mj) * (jj + mj + 2.0) / 4.0);
                            else if (mi == mj - 2)
                                constraints.Qp[iq][i * dim_p + j] = 0.5 * std::sqrt((jj + mj) * (jj - mj + 2.0) / 4.0);
                        }
                    }
                    for (int i = 0; i < dim_n; ++i)
                    {
                        for (int j = 0; j < dim_n; ++j)
                        {
                            const int oi = modelspace->Get_NeutronOrbitIndexInMscheme(i);
                            const int oj = modelspace->Get_NeutronOrbitIndexInMscheme(j);
                            if (modelspace->Orbits_n[oi].l != modelspace->Orbits_n[oj].l) continue;
                            if (modelspace->Orbits_n[oi].n != modelspace->Orbits_n[oj].n) continue;
                            const int ji = modelspace->Get_MSmatrix_2j(Neutron, i);
                            const int jj = modelspace->Get_MSmatrix_2j(Neutron, j);
                            if (ji != jj) continue;
                            const int mi = modelspace->Get_MSmatrix_2m(Neutron, i);
                            const int mj = modelspace->Get_MSmatrix_2m(Neutron, j);
                            if (mi == mj + 2)
                                constraints.Qn[iq][i * dim_n + j] = 0.5 * std::sqrt((jj - mj) * (jj + mj + 2.0) / 4.0);
                            else if (mi == mj - 2)
                                constraints.Qn[iq][i * dim_n + j] = 0.5 * std::sqrt((jj + mj) * (jj - mj + 2.0) / 4.0);
                        }
                    }
                }
                else if (constraints.names[iq] == "Jz")
                {
                    for (int i = 0; i < dim_p; ++i)
                    {
                        constraints.Qp[iq][i * dim_p + i] = 0.5 * modelspace->Get_MSmatrix_2m(Proton, i);
                    }
                    for (int i = 0; i < dim_n; ++i)
                    {
                        constraints.Qn[iq][i * dim_n + i] = 0.5 * modelspace->Get_MSmatrix_2m(Neutron, i);
                    }
                }
            }
        }
    }

    auto constraint_penalty = [&]() -> double
    {
        if (!use_constraints) return 0.0;
        double penalty = 0.0;
        for (size_t iq = 0; iq < constraints.names.size(); ++iq)
        {
            const double qp = (np > 0) ? cblas_ddot(static_cast<int>(np), rho_p, 1, constraints.Qp[iq].data(), 1) : 0.0;
            const double qn = (nn > 0) ? cblas_ddot(static_cast<int>(nn), rho_n, 1, constraints.Qn[iq].data(), 1) : 0.0;
            const double dq = qp + qn - constraints.targets[iq];
            penalty += 0.5 * constraint_strength * dq * dq;
        }
        return penalty;
    };

    auto apply_constraint_field = [&]()
    {
        if (!use_constraints) return;
        for (size_t iq = 0; iq < constraints.names.size(); ++iq)
        {
            const double qp = (np > 0) ? cblas_ddot(static_cast<int>(np), rho_p, 1, constraints.Qp[iq].data(), 1) : 0.0;
            const double qn = (nn > 0) ? cblas_ddot(static_cast<int>(nn), rho_n, 1, constraints.Qn[iq].data(), 1) : 0.0;
            const double dq = qp + qn - constraints.targets[iq];
            const double lambda = constraint_strength * dq;
            if (np > 0) cblas_daxpy(static_cast<int>(np), lambda, constraints.Qp[iq].data(), 1, FockTerm_p, 1);
            if (nn > 0) cblas_daxpy(static_cast<int>(nn), lambda, constraints.Qn[iq].data(), 1, FockTerm_n, 1);
        }
    };

    auto rebuild_fields = [&]() -> double
    {
        UpdateDensityMatrix();
        UpdateF();
        CalcEHF();
        const double obj = EHF + constraint_penalty();
        apply_constraint_field();
        return obj;
    };

    auto compute_displacement_residual = [&]() -> std::vector<double>
    {
        // Use copies because TransferOperatorToHFbasis mutates its arguments.
        std::vector<double> Fock_orb_p(FockTerm_p, FockTerm_p + dim_p * dim_p);
        std::vector<double> Fock_orb_n(FockTerm_n, FockTerm_n + dim_n * dim_n);
        TransferOperatorToHFbasis(Fock_orb_p.data(), Fock_orb_n.data());

        std::vector<double> r(nvec, 0.0);

        for (int h = 0; h < N_p; ++h)
        {
            for (int a = N_p; a < dim_p; ++a)
            {
                const double denom = std::fabs(Fock_orb_p[a * dim_p + a] - Fock_orb_p[h * dim_p + h]);
                double z = -Fock_orb_p[a * dim_p + h];
                if (denom > 1.0e-5) z /= denom;
                r[static_cast<size_t>(a) * dim_p + h] = z;
                r[static_cast<size_t>(h) * dim_p + a] = -z;
            }
        }

        for (int h = 0; h < N_n; ++h)
        {
            for (int a = N_n; a < dim_n; ++a)
            {
                const double denom = std::fabs(Fock_orb_n[a * dim_n + a] - Fock_orb_n[h * dim_n + h]);
                double z = -Fock_orb_n[a * dim_n + h];
                if (denom > 1.0e-5) z /= denom;
                r[np + static_cast<size_t>(a) * dim_n + h] = z;
                r[np + static_cast<size_t>(h) * dim_n + a] = -z;
            }
        }

        enforce_thouless_generator(r, dim_p, dim_n, N_p, N_n);
        return r;
    };

    auto restore_state = [&](const std::vector<double>& U_p_save,
                             const std::vector<double>& U_n_save,
                             const std::vector<double>& rho_p_save,
                             const std::vector<double>& rho_n_save,
                             const std::vector<double>& Fock_p_save,
                             const std::vector<double>& Fock_n_save,
                             const std::vector<double>& Vij_p_save,
                             const std::vector<double>& Vij_n_save,
                             double E_save, double e1_save, double e2_save)
    {
        if (np > 0)
        {
            std::copy(U_p_save.begin(), U_p_save.end(), U_p);
            std::copy(rho_p_save.begin(), rho_p_save.end(), rho_p);
            std::copy(Fock_p_save.begin(), Fock_p_save.end(), FockTerm_p);
            std::copy(Vij_p_save.begin(), Vij_p_save.end(), Vij_p);
        }
        if (nn > 0)
        {
            std::copy(U_n_save.begin(), U_n_save.end(), U_n);
            std::copy(rho_n_save.begin(), rho_n_save.end(), rho_n);
            std::copy(Fock_n_save.begin(), Fock_n_save.end(), FockTerm_n);
            std::copy(Vij_n_save.begin(), Vij_n_save.end(), Vij_n);
        }
        EHF = E_save;
        e1hf = e1_save;
        e2hf = e2_save;
    };

    auto apply_step = [&](const std::vector<double>& step) -> double
    {
        std::vector<double> Zp(np, 0.0);
        std::vector<double> Zn(nn, 0.0);
        if (np > 0) std::copy(step.begin(), step.begin() + np, Zp.begin());
        if (nn > 0) std::copy(step.begin() + np, step.end(), Zn.begin());

        // Keep the same convention as the validated gradient solver.
        UpdateU_Thouless_1st(Zp.data(), Zn.data());
        return rebuild_fields();
    };

    // Start from a diagonalized HF field once, then solve by orbital rotations.
    double objective = rebuild_fields();
    Diagonalize();
    objective = rebuild_fields();

    std::vector<double> residual = compute_displacement_residual();
    double rms = vector_norm(residual) * inv_sqrt_nvec;

    std::vector<double> residual_prev;
    std::vector<double> step_prev;
    std::deque<std::vector<double>> dF_history;
    std::deque<std::vector<double>> u_history;
    std::deque<double> w_history;

    std::cout << "  Modified Broyden HF: "
              << (use_constraints ? "constrained " : "")
              << "Thouless displacement, alpha=" << alpha
              << ", history=" << history_size
              << ", w0=" << w0;
    if (use_constraints) std::cout << ", k=" << constraint_strength;
    std::cout << std::endl;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        const double objective_start = objective;

        if (iterations < 5 || iterations % 10 == 0 || rms < tolerance)
        {
            std::cout << "    iter " << std::setw(4) << iterations
                      << "  rms=" << std::scientific << std::setprecision(3) << rms
                      << "  l1=" << l1_norm(residual)
                      << "  E=" << EHF;
            if (use_constraints) std::cout << "  Epen=" << objective;
            std::cout << "  hist=" << dF_history.size() << std::defaultfloat << std::endl;
        }

        if (rms < tolerance) break;

        if (!residual_prev.empty() && !step_prev.empty())
        {
            std::vector<double> dF(nvec, 0.0);
            for (size_t i = 0; i < nvec; ++i) dF[i] = residual[i] - residual_prev[i];

            const double dF_norm = vector_norm(dF);
            if (dF_norm > 1.0e-14)
            {
                std::vector<double> dx = step_prev;
                for (size_t i = 0; i < nvec; ++i)
                {
                    dF[i] /= dF_norm;
                    dx[i] /= dF_norm;
                }

                std::vector<double> u(nvec, 0.0);
                for (size_t i = 0; i < nvec; ++i) u[i] = alpha * dF[i] + dx[i];
                enforce_thouless_generator(u, dim_p, dim_n, N_p, N_n);

                const double rms_prev = std::max(1.0e-14, vector_norm(residual_prev) * inv_sqrt_nvec);
                const double weight = std::max(1.0, 1.0 / rms_prev);

                dF_history.push_back(std::move(dF));
                u_history.push_back(std::move(u));
                w_history.push_back(weight);
                while (static_cast<int>(dF_history.size()) > history_size)
                {
                    dF_history.pop_front();
                    u_history.pop_front();
                    w_history.pop_front();
                }
            }
        }

        std::vector<double> step(nvec, 0.0);
        for (size_t i = 0; i < nvec; ++i) step[i] = alpha * residual[i];

        bool proposed_broyden = false;
        const int m = static_cast<int>(dF_history.size());
        if (m >= 1)
        {
            std::vector<double> B(static_cast<size_t>(m) * static_cast<size_t>(m), 0.0);
            std::vector<double> gamma(m, 0.0);
            for (int i = 0; i < m; ++i)
            {
                gamma[i] = w_history[i] * dot_product(dF_history[i], residual);
                for (int j = 0; j < m; ++j)
                    B[static_cast<size_t>(i) * m + j] = w_history[i] * w_history[j] * dot_product(dF_history[i], dF_history[j]);
                B[static_cast<size_t>(i) * m + i] += w0 * w0;
            }

            if (solve_linear_system(B, gamma, m) && all_finite(gamma))
            {
                std::vector<double> candidate = step;
                for (int ih = 0; ih < m; ++ih)
                {
                    const double coeff = w_history[ih] * gamma[ih];
                    for (size_t i = 0; i < nvec; ++i) candidate[i] -= coeff * u_history[ih][i];
                }
                enforce_thouless_generator(candidate, dim_p, dim_n, N_p, N_n);
                if (all_finite(candidate))
                {
                    step.swap(candidate);
                    proposed_broyden = true;
                }
            }
            else
            {
                dF_history.clear();
                u_history.clear();
                w_history.clear();
            }
        }

        enforce_thouless_generator(step, dim_p, dim_n, N_p, N_n);
        cap_generator_element(step, max_thouless_element);

        const std::vector<double> U_p_save(U_p, U_p + dim_p * dim_p);
        const std::vector<double> U_n_save(U_n, U_n + dim_n * dim_n);
        const std::vector<double> rho_p_save(rho_p, rho_p + dim_p * dim_p);
        const std::vector<double> rho_n_save(rho_n, rho_n + dim_n * dim_n);
        const std::vector<double> Fock_p_save(FockTerm_p, FockTerm_p + dim_p * dim_p);
        const std::vector<double> Fock_n_save(FockTerm_n, FockTerm_n + dim_n * dim_n);
        const std::vector<double> Vij_p_save(Vij_p, Vij_p + dim_p * dim_p);
        const std::vector<double> Vij_n_save(Vij_n, Vij_n + dim_n * dim_n);
        const double E_save = EHF;
        const double e1_save = e1hf;
        const double e2_save = e2hf;

        double best_rms = std::numeric_limits<double>::infinity();
        double best_objective = std::numeric_limits<double>::infinity();
        std::vector<double> best_step;
        bool accepted = false;

        for (int family = 0; family < 2; ++family)
        {
            std::vector<double> base = step;
            if (family == 1)
            {
                base.assign(nvec, 0.0);
                for (size_t i = 0; i < nvec; ++i) base[i] = alpha * residual[i];
                enforce_thouless_generator(base, dim_p, dim_n, N_p, N_n);
                cap_generator_element(base, max_thouless_element);
            }

            for (double scale : {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125})
            {
                restore_state(U_p_save, U_n_save, rho_p_save, rho_n_save,
                              Fock_p_save, Fock_n_save, Vij_p_save, Vij_n_save,
                              E_save, e1_save, e2_save);

                std::vector<double> trial = base;
                for (double& x : trial) x *= scale;
                const double obj_trial = apply_step(trial);
                std::vector<double> r_trial = compute_displacement_residual();
                const double rms_trial = vector_norm(r_trial) * inv_sqrt_nvec;

                if (std::isfinite(rms_trial) && std::isfinite(obj_trial)
                    && (rms_trial < best_rms || obj_trial < best_objective))
                {
                    best_rms = rms_trial;
                    best_objective = obj_trial;
                    best_step = trial;
                }

                if (std::isfinite(rms_trial) && std::isfinite(obj_trial)
                    && (rms_trial < 0.995 * rms || obj_trial < objective_start))
                {
                    accepted = true;
                    break;
                }
            }
            if (accepted) break;
        }

        restore_state(U_p_save, U_n_save, rho_p_save, rho_n_save,
                      Fock_p_save, Fock_n_save, Vij_p_save, Vij_n_save,
                      E_save, e1_save, e2_save);

        if (best_step.empty())
        {
            dF_history.clear();
            u_history.clear();
            w_history.clear();
            best_step.assign(nvec, 0.0);
            for (size_t i = 0; i < nvec; ++i) best_step[i] = 0.05 * alpha * residual[i];
        }
        else if (!accepted)
        {
            dF_history.clear();
            u_history.clear();
            w_history.clear();
        }

        objective = apply_step(best_step);
        std::vector<double> new_residual = compute_displacement_residual();
        const double new_rms = vector_norm(new_residual) * inv_sqrt_nvec;

        residual_prev = residual;
        step_prev = best_step;
        residual = std::move(new_residual);
        rms = new_rms;

        if (proposed_broyden && !accepted)
        {
            dF_history.clear();
            u_history.clear();
            w_history.clear();
        }
    }

    rebuild_fields();
    Diagonalize();
    rebuild_fields();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged with " << (use_constraints ? "constrained " : "")
                  << "Thouless-gradient modified Broyden after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation did not converge with "
                  << (use_constraints ? "constrained " : "")
                  << "Thouless-gradient modified Broyden after "
                  << iterations << " iterations.\033[0m" << std::endl << std::endl;
    }

    if (use_constraints)
    {
        std::cout << "  Constraints:" << std::endl;
        for (size_t iq = 0; iq < constraints.names.size(); ++iq)
        {
            const double qp = (np > 0) ? cblas_ddot(static_cast<int>(np), rho_p, 1, constraints.Qp[iq].data(), 1) : 0.0;
            const double qn = (nn > 0) ? cblas_ddot(static_cast<int>(nn), rho_n, 1, constraints.Qn[iq].data(), 1) : 0.0;
            const double q = qp + qn;
            std::cout << "    " << std::setw(14) << constraints.names[iq]
                      << "  <Q>=" << std::fixed << std::setprecision(6) << q
                      << "  target=" << constraints.targets[iq]
                      << "  dQ=" << (q - constraints.targets[iq])
                      << std::defaultfloat << std::endl;
        }
    }

    PrintEHF();
}
