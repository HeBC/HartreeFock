#include "HartreeFock.h"

#include <limits>

namespace
{
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

//*********************************************************************
// Modified Broyden acceleration in Thouless particle-hole space.
//
// Important convention:
//   The mixed vector below is a Thouless *displacement*, not the raw gradient.
//   For a simple stable step we need Z = -alpha * gradient.  The previous
//   implementation used the opposite sign, which explains the immediate energy
//   increase and the non-convergent plateau.
void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    if (history_size < 1) history_size = 1;
    if (alpha <= 0.0) alpha = 0.25;
    if (alpha > 1.0) alpha = 1.0;
    if (w0 <= 0.0) w0 = 0.01;

    const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
    const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
    const size_t nvec = np + nn;
    if (nvec == 0) return;

    const double inv_sqrt_nvec = 1.0 / std::sqrt(static_cast<double>(nvec));
    const double max_thouless_element = 5.0e-2;

    auto rebuild_fields = [&]()
    {
        UpdateDensityMatrix();
        UpdateF();
        CalcEHF();
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
        if (N_p > 0)
        {
            std::copy(U_p_save.begin(), U_p_save.end(), U_p);
            std::copy(rho_p_save.begin(), rho_p_save.end(), rho_p);
            std::copy(Fock_p_save.begin(), Fock_p_save.end(), FockTerm_p);
            std::copy(Vij_p_save.begin(), Vij_p_save.end(), Vij_p);
        }
        if (N_n > 0)
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

    auto apply_step = [&](const std::vector<double>& step)
    {
        std::vector<double> Zp(np, 0.0);
        std::vector<double> Zn(nn, 0.0);
        if (N_p > 0) std::copy(step.begin(), step.begin() + np, Zp.begin());
        if (N_n > 0) std::copy(step.begin() + np, step.end(), Zn.begin());

        // The first-order Thouless update is used here deliberately.  It has the
        // same convention as the existing gradient solver and avoids a possible
        // left/right convention mismatch in the Padé update while debugging HF.
        UpdateU_Thouless_1st(Zp.data(), Zn.data());
        rebuild_fields();
    };

    // Start from a diagonalized HF field once, then solve by orbital rotations.
    rebuild_fields();
    Diagonalize();
    rebuild_fields();

    std::vector<double> residual = compute_displacement_residual();
    double rms = vector_norm(residual) * inv_sqrt_nvec;

    std::vector<double> residual_prev;
    std::vector<double> step_prev;
    std::deque<std::vector<double>> dF_history;
    std::deque<std::vector<double>> u_history;
    std::deque<double> w_history;

    std::cout << "  Modified Broyden HF debug: vector = Thouless displacement"
              << "  history_size = " << history_size
              << "  alpha = " << alpha
              << "  w0 = " << w0
              << "  tolerance = " << tolerance << std::endl;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        const double E_start = EHF;
        const double l1 = l1_norm(residual);

        if (iterations < 10 || iterations % 10 == 0 || rms < tolerance)
        {
            std::cout << "  Broyden iter " << std::setw(5) << iterations
                      << "  grad_l1 = " << std::scientific << std::setprecision(6) << l1
                      << "  grad_rms = " << rms
                      << "  E = " << EHF
                      << "  hist = " << dF_history.size()
                      << std::defaultfloat << std::endl;
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
        double best_E = std::numeric_limits<double>::infinity();
        std::vector<double> best_step;
        std::vector<double> best_residual;
        bool accepted = false;

        // Backtracking over Broyden step, then a plain damped gradient step as a fallback.
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
                apply_step(trial);
                std::vector<double> r_trial = compute_displacement_residual();
                const double rms_trial = vector_norm(r_trial) * inv_sqrt_nvec;

                if (std::isfinite(rms_trial) && rms_trial < best_rms)
                {
                    best_rms = rms_trial;
                    best_E = EHF;
                    best_step = trial;
                    best_residual = std::move(r_trial);
                }

                if (std::isfinite(rms_trial) && (rms_trial < rms || EHF < E_start))
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
            // Keep progress if this is the best available trial, but discard memory.
            dF_history.clear();
            u_history.clear();
            w_history.clear();
        }

        apply_step(best_step);
        std::vector<double> new_residual = compute_displacement_residual();
        const double new_rms = vector_norm(new_residual) * inv_sqrt_nvec;

        if (iterations < 10 || iterations % 10 == 0)
        {
            std::cout << "                 step = "
                      << (proposed_broyden ? "Broyden-Thouless" : "linear-Thouless")
                      << "  trial_rms = " << std::scientific << std::setprecision(6) << new_rms
                      << "  dE = " << std::fabs(EHF - E_start)
                      << std::defaultfloat << std::endl;
        }

        residual_prev = residual;
        step_prev = best_step;
        residual = std::move(new_residual);
        rms = new_rms;

        if (!accepted && best_rms > 1.2 * rms)
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
        std::cout << "  HF converged with Thouless-gradient modified Broyden after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation did not converge with Thouless-gradient modified Broyden after "
                  << iterations << " iterations.\033[0m" << std::endl << std::endl;
    }
    PrintEHF();
}
