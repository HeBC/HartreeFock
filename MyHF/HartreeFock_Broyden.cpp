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
        return std::sqrt(dot_product(a, a));
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

    void pack_blocks(const double* p, const double* n, int dim_p, int dim_n, std::vector<double>& x)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
        x.resize(np + nn);
        std::copy(p, p + np, x.begin());
        std::copy(n, n + nn, x.begin() + np);
    }

    void unpack_blocks(const std::vector<double>& x, int dim_p, int dim_n, double* p, double* n)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
        std::copy(x.begin(), x.begin() + np, p);
        std::copy(x.begin() + np, x.begin() + np + nn, n);
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
                {
                    z[offset + static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j)] = 0.0;
                }
            }
        }
    }

    void enforce_antisymmetric_block(std::vector<double>& z, size_t offset, int dim)
    {
        for (int i = 0; i < dim; ++i)
        {
            z[offset + static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(i)] = 0.0;
            for (int j = i + 1; j < dim; ++j)
            {
                const size_t ij = offset + static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j);
                const size_t ji = offset + static_cast<size_t>(j) * static_cast<size_t>(dim) + static_cast<size_t>(i);
                const double a = 0.5 * (z[ij] - z[ji]);
                z[ij] = a;
                z[ji] = -a;
            }
        }
    }

    void enforce_thouless_generator(std::vector<double>& z, int dim_p, int dim_n, int N_p, int N_n)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        zero_hh_pp_block(z, 0, dim_p, N_p);
        zero_hh_pp_block(z, np, dim_n, N_n);
        enforce_antisymmetric_block(z, 0, dim_p);
        enforce_antisymmetric_block(z, np, dim_n);
    }

    void cap_generator_norm(std::vector<double>& z, double max_norm)
    {
        double max_abs = 0.0;
        for (double x : z) max_abs = std::max(max_abs, std::fabs(x));
        if (max_abs > max_norm && max_abs > 0.0)
        {
            const double scale = max_norm / max_abs;
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
                {
                    std::swap(A[static_cast<size_t>(k) * n + j], A[static_cast<size_t>(piv) * n + j]);
                }
                std::swap(b[k], b[piv]);
            }
            const double diag = A[static_cast<size_t>(k) * n + k];
            for (int i = k + 1; i < n; ++i)
            {
                const double f = A[static_cast<size_t>(i) * n + k] / diag;
                A[static_cast<size_t>(i) * n + k] = 0.0;
                for (int j = k + 1; j < n; ++j)
                {
                    A[static_cast<size_t>(i) * n + j] -= f * A[static_cast<size_t>(k) * n + j];
                }
                b[i] -= f * b[k];
            }
        }
        for (int i = n - 1; i >= 0; --i)
        {
            double rhs = b[i];
            for (int j = i + 1; j < n; ++j)
            {
                rhs -= A[static_cast<size_t>(i) * n + j] * b[j];
            }
            const double diag = A[static_cast<size_t>(i) * n + i];
            if (std::fabs(diag) < tol) return false;
            b[i] = rhs / diag;
        }
        return true;
    }
}

//*********************************************************************
// Modified Broyden acceleration in the Thouless-gradient space.
//
// The HF variational condition is the vanishing particle-hole block of the
// Fock matrix in the current HF basis.  This is the vector that should be mixed
// in this class, because UpdateU_Thouless_pade() updates the orbital manifold
// directly and keeps the density idempotent.  Mixing the full density or the
// full Fock field includes large hh/pp pieces that are not the HF residual and
// pollutes the Broyden metric.
void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    if (history_size < 1) history_size = 1;
    if (alpha <= 0.0) alpha = 0.35;
    if (alpha > 1.0) alpha = 1.0;
    if (w0 <= 0.0) w0 = 0.01;

    const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
    const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
    const size_t nvec = np + nn;
    const double inv_sqrt_nvec = 1.0 / std::sqrt(static_cast<double>(nvec));
    const double max_thouless_element = 0.35;

    std::vector<double> residual(nvec, 0.0);
    std::vector<double> residual_prev;
    std::vector<double> step_prev;

    std::deque<std::vector<double>> dF_history;
    std::deque<std::vector<double>> u_history;
    std::deque<double> w_history;

    UpdateDensityMatrix();
    UpdateF();
    Diagonalize();
    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

    double previous_energy = EHF;
    double previous_rms = std::numeric_limits<double>::infinity();

    std::cout << "  Modified Broyden HF debug: vector = Thouless ph-gradient"
              << "  history_size = " << history_size
              << "  alpha = " << alpha
              << "  w0 = " << w0
              << "  tolerance = " << tolerance << std::endl;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        // FockTerm is rebuilt in the original m-scheme basis at the end of each
        // iteration.  Transform a copy of that field into the current HF basis;
        // the ph block is the actual HF residual.
        std::vector<double> Fock_orb_p(FockTerm_p, FockTerm_p + dim_p * dim_p);
        std::vector<double> Fock_orb_n(FockTerm_n, FockTerm_n + dim_n * dim_n);
        TransferOperatorToHFbasis(Fock_orb_p.data(), Fock_orb_n.data());

        std::fill(residual.begin(), residual.end(), 0.0);

        for (int i = 0; i < dim_p; ++i)
        {
            const bool occ_i = (i < N_p);
            for (int j = 0; j < dim_p; ++j)
            {
                const bool occ_j = (j < N_p);
                if (occ_i == occ_j) continue;

                const double denom = std::fabs(Fock_orb_p[i * dim_p + i] - Fock_orb_p[j * dim_p + j]);
                double value = Fock_orb_p[i * dim_p + j] * (occ_i ? 1.0 : -1.0);
                if (denom > 1.0e-5) value /= denom;
                residual[static_cast<size_t>(i) * static_cast<size_t>(dim_p) + static_cast<size_t>(j)] = value;
            }
        }

        for (int i = 0; i < dim_n; ++i)
        {
            const bool occ_i = (i < N_n);
            for (int j = 0; j < dim_n; ++j)
            {
                const bool occ_j = (j < N_n);
                if (occ_i == occ_j) continue;

                const double denom = std::fabs(Fock_orb_n[i * dim_n + i] - Fock_orb_n[j * dim_n + j]);
                double value = Fock_orb_n[i * dim_n + j] * (occ_i ? 1.0 : -1.0);
                if (denom > 1.0e-5) value /= denom;
                residual[np + static_cast<size_t>(i) * static_cast<size_t>(dim_n) + static_cast<size_t>(j)] = value;
            }
        }

        enforce_thouless_generator(residual, dim_p, dim_n, N_p, N_n);

        const double grad_l1 = l1_norm(residual);
        const double grad_rms = vector_norm(residual) * inv_sqrt_nvec;
        const double dE = std::fabs(EHF - previous_energy);

        if (iterations < 10 || iterations % 10 == 0 || grad_rms < tolerance || dE < tolerance)
        {
            std::cout << "  Broyden iter " << std::setw(5) << iterations
                      << "  grad_l1 = " << std::scientific << std::setprecision(6) << grad_l1
                      << "  grad_rms = " << grad_rms
                      << "  dE = " << dE
                      << "  E = " << EHF
                      << "  hist = " << dF_history.size()
                      << std::defaultfloat << std::endl;
        }

        if (iterations > 1 && (grad_rms < tolerance || dE < tolerance))
        {
            break;
        }

        if (!residual_prev.empty() && !step_prev.empty())
        {
            std::vector<double> dF(nvec, 0.0);
            for (size_t i = 0; i < nvec; ++i)
            {
                dF[i] = residual[i] - residual_prev[i];
            }

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
                for (size_t i = 0; i < nvec; ++i)
                {
                    u[i] = alpha * dF[i] + dx[i];
                }
                enforce_thouless_generator(u, dim_p, dim_n, N_p, N_n);

                const double prev_norm2 = dot_product(residual_prev, residual_prev) / static_cast<double>(nvec);
                const double rms_prev = std::sqrt(std::max(1.0e-28, prev_norm2));
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
        for (size_t i = 0; i < nvec; ++i)
        {
            step[i] = alpha * residual[i];
        }

        bool used_broyden = false;
        const int m = static_cast<int>(dF_history.size());
        if (m >= 2)
        {
            std::vector<double> B(static_cast<size_t>(m) * static_cast<size_t>(m), 0.0);
            std::vector<double> gamma(m, 0.0);

            for (int i = 0; i < m; ++i)
            {
                gamma[i] = w_history[i] * dot_product(dF_history[i], residual);
                for (int j = 0; j < m; ++j)
                {
                    B[static_cast<size_t>(i) * m + j] =
                        w_history[i] * w_history[j] * dot_product(dF_history[i], dF_history[j]);
                }
                B[static_cast<size_t>(i) * m + i] += w0 * w0;
            }

            if (solve_linear_system(B, gamma, m))
            {
                std::vector<double> candidate = step;
                for (int ih = 0; ih < m; ++ih)
                {
                    const double coeff = w_history[ih] * gamma[ih];
                    for (size_t i = 0; i < nvec; ++i)
                    {
                        candidate[i] -= coeff * u_history[ih][i];
                    }
                }

                enforce_thouless_generator(candidate, dim_p, dim_n, N_p, N_n);
                cap_generator_norm(candidate, max_thouless_element);
                if (all_finite(candidate))
                {
                    step.swap(candidate);
                    used_broyden = true;
                }
                else
                {
                    dF_history.clear();
                    u_history.clear();
                    w_history.clear();
                }
            }
        }

        enforce_thouless_generator(step, dim_p, dim_n, N_p, N_n);
        cap_generator_norm(step, max_thouless_element);

        std::vector<double> step_p(np, 0.0);
        std::vector<double> step_n(nn, 0.0);
        std::copy(step.begin(), step.begin() + np, step_p.begin());
        std::copy(step.begin() + np, step.end(), step_n.begin());

        previous_energy = EHF;
        UpdateU_Thouless_pade(step_p.data(), step_n.data());
        UpdateDensityMatrix();
        UpdateF();
        CalcEHF();

        if (used_broyden && grad_rms > 1.5 * previous_rms)
        {
            dF_history.clear();
            u_history.clear();
            w_history.clear();
        }

        if (iterations < 10 || iterations % 10 == 0)
        {
            std::cout << "                 step = " << (used_broyden ? "Broyden-Thouless" : "linear-Thouless") << std::endl;
        }

        residual_prev = residual;
        step_prev = step;
        previous_rms = grad_rms;
    }

    UpdateDensityMatrix();
    UpdateF();
    Diagonalize();
    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

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
