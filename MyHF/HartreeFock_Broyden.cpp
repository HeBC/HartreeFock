#include "HartreeFock.h"

#include <limits>

namespace
{
    double dot_product(const std::vector<double> &a, const std::vector<double> &b)
    {
        double value = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            value += a[i] * b[i];
        }
        return value;
    }

    double vector_norm(const std::vector<double> &a)
    {
        return std::sqrt(dot_product(a, a));
    }

    void pack_density(const double *rho_p, const double *rho_n,
                      int dim_p, int dim_n, std::vector<double> &x)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
        x.resize(np + nn);
        std::copy(rho_p, rho_p + np, x.begin());
        std::copy(rho_n, rho_n + nn, x.begin() + np);
    }

    void unpack_density(const std::vector<double> &x, int dim_p, int dim_n,
                        double *rho_p, double *rho_n)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
        std::copy(x.begin(), x.begin() + np, rho_p);
        std::copy(x.begin() + np, x.begin() + np + nn, rho_n);
    }

    void enforce_density_block(std::vector<double> &x, size_t offset, int dim, double target_trace)
    {
        for (int i = 0; i < dim; ++i)
        {
            for (int j = i + 1; j < dim; ++j)
            {
                const size_t ij = offset + static_cast<size_t>(i) * dim + j;
                const size_t ji = offset + static_cast<size_t>(j) * dim + i;
                const double avg = 0.5 * (x[ij] + x[ji]);
                x[ij] = avg;
                x[ji] = avg;
            }
        }

        double trace = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            trace += x[offset + static_cast<size_t>(i) * dim + i];
        }

        const double correction = (target_trace - trace) / static_cast<double>(dim);
        for (int i = 0; i < dim; ++i)
        {
            x[offset + static_cast<size_t>(i) * dim + i] += correction;
        }
    }

    bool solve_linear_system(std::vector<double> A, std::vector<double> &b, int n)
    {
        const double pivot_tol = 1.0e-14;

        for (int k = 0; k < n; ++k)
        {
            int pivot = k;
            double max_abs = std::fabs(A[static_cast<size_t>(k) * n + k]);
            for (int i = k + 1; i < n; ++i)
            {
                const double value = std::fabs(A[static_cast<size_t>(i) * n + k]);
                if (value > max_abs)
                {
                    max_abs = value;
                    pivot = i;
                }
            }

            if (max_abs < pivot_tol)
            {
                return false;
            }

            if (pivot != k)
            {
                for (int j = k; j < n; ++j)
                {
                    std::swap(A[static_cast<size_t>(k) * n + j],
                              A[static_cast<size_t>(pivot) * n + j]);
                }
                std::swap(b[k], b[pivot]);
            }

            const double diag = A[static_cast<size_t>(k) * n + k];
            for (int i = k + 1; i < n; ++i)
            {
                const double factor = A[static_cast<size_t>(i) * n + k] / diag;
                A[static_cast<size_t>(i) * n + k] = 0.0;
                for (int j = k + 1; j < n; ++j)
                {
                    A[static_cast<size_t>(i) * n + j] -= factor * A[static_cast<size_t>(k) * n + j];
                }
                b[i] -= factor * b[k];
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
            if (std::fabs(diag) < pivot_tol)
            {
                return false;
            }
            b[i] = rhs / diag;
        }

        return true;
    }
}

//*********************************************************************
// Fixed-point Hartree-Fock iteration accelerated by modified Broyden.
// The vector being mixed is x=(rho_p,rho_n), with residual F=rho_out-rho_in.
//
// The important safeguard is residual-tested mixing.  A Broyden vector is not
// accepted just because its norm/alignment looks reasonable; this routine
// explicitly evaluates the next SCF residual for the trial density and accepts
// the candidate with the smallest tested residual.
void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    if (history_size < 1)
    {
        history_size = 1;
    }
    if (alpha <= 0.0)
    {
        alpha = 1.0;
    }
    if (alpha > 1.0)
    {
        alpha = 1.0;
    }
    if (w0 <= 0.0)
    {
        w0 = 0.01;
    }

    const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
    const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
    const size_t nvec = np + nn;
    const double inv_sqrt_nvec = 1.0 / std::sqrt(static_cast<double>(nvec));
    const double residual_tolerance = std::max(tolerance, 1.0e-10);

    std::vector<double> x;
    std::vector<double> x_out(nvec, 0.0);
    std::vector<double> F(nvec, 0.0);
    std::vector<double> x_prev;
    std::vector<double> F_prev;

    std::deque<std::vector<double>> dF_history;
    std::deque<std::vector<double>> u_history;
    std::deque<double> w_history;

    UpdateDensityMatrix();
    pack_density(rho_p, rho_n, dim_p, dim_n, x);
    enforce_density_block(x, 0, dim_p, static_cast<double>(N_p));
    enforce_density_block(x, np, dim_n, static_cast<double>(N_n));

    auto evaluate_residual = [&](std::vector<double> candidate) -> double
    {
        enforce_density_block(candidate, 0, dim_p, static_cast<double>(N_p));
        enforce_density_block(candidate, np, dim_n, static_cast<double>(N_n));
        unpack_density(candidate, dim_p, dim_n, rho_p, rho_n);
        UpdateF();
        Diagonalize();
        UpdateDensityMatrix();

        std::vector<double> candidate_out(nvec, 0.0);
        pack_density(rho_p, rho_n, dim_p, dim_n, candidate_out);

        for (size_t i = 0; i < nvec; ++i)
        {
            candidate_out[i] -= candidate[i];
        }
        return vector_norm(candidate_out) * inv_sqrt_nvec;
    };

    const int broyden_start_iteration = 3;
    const std::vector<double> line_search_scales = {1.0, 0.8, 0.5, 0.25, 0.10};

    std::cout << "  Modified Broyden HF debug: history_size = " << history_size
              << " alpha = " << alpha << " w0 = " << w0
              << " residual_tol = " << residual_tolerance << std::endl;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        unpack_density(x, dim_p, dim_n, rho_p, rho_n);

        UpdateF();
        Diagonalize();
        UpdateDensityMatrix();
        pack_density(rho_p, rho_n, dim_p, dim_n, x_out);

        for (size_t i = 0; i < nvec; ++i)
        {
            F[i] = x_out[i] - x[i];
        }

        const double residual_norm = vector_norm(F) * inv_sqrt_nvec;
        const bool sp_energy_stagnated = CheckConvergence();

        if (iterations < 10 || iterations % 10 == 0 || residual_norm < residual_tolerance || sp_energy_stagnated)
        {
            std::cout << "  Broyden iter " << std::setw(5) << iterations
                      << "  rho_res = " << std::scientific << std::setprecision(6) << residual_norm
                      << "  hist = " << dF_history.size()
                      << "  sp_stag = " << (sp_energy_stagnated ? "yes" : "no")
                      << std::defaultfloat << std::endl;
        }

        if (residual_norm < residual_tolerance)
        {
            x = x_out;
            break;
        }

        if (!F_prev.empty())
        {
            std::vector<double> dF(nvec, 0.0);
            std::vector<double> dx(nvec, 0.0);
            for (size_t i = 0; i < nvec; ++i)
            {
                dF[i] = F[i] - F_prev[i];
                dx[i] = x[i] - x_prev[i];
            }

            const double dF_norm = vector_norm(dF);
            if (dF_norm > 1.0e-14)
            {
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

                const double weight = std::min(1.0e8, std::max(1.0, 1.0 / std::max(residual_norm, 1.0e-12)));
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

        // Simple fixed-point update: x_{k+1}=x_k+alpha F_k.
        std::vector<double> simple_step(nvec, 0.0);
        for (size_t i = 0; i < nvec; ++i)
        {
            simple_step[i] = alpha * F[i];
        }

        std::vector<double> broyden_step = simple_step;
        bool have_broyden_step = false;
        const int m = static_cast<int>(dF_history.size());
        if (iterations >= broyden_start_iteration && m > 0)
        {
            std::vector<double> A(static_cast<size_t>(m) * m, 0.0);
            std::vector<double> gamma(m, 0.0);

            for (int i = 0; i < m; ++i)
            {
                gamma[i] = w_history[i] * dot_product(dF_history[i], F);
                for (int j = 0; j < m; ++j)
                {
                    A[static_cast<size_t>(i) * m + j] =
                        w_history[i] * w_history[j] * dot_product(dF_history[i], dF_history[j]);
                }
                A[static_cast<size_t>(i) * m + i] += w0 * w0;
            }

            if (solve_linear_system(A, gamma, m))
            {
                for (int ih = 0; ih < m; ++ih)
                {
                    const double coeff = w_history[ih] * gamma[ih];
                    for (size_t i = 0; i < nvec; ++i)
                    {
                        broyden_step[i] -= coeff * u_history[ih][i];
                    }
                }
                have_broyden_step = std::isfinite(vector_norm(broyden_step));
            }
            else
            {
                std::cout << "  Warning: Broyden history matrix is singular; using tested simple mixing." << std::endl;
            }
        }

        // Residual-tested line search.  This fixes the previous bug where a
        // Broyden step was accepted using only a norm/alignment heuristic, which
        // allowed steps that immediately worsened the actual SCF residual.
        std::vector<double> best_x = x;
        double best_trial_residual = std::numeric_limits<double>::infinity();
        std::string best_label = "none";

        auto test_step = [&](const std::vector<double> &step, double scale, const std::string &label)
        {
            std::vector<double> candidate = x;
            for (size_t i = 0; i < nvec; ++i)
            {
                candidate[i] += scale * step[i];
            }
            const double trial_residual = evaluate_residual(candidate);
            if (trial_residual < best_trial_residual)
            {
                best_trial_residual = trial_residual;
                best_x = std::move(candidate);
                best_label = label;
            }
        };

        for (double scale : line_search_scales)
        {
            test_step(simple_step, scale, "simple(" + std::to_string(scale) + ")");
        }
        if (have_broyden_step)
        {
            for (double scale : line_search_scales)
            {
                test_step(broyden_step, scale, "Broyden(" + std::to_string(scale) + ")");
            }
        }

        x_prev = x;
        F_prev = F;
        x = best_x;

        if (iterations < 10 || iterations % 10 == 0)
        {
            std::cout << "                 accepted = " << best_label
                      << "  trial_rho_res = " << std::scientific << best_trial_residual
                      << std::defaultfloat << std::endl;
        }

        if (!std::isfinite(best_trial_residual))
        {
            std::cout << "\033[31m!!!! Warning: non-finite Broyden trial residual; aborting Broyden solve.\033[0m" << std::endl;
            iterations = maxiter;
            break;
        }
    }

    unpack_density(x, dim_p, dim_n, rho_p, rho_n);
    UpdateF();
    Diagonalize();
    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged with modified Broyden after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation did not converge with modified Broyden after "
                  << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
}
