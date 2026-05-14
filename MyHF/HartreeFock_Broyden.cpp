#include "HartreeFock.h"

#include <limits>

namespace
{
    double dot_product(const std::vector<double> &a, const std::vector<double> &b)
    {
        double value = 0.0;
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i)
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
        // Broyden mixing can introduce tiny nonsymmetric roundoff errors.  The
        // density matrix should remain real symmetric.  The trace correction
        // keeps the proton/neutron particle numbers fixed.
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
        // Row-major Gaussian elimination with partial pivoting.  The matrix is
        // small: its dimension is only the Broyden history size.
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
// Fixed-point Hartree-Fock iteration accelerated by Johnson's modified
// Broyden method.  The vector being mixed is the combined proton and
// neutron density matrix, x = (rho_p, rho_n).  The residual is
// F(x) = rho_out - rho_in, where rho_out is obtained by building the
// Fock matrix from rho_in, diagonalizing it, and reconstructing the
// occupied density matrix.
void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    if (history_size < 1)
    {
        history_size = 1;
    }
    if (alpha <= 0.0)
    {
        alpha = 0.7;
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

    double previous_residual_norm = std::numeric_limits<double>::infinity();
    const int broyden_start_iteration = 3;

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

        // Keep the same energy-based stopping condition used by Solve_diag().
        // Without this, Broyden may look unconverged only because the density
        // fixed-point residual is stricter than the original HF criterion.
        const bool energy_converged = CheckConvergence();
        if (energy_converged || residual_norm < tolerance)
        {
            x = x_out;
            break;
        }

        // If the residual jumps badly, discard the accumulated inverse-Jacobian
        // information.  This makes the solver fall back to damped diagonal HF
        // instead of letting a bad Broyden history dominate the update.
        if (residual_norm > 5.0 * previous_residual_norm)
        {
            dF_history.clear();
            u_history.clear();
            w_history.clear();
            F_prev.clear();
            x_prev.clear();
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

        std::vector<double> step(nvec, 0.0);
        for (size_t i = 0; i < nvec; ++i)
        {
            step[i] = alpha * F[i];
        }

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
                std::vector<double> broyden_step = step;
                for (int ih = 0; ih < m; ++ih)
                {
                    const double coeff = w_history[ih] * gamma[ih];
                    for (size_t i = 0; i < nvec; ++i)
                    {
                        broyden_step[i] -= coeff * u_history[ih][i];
                    }
                }

                // Do not allow the quasi-Newton correction to create a much
                // larger step than simple density mixing.  This is a practical
                // trust-radius safeguard for HF cases where plain diagonal HF
                // already converges.
                const double simple_step_norm = vector_norm(step) * inv_sqrt_nvec;
                const double broyden_step_norm = vector_norm(broyden_step) * inv_sqrt_nvec;
                if (std::isfinite(broyden_step_norm) && broyden_step_norm <= 3.0 * std::max(simple_step_norm, 1.0e-14))
                {
                    step.swap(broyden_step);
                }
            }
            else
            {
                std::cout << "  Warning: Broyden history matrix is singular; using simple density mixing in this iteration." << std::endl;
            }
        }

        x_prev = x;
        F_prev = F;
        previous_residual_norm = residual_norm;

        for (size_t i = 0; i < nvec; ++i)
        {
            x[i] += step[i];
            if (!std::isfinite(x[i]))
            {
                std::cout << "\033[31m!!!! Warning: non-finite density in Broyden iteration; aborting Broyden solve.\033[0m" << std::endl;
                iterations = maxiter;
                break;
            }
        }

        if (iterations >= maxiter)
        {
            break;
        }

        enforce_density_block(x, 0, dim_p, static_cast<double>(N_p));
        enforce_density_block(x, np, dim_n, static_cast<double>(N_n));
    }

    // Rebuild a self-consistent HF basis from the final mixed density.
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
