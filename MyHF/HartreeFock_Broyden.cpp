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
            if (!std::isfinite(x)) return false;
        return true;
    }

    void pack_density(const double* rho_p, const double* rho_n, int dim_p, int dim_n, std::vector<double>& x)
    {
        const size_t np = static_cast<size_t>(dim_p) * dim_p;
        const size_t nn = static_cast<size_t>(dim_n) * dim_n;
        x.resize(np + nn);
        std::copy(rho_p, rho_p + np, x.begin());
        std::copy(rho_n, rho_n + nn, x.begin() + np);
    }

    void unpack_density(const std::vector<double>& x, int dim_p, int dim_n, double* rho_p, double* rho_n)
    {
        const size_t np = static_cast<size_t>(dim_p) * dim_p;
        const size_t nn = static_cast<size_t>(dim_n) * dim_n;
        std::copy(x.begin(), x.begin() + np, rho_p);
        std::copy(x.begin() + np, x.begin() + np + nn, rho_n);
    }

    void enforce_density_block(std::vector<double>& x, size_t offset, int dim, double target_trace)
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
        double tr = 0.0;
        for (int i = 0; i < dim; ++i) tr += x[offset + static_cast<size_t>(i) * dim + i];
        const double corr = (target_trace - tr) / static_cast<double>(dim);
        for (int i = 0; i < dim; ++i) x[offset + static_cast<size_t>(i) * dim + i] += corr;
    }

    void enforce_density(std::vector<double>& x, int dim_p, int dim_n, int N_p, int N_n)
    {
        const size_t np = static_cast<size_t>(dim_p) * dim_p;
        enforce_density_block(x, 0, dim_p, static_cast<double>(N_p));
        enforce_density_block(x, np, dim_n, static_cast<double>(N_n));
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
                if (v > mx) { mx = v; piv = i; }
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
            for (int j = i + 1; j < n; ++j) rhs -= A[static_cast<size_t>(i) * n + j] * b[j];
            const double diag = A[static_cast<size_t>(i) * n + i];
            if (std::fabs(diag) < tol) return false;
            b[i] = rhs / diag;
        }
        return true;
    }
}

void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    if (history_size < 1) history_size = 1;
    if (alpha <= 0.0) alpha = 1.0;
    if (alpha > 1.0) alpha = 1.0;
    if (w0 <= 0.0) w0 = 0.01;

    const size_t np = static_cast<size_t>(dim_p) * dim_p;
    const size_t nn = static_cast<size_t>(dim_n) * dim_n;
    const size_t nvec = np + nn;
    const double inv_sqrt_nvec = 1.0 / std::sqrt(static_cast<double>(nvec));

    std::vector<double> x, x_out(nvec, 0.0), F(nvec, 0.0), x_prev, F_prev;
    std::deque<std::vector<double>> dF_history, u_history;
    std::deque<double> w_history;

    UpdateDensityMatrix();
    pack_density(rho_p, rho_n, dim_p, dim_n, x);
    enforce_density(x, dim_p, dim_n, N_p, N_n);

    double prev_diff_density = std::numeric_limits<double>::infinity();

    std::cout << "  Modified Broyden HF debug: history_size = " << history_size
              << " alpha = " << alpha << " w0 = " << w0
              << " density_tol = " << tolerance << std::endl;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        unpack_density(x, dim_p, dim_n, rho_p, rho_n);
        UpdateF();
        Diagonalize();
        UpdateDensityMatrix();
        pack_density(rho_p, rho_n, dim_p, dim_n, x_out);

        for (size_t i = 0; i < nvec; ++i) F[i] = x_out[i] - x[i];

        const double diff_density = l1_norm(F);
        const double rms_density = vector_norm(F) * inv_sqrt_nvec;
        CalcEHF();

        if (iterations < 10 || iterations % 10 == 0 || (diff_density < tolerance && iterations > 1))
        {
            std::cout << "  Broyden iter " << std::setw(5) << iterations
                      << "  diff_density = " << std::scientific << std::setprecision(6) << diff_density
                      << "  rms_density = " << rms_density
                      << "  hist = " << dF_history.size()
                      << std::defaultfloat << std::endl;
        }

        if (diff_density < tolerance && iterations > 1)
        {
            x = x_out;
            break;
        }

        if (!x_prev.empty() && !F_prev.empty())
        {
            std::vector<double> dx(nvec, 0.0), dF(nvec, 0.0);
            for (size_t i = 0; i < nvec; ++i)
            {
                dx[i] = x[i] - x_prev[i];
                dF[i] = F[i] - F_prev[i];
            }
            const double dF_norm = vector_norm(dF);
            if (dF_norm > 1.0e-14)
            {
                for (size_t i = 0; i < nvec; ++i)
                {
                    dx[i] /= dF_norm;
                    dF[i] /= dF_norm;
                }
                std::vector<double> u(nvec, 0.0);
                for (size_t i = 0; i < nvec; ++i) u[i] = alpha * dF[i] + dx[i];

                const double prev_norm2 = dot_product(F_prev, F_prev) / static_cast<double>(nvec);
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

        std::vector<double> x_linear(nvec, 0.0);
        for (size_t i = 0; i < nvec; ++i) x_linear[i] = x[i] + alpha * F[i];

        std::vector<double> x_next = x_linear;
        bool used_broyden = false;
        const int m = static_cast<int>(dF_history.size());
        if (m >= 2)
        {
            std::vector<double> B(static_cast<size_t>(m) * m, 0.0), gamma(m, 0.0);
            for (int i = 0; i < m; ++i)
            {
                gamma[i] = w_history[i] * dot_product(dF_history[i], F);
                for (int j = 0; j < m; ++j)
                    B[static_cast<size_t>(i) * m + j] = w_history[i] * w_history[j] * dot_product(dF_history[i], dF_history[j]);
                B[static_cast<size_t>(i) * m + i] += w0 * w0;
            }
            if (solve_linear_system(B, gamma, m))
            {
                std::vector<double> cand = x_linear;
                for (int h = 0; h < m; ++h)
                {
                    const double coeff = w_history[h] * gamma[h];
                    if (coeff != 0.0)
                        for (size_t i = 0; i < nvec; ++i) cand[i] -= coeff * u_history[h][i];
                }
                if (all_finite(cand))
                {
                    x_next.swap(cand);
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

        enforce_density(x_next, dim_p, dim_n, N_p, N_n);

        if (used_broyden && diff_density > 1.5 * prev_diff_density)
        {
            dF_history.clear();
            u_history.clear();
            w_history.clear();
        }

        if (iterations < 10 || iterations % 10 == 0)
            std::cout << "                 step = " << (used_broyden ? "Broyden" : "linear") << std::endl;

        x_prev = x;
        F_prev = F;
        prev_diff_density = diff_density;
        x = std::move(x_next);
    }

    unpack_density(x, dim_p, dim_n, rho_p, rho_n);
    UpdateF();
    Diagonalize();
    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
        std::cout << "  HF converged with modified Broyden after " << iterations << " iterations. " << std::endl;
    else
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation did not converge with modified Broyden after "
                  << iterations << " iterations.\033[0m" << std::endl << std::endl;
    PrintEHF();
}
