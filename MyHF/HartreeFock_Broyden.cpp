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

    void symmetrize_block(std::vector<double>& x, size_t offset, int dim)
    {
        for (int i = 0; i < dim; ++i)
        {
            for (int j = i + 1; j < dim; ++j)
            {
                const size_t ij = offset + static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j);
                const size_t ji = offset + static_cast<size_t>(j) * static_cast<size_t>(dim) + static_cast<size_t>(i);
                const double avg = 0.5 * (x[ij] + x[ji]);
                x[ij] = avg;
                x[ji] = avg;
            }
        }
    }

    void symmetrize_field(std::vector<double>& x, int dim_p, int dim_n)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        symmetrize_block(x, 0, dim_p);
        symmetrize_block(x, np, dim_n);
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
// Modified Broyden acceleration for the HF SCF loop.
//
// This class is orbital-state driven: Diagonalize() updates U, and
// UpdateDensityMatrix() builds an idempotent density from U.  Therefore the
// correct SCF vector to mix here is the Fock field, not the density.  The map is
//
//   Fock_in -> Diagonalize -> rho[U] -> UpdateF -> Fock_out,
//
// and the residual is R = Fock_out - Fock_in.  With alpha=1 and no Broyden
// history this reduces exactly to the diagonal HF fixed-point iteration.
void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    if (history_size < 1) history_size = 1;
    if (alpha <= 0.0) alpha = 0.5;
    if (alpha > 1.0) alpha = 1.0;
    if (w0 <= 0.0) w0 = 0.01;

    const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
    const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
    const size_t nvec = np + nn;
    const double inv_sqrt_nvec = 1.0 / std::sqrt(static_cast<double>(nvec));

    std::vector<double> x;
    std::vector<double> x_out(nvec, 0.0);
    std::vector<double> R(nvec, 0.0);
    std::vector<double> x_prev;
    std::vector<double> R_prev;

    std::deque<std::vector<double>> dR_history;
    std::deque<std::vector<double>> u_history;
    std::deque<double> w_history;

    UpdateDensityMatrix();
    UpdateF();
    pack_blocks(FockTerm_p, FockTerm_n, dim_p, dim_n, x);
    symmetrize_field(x, dim_p, dim_n);

    double prev_rms_field = std::numeric_limits<double>::infinity();

    std::cout << "  Modified Broyden HF debug: vector = Fock field"
              << "  history_size = " << history_size
              << "  alpha = " << alpha
              << "  w0 = " << w0
              << "  tolerance = " << tolerance << std::endl;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        unpack_blocks(x, dim_p, dim_n, FockTerm_p, FockTerm_n);
        Diagonalize();
        UpdateDensityMatrix();
        UpdateF();
        pack_blocks(FockTerm_p, FockTerm_n, dim_p, dim_n, x_out);
        symmetrize_field(x_out, dim_p, dim_n);

        for (size_t i = 0; i < nvec; ++i)
        {
            R[i] = x_out[i] - x[i];
        }

        const double diff_field = l1_norm(R);
        const double rms_field = vector_norm(R) * inv_sqrt_nvec;
        const bool sp_converged = CheckConvergence();

        if (iterations < 10 || iterations % 10 == 0 || sp_converged || rms_field < tolerance)
        {
            std::cout << "  Broyden iter " << std::setw(5) << iterations
                      << "  field_l1 = " << std::scientific << std::setprecision(6) << diff_field
                      << "  field_rms = " << rms_field
                      << "  hist = " << dR_history.size()
                      << "  sp_conv = " << (sp_converged ? "yes" : "no")
                      << std::defaultfloat << std::endl;
        }

        if (sp_converged || rms_field < tolerance)
        {
            x = x_out;
            break;
        }

        if (!x_prev.empty() && !R_prev.empty())
        {
            std::vector<double> dx(nvec, 0.0);
            std::vector<double> dR(nvec, 0.0);
            for (size_t i = 0; i < nvec; ++i)
            {
                dx[i] = x[i] - x_prev[i];
                dR[i] = R[i] - R_prev[i];
            }

            const double dR_norm = vector_norm(dR);
            if (dR_norm > 1.0e-14)
            {
                for (size_t i = 0; i < nvec; ++i)
                {
                    dx[i] /= dR_norm;
                    dR[i] /= dR_norm;
                }

                std::vector<double> u(nvec, 0.0);
                for (size_t i = 0; i < nvec; ++i)
                {
                    u[i] = alpha * dR[i] + dx[i];
                }

                const double prev_norm2 = dot_product(R_prev, R_prev) / static_cast<double>(nvec);
                const double rms_prev = std::sqrt(std::max(1.0e-28, prev_norm2));
                const double weight = std::max(1.0, 1.0 / rms_prev);

                dR_history.push_back(std::move(dR));
                u_history.push_back(std::move(u));
                w_history.push_back(weight);

                while (static_cast<int>(dR_history.size()) > history_size)
                {
                    dR_history.pop_front();
                    u_history.pop_front();
                    w_history.pop_front();
                }
            }
        }

        std::vector<double> x_linear(nvec, 0.0);
        for (size_t i = 0; i < nvec; ++i)
        {
            x_linear[i] = x[i] + alpha * R[i];
        }

        std::vector<double> x_next = x_linear;
        bool used_broyden = false;
        const int m = static_cast<int>(dR_history.size());

        if (m >= 2)
        {
            std::vector<double> B(static_cast<size_t>(m) * static_cast<size_t>(m), 0.0);
            std::vector<double> gamma(m, 0.0);

            for (int i = 0; i < m; ++i)
            {
                gamma[i] = w_history[i] * dot_product(dR_history[i], R);
                for (int j = 0; j < m; ++j)
                {
                    B[static_cast<size_t>(i) * m + j] =
                        w_history[i] * w_history[j] * dot_product(dR_history[i], dR_history[j]);
                }
                B[static_cast<size_t>(i) * m + i] += w0 * w0;
            }

            if (solve_linear_system(B, gamma, m))
            {
                std::vector<double> candidate = x_linear;
                for (int ih = 0; ih < m; ++ih)
                {
                    const double coeff = w_history[ih] * gamma[ih];
                    if (coeff != 0.0)
                    {
                        for (size_t i = 0; i < nvec; ++i)
                        {
                            candidate[i] -= coeff * u_history[ih][i];
                        }
                    }
                }

                if (all_finite(candidate))
                {
                    x_next.swap(candidate);
                    used_broyden = true;
                }
                else
                {
                    dR_history.clear();
                    u_history.clear();
                    w_history.clear();
                }
            }
        }

        symmetrize_field(x_next, dim_p, dim_n);

        if (used_broyden && rms_field > 1.5 * prev_rms_field)
        {
            dR_history.clear();
            u_history.clear();
            w_history.clear();
        }

        if (iterations < 10 || iterations % 10 == 0)
        {
            std::cout << "                 step = " << (used_broyden ? "Broyden-field" : "linear-field") << std::endl;
        }

        x_prev = x;
        R_prev = R;
        prev_rms_field = rms_field;
        x = std::move(x_next);
    }

    unpack_blocks(x, dim_p, dim_n, FockTerm_p, FockTerm_n);
    Diagonalize();
    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged with Fock-field modified Broyden after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation did not converge with Fock-field modified Broyden after "
                  << iterations << " iterations.\033[0m" << std::endl << std::endl;
    }
    PrintEHF();
}
