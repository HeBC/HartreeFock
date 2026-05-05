#include "HartreeFock.h"

namespace
{
    double dot_product_c(const std::vector<double> &a, const std::vector<double> &b)
    {
        double value = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            value += a[i] * b[i];
        }
        return value;
    }

    double vector_norm_c(const std::vector<double> &a)
    {
        return std::sqrt(dot_product_c(a, a));
    }

    void pack_density_c(const double *rho_p, const double *rho_n,
                        int dim_p, int dim_n, std::vector<double> &x)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
        x.resize(np + nn);
        std::copy(rho_p, rho_p + np, x.begin());
        std::copy(rho_n, rho_n + nn, x.begin() + np);
    }

    void unpack_density_c(const std::vector<double> &x, int dim_p, int dim_n,
                          double *rho_p, double *rho_n)
    {
        const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
        const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
        std::copy(x.begin(), x.begin() + np, rho_p);
        std::copy(x.begin() + np, x.begin() + np + nn, rho_n);
    }

    void enforce_density_block_c(std::vector<double> &x, size_t offset, int dim, double target_trace)
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

    bool solve_linear_system_c(std::vector<double> A, std::vector<double> &b, int n)
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
                    std::swap(A[static_cast<size_t>(k) * n + j], A[static_cast<size_t>(pivot) * n + j]);
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
// Modified Broyden density mixing with the same external constraints used
// by Solve_hybrid_Constraint().  The constrained Fock matrix is built as
//
//   F_constraint = F_HF - sum_i lambda_i Q_i,
//   lambda_i = constraint_strength * (target_i - <Q_i>).
//
// Broyden only mixes the density.  Each residual evaluation still rebuilds
// the constrained Fock matrix before diagonalization.
void HartreeFock::Solve_broyden_Constraint(int history_size, double alpha, double w0, double constraint_strength)
{
    if (history_size < 1)
    {
        history_size = 1;
    }
    if (alpha <= 0.0)
    {
        alpha = 0.2;
    }
    if (w0 <= 0.0)
    {
        w0 = 0.01;
    }
    if (constraint_strength <= 0.0)
    {
        constraint_strength = 0.1;
    }

    int number_of_Q = 0;
    std::vector<std::string> Qtype;

    if (modelspace->GetIsShapeConstrained() == true)
    {
        number_of_Q += 2;
        Qtype.push_back("QuadrupoleQ0");
        Qtype.push_back("QuadrupoleQ2");
    }
    if (modelspace->Get_Jx_constraint() == true)
    {
        number_of_Q += 1;
        Qtype.push_back("Jx");
    }
    if (modelspace->Get_Jz_constraint() == true)
    {
        number_of_Q += 1;
        Qtype.push_back("Jz");
    }

    if (number_of_Q == 0)
    {
        std::cout << "   No constraint loaded. Calling unconstrained Broyden solver." << std::endl;
        Solve_broyden(history_size, alpha, w0);
        return;
    }

    std::vector<std::vector<double>> QOperator_p(number_of_Q, std::vector<double>(dim_p * dim_p, 0.0));
    std::vector<std::vector<double>> QOperator_n(number_of_Q, std::vector<double>(dim_n * dim_n, 0.0));
    std::vector<double> targets(number_of_Q, 0.0);
    std::vector<double> deltaQs(number_of_Q, 0.0);
    std::vector<double> lambdas(number_of_Q, 0.0);

    auto it = std::find(Qtype.begin(), Qtype.end(), "QuadrupoleQ0");
    if (it != Qtype.end())
    {
        const int index = std::distance(Qtype.begin(), it);

        for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
        {
            const int Qindex = Ham->Q2MEs_p.Q0_list[i];
            const int ia = Ham->MSMEs.OB_p[Qindex].GetIndex_a();
            const int ib = Ham->MSMEs.OB_p[Qindex].GetIndex_b();
            QOperator_p[index][ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
        }
        for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
        {
            const int Qindex = Ham->Q2MEs_p.Q2_list[i];
            const int ia = Ham->MSMEs.OB_p[Qindex].GetIndex_a();
            const int ib = Ham->MSMEs.OB_p[Qindex].GetIndex_b();
            QOperator_p[index + 1][ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
        }
        for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
        {
            const int Qindex = Ham->Q2MEs_p.Q_2_list[i];
            const int ia = Ham->MSMEs.OB_p[Qindex].GetIndex_a();
            const int ib = Ham->MSMEs.OB_p[Qindex].GetIndex_b();
            QOperator_p[index + 1][ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
        }

        for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
        {
            const int Qindex = Ham->Q2MEs_n.Q0_list[i];
            const int ia = Ham->MSMEs.OB_n[Qindex].GetIndex_a();
            const int ib = Ham->MSMEs.OB_n[Qindex].GetIndex_b();
            QOperator_n[index][ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
        }
        for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
        {
            const int Qindex = Ham->Q2MEs_n.Q2_list[i];
            const int ia = Ham->MSMEs.OB_n[Qindex].GetIndex_a();
            const int ib = Ham->MSMEs.OB_n[Qindex].GetIndex_b();
            QOperator_n[index + 1][ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
        }
        for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
        {
            const int Qindex = Ham->Q2MEs_n.Q_2_list[i];
            const int ia = Ham->MSMEs.OB_n[Qindex].GetIndex_a();
            const int ib = Ham->MSMEs.OB_n[Qindex].GetIndex_b();
            QOperator_n[index + 1][ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
        }

        targets[index] = modelspace->GetShapeQ0();
        targets[index + 1] = modelspace->GetShapeQ2();
    }

    it = std::find(Qtype.begin(), Qtype.end(), "Jx");
    if (it != Qtype.end())
    {
        const int index = std::distance(Qtype.begin(), it);

        for (int i = 0; i < dim_p; i++)
        {
            for (int j = 0; j < dim_p; j++)
            {
                if (modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(i)].l ==
                    modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(j)].l)
                {
                    const int ji = modelspace->Get_MSmatrix_2j(Proton, i);
                    const int jj = modelspace->Get_MSmatrix_2j(Proton, j);
                    if (ji == jj)
                    {
                        const int mi = modelspace->Get_MSmatrix_2m(Proton, i);
                        const int mj = modelspace->Get_MSmatrix_2m(Proton, j);
                        if (mi == mj + 2)
                        {
                            QOperator_p[index][i * dim_p + j] = 0.5 * sqrt((jj - mj) * (jj + mj + 2.0) / 4.0);
                        }
                        else if (mi == mj - 2)
                        {
                            QOperator_p[index][i * dim_p + j] = 0.5 * sqrt((jj + mj) * (jj - mj + 2.0) / 4.0);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < dim_n; i++)
        {
            for (int j = 0; j < dim_n; j++)
            {
                if (modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(i)].l ==
                    modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(j)].l)
                {
                    const int ji = modelspace->Get_MSmatrix_2j(Neutron, i);
                    const int jj = modelspace->Get_MSmatrix_2j(Neutron, j);
                    if (ji == jj)
                    {
                        const int mi = modelspace->Get_MSmatrix_2m(Neutron, i);
                        const int mj = modelspace->Get_MSmatrix_2m(Neutron, j);
                        if (mi == mj + 2)
                        {
                            QOperator_n[index][i * dim_n + j] = 0.5 * sqrt((jj - mj) * (jj + mj + 2.0) / 4.0);
                        }
                        else if (mi == mj - 2)
                        {
                            QOperator_n[index][i * dim_n + j] = 0.5 * sqrt((jj + mj) * (jj - mj + 2.0) / 4.0);
                        }
                    }
                }
            }
        }
        targets[index] = sqrt(modelspace->GetTargetJx());
    }

    it = std::find(Qtype.begin(), Qtype.end(), "Jz");
    if (it != Qtype.end())
    {
        const int index = std::distance(Qtype.begin(), it);

        for (int i = 0; i < dim_p; i++)
        {
            for (int j = 0; j < dim_p; j++)
            {
                if (modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(i)].l ==
                        modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(j)].l &&
                    modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(i)].n ==
                        modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(j)].n &&
                    modelspace->Get_MSmatrix_2j(Proton, i) == modelspace->Get_MSmatrix_2j(Proton, j) &&
                    modelspace->Get_MSmatrix_2m(Proton, i) == modelspace->Get_MSmatrix_2m(Proton, j))
                {
                    QOperator_p[index][i * dim_p + j] = modelspace->Get_MSmatrix_2m(Proton, j) * 0.5;
                }
            }
        }
        for (int i = 0; i < dim_n; i++)
        {
            for (int j = 0; j < dim_n; j++)
            {
                if (modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(i)].l ==
                        modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(j)].l &&
                    modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(i)].n ==
                        modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(j)].n &&
                    modelspace->Get_MSmatrix_2j(Neutron, i) == modelspace->Get_MSmatrix_2j(Neutron, j) &&
                    modelspace->Get_MSmatrix_2m(Neutron, i) == modelspace->Get_MSmatrix_2m(Neutron, j))
                {
                    QOperator_n[index][i * dim_n + j] = modelspace->Get_MSmatrix_2m(Neutron, j) * 0.5;
                }
            }
        }
        targets[index] = modelspace->GetTargetJz();
    }

    auto apply_constraints_to_fock = [&]() {
        for (int iq = 0; iq < number_of_Q; ++iq)
        {
            double Qp = 0.0;
            double Qn = 0.0;
            CalOnebodyOperator(QOperator_p[iq].data(), QOperator_n[iq].data(), Qp, Qn);
            deltaQs[iq] = targets[iq] - Qp - Qn;
            lambdas[iq] = constraint_strength * deltaQs[iq];

            for (int a = 0; a < dim_p * dim_p; ++a)
            {
                FockTerm_p[a] -= lambdas[iq] * QOperator_p[iq][a];
            }
            for (int a = 0; a < dim_n * dim_n; ++a)
            {
                FockTerm_n[a] -= lambdas[iq] * QOperator_n[iq][a];
            }
        }
    };

    const size_t np = static_cast<size_t>(dim_p) * static_cast<size_t>(dim_p);
    const size_t nn = static_cast<size_t>(dim_n) * static_cast<size_t>(dim_n);
    const size_t nvec = np + nn;
    const double inv_sqrt_nvec = 1.0 / std::sqrt(static_cast<double>(nvec));
    const double constraint_tol = std::max(1.0e-6, std::sqrt(tolerance));

    std::vector<double> x;
    std::vector<double> x_out(nvec, 0.0);
    std::vector<double> F(nvec, 0.0);
    std::vector<double> x_prev;
    std::vector<double> F_prev;

    std::deque<std::vector<double>> dF_history;
    std::deque<std::vector<double>> u_history;
    std::deque<double> w_history;

    UpdateDensityMatrix();
    pack_density_c(rho_p, rho_n, dim_p, dim_n, x);
    enforce_density_block_c(x, 0, dim_p, static_cast<double>(N_p));
    enforce_density_block_c(x, np, dim_n, static_cast<double>(N_n));

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        unpack_density_c(x, dim_p, dim_n, rho_p, rho_n);

        UpdateF();
        apply_constraints_to_fock();
        Diagonalize();
        UpdateDensityMatrix();
        pack_density_c(rho_p, rho_n, dim_p, dim_n, x_out);

        for (size_t i = 0; i < nvec; ++i)
        {
            F[i] = x_out[i] - x[i];
        }

        double max_deltaQ = 0.0;
        for (double dq : deltaQs)
        {
            max_deltaQ = std::max(max_deltaQ, std::fabs(dq));
        }

        const double residual_norm = vector_norm_c(F) * inv_sqrt_nvec;
        if (residual_norm < tolerance && max_deltaQ < constraint_tol)
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

            const double dF_norm = vector_norm_c(dF);
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
        if (m > 0)
        {
            std::vector<double> A(static_cast<size_t>(m) * m, 0.0);
            std::vector<double> gamma(m, 0.0);

            for (int i = 0; i < m; ++i)
            {
                gamma[i] = w_history[i] * dot_product_c(dF_history[i], F);
                for (int j = 0; j < m; ++j)
                {
                    A[static_cast<size_t>(i) * m + j] =
                        w_history[i] * w_history[j] * dot_product_c(dF_history[i], dF_history[j]);
                }
                A[static_cast<size_t>(i) * m + i] += w0 * w0;
            }

            if (solve_linear_system_c(A, gamma, m))
            {
                for (int ih = 0; ih < m; ++ih)
                {
                    const double coeff = w_history[ih] * gamma[ih];
                    for (size_t i = 0; i < nvec; ++i)
                    {
                        step[i] -= coeff * u_history[ih][i];
                    }
                }
            }
            else
            {
                std::cout << "  Warning: constrained Broyden history matrix is singular; using simple density mixing in this iteration." << std::endl;
            }
        }

        x_prev = x;
        F_prev = F;

        for (size_t i = 0; i < nvec; ++i)
        {
            x[i] += step[i];
            if (!std::isfinite(x[i]))
            {
                std::cout << "\033[31m!!!! Warning: non-finite density in constrained Broyden iteration.\033[0m" << std::endl;
                iterations = maxiter;
                break;
            }
        }

        if (iterations >= maxiter)
        {
            break;
        }

        enforce_density_block_c(x, 0, dim_p, static_cast<double>(N_p));
        enforce_density_block_c(x, np, dim_n, static_cast<double>(N_n));
    }

    unpack_density_c(x, dim_p, dim_n, rho_p, rho_n);
    UpdateF();
    apply_constraints_to_fock();
    Diagonalize();
    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  Constrained HF converged with modified Broyden after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: constrained Hartree-Fock calculation did not converge with modified Broyden after "
                  << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "  Constraint residuals:" << std::endl;
    for (int iq = 0; iq < number_of_Q; ++iq)
    {
        double Qp = 0.0;
        double Qn = 0.0;
        CalOnebodyOperator(QOperator_p[iq].data(), QOperator_n[iq].data(), Qp, Qn);
        std::cout << "    " << Qtype[iq]
                  << " target = " << targets[iq]
                  << " value = " << Qp + Qn
                  << " delta = " << targets[iq] - Qp - Qn
                  << " lambda = " << lambdas[iq] << std::endl;
    }

    PrintEHF();
}
