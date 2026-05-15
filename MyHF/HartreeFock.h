#ifndef HartreeFock_h
#define HartreeFock_h 1

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <deque>
#include <mkl.h>
#include <omp.h>
using namespace std;

#include "ReadWriteFiles.h"

struct Triple
{
    double first;
    int second;
    int third;
    Triple(){};
    Triple(double f, int s, int t) : first(f), second(s), third(t) {}
};

class HartreeFock
{
public:
    int N_p, N_n, dim_p, dim_n;
    // method
    HartreeFock(Hamiltonian &H); /// Constructor
    ~HartreeFock();

    // Solve functions
    void Solve_gradient();
    void Solve_gradient_Constraint();
    void Solve_hybrid();
    void Solve_hybrid_Constraint(); // not work now!!!
    void Solve_broyden(int history_size = 8, double alpha = 0.5, double w0 = 0.01); /// Fixed-point HF with modified Broyden Thouless-gradient mixing
    void Solve_broyden_Constraint(int history_size = 8, double alpha = 0.2, double w0 = 0.01, double constraint_strength = 0.1); /// Modified Broyden with Q/J constraints in the Fock matrix
    void Solve_diag();              /// Diagonalize and UpdateF until convergence

    //---------------------
    // Tools
    void UpdateDensityMatrix();                                                                      /// Update the density matrix with the new coefficients C
    void UpdateDensityMatrix(const std::vector<int> proton_vec, const std::vector<int> neutron_vec); /// Update the density matrix with given orbits
    void UpdateDensityMatrix_DIIS();
    void UpdateU_hybrid();                                                                          /// Update the Unitary transformation matrix, hybrid method
    void UpdateF();                                                                                 /// Update the Fock matrix with the new transformation coefficients C
    void Diagonalize();                                                                             /// Diagonalize Fock term
    void CalcEHF();                                                                                 /// Calculate the HF energy.
    void CalcEHF(double constrainedQ);                                                              /// Calculate the HF energy with constrained
    double CalcEHF(const std::vector<int> proton_vec, const std::vector<int> neutron_vec);          // inidicate the orbits
    double CalcEHF_HForbits(const std::vector<int> proton_vec, const std::vector<int> neutron_vec); // cal E on HF orbits
    void TransferOperatorToHFbasis(double *Op_p, double *Op_n);
    void CalOnebodyOperator(double *Op_p, double *Op_n, double &Qp, double &Qn);
    void Operator_ph(double *Op_p, double *Op_n);
    void PrintEHF();
    void PrintQudrapole();                            /// Print qudrapole moment
    void HF_ShapeCoefficients_Lab();                  /// Print beta and gamma of shape
    void HF_ShapeCoefficients_calr2_Lab();
    void Print_Jz();                                  /// Print <Jz>
    bool CheckConvergence();                          /// check the HF single SP
    void Reset_U();                                   /// use identical U matrix
    void RandomTransformationU(int RandomSeed = 525); /// Random transformation matrix U
    void UpdateTolerance(double T) { this->tolerance = T; };
    void UpdateGradientStepSize(double size) { gradient_eta = size; };
    Hamiltonian Residual_H();                   /// get residual interaction
    Hamiltonian TransformToHFBasis(const Hamiltonian& HamIn);
    void SetMaxIteration(int num){maxiter = num;};

    // gradient method
    void Cal_Gradient(double *Z_p, double *Z_n);
    void Cal_Gradient_preconditioned(double *Z_p, double *Z_n);
    void Cal_Gradient_preconditioned_SRG(double *Z_p, double *Z_n);
    void Cal_Gradient_given_gradient(double *Z_p, double *Z_n);
    void Cal_Gradient_preconditioned_given_gradient(double *Z_p, double *Z_n);
    void UpdateU_Thouless_pade(double *Z_p, double *Z_n); // THouless by using pade approximation
    void UpdateU_Thouless_1st(double *Z_p, double *Z_n);  // Thouless up to first order

    // output states
    void SaveHoleParameters(string filename);
    void SaveParticleHoleStates(int Num);
    std::vector<std::vector<int>> generateCombinations(const std::vector<int> &numbers, int n);
    std::vector<int> GetHoleList(int Isospin);
    std::vector<int> GetParticleList(int Isospin);
    std::vector<int> ConstructParticleHoleState(int isospin, const std::vector<int> &hole_vec, const std::vector<int> &part_vec);

    /// debug code
    void Check_orthogonal_U_p(int i, int j);
    void Check_orthogonal_U_n(int i, int j);
    void Check_matrix(int dim, double *Matrix);
    void PrintParameters_Hole();
    void PrintAllParameters();
    void PrintDensity();
    void PrintFockMatrix();
    void PrintVtb();
    void PrintAllHFEnergies();
    void PrintHoleOrbitsIndex();
    void PrintOccupationHO_jorbit();
    void CheckDensity();
    void PrintOccupationHO();

private:
    ModelSpace *modelspace;           /// Model Space
    Hamiltonian *Ham;                 /// Hamiltonian
    double *U_p, *U_n;                /// transformation coefficients, 1st index is ho basis, 2nd = HF basis
    double *rho_p, *rho_n;            /// density matrix rho_ij, the index in order of dim_p * dim_p dim_n * dim_n
    double *FockTerm_p, *FockTerm_n;  /// Fock matrix
    double *Vij_p, *Vij_n;            /// Two body term
    double *T_term_p = nullptr, *T_term_n = nullptr;   /// SP energies       
    double tolerance;                 /// tolerance for convergence
    int iterations;                   /// record iterations used in Solve()
    int maxiter = 1000;               /// max number of iteration
    int *holeorbs_p, *holeorbs_n;     /// record the hole orbit in Hatree Fock space// 1 for hole, 0 for particle
    double *energies, *prev_energies; /// vector of single particle energies [Proton, Neutron]
    double EHF;                       /// Hartree-Fock energy (Normal-ordered 0-body term)
    double e1hf;                      /// One-body contribution to EHF
    double e2hf;                      /// Two-body contribution to EHF
    double eta = 1.0;                 /// 1. will be Diagonalization method, use a small number
    double gradient_eta = 0.1;        // eta for steepest descent method.
    std::deque<std::vector<double>> DIIS_density_mats_p, DIIS_density_mats_n;
    ///< Save density matrix from past iterations for DIIS
    std::deque<std::vector<double>> DIIS_error_mats_p, DIIS_error_mats_n;
    ///< Save error from past iterations for DIIS
    double frobenius_norm(const std::vector<double> &A);
    static bool compareTriples(const Triple &t1, const Triple &t2);
    void gram_schmidt(double *vectors, int num_vectors, int vector_size);
    void generateCombinationsRecursive(const std::vector<int> &numbers, std::vector<int> &combination,
                                       int startIndex, int n, std::vector<std::vector<int>> &combinations);
};

namespace HartreeFockBroydenDetail
{
inline double dot(const std::vector<double> &a, const std::vector<double> &b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

inline double norm(const std::vector<double> &a)
{
    return std::sqrt(std::max(0.0, dot(a, a)));
}

inline double l1_norm(const std::vector<double> &a)
{
    double s = 0.0;
    for (double x : a)
        s += std::abs(x);
    return s;
}

inline bool all_finite(const std::vector<double> &a)
{
    for (double x : a)
        if (!std::isfinite(x))
            return false;
    return true;
}

inline void axpy(double a, const std::vector<double> &x, std::vector<double> &y)
{
    for (size_t i = 0; i < y.size(); ++i)
        y[i] += a * x[i];
}

inline void scale_to_trust_radius(std::vector<double> &x, double max_rms)
{
    if (x.empty() || max_rms <= 0.0)
        return;
    const double rms = norm(x) / std::sqrt(static_cast<double>(x.size()));
    if (rms > max_rms)
    {
        const double s = max_rms / rms;
        for (double &v : x)
            v *= s;
    }
}
}

inline void HartreeFock::Solve_broyden(int history_size, double alpha, double w0)
{
    using namespace HartreeFockBroydenDetail;

    if (history_size < 1)
        history_size = 1;

    const int size_p = dim_p * dim_p;
    const int size_n = dim_n * dim_n;
    const int nvec = size_p + size_n;
    if (nvec == 0)
        return;

    std::deque<std::vector<double>> dF_hist;
    std::deque<std::vector<double>> u_hist;
    std::deque<double> w_hist;
    std::vector<double> F_prev(nvec, 0.0);
    std::vector<double> last_step(nvec, 0.0);
    bool have_prev = false;

    auto compute_thouless_gradient = [&]() -> std::vector<double>
    {
        UpdateDensityMatrix();
        UpdateF();
        CalcEHF();

        // Work in the current HF orbital basis. TransferOperatorToHFbasis mutates
        // FockTerm_{p,n}, so CalcEHF() must be done before this call.
        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);

        std::vector<double> g(nvec, 0.0);
        if (N_p > 0)
            cblas_dcopy(size_p, FockTerm_p, 1, g.data(), 1);
        if (N_n > 0)
            cblas_dcopy(size_n, FockTerm_n, 1, g.data() + size_p, 1);

        // Keep only the particle-hole / hole-particle residual.  For a symmetric
        // Fock matrix this is anti-symmetric and is directly a Thouless generator.
        Operator_ph(g.data(), g.data() + size_p);
        return g;
    };

    auto apply_thouless_step = [&](const std::vector<double> &step)
    {
        std::vector<double> Zp(size_p, 0.0);
        std::vector<double> Zn(size_n, 0.0);
        if (N_p > 0)
            cblas_dcopy(size_p, step.data(), 1, Zp.data(), 1);
        if (N_n > 0)
            cblas_dcopy(size_n, step.data() + size_p, 1, Zn.data(), 1);

        // Padé keeps the orbital rotation closer to unitary than the first-order
        // update when Broyden proposes a nontrivial correction.
        UpdateU_Thouless_pade(Zp.data(), Zn.data());
    };

    std::cout << "  Modified Broyden HF debug: vector = Thouless ph-gradient"
              << "  history_size = " << history_size
              << "  alpha = " << alpha
              << "  w0 = " << w0
              << "  tolerance = " << tolerance << std::endl;

    iterations = 0;
    double E_previous = 0.0;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        std::vector<double> F = compute_thouless_gradient();
        const double grad_l1 = l1_norm(F);
        const double grad_rms = norm(F) / std::sqrt(static_cast<double>(nvec));
        const double dE = (iterations == 0) ? 0.0 : std::abs(EHF - E_previous);

        if (iterations < 10 || iterations % 10 == 0)
        {
            std::cout << "  Broyden iter " << std::setw(5) << iterations
                      << "  grad_l1 = " << std::scientific << std::setprecision(6) << grad_l1
                      << "  grad_rms = " << grad_rms
                      << "  dE = " << dE
                      << "  E = " << EHF
                      << "  hist = " << dF_hist.size() << std::defaultfloat << std::endl;
        }

        if (grad_rms < tolerance)
            break;

        if (have_prev)
        {
            std::vector<double> dF(nvec, 0.0);
            for (int i = 0; i < nvec; ++i)
                dF[i] = F[i] - F_prev[i];

            const double ndF = norm(dF);
            if (ndF > 1.0e-14)
            {
                for (int i = 0; i < nvec; ++i)
                    dF[i] /= ndF;

                // Johnson's Δn is the change in the iterated variable.  For the
                // local Thouless implementation this is the previous applied
                // Thouless step, not the difference of two gradients.
                std::vector<double> dx(last_step);
                for (double &x : dx)
                    x /= ndF;

                std::vector<double> u(dx);
                axpy(alpha, dF, u); // u = alpha*dF + dx, for G^(1)=alpha I

                const double rms_prev = norm(F_prev) / std::sqrt(static_cast<double>(nvec));
                const double w = std::max(1.0, 1.0 / std::max(rms_prev, 1.0e-14));

                dF_hist.push_back(dF);
                u_hist.push_back(u);
                w_hist.push_back(w);
                while (static_cast<int>(dF_hist.size()) > history_size)
                {
                    dF_hist.pop_front();
                    u_hist.pop_front();
                    w_hist.pop_front();
                }
            }
        }

        std::vector<double> step(F);
        for (double &x : step)
            x *= alpha;
        std::string step_name = "linear-Thouless";

        const int m = static_cast<int>(dF_hist.size());
        if (m >= 1)
        {
            std::vector<double> B(m * m, 0.0);
            std::vector<double> rhs(m, 0.0);
            const double w02 = w0 * w0;
            for (int i = 0; i < m; ++i)
            {
                rhs[i] = w_hist[i] * dot(dF_hist[i], F);
                for (int j = 0; j < m; ++j)
                    B[i * m + j] = w_hist[i] * w_hist[j] * dot(dF_hist[i], dF_hist[j]);
                B[i * m + i] += w02;
            }

            std::vector<int> ipiv(m, 0);
            const int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, m, 1, B.data(), m, ipiv.data(), rhs.data(), 1);
            if (info == 0 && all_finite(rhs))
            {
                for (int i = 0; i < m; ++i)
                    axpy(-w_hist[i] * rhs[i], u_hist[i], step);
                step_name = "Broyden-Thouless";
            }
            else
            {
                dF_hist.clear();
                u_hist.clear();
                w_hist.clear();
            }
        }

        if (!all_finite(step))
        {
            step = F;
            for (double &x : step)
                x *= alpha;
            dF_hist.clear();
            u_hist.clear();
            w_hist.clear();
            step_name = "linear-Thouless";
        }

        // A conservative trust radius prevents one bad Broyden solve from making
        // an enormous Thouless rotation.  It is inactive for the small steps in
        // normal convergence.
        scale_to_trust_radius(step, 5.0e-2);

        if (iterations < 10 || iterations % 10 == 0)
            std::cout << "                 step = " << step_name << std::endl;

        E_previous = EHF;
        apply_thouless_step(step);
        last_step = step;
        F_prev = F;
        have_prev = true;
    }

    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation did not converge with Thouless-gradient modified Broyden after "
                  << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
}

#endif