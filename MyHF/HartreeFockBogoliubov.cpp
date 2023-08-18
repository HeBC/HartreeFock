/*
                    _ooOoo_
                   o8888888o
                   88" . "88
                   (| -_- |)
                   O\  =  /O
                ____/`---'\____
              .'  \\|     |//  `.
             /  \\|||  :  |||//  \
            /  _||||| -:- |||||-  \
            |   | \\\  -  /// |   |
            | \_|  ''\---/''  |   |
            \  .-\__  `-`  ___/-. /
          ___`. .'  /--.--\  `. . __
       ."" '<  `.___\_<|>_/___.'  >'"".
      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
      \  \ `-.   \_ __\ /__ _/   .-` /  /
======`-.____`-.___\_____/___.-`____.-'======
                   `=---='
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Codes are far away from bugs
          with the Buddha protecting
*/

///////////////////////////////////////////////////////////////////////////////////
//   Deformed HFB code for the valence space calculation
//   Copyright (C) 2023  Bingcheng He
///////////////////////////////////////////////////////////////////////////////////

#include "ModelSpace.h"
#include "ReadWriteFiles.h"
#include "AngMom.h"

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

#include <omp.h>
using namespace std;

// Define custom reductions for tempValue_e1, tempValue_e2, and tempValue_epairing
#pragma omp declare reduction(+ : ComplexNum : omp_out += omp_in)

struct Triple
{
    double first;
    int second;
    int third;
    Triple(){};
    Triple(double f, int s, int t) : first(f), second(s), third(t) {}
};

class HartreeFockBogoliubov
{
public:
    int N_p, N_n, dim_p, dim_n;

    // Constructor
    HartreeFockBogoliubov(Hamiltonian &H);
    ~HartreeFockBogoliubov();

    // Solve
    void Solve_diag();
    void Solve_gradient();
    void Solve_gradient_Constraint();

    // VAP  PNP
    void Solve_gradient_PNP();
    void ParticleNumberProjection();
    // Tools
    void UpdateDensityMatrix();
    void UpdatePotential(); // update Gamma and Delta
    bool CheckConvergence();
    void SetCanonical_UV(); // use configuration of canonical form of U and V
    void SetCanonical_UV_Random(int RandomSeed);
    void SetByNpNn_UV();
    void UpdateOccupationConstant(double NewV_p, double NewV_n); // update occupation
    void Diagonalize();
    void HFB_H00();
    void PrintEHFB();
    void UpdateTolerance(double t) { tolerance = t; };
    void UpdateGradientStepSize(double stepSize) { gradient_eta = stepSize; };
    void Get_Z_matrix(int dim, ComplexNum *U, ComplexNum *V, ComplexNum *Z); //  input dim, U, V; output Z
    void InverseMatrix(int dim, ComplexNum *U, ComplexNum *U1);
    void InverseMatrix(int dim, ComplexNum *U);
    void Rotate_UV_PNP(int dim, ComplexNum *U, double angle);
    void Rotate_Z_PNP(int dim, ComplexNum *U, double angle); // Zϕ = exp( i 4π ϕ ) Z, where ϕ in range from 0 to 1
    ComplexNum Determinant(int dim, ComplexNum *Matrix);
    ComplexNum HamiltonianME_DifferentStates(ComplexNum *Z_p, ComplexNum *Z1_p, ComplexNum *Z_n, ComplexNum *Z1_n);
    ComplexNum NormME_DifferentStates(double angle_p, double angle_n);

    // gradient method
    void HFB_H11(ComplexNum *H11_p, ComplexNum *H11_n);
    void HFB_H20(ComplexNum *H20_p, ComplexNum *H20_n);
    void HFB_F00(double *F_p, double *F_n, double &F00_p, double &F00_n); // F00 part of the Hermitian one-body operator F = \sum_ij f_ij C^+_i C_j
    void HFB_F20(ComplexNum *F_p, ComplexNum *F_n);                       // F20 part of the Hermitian one-body operato
    void UpdateUV_Thouless(ComplexNum *Z_p, ComplexNum *Z_n);
    void Cal_Gradient_SteepestDescent(ComplexNum *Z_p, ComplexNum *Z_n);
    void Cal_Gradient_Preconditioning(ComplexNum *Z_p, ComplexNum *Z_n);
    void gradient_DifferentStates(ComplexNum *Z_p, ComplexNum *Z_n);

    /// debug
    void Check_Unitarity();
    void PrintDensityMatrix();
    void PrintTransformMatrix();
    void PrintPotential();
    void Check_matrix(int dim, ComplexNum *Matrix);
    void Check_matrix(int dim, double *Matrix);
    bool isSkewMatrix(const ComplexNum *matrix, int rows, int cols);
    bool isIdentityMatrix(const ComplexNum *matrix, int rows, int cols);
    bool isSymmetricMatrix(const ComplexNum *matrix, int rows, int cols);
    bool isOrthogonalMatrix(const ComplexNum *matrix, int number_vector, int vector_size);
    void gram_schmidt(ComplexNum *vectors, int num_vectors, int vector_size);

private:
    ModelSpace *modelspace; /// Model Space
    Hamiltonian *Ham;       /// Hamiltonian
    ///------------------------------------------------------------------------------------------
    /// Beta_k^+ = \sum_l U_{lk} C_l^+ + V_{lk} C_l                        (7.1) in Peter Ring's book
    ComplexNum *V_p, *V_n;         /// transformation coefficients, 1st index is ho basis, 2nd = HFB basis
    ComplexNum *U_p, *U_n;         /// transformation coefficients, 1st index is ho basis, 2nd = HFB basis
    ComplexNum *rho_p, *rho_n;     /// density matrix rho_ij, the size of rho is dim_p * dim_p dim_n * dim_n
    ComplexNum *kappa_p, *kappa_n; /// pairing tensor (anomalous density) kappa_ij, the size of kappa is dim_p * dim_p dim_n * dim_n
    ComplexNum *Gamma_p, *Gamma_n; /// Gamma matrix
    ComplexNum *Delta_p, *Delta_n; /// Delta matrix

    double *T_term_p = nullptr, *T_term_n = nullptr; /// SP energies
    double tolerance;                                /// tolerance for convergence
    int iterations;                                  /// record iterations used in Solve()
    int maxiter = 1000;                              /// max number of iteration
    double *energies, *prev_energies;                /// vector of single particle energies [Proton, Neutron]

    double UnOccupiedConstantU_p = 0.8; ///
    double OccupiedConstantV_p = 0.6;   ///  OccupiedConstantV = Sqrt( 1 - UnOccupiedConstantU ^2 )
    double UnOccupiedConstantU_n = 0.8; ///
    double OccupiedConstantV_n = 0.6;   ///  OccupiedConstantV = Sqrt( 1 - UnOccupiedConstantU ^2 )
    double E_hfb = 10000;               /// The energy will update in the function HFB_H00()
    double e1, e2, epairing;

    double gradient_eta = 0.1; // eta for steepest descent method.
    static bool compareTriples(const Triple &t1, const Triple &t2);
    void EigenProblem(int n, ComplexNum *A, double *EigenValue, ComplexNum *EigenVector);
    void EigenProblem(int n, ComplexNum *A, double *EigenValue);
    void remove_row_and_column_inplace(int m, int n, ComplexNum *A, int row_to_remove, int col_to_remove);
    void remove_element_inplace(int dim, ComplexNum *A, int index_to_remove);
};

HartreeFockBogoliubov::HartreeFockBogoliubov(Hamiltonian &H)
    : Ham(&H), modelspace(H.GetModelSpace()), tolerance(1e-8)
{
    this->N_p = modelspace->GetProtonNum();
    this->N_n = modelspace->GetNeutronNum();
    this->dim_p = modelspace->Get_MScheme_dim(Proton);
    this->dim_n = modelspace->Get_MScheme_dim(Neutron);
    ///  transformation matrix
    this->V_p = (ComplexNum *)mkl_malloc((dim_p * dim_p) * sizeof(ComplexNum), 64);
    memset(V_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    this->V_n = (ComplexNum *)mkl_malloc((dim_n * dim_n) * sizeof(ComplexNum), 64);
    memset(V_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    this->U_p = (ComplexNum *)mkl_malloc((dim_p * dim_p) * sizeof(ComplexNum), 64);
    memset(U_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    this->U_n = (ComplexNum *)mkl_malloc((dim_n * dim_n) * sizeof(ComplexNum), 64);
    memset(U_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    /// density
    this->rho_p = (ComplexNum *)mkl_malloc((dim_p) * (dim_p) * sizeof(ComplexNum), 64);
    memset(rho_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    this->rho_n = (ComplexNum *)mkl_malloc((dim_n) * (dim_n) * sizeof(ComplexNum), 64);
    memset(rho_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    this->kappa_p = (ComplexNum *)mkl_malloc((dim_p) * (dim_p) * sizeof(ComplexNum), 64);
    memset(kappa_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    this->kappa_n = (ComplexNum *)mkl_malloc((dim_n) * (dim_n) * sizeof(ComplexNum), 64);
    memset(kappa_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    ///  potential matrix
    this->Gamma_p = (ComplexNum *)mkl_malloc((dim_p) * (dim_p) * sizeof(ComplexNum), 64);
    memset(Gamma_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    this->Gamma_n = (ComplexNum *)mkl_malloc((dim_n) * (dim_n) * sizeof(ComplexNum), 64);
    memset(Gamma_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    this->Delta_p = (ComplexNum *)mkl_malloc((dim_p) * (dim_p) * sizeof(ComplexNum), 64);
    memset(Delta_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    this->Delta_n = (ComplexNum *)mkl_malloc((dim_n) * (dim_n) * sizeof(ComplexNum), 64);
    memset(Delta_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));

    // One body part
    T_term_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    T_term_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    memset(T_term_p, 0, (dim_p) * (dim_p) * sizeof(double));
    memset(T_term_n, 0, (dim_n) * (dim_n) * sizeof(double));

    for (size_t i = 0; i < Ham->OBEs_p.size(); i++)
    {
        int index_a, index_b;
        index_a = Ham->OBEs_p[i].GetIndex_a();
        index_b = Ham->OBEs_p[i].GetIndex_b();
        double temp_E = Ham->OBEs_p[i].GetE();
        int temp_j = modelspace->Orbits_p[index_a].j2;
        int M_i = modelspace->LookupStartingPoint(Proton, index_a);
        int M_j = modelspace->LookupStartingPoint(Proton, index_b);
        for (size_t j = 0; j < temp_j + 1; j++)
        {
            T_term_p[(M_i + j) * dim_p + (M_j + j)] = temp_E;
        }
    }
    for (size_t i = 0; i < Ham->OBEs_n.size(); i++)
    {
        int index_a, index_b;
        index_a = Ham->OBEs_n[i].GetIndex_a();
        index_b = Ham->OBEs_n[i].GetIndex_b();
        double temp_E = Ham->OBEs_n[i].GetE();
        int temp_j = modelspace->Orbits_n[index_a].j2;
        int M_i = modelspace->LookupStartingPoint(Neutron, index_a);
        int M_j = modelspace->LookupStartingPoint(Neutron, index_b);
        for (size_t j = 0; j < temp_j + 1; j++)
            T_term_n[(M_i + j) * dim_n + (M_j + j)] = temp_E;
    }

    // SPE in ascending order
    std::vector<Triple> SPEpairs_p(dim_p);
    for (int i = 0; i < dim_p; ++i)
    {
        // std::cout << i << "  " << T_term[i] << "  " << modelspace->Get_MSmatrix_2j(Proton, i) << "  " << modelspace->Get_MSmatrix_2m(Proton, i) << std::endl;
        SPEpairs_p[i] = Triple(T_term_p[i * dim_p + i], i, modelspace->Get_MSmatrix_2m(Proton, i));
    }
    std::sort(SPEpairs_p.begin(), SPEpairs_p.end(), compareTriples);

    //-----------
    // Sort the vector of pairs based on the first element of each pair, which is the value of the
    // element in the original array.
    std::vector<Triple> SPEpairs_n(dim_n);
    for (int i = 0; i < dim_n; ++i)
    {
        // std::cout << i << "  " << T_term[i] << "  " << modelspace->Get_MSmatrix_2j(Proton, i) << "  " << modelspace->Get_MSmatrix_2m(Proton, i) << std::endl;
        SPEpairs_n[i] = Triple(T_term_n[i * dim_n + i], i, modelspace->Get_MSmatrix_2m(Neutron, i));
    }
    std::sort(SPEpairs_n.begin(), SPEpairs_n.end(), compareTriples);

    this->energies = (double *)mkl_malloc((2 * dim_p + 2 * dim_n) * sizeof(double), 64);
    this->prev_energies = (double *)mkl_malloc((2 * dim_p + 2 * dim_n) * sizeof(double), 64);
    memset(energies, 1000., (2 * dim_p + 2 * dim_n) * sizeof(double));
    memset(prev_energies, 0, (2 * dim_p + 2 * dim_n) * sizeof(double));
    this->SetCanonical_UV(); // initial U and V matrix
}

HartreeFockBogoliubov::~HartreeFockBogoliubov()
{
    mkl_free(V_p);
    mkl_free(V_n);
    mkl_free(U_p);
    mkl_free(U_n);
    mkl_free(rho_p);
    mkl_free(rho_n);
    mkl_free(kappa_p);
    mkl_free(kappa_n);
    mkl_free(Gamma_p);
    mkl_free(Gamma_n);
    mkl_free(Delta_p);
    mkl_free(Delta_n);
    mkl_free(energies);
    mkl_free(prev_energies);
    if (T_term_p != nullptr)
    {
        mkl_free(T_term_p);
    }
    if (T_term_n != nullptr)
    {
        mkl_free(T_term_n);
    }
}

bool HartreeFockBogoliubov::compareTriples(const Triple &t1, const Triple &t2)
{

    if (fabs(t1.first - t2.first) < 1.e-5)
    {
        if (abs(t1.third) < abs(t2.third))
        {
            return true;
        }
        else
            return false;
    }
    else
    {
        if (t1.first < t2.first)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

///*********************************************************************
/// one-body density matrix, SEE Peter Ring Eq.(7.23) page 251
/// ρ = V^∗  V^t   density matrix
/// κ = V^∗  U^t   pairing tensor (anomalous density)
/// A 2M dimensional space
/// [ Beta   ] = [ U^+  V^+ ] [ c   ]
/// [ Beta^+ ]   [ V^T  U^T ] [ c^+ ]
/// where ρ is hermitatian
/// and κ is skew symmetric
void HartreeFockBogoliubov::UpdateDensityMatrix()
{
    /// density matrix
    ComplexNum alpha = ComplexNum(1.0, 0.0); // (1.0 + 0.0i)
    ComplexNum beta = ComplexNum(0.0, 0.0);  // (1.0 + 0.0i)
    /// First calculate transpose of rho
    /// rho^T = V V^H, H represent hermitatian conjugate
    /// Then perform a in-place transpose
    if (N_p > 0)
    {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim_p, dim_p, dim_p, &alpha, V_p, dim_p, V_p, dim_p, &beta, rho_p, dim_p);
        mkl_zimatcopy('R', 'T', dim_p, dim_p, alpha, rho_p, dim_p, dim_p);
    }
    if (N_n > 0)
    {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim_n, dim_n, dim_n, &alpha, V_n, dim_n, V_n, dim_n, &beta, rho_n, dim_n);
        mkl_zimatcopy('R', 'T', dim_n, dim_n, alpha, rho_n, dim_n, dim_n);
    }

    /// pairing tensor
    /// First calculate transpose of k
    /// k^T = U V^H, H represent hermitatian conjugate
    /// Then perform a in-place transpose
    if (N_p > 0)
    {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim_p, dim_p, dim_p, &alpha, U_p, dim_p, V_p, dim_p, &beta, kappa_p, dim_p);
        mkl_zimatcopy('R', 'T', dim_p, dim_p, alpha, kappa_p, dim_p, dim_p);
    }
    if (N_n > 0)
    {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim_n, dim_n, dim_n, &alpha, U_n, dim_n, V_n, dim_n, &beta, kappa_n, dim_n);
        mkl_zimatcopy('R', 'T', dim_n, dim_n, alpha, kappa_n, dim_n, dim_n);
    }
}

//*********************************************************************
///  [See Peter Ring eq 7.41] page 254
///   Gamma_{ij} = \sum_{a b}  \rho_{ab} \bar{V}^{(2)}_{ibja}
///   Delta_{ij} = 1/2 * \sum_{a b} \kappa_{ab} \bar{V}^{(2)}_{ijab}
void HartreeFockBogoliubov::UpdatePotential()
{
    memset(Gamma_p, 0, dim_p * dim_p * sizeof(ComplexNum));
    memset(Gamma_n, 0, dim_n * dim_n * sizeof(ComplexNum));
    memset(Delta_p, 0, dim_p * dim_p * sizeof(ComplexNum));
    memset(Delta_n, 0, dim_n * dim_n * sizeof(ComplexNum));
    // use a skewed stored Vpp and Vnn
    double *Vpp, *Vpn, *Vnn;
    Vpp = Ham->MSMEs.GetVppPrt();
    Vnn = Ham->MSMEs.GetVnnPrt();
    Vpn = Ham->MSMEs.GetVpnPrt();
    std::vector<ComplexNum> Temp_rho_p(dim_p * dim_p, 0);
    std::vector<ComplexNum> Temp_rho_n(dim_n * dim_n, 0);
    mkl_zomatcopy('R', 'T', dim_p, dim_p, 1.0, rho_p, dim_p, Temp_rho_p.data(), dim_p);
    mkl_zomatcopy('R', 'T', dim_n, dim_n, 1.0, rho_n, dim_n, Temp_rho_n.data(), dim_n);

    // Proton subspace
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t l = 0; l < dim_p; l++)
                {
                    // add Vpp term
                    Gamma_p[i * dim_p + j] += Vpp[i * dim_p * dim_p * dim_p + j * dim_p * dim_p + k * dim_p + l] * Temp_rho_p[k * dim_p + l];

                    // pairing
                    Delta_p[i * dim_p + j] += 0.5 * Vpp[i * dim_p * dim_p * dim_p + k * dim_p * dim_p + j * dim_p + l] * kappa_p[k * dim_p + l];
                }
            }
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t l = 0; l < dim_n; l++)
                {
                    // add Vpn term
                    Gamma_p[i * dim_p + j] += Vpn[dim_p * dim_n * dim_n * i + dim_n * dim_n * j + k * dim_n + l] * Temp_rho_n[k * dim_n + l];
                }
            }
        }
    }

    // Neutron subspace
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t l = 0; l < dim_n; l++)
                {
                    // add Vnn term
                    Gamma_n[i * dim_n + j] += Vnn[dim_n * dim_n * dim_n * i + dim_n * dim_n * j + k * dim_n + l] * Temp_rho_n[k * dim_n + l];

                    Delta_n[i * dim_n + j] += 0.5 * Vnn[i * dim_n * dim_n * dim_n + k * dim_n * dim_n + j * dim_n + l] * kappa_n[k * dim_n + l];
                }
            }

            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t l = 0; l < dim_p; l++)
                {
                    // add Vpn term
                    Gamma_n[i * dim_n + j] += Vpn[dim_p * dim_n * dim_n * k + dim_n * dim_n * l + i * dim_n + j] * Temp_rho_p[k * dim_n + l];
                }
            }
        }
    }
}

/// Gram Schimidt procedure
/// in place transform
void HartreeFockBogoliubov::gram_schmidt(ComplexNum *vectors, int num_vectors, int vector_size)
{
    int i, j, k;
    ComplexNum temp;

    // Normalize the first vector (vectors[0])
    ComplexNum norm;
    cblas_zdotc_sub(vector_size, vectors, num_vectors, vectors, num_vectors, &norm);
    norm = 1. / sqrt(norm);
    cblas_zscal(vector_size, &norm, vectors, num_vectors);

    // Gram-Schmidt process for the remaining vectors
    for (i = 1; i < num_vectors; i++)
    {
        for (j = 0; j < i; j++)
        {
            // Compute the inner product between vectors[j] and vectors[i]
            ComplexNum inner_product;
            cblas_zdotc_sub(vector_size, vectors + j, num_vectors, vectors + i, num_vectors, &inner_product);

            // Subtract the projection of vectors[i] onto vectors[j]
            for (k = 0; k < vector_size; k++)
            {
                temp = inner_product * vectors[j + k * num_vectors];
                vectors[i + k * num_vectors] -= temp;
            }
        }

        // Normalize vectors[i]
        cblas_zdotc_sub(vector_size, vectors + i, num_vectors, vectors + i, num_vectors, &norm);
        norm = 1. / sqrt(norm);
        cblas_zscal(vector_size, &norm, vectors + i, num_vectors);
    }
}

//*********************************************************************
/// Diagonalize and update the U and V matrix
/// h = T + Gamma
/// (   h         Delta   )( U_k )  = ( U_k ) E_k
/// (  -Delta^*   -h^*    )( V_k )  = ( V_k )
void HartreeFockBogoliubov::Solve_diag()
{
    iterations = 0; // count number of iterations
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        UpdateDensityMatrix();
        UpdatePotential();
        // PrintDensityMatrix();
        // PrintPotential();
        // Check_Unitarity();

        // HFB_H00();
        Diagonalize();
        // Diagonalize_h_part();

        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        if (CheckConvergence())
            break;
    }

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HFB converged after " << iterations << " iterations. " << std::endl;
        UpdateDensityMatrix();
        UpdatePotential();
        HFB_H00();
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock-Bogoliubov calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
}

//*********************************************************************
// Gradient method
// First order gradient are considered
// the thouless theorem is adopted
// |phi> = e^( Z b^+ b^+ ) |phi_0>   (Zij = - eta H20  i<j )
void HartreeFockBogoliubov::Solve_gradient()
{
    double E_previous = 1.e10;
    tolerance = 1.e-5;
    iterations = 0; // count number of iterations
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        E_previous = this->E_hfb;
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
        UpdatePotential();
        HFB_H00();
        std::vector<ComplexNum> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<ComplexNum> Z_n(dim_n * dim_n, 0);
        HFB_H20(Z_p.data(), Z_n.data());
        // Cal_Gradient_SteepestDescent(Z_p.data(), Z_n.data());
        Cal_Gradient_Preconditioning(Z_p.data(), Z_n.data());
        UpdateUV_Thouless(Z_p.data(), Z_n.data());

        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        if (fabs(E_previous - E_hfb) < this->tolerance)
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdatePotential();
    HFB_H00();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock-Bogoliubov calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHFB();
    return;
}

void HartreeFockBogoliubov::remove_row_and_column_inplace(int m, int n, ComplexNum *A, int row_to_remove, int col_to_remove)
{
    // Remove the specified row
    for (size_t i = row_to_remove; i < m - 1; i++)
    {
        int step = 1;
        zcopy(&m, A + (i + 1) * m, &step, A + i * m, &step);
    }
    // Remove the specified column
    for (size_t i = col_to_remove; i < n - 1; i++)
    {
        zcopy(&n, A + (i + 1), &n, A + i, &n);
    }

    for (size_t i = 0; i < m; i++)
    {
        A[m - 1 + m * i] = 0;
        A[m * (n - 1) + i] = 0;
    }
}

void HartreeFockBogoliubov::remove_element_inplace(int dim, ComplexNum *A, int index_to_remove)
{
    // Remove the specified row
    for (size_t i = index_to_remove; i < dim - 1; i++)
    {
        int step = 1;
        A[i] = A[i + 1];
    }

    A[dim - 1] = 0;
}

// constraint HFB
// Hc = H0 - lamda * Q
// the parameter lamda will bring the expectation value of <Q> to the correct
// value Q0. This is achieved by solving the equation  <Q> - Q0 = Z Q^20
// in the above equation, we just keep the first order correction of thouless
// transformation. When inlcuding multi-constaints, the accosicated linear
// equation is solved to determined the lamdas
// Suppose we have N constraints, <Qi> is the ith one-body constraint
// The Hamiltonian is H =  H0 - \sum_i  λi Qi
// If convergence is achieved. we find, as in a variation whh a Lagrange parameter λ.
// ( H0 - \sum_i λi Qi )^20 = 0
// By using the thouless theorem, the gradient is given by
// Z = - η ( H0 - \sum_i λi Qi )^20
// For a specific ith one-body constraint, we have
// <Qi> - qi = Z Qi^20 =   - η ( H0 - \sum_j λj Qj )^20 Qi^20
// The λj will determined by soloving the above linear equation in two steps.
// First, we assume we arrive at the point <Qi> = qi, where qi is our target.
// the λj is given by the following set of linear equations
// H0^20  Qi^20 = \sum_j λj Qj^20  Qi^20
// Then, we correct the expectation of one-body operator <Qi> by soloving the
// equations
// <Qi> - qi = H0^20 Qi^20 - \sum_j λ^cor_j Qj^20  Qi^20
// Therefore, the new gradient is
// Z = -η ( H0^20 - \sum_i λi Qi^20 ) - \sum_i λ^cor_i Qi^20
void HartreeFockBogoliubov::Solve_gradient_Constraint()
{
    double E_previous = 1.e10;
    iterations = 0; // count number of iterations
    UpdateTolerance(1.e-5);
    UpdateGradientStepSize(0.06);
    //--------------------------------------------------------------------------
    // include Constrains
    int number_of_Q = 0;
    int number_of_Q_p = 0, number_of_Q_n = 0;
    std::vector<std::string> Qtype;
    if (modelspace->Get_ParticleNumberConstrained())
    {
        number_of_Q += 2;                    //  for Np and Nn
        Qtype.push_back("ParticleNumberNp"); // Constraint N_p
        Qtype.push_back("ParticleNumberNn"); // Constraint N_n
        number_of_Q_p += 1;
        number_of_Q_n += 1;
    }
    if (modelspace->GetIsShapeConstrained())
    {
        number_of_Q += 2; //  for Q0 and Q2
        Qtype.push_back("QuadrupoleQ0");
        Qtype.push_back("QuadrupoleQ2");
        number_of_Q_p += 2;
        number_of_Q_n += 2;
    }

    //--------------------------------------------------------------------------
    // load Operators
    if (number_of_Q == 0)
    {
        std::cout << "   No constrain loaded!" << std::endl;
        exit(0);
    }
    std::vector<std::vector<double>> QOperator_p(number_of_Q, std::vector<double>(dim_p * dim_p, 0));
    std::vector<std::vector<double>> QOperator_n(number_of_Q, std::vector<double>(dim_n * dim_n, 0));
    std::vector<double> targets(number_of_Q, 0);
    std::vector<double> deltaQs(number_of_Q, 0);

    /// Load Particle Number operator
    auto it = std::find(Qtype.begin(), Qtype.end(), "ParticleNumberNp");
    if (it != Qtype.end())
    {
        // Element found
        int index = std::distance(Qtype.begin(), it);
        // std::cout << "Element " << Qtype[index] << " found at index " << index << std::endl;
        // Load Proton Particle Number operator
        memset(QOperator_p[index].data(), 0, (dim_p * dim_p) * sizeof(double));
        for (size_t i = 0; i < dim_p; i++)
        {
            QOperator_p[index][i * dim_p + i] = 1.;
        }
        memset(QOperator_p[index + 1].data(), 0, (dim_p * dim_p) * sizeof(double));
        for (size_t i = 0; i < dim_p; i++)
        {
            // QOperator_p[index + 1][i * dim_p + i] = 1.;
        }

        // Load Neutron Particle Number operator
        memset(QOperator_n[index].data(), 0, (dim_n * dim_n) * sizeof(double));
        for (size_t i = 0; i < dim_n; i++)
        {
            // QOperator_n[index][i * dim_n + i] = 1.;
        }
        memset(QOperator_n[index + 1].data(), 0, (dim_n * dim_n) * sizeof(double));
        for (size_t i = 0; i < dim_n; i++)
        {
            QOperator_n[index + 1][i * dim_n + i] = 1.;
        }

        // load targets
        targets[index] = N_p;
        targets[index + 1] = N_n;
    }

    /// Load Quadrupole operator
    it = std::find(Qtype.begin(), Qtype.end(), "QuadrupoleQ0");
    if (it != Qtype.end())
    {
        // Element found
        int index = std::distance(Qtype.begin(), it);
        // std::cout << "Element " << Qtype[index] << " found at index " << index << std::endl;
        // Load Proton Quadrupole
        memset(QOperator_p[index].data(), 0, (dim_p * dim_p) * sizeof(double));
        for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
        {
            int Qindex = Ham->Q2MEs_p.Q0_list[i];          // index of M scheme One body operator
            int ia = Ham->MSMEs.OB_p[Qindex].GetIndex_a(); // index of a in M scheme
            int ib = Ham->MSMEs.OB_p[Qindex].GetIndex_b();
            QOperator_p[index][ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
        }
        memset(QOperator_p[index + 1].data(), 0, (dim_p * dim_p) * sizeof(double));
        for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
        {
            int Qindex = Ham->Q2MEs_p.Q2_list[i];          // index of M scheme One body operator
            int ia = Ham->MSMEs.OB_p[Qindex].GetIndex_a(); // index of a in M scheme
            int ib = Ham->MSMEs.OB_p[Qindex].GetIndex_b();
            QOperator_p[index + 1][ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
        }
        for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
        {
            int Qindex = Ham->Q2MEs_p.Q_2_list[i];         // index of M scheme One body operator
            int ia = Ham->MSMEs.OB_p[Qindex].GetIndex_a(); // index of a in M scheme
            int ib = Ham->MSMEs.OB_p[Qindex].GetIndex_b();
            QOperator_p[index + 1][ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
        }
        // Load neutron Quadrupole
        memset(QOperator_n[index].data(), 0, (dim_n * dim_n) * sizeof(double));
        for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
        {
            int Qindex = Ham->Q2MEs_n.Q0_list[i];          // index of M scheme One body operator
            int ia = Ham->MSMEs.OB_n[Qindex].GetIndex_a(); // index of a in M scheme
            int ib = Ham->MSMEs.OB_n[Qindex].GetIndex_b();
            QOperator_n[index][ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
        }
        memset(QOperator_n[index + 1].data(), 0, (dim_n * dim_n) * sizeof(double));
        for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
        {
            int Qindex = Ham->Q2MEs_n.Q2_list[i];          // index of M scheme One body operator
            int ia = Ham->MSMEs.OB_n[Qindex].GetIndex_a(); // index of a in M scheme
            int ib = Ham->MSMEs.OB_n[Qindex].GetIndex_b();
            QOperator_n[index + 1][ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
        }
        for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
        {
            int Qindex = Ham->Q2MEs_n.Q_2_list[i];         // index of M scheme One body operator
            int ia = Ham->MSMEs.OB_n[Qindex].GetIndex_a(); // index of a in M scheme
            int ib = Ham->MSMEs.OB_n[Qindex].GetIndex_b();
            QOperator_n[index + 1][ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
        }
        // load targets
        targets[index] = modelspace->GetShapeQ0();
        targets[index + 1] = modelspace->GetShapeQ2();
    }

    /// Load operator ...
    /// ...

    //--------------------------------------------------------------------------
    // Solve constrained HF
    UpdateDensityMatrix(); // Update density matrix
    Check_Unitarity();
    UpdatePotential();
    // intitial expectation values of targets
    for (size_t i = 0; i < number_of_Q; i++)
    {
        double tempQp, tempQn;
        HFB_F00(QOperator_p[i].data(), QOperator_n[i].data(), tempQp, tempQn);
        deltaQs[i] = tempQp + tempQn - targets[i];
    }
    HFB_H00();
    for (iterations = 0; iterations < maxiter; ++iterations)
    {

        E_previous = this->E_hfb;
        std::vector<ComplexNum> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<ComplexNum> Z_n(dim_n * dim_n, 0);
        std::vector<ComplexNum> H20_p(dim_p * dim_p, 0);
        std::vector<ComplexNum> H20_n(dim_n * dim_n, 0);
        HFB_H20(H20_p.data(), H20_n.data());
        Z_p = H20_p;
        Z_n = H20_n;

        // contribution from constrian
        std::vector<std::vector<ComplexNum>> Q20_p(number_of_Q, std::vector<ComplexNum>(dim_p * dim_p, 0));
        std::vector<std::vector<ComplexNum>> Q20_n(number_of_Q, std::vector<ComplexNum>(dim_n * dim_n, 0));
        for (size_t i = 0; i < number_of_Q; i++)
        {
            cblas_dcopy(dim_p * dim_p, QOperator_p[i].data(), 1, (double *)Q20_p[i].data(), 2); // check this
            cblas_dcopy(dim_n * dim_n, QOperator_n[i].data(), 1, (double *)Q20_n[i].data(), 2);
            HFB_F20(Q20_p[i].data(), Q20_n[i].data());
        }

        //---------------------------------------------
        std::vector<ComplexNum> A_p(number_of_Q * number_of_Q, 0);
        std::vector<ComplexNum> A_n(number_of_Q * number_of_Q, 0);
        std::vector<ComplexNum> b_p(number_of_Q, 0);
        std::vector<ComplexNum> b_n(number_of_Q, 0);
        // prepare matrix
        for (size_t i = 0; i < number_of_Q; i++)
        {
            for (size_t j = i; j < number_of_Q; j++)
            {
                // QQ part proton
                ComplexNum QQ_complex = 0.;
                // #pragma omp parallel for
                for (size_t k = 0; k < dim_p; k++)
                {
                    for (size_t l = 0; l < dim_p; l++)
                    {
                        QQ_complex += Q20_p[i][k * dim_p + l] * std::conj(Q20_p[j][k * dim_p + l]) + Q20_p[j][k * dim_p + l] * std::conj(Q20_p[i][k * dim_p + l]);
                    }
                }
                A_p[i * number_of_Q + j] = 0.5 * QQ_complex;
                // QQ part proton
                QQ_complex = 0.;
                // #pragma omp parallel for
                for (size_t k = 0; k < dim_n; k++)
                {
                    for (size_t l = 0; l < dim_n; l++)
                    {
                        QQ_complex += Q20_n[i][k * dim_n + l] * std::conj(Q20_n[j][k * dim_n + l]) + Q20_n[j][k * dim_n + l] * std::conj(Q20_n[i][k * dim_n + l]);
                    }
                }
                A_n[i * number_of_Q + j] = 0.5 * QQ_complex;
                if (i != j)
                {
                    A_p[j * number_of_Q + i] = A_p[i * number_of_Q + j];
                    A_n[j * number_of_Q + i] = A_n[i * number_of_Q + j];
                }
            }

            // HQ proton
            ComplexNum HQ_complex = 0.;
            // #pragma omp parallel for
            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t l = 0; l < dim_p; l++)
                {
                    HQ_complex += Q20_p[i][k * dim_p + l] * std::conj(H20_p[k * dim_p + l]) + H20_p[k * dim_p + l] * std::conj(Q20_p[i][k * dim_p + l]);
                }
            }
            // b_p[i] = deltaQs[i] / gradient_eta + HQ_complex;
            b_p[i] = 0.5 * HQ_complex;
            // HQ neutron
            HQ_complex = 0.;
            // #pragma omp parallel for
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t l = 0; l < dim_n; l++)
                {
                    HQ_complex += Q20_n[i][k * dim_n + l] * std::conj(H20_n[k * dim_n + l]) + H20_n[k * dim_n + l] * std::conj(Q20_n[i][k * dim_n + l]);
                }
            }
            // b_n[i] = deltaQs[i] / gradient_eta + HQ_complex;
            b_n[i] = 0.5 * HQ_complex;
        }

        // remove singularity
        for (size_t i = 0; i < number_of_Q; i++)
        {
            if (Qtype[i] == "ParticleNumberNn")
            {
                // Check_matrix(number_of_Q, A_p.data());
                remove_row_and_column_inplace(number_of_Q, number_of_Q, A_p.data(), i, i);
                remove_element_inplace(number_of_Q, b_p.data(), i);
                // Check_matrix(number_of_Q, A_p.data());
            }
            if (Qtype[i] == "ParticleNumberNp")
            {
                // Check_matrix(number_of_Q, A_n.data());
                remove_row_and_column_inplace(number_of_Q, number_of_Q, A_n.data(), i, i);
                remove_element_inplace(number_of_Q, b_n.data(), i);
                // Check_matrix(number_of_Q, A_n.data());
            }
        }

        std::vector<ComplexNum> Acopy_p(number_of_Q * number_of_Q, 0);
        std::vector<ComplexNum> Acopy_n(number_of_Q * number_of_Q, 0);
        std::vector<int> ipiv_p(number_of_Q, 0);
        std::vector<int> ipiv_n(number_of_Q, 0);
        Acopy_p = A_p;
        Acopy_n = A_n;

        if (LAPACKE_zgesv(LAPACK_ROW_MAJOR, number_of_Q_p, 1, Acopy_p.data(), number_of_Q, ipiv_p.data(), b_p.data(), 1) != 0)
        {
            std::cout << "  Proton Linear equation error!" << std::endl;
            exit(0);
        }
        if (LAPACKE_zgesv(LAPACK_ROW_MAJOR, number_of_Q_n, 1, Acopy_n.data(), number_of_Q, ipiv_n.data(), b_n.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }
        int count_p = 0;
        int count_n = 0;
        for (size_t i = 0; i < number_of_Q; i++)
        {
            if (fabs(b_p[i].imag()) > 1.e-7 or fabs(b_n[i].imag()) > 1.e-7)
            {
                std::cout << "   The lamda have a imag part!! " << std::endl;
                exit(0);
            }
            ComplexNum alpha;

            if (Qtype[i] != "ParticleNumberNn")
            {
                alpha = -b_p[count_p];
                cblas_zaxpy(dim_p * dim_p, &alpha, Q20_p[i].data(), 1, Z_p.data(), 1);
                count_p++;
            }

            if (Qtype[i] != "ParticleNumberNp")
            {
                alpha = -b_n[count_n];
                cblas_zaxpy(dim_n * dim_n, &alpha, Q20_n[i].data(), 1, Z_n.data(), 1);
                count_n++;
            }
        }
        // Cal_Gradient_SteepestDescent(Z_p.data(), Z_n.data());
        Cal_Gradient_Preconditioning(Z_p.data(), Z_n.data());

        //--------------------------------
        Acopy_p = A_p;
        Acopy_n = A_n;
        count_p = 0;
        count_n = 0;
        for (size_t i = 0; i < number_of_Q; i++)
        {
            if (Qtype[i] != "ParticleNumberNn")
            {
                b_p[count_p] = deltaQs[i];
                count_p++;
            }
            // std::cout << b_p[i] << std::endl;
            // Check_matrix(number_of_Q, A_p.data());
            if (Qtype[i] != "ParticleNumberNp")
            {
                b_n[count_n] = deltaQs[i];
                count_n++;
            }
        }
        /// correct expectation of operator
        if (LAPACKE_zgesv(LAPACK_ROW_MAJOR, number_of_Q_p, 1, Acopy_p.data(), number_of_Q, ipiv_p.data(), b_p.data(), 1) != 0)
        {
            std::cout << "  Proton Linear equation error!" << std::endl;
            exit(0);
        }
        if (LAPACKE_zgesv(LAPACK_ROW_MAJOR, number_of_Q_p, 1, Acopy_n.data(), number_of_Q, ipiv_n.data(), b_n.data(), 1) != 0)
        {
            std::cout << "  Neutron Linear equation error!" << std::endl;
            exit(0);
        }
        count_p = 0;
        count_n = 0;
        for (size_t i = 0; i < number_of_Q; i++)
        {
            ComplexNum alpha;
            if (Qtype[i] != "ParticleNumberNn")
            {
                alpha = -b_p[count_p];
                cblas_zaxpy(dim_p * dim_p, &alpha, Q20_p[i].data(), 1, Z_p.data(), 1);
                count_p++;
            }
            if (Qtype[i] != "ParticleNumberNp")
            {
                alpha = -b_n[count_n];
                cblas_zaxpy(dim_n * dim_n, &alpha, Q20_n[i].data(), 1, Z_n.data(), 1);
                count_n++;
            }
        }

        UpdateUV_Thouless(Z_p.data(), Z_n.data());
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
        UpdatePotential();     // Update the Fock matrix

        // PrintDensityMatrix();
        // PrintTransformMatrix();

        for (size_t i = 0; i < number_of_Q; i++)
        {
            double tempQp, tempQn;
            HFB_F00(QOperator_p[i].data(), QOperator_n[i].data(), tempQp, tempQn);
            deltaQs[i] = tempQp + tempQn - targets[i];
            // std::cout << i << "   " << tempQp << "  " << tempQn << "  " << deltaQs[i] << std::endl;
            // std::cout << "  " << i << "   " << Qtype[i] << "    " << tempQp << "  " << tempQn << "  " << deltaQs[i] << std::endl;
        }
        // std::cout << std::endl;
        HFB_H00();
        //   std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        //   if (CheckConvergence())
        if (fabs(E_previous - E_hfb) < this->tolerance)
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdatePotential();     // Update the Fock matrix
    HFB_H00();
    std::cout << "  "
              << "Qi"
              << "  "
              << "Constraints type"
              << "        "
              << "Qp"
              << "        "
              << "Qn"
              << "       "
              << "Delta Q" << std::endl;
    std::cout << std::setprecision(4);
    for (size_t i = 0; i < number_of_Q; i++)
    {
        double tempQp, tempQn;
        HFB_F00(QOperator_p[i].data(), QOperator_n[i].data(), tempQp, tempQn);
        deltaQs[i] = tempQp + tempQn - targets[i];

        std::cout << std::fixed << "  " << i << "   " << std::setw(16) << Qtype[i] << "    " << std::setw(8) << tempQp << "   " << std::setw(8) << tempQn << "    " << deltaQs[i] << std::endl;
    }
    std::cout << std::endl;
    if (iterations < maxiter)
    {
        std::cout << "  HFB converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Constrained Hartree-Fock-Bogoliubov calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHFB();
}

void HartreeFockBogoliubov::Solve_gradient_PNP()
{
    QuadratureClass PNP_mesh_p, PNP_mesh_n;
    AngMomProjection angMomProjectionInstance;
    // M = max (1/2 N, Omega - 1/2 N) + 1
    // See more in Nuclear Physics A332 (1979) 69-51
    int meshDim_p = std::max(N_p / 2, dim_p - N_p / 2) + 1;
    int meshDim_n = std::max(N_n / 2, dim_n - N_n / 2) + 1;
    PNP_mesh_p.SetNumber(meshDim_p);
    PNP_mesh_n.SetNumber(meshDim_n);
    angMomProjectionInstance.Generate_GQ_Mesh(PNP_mesh_p, "legendre");
    angMomProjectionInstance.Generate_GQ_Mesh(PNP_mesh_n, "legendre");

    std::vector<ComplexNum> Z0_p(dim_p * dim_p);
    std::vector<ComplexNum> Z1_p(dim_p * dim_p);
    std::vector<ComplexNum> Z0_n(dim_n * dim_n);
    std::vector<ComplexNum> Z1_n(dim_n * dim_n);

    UpdateDensityMatrix();
    ComplexNum HFBh00 = 0.;
    ComplexNum HFBnorm = 0.;
    // Get unrotated Z matrix
    Get_Z_matrix(dim_p, U_p, V_p, Z0_p.data());
    Get_Z_matrix(dim_n, U_n, V_n, Z0_n.data());
    for (size_t gauge_p = 0; gauge_p < PNP_mesh_p.GetTotalNumber(); gauge_p++)
    {
        for (size_t gauge_n = 0; gauge_n < PNP_mesh_n.GetTotalNumber(); gauge_n++)
        {
            double angle_p, angle_n;
            ComplexNum Weight_p, Weight_n;
            angle_p = PNP_mesh_p.GetX(gauge_p);
            angle_n = PNP_mesh_n.GetX(gauge_n);
            Weight_p = PNP_mesh_p.GetWeight(gauge_p) * std::exp(M_PI * angle_p * N_p * ComplexNum(0, -2));
            Weight_n = PNP_mesh_n.GetWeight(gauge_n) * std::exp(M_PI * angle_n * N_n * ComplexNum(0, -2));
            // std::cout << gauge_p << "  " << gauge_n << "  " << angle_p << "    " << angle_n << "   " << Weight_p << Weight_n << std::endl;

            // Z and Zg
            cblas_zcopy(dim_p * dim_p, Z0_p.data(), 1, Z1_p.data(), 1);
            Rotate_Z_PNP(dim_p, Z1_p.data(), angle_p);
            cblas_zcopy(dim_n * dim_n, Z0_n.data(), 1, Z1_n.data(), 1);
            Rotate_Z_PNP(dim_n, Z1_n.data(), angle_n);

            ComplexNum HFBnormtemp = NormME_DifferentStates(angle_p, angle_n);
            HFBh00 += Weight_p * Weight_n * HFBnormtemp * HamiltonianME_DifferentStates(Z0_p.data(), Z1_p.data(), Z0_n.data(), Z1_n.data());
            HFBnorm += Weight_p * Weight_n * HFBnormtemp;

            // energy gradient
        }
    }

    std::cout << HFBh00 / HFBnorm << std::endl;
    return;
}

// this function will update Gamma and Delta functions!
void HartreeFockBogoliubov::ParticleNumberProjection()
{
    QuadratureClass PNP_mesh_p, PNP_mesh_n;
    AngMomProjection angMomProjectionInstance;
    // M = max (1/2 N, Omega - 1/2 N) + 1
    // See more in Nuclear Physics A332 (1979) 69-51
    int meshDim_p = std::max(N_p / 2, dim_p - N_p / 2) + 1;
    int meshDim_n = std::max(N_n / 2, dim_n - N_n / 2) + 1;
    PNP_mesh_p.SetNumber(meshDim_p);
    PNP_mesh_n.SetNumber(meshDim_n);
    angMomProjectionInstance.Generate_GQ_Mesh(PNP_mesh_p, "legendre");
    angMomProjectionInstance.Generate_GQ_Mesh(PNP_mesh_n, "legendre");

    std::vector<ComplexNum> Z0_p(dim_p * dim_p);
    std::vector<ComplexNum> Z1_p(dim_p * dim_p);
    std::vector<ComplexNum> Z0_n(dim_n * dim_n);
    std::vector<ComplexNum> Z1_n(dim_n * dim_n);

    UpdateDensityMatrix();
    ComplexNum HFBh00 = 0.;
    ComplexNum HFBnorm = 0.;
    // Get unrotated Z matrix
    Get_Z_matrix(dim_p, U_p, V_p, Z0_p.data());
    Get_Z_matrix(dim_n, U_n, V_n, Z0_n.data());
    for (size_t gauge_p = 0; gauge_p < PNP_mesh_p.GetTotalNumber(); gauge_p++)
    {
        for (size_t gauge_n = 0; gauge_n < PNP_mesh_n.GetTotalNumber(); gauge_n++)
        {
            double angle_p, angle_n;
            ComplexNum Weight_p, Weight_n;
            angle_p = PNP_mesh_p.GetX(gauge_p);
            angle_n = PNP_mesh_n.GetX(gauge_n);
            Weight_p = PNP_mesh_p.GetWeight(gauge_p) * std::exp(M_PI * angle_p * N_p * ComplexNum(0, -2));
            Weight_n = PNP_mesh_n.GetWeight(gauge_n) * std::exp(M_PI * angle_n * N_n * ComplexNum(0, -2));
            // std::cout << gauge_p << "  " << gauge_n << "  " << angle_p << "    " << angle_n << "   " << Weight_p << Weight_n << std::endl;

            // Z and Zg
            cblas_zcopy(dim_p * dim_p, Z0_p.data(), 1, Z1_p.data(), 1);
            Rotate_Z_PNP(dim_p, Z1_p.data(), angle_p);
            cblas_zcopy(dim_n * dim_n, Z0_n.data(), 1, Z1_n.data(), 1);
            Rotate_Z_PNP(dim_n, Z1_n.data(), angle_n);

            ComplexNum HFBnormtemp = NormME_DifferentStates(angle_p, angle_n);
            HFBh00 += Weight_p * Weight_n * HFBnormtemp * HamiltonianME_DifferentStates(Z0_p.data(), Z1_p.data(), Z0_n.data(), Z1_n.data());
            HFBnorm += Weight_p * Weight_n * HFBnormtemp;
        }
    }
    std::cout << "  Projected Energy: " << HFBh00 / HFBnorm << std::endl;
    return;
}

// update Occupation inital V and U
void HartreeFockBogoliubov::UpdateOccupationConstant(double NewV_p, double NewV_n)
{
    if (NewV_p > 1. or NewV_p <= 0.)
    {
        std::cout << "   The new unOccupy coefficient is not suitable! " << NewV_p << std::endl;
        exit(0);
    }
    OccupiedConstantV_p = NewV_p;
    UnOccupiedConstantU_p = sqrt(1. - NewV_p * NewV_p);
    if (NewV_n > 1. or NewV_n <= 0.)
    {
        std::cout << "   The new unOccupy coefficient is not suitable! " << NewV_n << std::endl;
        exit(0);
    }
    OccupiedConstantV_n = NewV_n;
    UnOccupiedConstantU_n = sqrt(1. - NewV_n * NewV_n);
    return;
}

///----------------------------------------------
/// use Canonical U V matrix
/// U diagonal, Uij = u_constant δij . The nonzero entries of the
/// V are all equal to ±v = ± sqrt(1 − u2), and are in positions
/// corresponding to pairing in the neutron-neutron channel and
/// the proton-proton channel.
void HartreeFockBogoliubov::SetCanonical_UV()
{
    // U matrix
    memset(U_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_p; i++)
    {
        U_p[i * dim_p + i] = UnOccupiedConstantU_p;
    }
    memset(U_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_n; i++)
    {
        U_n[i * dim_n + i] = UnOccupiedConstantU_n;
    }

    // V matrix
    memset(V_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_p - 1; i += 2)
    {
        V_p[i * dim_p + (i + 1)] = OccupiedConstantV_p;
        V_p[(i + 1) * dim_p + (i)] = -OccupiedConstantV_p;
    }
    memset(V_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_n - 1; i += 2)
    {
        V_n[i * dim_n + (i + 1)] = OccupiedConstantV_n;
        V_n[(i + 1) * dim_n + (i)] = -OccupiedConstantV_n;
    }
    return;
}

void HartreeFockBogoliubov::SetCanonical_UV_Random(int RandomSeed)
{
    srand(RandomSeed); // seed the random number generator with a seed
    double RandomValue;
    // Proton
    memset(U_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    memset(V_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_p - 1; i += 2)
    {
        RandomValue = (rand() % 1000) / 1000.;
        U_p[i * dim_p + i] = RandomValue;
        U_p[(i + 1) * dim_p + (i + 1)] = RandomValue;
        RandomValue = sqrt(1. - RandomValue * RandomValue);
        V_p[i * dim_p + (i + 1)] = RandomValue;
        V_p[(i + 1) * dim_p + (i)] = -RandomValue;
    }

    // Neutron
    memset(U_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    memset(V_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_n - 1; i += 2)
    {
        RandomValue = (rand() % 1000) / 1000.;
        U_n[i * dim_n + i] = RandomValue;
        U_n[(i + 1) * dim_n + (i + 1)] = RandomValue;
        RandomValue = sqrt(1. - RandomValue * RandomValue);
        V_n[i * dim_n + (i + 1)] = RandomValue;
        V_n[(i + 1) * dim_n + (i)] = -RandomValue;
    }
    return;
}

void HartreeFockBogoliubov::SetByNpNn_UV()
{
    // U matrix
    memset(U_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    for (size_t i = 0; i < N_p; i++)
    {
        U_p[i * dim_p + i] = UnOccupiedConstantU_p;
        // U_p[i * dim_p + i] = 1.;
    }
    memset(U_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    for (size_t i = 0; i < N_n; i++)
    {
        U_n[i * dim_n + i] = UnOccupiedConstantU_n;
        // U_n[i * dim_n + i] = 1.;
    }

    // V matrix
    memset(V_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    for (size_t i = 0; i < N_p; i += 2)
    {
        V_p[i * dim_p + (i + 1)] = OccupiedConstantV_n;
        V_p[(i + 1) * dim_p + (i)] = -OccupiedConstantV_n;
    }
    memset(V_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    for (size_t i = 0; i < N_n; i += 2)
    {
        V_n[i * dim_n + (i + 1)] = OccupiedConstantV_n;
        V_n[(i + 1) * dim_n + (i)] = -OccupiedConstantV_n;
    }

    return;
}

/// Eigen problem of a general matrix
/// n is the dimension
/// lda is the leading order of the matrix, also fixed as n
/// return the Eigenvector in row-major format, each vector is stored as a column
void HartreeFockBogoliubov::EigenProblem(int n, ComplexNum *A, double *EigenValue, ComplexNum *EigenVector)
{
    std::vector<ComplexNum> eigenvalues(n, ComplexNum(0., 0.));
    std::vector<ComplexNum> eigenvectors(n * n, ComplexNum(0., 0.));
    std::vector<double> eigenvalues_real(n, 0);

    // Call LAPACKE_zgeev to compute eigenvalues and eigenvectors
    lapack_int info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'V', n, A, n, eigenvalues.data(), NULL, n, eigenvectors.data(), n);

    /* Check for convergence */
    if (info == 0)
    {
        // Diagonalization successful
        // check imag part
        for (size_t i = 0; i < n; i++)
        {
            if (fabs(eigenvalues[i].imag()) > 1.e-8)
            {
                std::cout << "  Eigenvalue have a imagine part" << std::endl;
                exit(0);
            }
            eigenvalues_real[i] = eigenvalues[i].real();
        }

        // Sort the eigenvalues in ascending order based on their real values
        std::vector<std::pair<double, int>> eigenvalue_indices;
        for (int i = 0; i < n; ++i)
        {
            eigenvalue_indices.push_back(std::make_pair(eigenvalues_real[i], i));
        }
        std::sort(eigenvalue_indices.begin(), eigenvalue_indices.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        // Print sorted eigenvalues and corresponding eigenvectors
        // std::cout << "Eigenvalues:" << std::endl;
        for (const auto &eigenvalue_index : eigenvalue_indices)
        {
            int index = std::distance(eigenvalue_indices.begin(), std::find(eigenvalue_indices.begin(), eigenvalue_indices.end(), eigenvalue_index));
            // std::cout << index << "   " << eigenvalue_index.first << " \n";
            EigenValue[index] = eigenvalue_index.first;
        }
        // std::cout << std::endl;

        // std::cout << "Eigenvectors:" << std::endl;
        for (const auto &eigenvalue_index : eigenvalue_indices)
        {
            int index = std::distance(eigenvalue_indices.begin(), std::find(eigenvalue_indices.begin(), eigenvalue_indices.end(), eigenvalue_index));
            int idx = eigenvalue_index.second;
            for (int j = 0; j < n; ++j)
            {
                // std::cout << eigenvectors[idx * n + j] << " ";
                //  EigenVector[j * n + index] = eigenvectors[j * n + idx];
                EigenVector[j * n + index] = eigenvectors[j * n + idx];
            }
            // std::cout << std::endl;
        }
        gram_schmidt(EigenVector, n, n);
    }
    else
    {
        std::cout << "Diagonalization failed. Error code: " << info << std::endl;
    }
    return;
}
/// Eigen values only!!!
void HartreeFockBogoliubov::EigenProblem(int n, ComplexNum *A, double *EigenValue)
{
    std::vector<ComplexNum> eigenvalues(n, ComplexNum(0., 0.));
    std::vector<ComplexNum> eigenvectors(n * n, ComplexNum(0., 0.));
    std::vector<double> eigenvalues_real(n, 0);

    // Call LAPACKE_zgeev to compute eigenvalues
    lapack_int info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', n, A, n, eigenvalues.data(), NULL, n, NULL, n);

    /* Check for convergence */
    if (info == 0)
    {
        // Diagonalization successful
        // check imag part
        for (size_t i = 0; i < n; i++)
        {
            if (fabs(eigenvalues[i].imag()) > 1.e-8)
            {
                std::cout << "  Eigenvalue have a imagine part" << std::endl;
                exit(0);
            }
            eigenvalues_real[i] = eigenvalues[i].real();
        }

        // Sort the eigenvalues in ascending order based on their real values
        std::vector<std::pair<double, int>> eigenvalue_indices;
        for (int i = 0; i < n; ++i)
        {
            eigenvalue_indices.push_back(std::make_pair(eigenvalues_real[i], i));
        }
        std::sort(eigenvalue_indices.begin(), eigenvalue_indices.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        // Print sorted eigenvalues and corresponding eigenvectors
        // std::cout << "Eigenvalues:" << std::endl;
        for (const auto &eigenvalue_index : eigenvalue_indices)
        {
            int index = std::distance(eigenvalue_indices.begin(), std::find(eigenvalue_indices.begin(), eigenvalue_indices.end(), eigenvalue_index));
            // std::cout << index << "   " << eigenvalue_index.first << " \n";
            EigenValue[index] = eigenvalue_index.first;
        }
        // std::cout << std::endl;
    }
    else
    {
        std::cout << "Diagonalization failed. Error code: " << info << std::endl;
    }
    return;
}

//*********************************************************************
/// [ See Peter Ring eq.7.39  on page 254 ]
/// Diagonalize the HFB matrix
/// h = T + Gamma
/// if E_k > 0
/// (   h         Delta   )( U_k )  = ( U_k ) E_k
/// (  -Delta^*   -h^*    )( V_k )  = ( V_k )
/// for E_k < 0 ( HOLE STATES)
/// (   h         Delta   )( V^*_k )  = ( V^*_k ) E_k
/// (  -Delta^*   -h^*    )( U^*_k )  = ( U^*_k )
void HartreeFockBogoliubov::Diagonalize()
{
    // move energies
    cblas_dcopy(2 * dim_p + 2 * dim_n, energies, 1, prev_energies, 1);

    /// declare array
    std::vector<ComplexNum> HFBE_p(2 * dim_p * 2 * dim_p, 0);
    std::vector<ComplexNum> HFBE_n(2 * dim_n * 2 * dim_n, 0);

    /// construct HFB equation
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            /// [ h  ]
            /// [    ]
            HFBE_p[i * (2 * dim_p) + j] = T_term_p[i * dim_p + j] + Gamma_p[i * dim_p + j];

            /// [   d ]
            /// [     ]
            HFBE_p[i * (2 * dim_p) + j + dim_p] = Delta_p[i * dim_p + j];

            /// [      ]
            /// [ -d   ]
            HFBE_p[(i + dim_p) * (2 * dim_p) + j] = -std::conj(Delta_p[i * dim_p + j]);

            /// [      ]
            /// [   -h ]
            HFBE_p[(i + dim_p) * (2 * dim_p) + j + dim_p] = -T_term_p[i * dim_p + j] - std::conj(Gamma_p[i * dim_p + j]);
        }
    }

    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            /// [ h  ]
            /// [    ]
            HFBE_n[i * (2 * dim_n) + j] = T_term_n[i * dim_n + j] + Gamma_n[i * dim_n + j];

            /// [   d ]
            /// [     ]
            HFBE_n[i * (2 * dim_n) + j + dim_n] = Delta_n[i * dim_n + j];

            /// [      ]
            /// [ -d   ]
            HFBE_n[(i + dim_n) * (2 * dim_n) + j] = -std::conj(Delta_n[i * dim_n + j]);

            /// [      ]
            /// [   -h ]
            HFBE_n[(i + dim_n) * (2 * dim_n) + j + dim_n] = -T_term_n[i * dim_n + j] - std::conj(Gamma_n[i * dim_n + j]);
        }
    }

    /// Diag Proton HFB equation
    std::vector<ComplexNum> EigenVector_p(4 * dim_p * dim_p, ComplexNum(0, 0));
    EigenProblem(2 * dim_p, HFBE_p.data(), energies, EigenVector_p.data());

    /// Diag Neutron HFB equation
    std::vector<ComplexNum> EigenVector_n(4 * dim_n * dim_n, ComplexNum(0, 0));
    EigenProblem(2 * dim_n, HFBE_n.data(), energies + 2 * dim_p, EigenVector_n.data());

    // std::vector<ComplexNum> CU_p(dim_p * dim_p, ComplexNum(0, 0));
    // std::vector<ComplexNum> CV_p(dim_p * dim_p, ComplexNum(0, 0));
    // std::vector<ComplexNum> tempUV_p(dim_p * dim_p, ComplexNum(0, 0));

    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            // CU_p[i * dim_p + j] = EigenVector_p[i * (2 * dim_p) + j];
            // CV_p[i * dim_p + j] = EigenVector_p[(i + dim_p) * (2 * dim_p) + j];

            V_p[i * dim_p + j] = std::conj(EigenVector_p[i * (2 * dim_p) + j]);
            U_p[i * dim_p + j] = std::conj(EigenVector_p[(i + dim_p) * (2 * dim_p) + j]);
        }
    }

    if (N_p > 0)
    {
        /*
cblas_dcopy(dim_p * dim_p, U_p, 1, tempUV_p.data(), 1);
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., tempUV_p.data(), dim_p, CU_p.data(), dim_p, 0, U_p, dim_p);
cblas_dcopy(dim_p * dim_p, V_p, 1, tempUV_p.data(), 1);
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., tempUV_p.data(), dim_p, CV_p.data(), dim_p, 0, V_p, dim_p);
*/
    }

    // std::vector<double> CU_n(dim_n * dim_n, 0);
    // std::vector<double> CV_n(dim_n * dim_n, 0);
    // std::vector<double> tempUV_n(dim_n * dim_n, 0);

    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            // CU_n[i * dim_n + j] = EigenVector_n[i * (2 * dim_n) + j];
            // CV_n[i * dim_n + j] = EigenVector_n[(i + dim_n) * (2 * dim_n) + j];

            V_n[i * dim_n + j] = std::conj(EigenVector_n[i * (2 * dim_n) + j]);
            U_n[i * dim_n + j] = std::conj(EigenVector_n[(i + dim_n) * (2 * dim_n) + j]);
        }
    }
    if (N_n > 0)
    {
        /*
        cblas_dcopy(dim_n * dim_n, U_n, 1, tempUV_n.data(), 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., tempUV_n.data(), dim_n, CU_n.data(), dim_n, 0, U_n, dim_n);
        cblas_dcopy(dim_n * dim_n, V_n, 1, tempUV_n.data(), 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., tempUV_n.data(), dim_n, CV_n.data(), dim_n, 0, V_n, dim_n);
        */
    }
    return;
}

bool HartreeFockBogoliubov::CheckConvergence()
{
    double ediff = 0;
    std::vector<double> Ediffarray(2 * dim_p + 2 * dim_n, 0);
    vdSub(2 * dim_p + 2 * dim_n, energies, prev_energies, Ediffarray.data());
    for (size_t i = 0; i < 2 * dim_p + 2 * dim_n; i++)
    {
        ediff += Ediffarray[i] * Ediffarray[i];
    }
    // std::cout << ediff << std::endl;
    return (sqrt(ediff) < tolerance);
}

//*********************************************************************
// HFB energy of vaccum state
// H00 = Tr(T rho  + 0.5 Gamma rho - 0.5 Delta kappa^* )
// See more in Peter Ring's book eq. E.20
void HartreeFockBogoliubov::HFB_H00()
{
    ComplexNum tempValue_e1, tempValue_e2, tempValue_epairing;
    /// proton
    tempValue_e1 = ComplexNum(0, 0);
    tempValue_e2 = ComplexNum(0, 0);
    tempValue_epairing = ComplexNum(0, 0);
    // Use OpenMP to parallelize the two-fold for loop with collapse(2) clause
    // #pragma omp parallel for collapse(2) reduction(+ : tempValue_e1, tempValue_e2, tempValue_epairing)
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            tempValue_e1 += T_term_p[i * dim_p + j] * rho_p[j * dim_p + i];
            tempValue_e2 += Gamma_p[i * dim_p + j] * rho_p[j * dim_p + i];
            tempValue_epairing += Delta_p[i * dim_p + j] * std::conj(kappa_p[j * dim_p + i]);
        }
    }
    if (fabs(tempValue_e1.imag()) > 1.e-7)
    {
        std::cout << "   e1 have a imag part!" << std::endl;
        exit(0);
    }
    e1 = tempValue_e1.real();
    if (fabs(tempValue_e2.imag()) > 1.e-7)
    {
        std::cout << "   e2 have a imag part!" << std::endl;
        exit(0);
    }
    e2 = tempValue_e2.real();
    if (fabs(tempValue_epairing.imag()) > 1.e-7)
    {
        std::cout << "   e_pairing have a imag part!" << std::endl;
        exit(0);
    }
    epairing = tempValue_epairing.real();

    ///-------------------------------------------------------
    /// neutron
    tempValue_e1 = ComplexNum(0, 0);
    tempValue_e2 = ComplexNum(0, 0);
    tempValue_epairing = ComplexNum(0, 0);
    // Use OpenMP to parallelize the two-fold for loop with collapse(2) clause
    // #pragma omp parallel for collapse(2) reduction(+ : tempValue_e1, tempValue_e2, tempValue_epairing)
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            tempValue_e1 += T_term_n[i * dim_n + j] * rho_n[j * dim_n + i];
            tempValue_e2 += Gamma_n[i * dim_n + j] * rho_n[j * dim_n + i];
            tempValue_epairing += Delta_n[i * dim_n + j] * std::conj(kappa_n[j * dim_n + i]);
        }
    }
    if (fabs(tempValue_e1.imag()) > 1.e-7)
    {
        std::cout << "   e1 have a imag part!" << std::endl;
        exit(0);
    }
    e1 += tempValue_e1.real();

    if (fabs(tempValue_e2.imag()) > 1.e-7)
    {
        std::cout << "   e2 have a imag part!" << std::endl;
        exit(0);
    }
    e2 += tempValue_e2.real();

    if (fabs(tempValue_epairing.imag()) > 1.e-7)
    {
        std::cout << "   e_pairing have a imag part!" << std::endl;
        exit(0);
    }
    epairing += tempValue_epairing.real();

    this->E_hfb = e1 + 0.5 * e2 - 0.5 * epairing;
}

void HartreeFockBogoliubov::PrintEHFB()
{
    std::cout << std::fixed << std::setprecision(7);
    std::cout << "  One body term = " << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(12) << e1 << std::endl;
    std::cout << "  Two body term = " << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(12) << 0.5 * e2 << std::endl;
    std::cout << "  Pairing  term = " << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(12) << 0.5 * epairing << std::endl;
    std::cout << "  E_HFB         = " << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(12) << E_hfb << std::endl;
}

// HFB One-quasiparticle state
// Return H11
// H11 = U^+ h U - V^+ h^T V  + U^+ Δ V  -  V^+ Δ^* U
// This procedure will add Gamma and T matrix, and store it in the Gamma
void HartreeFockBogoliubov::HFB_H11(ComplexNum *H11_p, ComplexNum *H11_n)
{
    // proton H11
    memset(H11_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t l = 0; l < dim_p; l++)
        {
            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t j = 0; j < dim_p; j++)
                {
                    // U^+ h V^*  = \sum_{il} U^+_ij h_jk U_kl
                    //            = \sum_{il} U^*_ji h_jk U_kl
                    H11_p[i * dim_p + l] += std::conj(U_p[j * dim_p + i]) * (Gamma_p[j * dim_p + k] + T_term_p[j * dim_p + k]) * U_p[k * dim_p + l];

                    //-V^+ h^T U^* = \sum_{il} V^+_ij h^T_jk V_kl
                    //             = \sum_{il} V^*_ji h_kj   V_kl
                    H11_p[i * dim_p + l] -= std::conj(V_p[j * dim_p + i]) * (Gamma_p[k * dim_p + j] + T_term_p[k * dim_p + j]) * V_p[k * dim_p + l];

                    // U^+ Δ U^*  = \sum_{il} U^+_ij Δ_jk V_kl
                    //            = \sum_{il} U^*_ji Δ_jk V_kl
                    H11_p[i * dim_p + l] += std::conj(U_p[j * dim_p + i]) * Delta_p[j * dim_p + k] * V_p[k * dim_p + l];

                    //-V^+ Δ^* V^* = \sum_{il} V^+_ij Δ^*_jk U_kl
                    //             = \sum_{il} V^*_ji Δ^*_jk U_kl
                    H11_p[i * dim_p + l] -= std::conj(V_p[j * dim_p + i]) * std::conj(Delta_p[j * dim_p + k]) * U_p[k * dim_p + l];
                }
            }
        }
    }

    // neutron H20
    memset(H11_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t l = 0; l < dim_n; l++)
        {
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t j = 0; j < dim_n; j++)
                {
                    // U^+ h V^*  = \sum_{il} U^+_ij h_jk U_kl
                    //            = \sum_{il} U^*_ji h_jk U_kl
                    H11_n[i * dim_n + l] += std::conj(U_n[j * dim_n + i]) * (Gamma_n[j * dim_n + k] + T_term_n[j * dim_n + k]) * U_n[k * dim_n + l];

                    //-V^+ h^T U^* = \sum_{il} V^+_ij h^T_jk V_kl
                    //             = \sum_{il} V^*_ji h_kj   V_kl
                    H11_n[i * dim_n + l] -= std::conj(V_n[j * dim_n + i]) * (Gamma_n[k * dim_n + j] + T_term_n[k * dim_n + j]) * V_n[k * dim_n + l];

                    // U^+ Δ U^*  = \sum_{il} U^+_ij Δ_jk V_kl
                    //            = \sum_{il} U^*_ji Δ_jk V_kl
                    H11_n[i * dim_n + l] += std::conj(U_n[j * dim_n + i]) * Delta_n[j * dim_n + k] * V_n[k * dim_n + l];

                    //-V^+ Δ^* V^* = \sum_{il} V^+_ij Δ^*_jk U_kl
                    //             = \sum_{il} V^*_ji Δ^*_jk U_kl
                    H11_n[i * dim_n + l] -= std::conj(V_n[j * dim_n + i]) * std::conj(Delta_n[j * dim_n + k]) * U_n[k * dim_n + l];
                }
            }
        }
    }
    return;
}

// HFB two-quasiparticle state
// Return H20
// H2O = U^+ h V^* - V^+ h^T U^*  + U^+ Δ U^*  -  V^+ Δ^* V^*
void HartreeFockBogoliubov::HFB_H20(ComplexNum *H20_p, ComplexNum *H20_n)
{
    // proton H20
    memset(H20_p, 0, (dim_p) * (dim_p) * sizeof(ComplexNum));
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t l = 0; l < i; l++)
        {
            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t j = 0; j < dim_p; j++)
                {
                    // U^+ h V^*  = \sum_{il} U^+_ij h_jk V^*_kl
                    //            = \sum_{il} U^*_ji h_jk V^*_kl
                    H20_p[i * dim_p + l] += std::conj(U_p[j * dim_p + i]) * (Gamma_p[j * dim_p + k] + T_term_p[j * dim_p + k]) * std::conj(V_p[k * dim_p + l]);

                    //-V^+ h^T U^* = \sum_{il} V^+_ij h^T_jk U^*_kl
                    //             = \sum_{il} V^*_ji h_kj U^*_kl
                    H20_p[i * dim_p + l] -= std::conj(V_p[j * dim_p + i]) * (Gamma_p[k * dim_p + j] + T_term_p[k * dim_p + j]) * std::conj(U_p[k * dim_p + l]);

                    // U^+ Δ U^*  = \sum_{il} U^+_ij Δ_jk U^*_kl
                    //            = \sum_{il} U^*_ji Δ_jk U^*_kl
                    H20_p[i * dim_p + l] += std::conj(U_p[j * dim_p + i]) * Delta_p[j * dim_p + k] * std::conj(U_p[k * dim_p + l]);

                    //-V^+ Δ^* V^* = \sum_{il} V^+_ij Δ^*_jk V^*_kl
                    //             = \sum_{il} V^*_ji Δ^*_jk V^*_kl
                    H20_p[i * dim_p + l] -= std::conj(V_p[j * dim_p + i]) * std::conj(Delta_p[j * dim_p + k]) * std::conj(V_p[k * dim_p + l]);
                }
            }
            H20_p[l * dim_p + i] = -H20_p[i * dim_p + l];
        }
    }

    if (!isSkewMatrix(H20_p, dim_p, dim_p))
        Check_matrix(dim_p, H20_p);

    // neutron H20
    memset(H20_n, 0, (dim_n) * (dim_n) * sizeof(ComplexNum));
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t l = 0; l < i; l++)
        {
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t j = 0; j < dim_n; j++)
                {
                    // U^+ h V^*  = \sum_{il} U^+_ij h_jk V^*_kl
                    //            = \sum_{il} U^*_ji h_jk V^*_kl
                    H20_n[i * dim_n + l] += std::conj(U_n[j * dim_n + i]) * (Gamma_n[j * dim_n + k] + T_term_n[j * dim_n + k]) * std::conj(V_n[k * dim_n + l]);

                    //-V^+ h^T U^* = \sum_{il} V^+_ij h^T_jk U^*_kl
                    //             = \sum_{il} V^*_ji h_kj U^*_kl
                    H20_n[i * dim_n + l] -= std::conj(V_n[j * dim_n + i]) * (Gamma_n[k * dim_n + j] + T_term_n[k * dim_n + j]) * std::conj(U_n[k * dim_n + l]);

                    // U^+ Δ U^*  = \sum_{il} U^+_ij Δ_jk U^*_kl
                    //            = \sum_{il} U^*_ji Δ_jk U^*_kl
                    H20_n[i * dim_n + l] += std::conj(U_n[j * dim_n + i]) * Delta_n[j * dim_n + k] * std::conj(U_n[k * dim_n + l]);

                    //-V^+ Δ^* V^* = \sum_{il} V^+_ij Δ^*_jk V^*_kl
                    //             = \sum_{il} V^*_ji Δ^*_jk V^*_kl
                    H20_n[i * dim_n + l] -= std::conj(V_n[j * dim_n + i]) * std::conj(Delta_n[j * dim_n + k]) * std::conj(V_n[k * dim_n + l]);
                }
            }
            H20_n[l * dim_n + i] = -H20_n[i * dim_n + l];
        }
    }

    if (!isSkewMatrix(H20_n, dim_n, dim_n))
        Check_matrix(dim_n, H20_n);
    return;
}

// F00 part of the Hermitian one-body operator
// F = \sum_ij f_ij C^+_i C_j
// F00 = Tr( f * rho )
// See more in Peter Ring's book ( E12, page 613 )
void HartreeFockBogoliubov::HFB_F00(double *F_p, double *F_n, double &F00_p, double &F00_n)
{
    // Proton Part
    ComplexNum Temp_F00 = 0;
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            Temp_F00 += F_p[i * dim_p + j] * rho_p[j * dim_p + i];
            // std::cout<<  F_p[i * dim_p + j] <<  "    " << rho_p[j * dim_p + i] <<std::endl;
        }
    }
    if (fabs(Temp_F00.imag()) > 1.e-7)
    {
        std::cout << "  Proton F00 have a imaginary part! " << std::endl;
    }
    F00_p = Temp_F00.real();

    // std::cout<< "N_p  "<< Temp_F00<<std::endl;

    // Neutron Part
    Temp_F00 = 0;
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            Temp_F00 += F_n[i * dim_n + j] * rho_n[j * dim_n + i];
        }
    }
    if (fabs(Temp_F00.imag()) > 1.e-7)
    {
        std::cout << "  Neutron F00 have a imaginary part! " << std::endl;
    }
    F00_n = Temp_F00.real();
}

// F20 part of the Hermitian scalar one-body operator
// F = \sum_ij f_ij C^+_i C_j
// F20 = U^+ f V^* - V^+ f^T U^*
// See more in Peter Ring's book ( E14, page 613 )
void HartreeFockBogoliubov::HFB_F20(ComplexNum *F_p, ComplexNum *F_n)
{
    std::vector<ComplexNum> F20_p(dim_p * dim_p, 0);
    std::vector<ComplexNum> F20_n(dim_n * dim_n, 0);
    // Proton Part
    // #pragma omp parallel for
    /// U^+ f V^* - V^+ f^T U^*
    /// U^*_ki f_kl V^*_lj - V^*_ki f_lk U^*_lj
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t l = 0; l < dim_p; l++)
                {
                    F20_p[i * dim_p + j] += std::conj(U_p[k * dim_p + i]) * F_p[k * dim_p + l] * std::conj(V_p[l * dim_p + j]) - std::conj(V_p[k * dim_p + i]) * F_p[l * dim_p + k] * std::conj(U_p[l * dim_p + j]);
                }
            }
        }
    }

    // Neutron Part
#pragma omp parallel for
    /// U^+ f V^* - V^+ f^T U^*
    /// U^*_ki f_kl V^*_lj - V^*_ki f_lk U^*_lj
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t l = 0; l < dim_n; l++)
                {
                    F20_n[i * dim_n + j] += std::conj(U_n[k * dim_n + i]) * F_n[k * dim_n + l] * std::conj(V_n[l * dim_n + j]) - std::conj(V_n[k * dim_n + i]) * F_n[l * dim_n + k] * std::conj(U_n[l * dim_n + j]);
                }
            }
        }
    }
    cblas_zcopy(dim_p * dim_p, F20_p.data(), 1, F_p, 1);
    cblas_zcopy(dim_n * dim_n, F20_n.data(), 1, F_n, 1);
}

//*********************************************************************
// The update from U to U' can be expressed as a Thouless transformation of U
// U' = e^Z U
// U' = (U + V^∗ Z^∗)
// V' = (V + U^∗ Z^∗)
// the matrices of U and V are stored in the format
// ZUV = [ U ]
//       [ V ]
// this vector is orthogonalized by a Gram Schmidt method
void HartreeFockBogoliubov::UpdateUV_Thouless(ComplexNum *Z_p, ComplexNum *Z_n)
{
    std::vector<ComplexNum> ZUV_p(2 * dim_p * dim_p, 0);
    // std::vector<ComplexNum> ZV_p(dim_p * dim_p, 0);
    std::vector<ComplexNum> ZUV_n(2 * dim_n * dim_n, 0);
    // std::vector<ComplexNum> ZV_n(dim_n * dim_n, 0);

    // Proton UV
    cblas_zcopy(dim_p * dim_p, U_p, 1, ZUV_p.data(), 1);
    cblas_zcopy(dim_p * dim_p, V_p, 1, ZUV_p.data() + dim_p * dim_p, 1);

    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            for (size_t k = 0; k < dim_p; k++)
            {
                // U part
                ZUV_p[i * dim_p + j] += std::conj(V_p[i * dim_p + k] * Z_p[k * dim_p + j]);
                // V part
                ZUV_p[i * dim_p + j + dim_p * dim_p] += std::conj(U_p[i * dim_p + k] * Z_p[k * dim_p + j]);
            }
        }
    }
    gram_schmidt(ZUV_p.data(), dim_p, dim_p * 2);
    isOrthogonalMatrix(ZUV_p.data(), dim_p, dim_p * 2);
    cblas_zcopy(dim_p * dim_p, ZUV_p.data(), 1, U_p, 1);
    cblas_zcopy(dim_p * dim_p, ZUV_p.data() + dim_p * dim_p, 1, V_p, 1);

    // Neutron UV
    cblas_zcopy(dim_n * dim_n, U_n, 1, ZUV_n.data(), 1);
    cblas_zcopy(dim_n * dim_n, V_n, 1, ZUV_n.data() + dim_n * dim_n, 1);
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            for (size_t k = 0; k < dim_n; k++)
            {
                // U part
                ZUV_n[i * dim_n + j] += std::conj(V_n[i * dim_n + k] * Z_n[k * dim_n + j]);
                // V part
                ZUV_n[i * dim_n + j + dim_n * dim_n] += std::conj(U_n[i * dim_n + k] * Z_n[k * dim_n + j]);
            }
        }
    }
    gram_schmidt(ZUV_n.data(), dim_n, dim_n * 2);
    isOrthogonalMatrix(ZUV_n.data(), dim_n, dim_n * 2);
    cblas_zcopy(dim_n * dim_n, ZUV_n.data(), 1, U_n, 1);
    cblas_zcopy(dim_n * dim_n, ZUV_n.data() + dim_n * dim_n, 1, V_n, 1);
    return;
}

//*********************************************************************
// Z_ij = - η (∂E)/(∂Z_ij)
// Given the gradient, the simplest algorithm to update U is the steepest descent method
// where η is a fixed small number
void HartreeFockBogoliubov::Cal_Gradient_SteepestDescent(ComplexNum *Z_p, ComplexNum *Z_n)
{
    cblas_zdscal(dim_p * dim_p, -gradient_eta, Z_p, 1);
    cblas_zdscal(dim_n * dim_n, -gradient_eta, Z_n, 1);
    return;
}

// to improve the efficiency of the iteration
// process is to divide the elements of Hc
// 20 by preconditioning
// Z_ij = -  η /p_ij *  (∂E)/(∂Z_ij)
// pij = max(Ei + Ej , Emin), where Ei is the eigenEnergies of
// one quasi-particle operator H11
// where Emin is a numerical parameter of the order of 1–2 MeV, we fixed it as 1.5 MeV
// See more in  PHYSICAL REVIEW C 84, 014312 (2011)
void HartreeFockBogoliubov::Cal_Gradient_Preconditioning(ComplexNum *Z_p, ComplexNum *Z_n)
{
    double Emin = 2.;                                // MeV
    std::vector<ComplexNum> H11_p(dim_p * dim_p, 0); // H11
    std::vector<ComplexNum> H11_n(dim_n * dim_n, 0);
    std::vector<double> EigenValue_p(dim_p, 0);
    std::vector<double> EigenValue_n(dim_n, 0);
    HFB_H11(H11_p.data(), H11_n.data());
    EigenProblem(dim_p, H11_p.data(), EigenValue_p.data());
    EigenProblem(dim_n, H11_n.data(), EigenValue_n.data());
    double factor;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            // factor = std::max(fabs(EigenValue_p[i]) + fabs(EigenValue_p[j]), Emin);
            factor = std::max((EigenValue_p[i]) + (EigenValue_p[j]), Emin);
            Z_p[i * dim_p + j] *= -gradient_eta / factor;
            Z_p[j * dim_p + i] *= -gradient_eta / factor;
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            // factor = std::max(fabs(EigenValue_n[i]) + fabs(EigenValue_n[j]), Emin);
            factor = std::max((EigenValue_n[i]) + (EigenValue_n[j]), Emin);
            Z_n[i * dim_n + j] *= -gradient_eta / factor;
            Z_n[j * dim_n + i] *= -gradient_eta / factor;
        }
    }
    return;
}

// the HFB vacuum can be rewritten by the thouless theorem
// |HFB> = e^Zc+c+ |0>, where c+ is the creation operator on
// HO basis.
// Z = (V U^-1)*
void HartreeFockBogoliubov::Get_Z_matrix(int dim, ComplexNum *U, ComplexNum *V, ComplexNum *Z)
{
    memset(Z, 0, (dim) * (dim) * sizeof(ComplexNum));
    std::vector<ComplexNum> U1(dim * dim);
    InverseMatrix(dim_p, U, U1.data());
    /// Z = V U^-1
    ComplexNum alpha(1, 0), beta(0.0, 0.0);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha, V, dim, U1.data(), dim, &beta, Z, dim);
    // Apply complex conjugate in-place
    mkl_zimatcopy('C', 'N', dim, dim, alpha, Z, dim, dim);
    return;
}

void HartreeFockBogoliubov::InverseMatrix(int dim, ComplexNum *U, ComplexNum *U1)
{
    // Define the parameters for the matrix
    MKL_INT n = dim;   // Size of the matrix
    MKL_INT lda = dim; // Leading dimension of the matrix
    MKL_INT ipiv[dim]; // Pivot indices
    MKL_INT lwork = dim * dim;
    cblas_zcopy(dim * dim, U, 1, U1, 1);
    // Calculate the optimal workspace size
    MKL_INT info;
    zgetrf(&n, &n, U1, &lda, ipiv, &info); // LU factorization
    if (info != 0)
    {
        std::cerr << "zgetrf failed with code " << info << std::endl;
        return;
    }

    ComplexNum *work = new ComplexNum[lwork];
    // Calculate the inverse
    zgetri(&n, U1, &lda, ipiv, work, &lwork, &info);
    if (info != 0)
    {
        std::cerr << "zgetri failed with code " << info << std::endl;
        return;
    }

    // Output the inverse matrix
    /*
    std::cout << "Inverse matrix:" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << U1[i * n + j].real() << "  " << U1[i * n + j].imag() << "i\t";
        }
        std::cout << std::endl;
    }
    */
    delete[] work; // Deallocate the workspace memory
    return;
}

void HartreeFockBogoliubov::InverseMatrix(int dim, ComplexNum *U)
{
    // Define the parameters for the matrix
    MKL_INT n = dim;   // Size of the matrix
    MKL_INT lda = dim; // Leading dimension of the matrix
    MKL_INT ipiv[dim]; // Pivot indices
    MKL_INT lwork = dim * dim;
    // Calculate the optimal workspace size
    MKL_INT info;
    zgetrf(&n, &n, U, &lda, ipiv, &info); // LU factorization
    if (info != 0)
    {
        std::cerr << "zgetrf failed with code " << info << std::endl;
        return;
    }

    ComplexNum *work = new ComplexNum[lwork];
    // Calculate the inverse
    zgetri(&n, U, &lda, ipiv, work, &lwork, &info);
    if (info != 0)
    {
        std::cerr << "zgetri failed with code " << info << std::endl;
        return;
    }
    delete[] work; // Deallocate the workspace memory
    return;
}

// Uϕ = D(ϕ) U     V-ϕ = D(ϕ) V
// D(ϕ) = exp( i 2π ϕ ) where ϕ in range from 0 to 1
void HartreeFockBogoliubov::Rotate_UV_PNP(int dim, ComplexNum *U, double angle)
{
    ComplexNum alpha = std::exp(ComplexNum(0, 2) * M_PI * angle);
    cblas_zscal(dim * dim, &alpha, U, 1);
    return;
}

// Z = (V U^-1)*
// Zϕ = exp( i 4π ϕ ) Z, where ϕ in range from 0 to 1
void HartreeFockBogoliubov::Rotate_Z_PNP(int dim, ComplexNum *U, double angle)
{
    ComplexNum alpha = std::exp(ComplexNum(0, 4) * M_PI * angle);
    cblas_zscal(dim * dim, &alpha, U, 1);
    return;
}

ComplexNum HartreeFockBogoliubov::Determinant(int dim, ComplexNum *Matrix)
{
    MKL_INT n = dim; // Size of the matrix
    MKL_INT ipiv[n];
    MKL_INT info;

    // Perform LU factorization using LAPACKE_zgetrf
    info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, Matrix, n, ipiv);

    if (info != 0)
    {
        std::cerr << "LAPACKE_zgetrf in Determinant function failed with code " << info << std::endl;
        exit(0);
    }

    // Calculate the determinant using the LU factors
    std::complex<double> determinant = 1.0;
    for (MKL_INT i = 0; i < n; ++i)
    {
        determinant *= Matrix[i * n + i];
    }

    // Output the determinant
    // std::cout << "Determinant of the complex matrix: " << determinant.real() << "+" << determinant.imag() << "i" << std::endl;
    return determinant;
}

// this function will update Gamma and Delta functions!
// ρ_ij = <φ|c†_j c_i|φ'> / <φ|φ'> = −Z' (1 − Z^∗ Z' )−1 Z^∗
// k_ij = <φ|c_j c_i|φ'> / <φ|φ'> = Z' (1 − Z^∗ Z' )−1
// k'_ij = <φ|c†_j c†_i|φ'> / <φ|φ'> = (1 − Z^∗ Z' )−1 Z^∗
ComplexNum HartreeFockBogoliubov::HamiltonianME_DifferentStates(ComplexNum *Z_p, ComplexNum *Z1_p, ComplexNum *Z_n, ComplexNum *Z1_n)
{
    std::vector<ComplexNum> rho_diff_p(dim_p * dim_p);
    std::vector<ComplexNum> kappa_diff_p(dim_p * dim_p);
    std::vector<ComplexNum> kappa1_diff_p(dim_p * dim_p);
    std::vector<ComplexNum> rho_diff_n(dim_n * dim_n);
    std::vector<ComplexNum> kappa_diff_n(dim_n * dim_n);
    std::vector<ComplexNum> kappa1_diff_n(dim_n * dim_n);
    std::vector<ComplexNum> ZZ_p(dim_p * dim_p);
    std::vector<ComplexNum> ZZ_n(dim_n * dim_n);
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            ZZ_p[i * dim_p + j] = 0.;
            if (i == j)
            {
                ZZ_p[i * dim_p + i] = 1.;
            }
            for (size_t k = 0; k < dim_p; k++)
            {
                ZZ_p[i * dim_p + j] -= std::conj(Z_p[i * dim_p + k]) * Z1_p[k * dim_p + j];
            }
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            ZZ_n[i * dim_n + j] = 0.;
            if (i == j)
            {
                ZZ_n[i * dim_n + i] = 1.;
            }
            for (size_t k = 0; k < dim_n; k++)
            {
                ZZ_n[i * dim_n + j] -= std::conj(Z_n[i * dim_n + k]) * Z1_n[k * dim_n + j];
            }
        }
    }
    InverseMatrix(dim_p, ZZ_p.data());
    InverseMatrix(dim_n, ZZ_n.data());

    // rho
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            rho_diff_p[i * dim_p + j] = 0.;
            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t l = 0; l < dim_p; l++)
                {
                    rho_diff_p[i * dim_p + j] -= Z1_p[i * dim_p + k] * ZZ_p[k * dim_p + l] * std::conj(Z_p[l * dim_p + j]);
                }
            }
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            rho_diff_n[i * dim_n + j] = 0.;
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t l = 0; l < dim_n; l++)
                {
                    rho_diff_n[i * dim_n + j] -= Z1_n[i * dim_n + k] * ZZ_n[k * dim_n + l] * std::conj(Z_n[l * dim_n + j]);
                }
            }
        }
    }
    // kappa
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            kappa_diff_p[i * dim_p + j] = 0.;
            for (size_t k = 0; k < dim_p; k++)
            {
                kappa_diff_p[i * dim_p + j] += Z1_p[i * dim_p + k] * ZZ_p[k * dim_p + j];
            }
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            kappa_diff_n[i * dim_n + j] = 0.;
            for (size_t k = 0; k < dim_p; k++)
            {
                kappa_diff_n[i * dim_n + j] += Z1_n[i * dim_n + k] * ZZ_n[k * dim_n + j];
            }
        }
    }
    // kappa1
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            kappa1_diff_p[i * dim_p + j] = 0.;
            for (size_t k = 0; k < dim_p; k++)
            {
                kappa1_diff_p[i * dim_p + j] += ZZ_p[i * dim_p + k] * std::conj(Z_p[k * dim_p + j]);
            }
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            kappa1_diff_n[i * dim_n + j] = 0.;
            for (size_t k = 0; k < dim_n; k++)
            {
                kappa1_diff_n[i * dim_n + j] += ZZ_n[i * dim_n + k] * std::conj(Z_n[k * dim_n + j]);
            }
        }
    }

    ///  [See Peter Ring eq 7.41] page 254
    ///   Gamma_{ij} = \sum_{a b}  \rho_{ab} \bar{V}^{(2)}_{ibja}
    ///   Delta_{ij} = 1/2 * \sum_{a b} \kappa_{ab} \bar{V}^{(2)}_{ijab}
    memset(Gamma_p, 0, dim_p * dim_p * sizeof(ComplexNum));
    memset(Gamma_n, 0, dim_n * dim_n * sizeof(ComplexNum));
    memset(Delta_p, 0, dim_p * dim_p * sizeof(ComplexNum));
    memset(Delta_n, 0, dim_n * dim_n * sizeof(ComplexNum));
    // use a skewed stored Vpp and Vnn
    double *Vpp, *Vpn, *Vnn;
    Vpp = Ham->MSMEs.GetVppPrt();
    Vnn = Ham->MSMEs.GetVnnPrt();
    Vpn = Ham->MSMEs.GetVpnPrt();
    std::vector<ComplexNum> Temp_rho_p(dim_p * dim_p, 0);
    std::vector<ComplexNum> Temp_rho_n(dim_n * dim_n, 0);
    mkl_zomatcopy('R', 'T', dim_p, dim_p, 1.0, rho_diff_p.data(), dim_p, Temp_rho_p.data(), dim_p);
    mkl_zomatcopy('R', 'T', dim_n, dim_n, 1.0, rho_diff_n.data(), dim_n, Temp_rho_n.data(), dim_n);

    // Proton subspace
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t l = 0; l < dim_p; l++)
                {
                    // add Vpp term
                    Gamma_p[i * dim_p + j] += Vpp[i * dim_p * dim_p * dim_p + j * dim_p * dim_p + k * dim_p + l] * Temp_rho_p[k * dim_p + l];

                    // pairing
                    Delta_p[i * dim_p + j] += 0.5 * Vpp[i * dim_p * dim_p * dim_p + k * dim_p * dim_p + j * dim_p + l] * kappa_diff_p[k * dim_p + l];
                }
            }
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t l = 0; l < dim_n; l++)
                {
                    // add Vpn term
                    Gamma_p[i * dim_p + j] += Vpn[dim_p * dim_n * dim_n * i + dim_n * dim_n * j + k * dim_n + l] * Temp_rho_n[k * dim_n + l];
                }
            }
        }
    }

    // Neutron subspace
    // #pragma omp parallel for
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            for (size_t k = 0; k < dim_n; k++)
            {
                for (size_t l = 0; l < dim_n; l++)
                {
                    // add Vnn term
                    Gamma_n[i * dim_n + j] += Vnn[dim_n * dim_n * dim_n * i + dim_n * dim_n * j + k * dim_n + l] * Temp_rho_n[k * dim_n + l];

                    Delta_n[i * dim_n + j] += 0.5 * Vnn[i * dim_n * dim_n * dim_n + k * dim_n * dim_n + j * dim_n + l] * kappa_diff_n[k * dim_n + l];
                }
            }

            for (size_t k = 0; k < dim_p; k++)
            {
                for (size_t l = 0; l < dim_p; l++)
                {
                    // add Vpn term
                    Gamma_n[i * dim_n + j] += Vpn[dim_p * dim_n * dim_n * k + dim_n * dim_n * l + i * dim_n + j] * Temp_rho_p[k * dim_n + l];
                }
            }
        }
    }

    //**************************************************
    // Tr(tρ + 1/2 Gamma ρ − 1 /2 κ' Delta)
    ComplexNum tempValue_e1, tempValue_e2, tempValue_epairing, HFBh00;
    /// proton
    tempValue_e1 = ComplexNum(0, 0);
    tempValue_e2 = ComplexNum(0, 0);
    tempValue_epairing = ComplexNum(0, 0);
    // Use OpenMP to parallelize the two-fold for loop with collapse(2) clause
    // #pragma omp parallel for collapse(2) reduction(+ : tempValue_e1, tempValue_e2, tempValue_epairing)
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            tempValue_e1 += T_term_p[i * dim_p + j] * rho_diff_p[j * dim_p + i];
            tempValue_e2 += Gamma_p[i * dim_p + j] * rho_diff_p[j * dim_p + i];
            tempValue_epairing += kappa1_diff_p[i * dim_p + j] * Delta_p[i * dim_p + j];
        }
    }

    ///-------------------------------------------------------
    /// neutron
    // Use OpenMP to parallelize the two-fold for loop with collapse(2) clause
    // #pragma omp parallel for collapse(2) reduction(+ : tempValue_e1, tempValue_e2, tempValue_epairing)
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            tempValue_e1 += T_term_n[i * dim_n + j] * rho_diff_n[j * dim_n + i];
            tempValue_e2 += Gamma_n[i * dim_n + j] * rho_diff_n[j * dim_n + i];
            tempValue_epairing += kappa1_diff_n[i * dim_n + j] * Delta_n[i * dim_n + j];
        }
    }
    HFBh00 = tempValue_e1 + 0.5 * tempValue_e2 - 0.5 * tempValue_epairing;

    // std::cout << tempValue_e1 <<  0.5 * tempValue_e2 <<  - 0.5 * tempValue_epairing << HFBh00 << std::endl;
    return HFBh00;
}

// overlap intergral between two different configruation
// <φ | φ'(ϕ)> = ( exp^( -2 i ϕ ) +  rho ( 1 - exp^(-2iϕ) ) )^-1
ComplexNum HartreeFockBogoliubov::NormME_DifferentStates(double angle_p, double angle_n)
{
    //**************************************************
    // <φ | φ'(ϕ)> = ( exp^( -2 i ϕ ) +  rho ( 1 - exp^(-2iϕ) ) )^-1
    std::vector<ComplexNum> Cphi_p(dim_p * dim_p);
    std::vector<ComplexNum> Cphi_n(dim_n * dim_n);
    ComplexNum factor_p, factor_n;
    // exp^( -2 i ϕ  )
    factor_p = 1. - std::exp(ComplexNum(0, -4.) * M_PI * angle_p);
    factor_n = 1. - std::exp(ComplexNum(0, -4.) * M_PI * angle_n);
    cblas_zcopy(dim_p * dim_p, rho_p, 1, Cphi_p.data(), 1);
    cblas_zcopy(dim_n * dim_n, rho_n, 1, Cphi_n.data(), 1);
    cblas_zscal(dim_p * dim_p, &factor_p, Cphi_p.data(), 1);
    cblas_zscal(dim_n * dim_n, &factor_n, Cphi_n.data(), 1);
    factor_p = std::exp(ComplexNum(0, -4.) * M_PI * angle_p);
    factor_n = std::exp(ComplexNum(0, -4.) * M_PI * angle_n);
    for (size_t i = 0; i < dim_p; i++)
    {
        Cphi_p[i * dim_p + i] += factor_p;
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        Cphi_n[i * dim_n + i] += factor_n;
    }
    InverseMatrix(dim_p, Cphi_p.data());
    InverseMatrix(dim_n, Cphi_n.data());
    // exp^( i M phi  )
    factor_p = std::exp(ComplexNum(0, 2.) * M_PI * angle_p * (double)dim_p);
    factor_n = std::exp(ComplexNum(0, 2.) * M_PI * angle_n * (double)dim_n);

    ComplexNum HFBNorm;
    // std::cout << HFBh00 << std::endl;
    HFBNorm = factor_p / sqrt(Determinant(dim_p, Cphi_p.data())) * factor_n / sqrt(Determinant(dim_n, Cphi_n.data()));
    return HFBNorm;
}

// gradient of matrix element between two different states ∂<φ| / ∂Z^∗ ( H − Eb) |φ>
// ∂<φ| / ∂Z^∗ ( H − Eb) |φ> = <φ|φ'> (U†_D ( e + Gamma ) V^*_D − V†_D ( t + Gamma )^T U^∗_D
// + U†_D U^∗_D − V^†_D Delta' V^*_D ) − Z_D < φ'| H − Eb |φ >
void HartreeFockBogoliubov::gradient_DifferentStates(ComplexNum *Z_p, ComplexNum *Z1_p, ComplexNum *Z_n, ComplexNum *Z1_n, ComplexNum *gradientZ_p, ComplexNum *gradientZ_n)
{
    std::vector<ComplexNum> UD_p(dim_p * dim_p);
    std::vector<ComplexNum> VD_p(dim_p * dim_p);
    std::vector<ComplexNum> ZD_p(dim_p * dim_p);
    std::vector<ComplexNum> UD_n(dim_n * dim_n);
    std::vector<ComplexNum> VD_n(dim_n * dim_n);
    std::vector<ComplexNum> ZD_n(dim_n * dim_n);
    std::vector<ComplexNum> Del1_p(dim_p * dim_p);
    std::vector<ComplexNum> Del1_n(dim_n * dim_n);

    // ZD
    memset(ZD_p.data(), 0, dim_p * dim_p * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            /* code */
        }
    }

    // UD
    memset(UD_p.data(), 0, dim_p * dim_p * sizeof(ComplexNum));
    for (size_t i = 0; i < dim_p; i++)
    {
        /* code */
    }
}

//**********************************************************************
void HartreeFockBogoliubov::PrintTransformMatrix()
{
    std::cout << "   Proton:" << std::endl;
    std::cout << "   [HO basis,  HFB basis]" << std::endl;
    std::cout << "   U_p matrix" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << U_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   V_p matrix" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << V_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   U_n matrix" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << U_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   V_n matrix" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << V_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return;
}

void HartreeFockBogoliubov::PrintDensityMatrix()
{
    std::cout << "   Proton:" << std::endl;
    std::cout << "   rho_p matrix" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << rho_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   kappa_p matrix" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << kappa_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   rho_n matrix" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << rho_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   Kappa_n matrix" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << kappa_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return;
}

// Function to check if a matrix (one-dimensional vector) is an identity matrix
bool HartreeFockBogoliubov::isIdentityMatrix(const ComplexNum *matrix, int rows, int cols)
{
    // Identity matrix must be square
    if (rows != cols)
    {
        return false;
    }

    // Check if the main diagonal elements are 1, and other elements are 0
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (i == j && fabs(matrix[i * cols + j].real() - 1.0) > 0.00001)
            {
                return false; // Diagonal elements should be 1
            }
            else if (i != j && fabs(matrix[i * cols + j].real()) > 0.00001)
            {
                return false; // Non-diagonal elements should be 0
            }
        }
    }

    // All conditions are satisfied, so it is an identity matrix
    return true;
}

// Function to check if a matrix (one-dimensional vector) is a Skew matrix
bool HartreeFockBogoliubov::isSkewMatrix(const ComplexNum *matrix, int rows, int cols)
{
    // Identity matrix must be square
    if (rows != cols)
    {
        return false;
    }

    // Check if the main diagonal elements are 1, and other elements are 0
    for (int i = 0; i < rows; ++i)
    {
        for (int j = i; j < cols; ++j)
        {
            if (i == j && fabs(matrix[i * cols + j].real()) > 0.00001)
            {
                return false; // Diagonal elements should be 1
            }
            else if (i != j && fabs(matrix[i * cols + j].real() + matrix[j * cols + i].real()) > 0.00001)
            {
                return false; // Non-diagonal elements should be 0
            }
        }
    }

    // All conditions are satisfied, so it is an identity matrix
    return true;
}

// Function to check if a matrix (one-dimensional vector) is a Symmetric matrix
bool HartreeFockBogoliubov::isSymmetricMatrix(const ComplexNum *matrix, int rows, int cols)
{
    // Identity matrix must be square
    if (rows != cols)
    {
        return false;
    }

    // Check if the main diagonal elements are 1, and other elements are 0
    for (int i = 0; i < rows; ++i)
    {
        for (int j = i; j < cols; ++j)
        {
            if (i == j && fabs(matrix[i * cols + j].real()) < 0.00001)
            {
                return false; // Diagonal elements should be 1
            }
            else if (i != j && fabs(matrix[i * cols + j].real() - matrix[j * cols + i].real()) > 0.00001)
            {
                return false; // Non-diagonal elements should be 0
            }
        }
    }

    // All conditions are satisfied, so it is an identity matrix
    return true;
}

// Function to check if a matrix (one-dimensional vector) is a orthogonal matrix
bool HartreeFockBogoliubov::isOrthogonalMatrix(const ComplexNum *matrix, int number_vector, int vector_size)
{
    // Check if the main diagonal elements are 1, and other elements are 0
    for (int i = 0; i < number_vector; ++i)
    {
        for (int j = i; j < number_vector; ++j)
        {
            ComplexNum tempValue;
            cblas_zdotc_sub(vector_size, matrix + i, number_vector, matrix + j, number_vector, &tempValue);
            if (i != j and tempValue.real() > 0.00001)
            {
                std::cout << "  The vector " << i << " and vector " << j << " are not orthogonal!   " << tempValue.real() << std::endl;
                return false; // Diagonal elements should be 1
            }
            else if (i == j and fabs(tempValue.real() - 1.0) > 0.00001)
            {
                std::cout << "  The vector " << i << " is not normalized    " << tempValue.real() << "  " << tempValue.imag() << std::endl;
                return false; // Diagonal elements should be 1
            }
        }
    }
    // All conditions are satisfied, so it is an identity matrix
    return true;
}

void HartreeFockBogoliubov::Check_matrix(int dim, ComplexNum *Matrix)
{
    std::cout << "   Chcecking matrix:  dim: " << dim << std::endl;
    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << Matrix[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void HartreeFockBogoliubov::Check_matrix(int dim, double *Matrix)
{
    std::cout << "   Chcecking matrix:  dim: " << dim << std::endl;
    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << Matrix[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/// U^T V + V^T U  = 0
/// U V^+ + V* U^T = 0
/// U^+ U + V^+ V  = 1
/// U U^+ + V* V^T = 1  //CHECK THIS
void HartreeFockBogoliubov::Check_Unitarity()
{
    /// proton
    std::vector<ComplexNum> checkU_p(dim_p * dim_p);
    std::vector<ComplexNum> checkV_p(dim_p * dim_p);
    ComplexNum alpha = ComplexNum(1, 0);
    ComplexNum beta = ComplexNum(0, 0);

    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, dim_p, dim_p, dim_p, &alpha, V_p, dim_p, V_p, dim_p, &beta, checkV_p.data(), dim_p);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim_p, dim_p, dim_p, &alpha, U_p, dim_p, U_p, dim_p, &beta, checkU_p.data(), dim_p);
    mkl_zimatcopy('R', 'T', dim_p, dim_p, ComplexNum(1, 0), checkV_p.data(), dim_p, dim_p);
    cblas_zaxpy(dim_p * dim_p, &alpha, checkV_p.data(), 1, checkU_p.data(), 1);

    /// neutron
    std::vector<ComplexNum> checkU_n(dim_n * dim_n);
    std::vector<ComplexNum> checkV_n(dim_n * dim_n);
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, dim_n, dim_n, dim_n, &alpha, V_n, dim_n, V_n, dim_n, &beta, checkV_n.data(), dim_n);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim_n, dim_n, dim_n, &alpha, U_n, dim_n, U_n, dim_n, &beta, checkU_n.data(), dim_n);
    mkl_zimatcopy('R', 'T', dim_n, dim_n, ComplexNum(1, 0), checkV_n.data(), dim_n, dim_n);
    cblas_zaxpy(dim_n * dim_n, &alpha, checkV_n.data(), 1, checkU_n.data(), 1);

    if (!isIdentityMatrix(checkU_p.data(), dim_p, dim_p))
    {
        Check_matrix(dim_p, checkU_p.data());
        std::cout << "   The proton matrix is not unit! " << std::endl;
        exit(0);
    }
    if (!isIdentityMatrix(checkU_n.data(), dim_n, dim_n))
    {
        Check_matrix(dim_n, checkU_n.data());
        std::cout << "   The neutron matrix is not unit! " << std::endl;
        exit(0);
    }

    return;
}

void HartreeFockBogoliubov::PrintPotential()
{
    std::cout << "   Proton:" << std::endl;
    std::cout << "   Gamma_p matrix" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << Gamma_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   Delta_p matrix" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << Delta_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   Gamma_n matrix" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << Gamma_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   Delta_n matrix" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << Delta_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return;
}

int main(int argc, char *argv[])
{
    ReadWriteFiles rw;
    ModelSpace MS;
    Hamiltonian Hinput(MS);
    // Read OSLO format interaction
    // rw.Read_OSLO_HF_input("InputFile_OSLO.dat", MS, Hinput);

    // read Kshell format interaction
    rw.Read_KShell_HFB_input("Input_HFB.txt", MS, Hinput);

    // print information
    MS.PrintAllParameters_HFB();
    Hinput.PrintHamiltonianInfo_pn();

    //----------------------------------------------
    HartreeFockBogoliubov hfb(Hinput);
    // hfb.UpdateOccupationConstant(hfb.N_p * 1. / hfb.dim_p, hfb.N_n * 1. / hfb.dim_n);
    // hfb.SetCanonical_UV();
    hfb.SetCanonical_UV_Random(20);
    // hfb.UpdateDensityMatrix();
    // hfb.PrintDensityMatrix();
    // hfb.Solve_gradient();

    hfb.Solve_gradient_Constraint();

    hfb.ParticleNumberProjection();

    // hfb.Solve_diag();
    // hfb.PrintDensityMatrix();
    // hfb.PrintTransformMatrix();

    return 0;
}
