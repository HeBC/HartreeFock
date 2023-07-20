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
    void Solve_diag();              /// Diagonalize and UpdateF until convergence
    void Solve_noCore();            /// the one body part has been modified for no core!

    //---------------------
    // Tools
    void UpdateU_hybrid(); /// Update the Unitary transformation matrix, hybrid method
    void UpdateU_Qconstraint(double deltaQ, double *O_p, double *O_n);
    void UpdateDensityMatrix(); /// Update the density matrix with the new coefficients C
    void UpdateDensityMatrix(const std::vector<int> proton_vec, const std::vector<int> neutron_vec);
    void UpdateDensityMatrix_DIIS();
    void UpdateF(); /// Update the Fock matrix with the new transformation coefficients C
    void UpdateF_noCore();
    void UpdateF_FromQ(double *O_p, double *O_n);
    void Diagonalize();                                                                             /// Diagonalize Fock term
    void CalcEHF();                                                                                 /// Calculate the HF energy.
    void CalcEHF_noCore();                                                                          /// Calculate the HF energy.
    void CalcEHF(double constrainedQ);                                                              /// Calculate the HF energy with constrained
    double CalcEHF(const std::vector<int> proton_vec, const std::vector<int> neutron_vec);          // inidicate the orbits
    double CalcEHF_HForbits(const std::vector<int> proton_vec, const std::vector<int> neutron_vec); // cal E on HF orbits
    void TransferOperatorToHFbasis(double *Op_p, double *Op_n);
    void CalOnebodyOperator(double *Op_p, double *Op_n, double &Qp, double &Qn);
    void Operator_ph(double *Op_p, double *Op_n);
    void PrintEHF();
    void PrintQudrapole();                            /// Print qudrapole moment
    void Print_Jz();                                  /// Print <Jz>
    bool CheckConvergence();                          /// check the HF single SP
    void Reset_U();                                   /// use identical U matrix
    void RandomTransformationU(int RandomSeed = 525); // Random transformation matrix U
    void UpdateTolerance(double T) { this->tolerance = T; };
    void UpdateGradientStepSize(double size) { gradient_eta = size; };

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
    void CheckDensity();
    void PrintOccupationHO();

private:
    ModelSpace *modelspace;          /// Model Space
    Hamiltonian *Ham;                /// Hamiltonian
    double *U_p, *U_n;               /// transformation coefficients, 1st index is ho basis, 2nd = HF basis
    double *rho_p, *rho_n;           /// density matrix rho_ij, the index in order of dim_p * dim_p dim_n * dim_n
    double *FockTerm_p, *FockTerm_n; /// Fock matrix
    double *Vij_p, *Vij_n;           /// Two body term              
    double *T_term_p = nullptr,       /// SP energies
           *T_term_n = nullptr;       /// SP energies for no core
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

#endif