#ifndef CalHF_h
#define CalHF_h 1

////////////////////////////////////////////////////////////////
/// Calculate all kinds of MEs with the Pfaffian techniques  ///
////////////////////////////////////////////////////////////////

#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "mpi.h"
#include "ModelSpace.h"
#include "Hamiltonian.h"
#include "AngMom.h"
#include "HFbasis.h"

namespace HF_Pfaffian_Tools
{
    // calculate the s matrix for overlap   [ 0    M ]
    //                                      [ -M^T 0 ]
    void S_matrix_overlap(HFbasis &Bra, HFbasis &Ket, double *M);
    // for complex number
    void S_matrix_overlap_complex(HFbasis &Bra, HFbasis &Ket, ComplexNum *M);
    // the transform matrix elemenmts have inner structure, use the parameter inner_dim
    // indicate the inner dimension
    void S_matrix_overlap(HFbasis &Bra, HFbasis &Ket, double *M, int inner_dim);
    // End overlap

    // S matrix for Onebody operator <a|O_ij|b>
    // the overlap part should be evaluated separately
    void S_matrix_OneBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j);
    // for complex number
    void S_matrix_OneBody_MEs_complex(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, int i, int j);
    // the transform matrix elemenmts have inner structure, use the parameter inner_dim
    // indicate the inner dimension
    void S_matrix_OneBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j, int inner_dim);
    // S matrix for OneBody operator
    // the array M should be zero first!
    void S_matrix_FullOvl_OneBody(HFbasis &Bra, HFbasis &Ket, double *M, double *M_ovl);
    void S_matrix_FullOvl_OneBody(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, ComplexNum *M_ovl);
    // End OneBody

    // S matrix for Onebody operator <a|C^+_i C^+_j C_k C_l|b>
    // the overlap part should be evaluated separately
    void S_matrix_TwoBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j, int k, int l);
    // for complex number
    void S_matrix_TwoBody_MEs_complex(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, int i, int j, int k, int l);
    // the transform matrix elemenmts have inner structure, use the parameter inner_dim
    // indicate the inner dimension
    void S_matrix_TwoBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j, int k, int l, int inner_dim);

    // S matrix for TwoBody operator
    // the array M should be zero first!
    void S_matrix_FullOvl_TwoBody(HFbasis &Bra, HFbasis &Ket, double *M, double *M_ovl);
    void S_matrix_FullOvl_TwoBody(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, ComplexNum *M_ovl);
    // End Twobody

    // out-place Pfaffian
    double PFAFFIAN(int N, const double *Smatrix);         // return the pfaffian value of the matrix S, N is the dim of S
    ComplexNum PFAFFIAN(int N, const ComplexNum *Smatrix); // return the pfaffian value of the matrix S, N is the dim of S

    // normalize the basis
    void normalizeBais(double *x, int N, int dim, int inner_dim);

    // functions
    double CalHFKernels(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket);
    ComplexNum CalHFKernels_Complex(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket);
    void CalHFKernels_Complex(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket, ComplexNum &HamME, ComplexNum &OvlME);

    void CalHFShape(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket);
    double CalHFOverlap(PNbasis &Bra, PNbasis &Ket);
    ComplexNum CalHFOverlap_Complex(PNbasis &Bra, PNbasis &Ket);
    double CalHF_Hamiltonian(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket);
    void CalHF_Q2(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket, double *Q_array);

    // deprecated
    double CalHFKernels_Advanced(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket);

};

namespace HFB_Pfaffian_Tools
{
    // <φ0|φ1> = (−1)^N(N+1)/2 pf(M), where M is a 2N X 2N skew-symmetric matrix
    // M = [ M1   -I ]
    //     [  I   M0* ]
    // where M1 and M0 come from HFB wave function |φ1> = exp( 1/2 sum_kk' M1_kk' a^+_k a^+_k' ) |0>
    ComplexNum Compute_Overlap(int N, ComplexNum *M0, ComplexNum *M1);

};

#endif
