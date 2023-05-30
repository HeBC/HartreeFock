
#ifndef NPSMCommutator_h
#define NPSMCommutator_h 1

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ModelSpace.h"
#include "Hamiltonian.h"
#include "MultiPairs.h"
#include "mkl.h"

namespace NPSMCommutator
{
    void MatrixCope(ComplexNum *matrix_a, const ComplexNum *matrix_b, int number);
    void DoubleCommutator(ComplexNum *a, ComplexNum *b, ComplexNum *c, ComplexNum *p4, int dim);
    void MKLDoubleCommutator(ComplexNum *a, ComplexNum *b, ComplexNum *c, ComplexNum *p4, int dim);
    ComplexNum Cal_Overlap(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR);
    ComplexNum CalPairingME(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, ComplexNum *A_dagger, ComplexNum *A_annihilation);
    ComplexNum CalOneBodyOperator(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, const ComplexNum *OB_operator);
    ComplexNum Prepare_pairingME(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m);
    ComplexNum Prepare_Collective_pairingME(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m);
    ComplexNum Prepare_OB_ME(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m);

    ComplexNum Cal_Overlap_parity(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity);
    ComplexNum CalOneBodyOperator_parity(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, ComplexNum *OB_operator);
    ComplexNum Prepare_Collective_pairingME_parity(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, int Hindex, int m);
    ComplexNum Prepare_pairingME_parity(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, int Hindex, int m);
    void Prepare_OB_ME_parity(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, int Hindex, int m, ComplexNum *res);

    ComplexNum CalPairingME_SamePairs(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, ComplexNum *A_dagger, ComplexNum *A_annihilation);
    ComplexNum CalOneBodyOperator_SamePairs(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, const ComplexNum *OB_operator);
    ComplexNum Prepare_Collective_pairingME_SamePairs(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m);
    ComplexNum Prepare_pairingME_SamePairs(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m);
    ComplexNum CalOneBodyOperator_parity_SamePairs(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, ComplexNum *OB_operator);
    ComplexNum Prepare_OB_ME_SamePairs(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m);

}

#endif
