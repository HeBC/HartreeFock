
#ifndef MultiPairs_h
#define MultiPairs_h 1

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ModelSpace.h"
#include "AngMom.h"
#include "mkl.h"

class MultiPairs
{
private:
    ModelSpace *ms;
    ComplexNum **Basis;
    int N, dim, dim2, isospin;
    ComplexNum ***GetPointer_initial() { return &Basis; };
    AngMomProjection *AMProjector;
    string ReadMatrixPath_p = "Input/proton/PS_";
    string ReadMatrixPath_n = "Input/neutron/PS_";
    string GetMatrixFilename(int nth_pair, int Order);
    void MatrixCope(ComplexNum *Matrix_a, ComplexNum *Matrix_b);
    bool Malloced_Memory = false;

public:
    // structor
    MultiPairs(){};
    ~MultiPairs();
    MultiPairs(ModelSpace &ms, int isospin);
    MultiPairs(int N, ModelSpace &ms, int isospin);
    MultiPairs(ModelSpace *ms, AngMomProjection *prt, int isospin);
    MultiPairs(MultiPairs &anotherMP);

    // Copy basis
    MultiPairs &operator=(MultiPairs &rhs);

    // method
    ComplexNum **GetPointer() { return Basis; };
    ComplexNum *GetPointer(int N_index) { return Basis[N_index]; };
    int GetIsospin() { return isospin; };
    int GetDim() { return dim; };   // retrun M scheme matrix dimension
    int GetDim2() { return dim2; }; // retrun M scheme matrix dimension
    int GetPairNumber() { return N; };
    void FreeMemory();
    void MallocMemory(ComplexNum ***pt);
    void GetAngMomProjection_prt(AngMomProjection *prt) { AMProjector = prt; };

    // manipulating pair-structure
    void ZeroPairStructure();
    void ZeroPairStructure(int i);
    void ZeroPairStructure(ComplexNum *ystruc);
    void Build_Basis_QRdecomp_Diff_Iden(const std::vector<double> &x);
    void Build_Basis_QRdecomp_Diff(int isospin, const std::vector<double> &x);
    void Build_Basis_Diff_Iden(const std::vector<double> &x);
    void Build_Basis_Diff(int isospin, const std::vector<double> &x);
    void Build_Basis_SamePairs(int isospin, const std::vector<double> &x);
    void Build_Basis_MSchemePairs_SamePairs(int isospin, const std::vector<double> &x);
    void Build_Basis_MSchemePairs_Diff(int isospin, const std::vector<double> &x);
    void Build_Basis_BorkenJSchemePairs_SamePairs(int isospin, const std::vector<double> &x);
    void Build_Basis_BorkenJSchemePairs_Diff(int isospin, const std::vector<double> &x);

    void RotatedPairs(int alpha, int beta, int gamma);
    void ParityProjection();
    void MatrixCope(ComplexNum **Matrix_a, ComplexNum **Matrix_b);                            // A = B
    void VectorDotVector(int num, ComplexNum *Matrix_a, ComplexNum *Matrix_b, ComplexNum *Y); // Y = a*b
    void ReadPairs(int order);
    void PrintAllParameters();
};

#endif
