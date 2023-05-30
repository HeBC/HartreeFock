#ifndef GCM_h
#define GCM_h 1

#include "mpi.h"
#include <string>

#include "ModelSpace.h"
#include "MultiPairs.h"
#include "ReadWriteFiles.h"
#include "NPSMCommutator.h"
#include "MatrixIndex.h"

class CountEvaluationsGCM
{
public:
    // methods
    CountEvaluationsGCM(ModelSpace &ms);
    ~CountEvaluationsGCM();
    bool CountCals();
    void SetFilename(string input) { OutPutString = input; };
    int GetSavedNum() { return saved_number; };
    string GetFilename() { return OutPutString; };

private:
    ModelSpace *ms;
    int total_steps;
    int count_cals = 0;
    int saved_number = 0;
    string saved_path = "Output/GCM_output";
    string OutPutString;
};

class Cal_GCM_projection
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    AngMomProjection *AngMomProj;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;

    // method
    Cal_GCM_projection(){};
    Cal_GCM_projection(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj, MatrixIndex &MyIndex, MultiPairs &Stored_p, MultiPairs &Stored_n); // projection
    ~Cal_GCM_projection();

    std::vector<double> Cal_Overlap_before_Porjection(); // Test linear denpendentce
    std::vector<double> Cal_All_MEs_pn();                // VBP
    std::vector<int> SelectBasis();                      // pick linear independent basis
    double Get_Overlap_dependence() { return Overlap_dependence; };

private:
    string Ovl_filename = "Output/OvlME.dat";
    string Ham_filename = "Output/HamME.dat";
    string Saved_Ovl_filename = "Input/OvlME.dat";
    string Saved_Ham_filename = "Input/HamME.dat";
    string PairMatrixPath = "Output/PS_";
    string CheckMatrix = "Output/CheckOvlME.dat";
    string CheckHMatrix = "Output/CheckHamME.dat";

    void SaveData_pn(ComplexNum *OvlME, ComplexNum *HamME);
    void Build_Matrix(int dim, ComplexNum *ele, ComplexNum *NewEle);
    std::vector<double> DealTotalHamiltonianMatrix(ComplexNum *OvlME, ComplexNum *HamME);
    std::vector<double> EigenValues(int dim, ComplexNum *Ovl, ComplexNum *Ham);

    // basis constructed by different pairs
    void Calculate_ME_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME);
    ComplexNum Calculate_Vpn_MEs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn);
    ComplexNum Calculate_Vpn_MEs_pairty(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n);

    // basis constructed by same pairs
    void Calculate_ME_SamePairs_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME);
    ComplexNum Calculate_Vpn_MEs_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn);
    ComplexNum Calculate_Vpn_MEs_pairty_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n);

    // Evaluate overlaps for linear dependence analysis
    void MPI_Calculate_Overlap(int i, ComplexNum *MyOvlME);
    std::vector<double> AnalysisOverlap(ComplexNum *Ovl);
    void PickBasis(ComplexNum *Ovl, std::vector<int> &Index);
    void Build_Matrix_BasisPicking(int Original_dim, int dim, std::vector<int> &Index, ComplexNum *ele, ComplexNum *NewEle);
    double Overlap_dependence = 1.e-4;
};

#endif