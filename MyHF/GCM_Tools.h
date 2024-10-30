#ifndef GCM_h
#define GCM_h 1

#include "mpi.h"
#include <string>
#include <stdio.h>
#include <algorithm>

#include "ModelSpace.h"
#include "ReadWriteFiles.h"
#include "HFbasis.h"
#include "AngMom.h"
#include "Pfaffian_tools.h"

using namespace HF_Pfaffian_Tools;

class MatrixIndex
{
private:
    void InitializeHindex(Hamiltonian &Ham);
    void InitializeHindex_Iden(Hamiltonian &Ham);
    ModelSpace *ms;
    AngMomProjection *AMproj;
    // Hamiltonian *Ham;

public:
    int ME_total;
    int Ovl_total;
    int *MEindex_i, *MEindex_j;                            // index of ME_{ij}
    MatrixIndex(ModelSpace &ms, AngMomProjection &AMproj); // GCM projection
    ~MatrixIndex();
};

class GCM_Projection
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    AngMomProjection *AngMomProj;
    MatrixIndex *MyIndex = nullptr;
    PNbasis *basis_stored = nullptr;
    std::vector<double> GCM_results_E; // record the projected Es
    std::vector<int>   GCM_results_J2; // record the projected Js

    // methods
    GCM_Projection(){};
    GCM_Projection(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj);
    ~GCM_Projection();

    void ReadBasis(string Path);   // initial index and array // must run this
    void DoCalculation();          // Do projection for GCM
    void DoCalculation_LAmethod(); // Do projection for GCM with linear algebra method
    void PrintResults();
    void PrintInfo();
    void Do_Projection();
    double Get_Overlap_dependence() { return Overlap_dependence_threshold; };
    // int SelectBasis();                   // select orthogonal basis, return the number of taken basis
    // void OptimizeConfigruation();        // Generate a orthogonal configruation; Orthogonal to one configuration
    // void GnerateNewOrthogonalConfigruation();
    // void Cal_angle_between_2configruations();
    // void Cal_CosAs(); // return cos for all configurations

private:
    std::vector<double> E_calculated;              // record the naive energies
    std::vector<std::vector<double>> record_paras; // record the parameters
    string Output_Ovl_filename = "Output/GCM_OvlME.dat";
    string Output_Ham_filename = "Output/GCM_HamME.dat";
    string CheckOvlMatrix = "Output/CheckOvlME.dat";
    string CheckHMatrix = "Output/CheckHamME.dat";

    // method
    std::vector<double> DealTotalHamiltonianMatrix(ComplexNum *OvlME, ComplexNum *HamME);
    void SolveLinearEquationMatrix(int NumJ, ComplexNum *OvlME, ComplexNum *HamME);
    void Build_Matrix(int dim, ComplexNum *ele, ComplexNum *NewEle);
    void Build_Matrix(int dim, double *ele, double *NewEle);
    std::vector<double> EigenValues(int dim, ComplexNum *Ovl, ComplexNum *Ham);
    void SaveData(ComplexNum *OvlME, ComplexNum *HamME);
    void CheckConfigruationValid(ComplexNum *OvlME);
    std::vector<double> Cal_Overlap_before_Porjection();
    std::vector<double> AnalysisOverlap(double *Ovl);
    void solveLinearSystem(int n, double *A, double *b, double *x);
    void solveLinearSystem(int n, ComplexNum *A, ComplexNum *b, ComplexNum *x); // A x = b
    // double overlap_cosA(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
    // double overlap_nlopt(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
    // double Cal_cosAB_between_2configruations(const std::vector<double> &x, const std::vector<double> &y);
    // double overlap_cosAB(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);

    double Overlap_dependence_threshold = 1.e-4;
};


void mpi_initialize();
void mpi_finalize();


#endif