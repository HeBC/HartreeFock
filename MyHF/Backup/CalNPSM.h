#ifndef CalNPSM_h
#define CalNPSM_h 1

#include <random>
#include <nlopt.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "mpi.h"
#include "ModelSpace.h"
#include "Hamiltonian.h"
#include "MultiPairs.h"
#include "AngMom.h"
#include "NPSMCommutator.h"
#include "ReadWriteFiles.h"
#include "GCM_Toos.h"
#include "MatrixIndex.h"

#include "MCMC.h"
using namespace MarkovChainMC;

//////////////////////////// Minuit
#include "Minuit/FCNBase.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnPrint.h"
#include "Minuit/MnUserParameters.h"
#include "Minuit/MnSimplex.h"

class CalNPSM
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    AngMomProjection *AngMomProj;
    MatrixIndex *MyIndex;
    MultiPairs *MP_p, *MP_n, *MP_stored_p, *MP_stored_n;

    // method
    CalNPSM(){};
    CalNPSM(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj, MatrixIndex &MyIndex, MultiPairs &Stored_p, MultiPairs &Stored_n); // projection
    CalNPSM(ModelSpace &ms, Hamiltonian &Ham, MatrixIndex &MyIndex, MultiPairs &Stored_p, MultiPairs &Stored_n);                               // no projection
    ~CalNPSM();

    double Cal_All_MEs_pn(const std::vector<double> &x);
    double Cal_All_MEs_pn_NoProjection(const std::vector<double> &x);
    double Cal_All_MEs_SamePairs_pn(const std::vector<double> &x);
    double Cal_All_MEs_SamePairs_pn_NoJProjection(const std::vector<double> &x);

    // half-closed
    double Cal_All_MEs_Iden(const std::vector<double> &x);
    double Cal_All_MEs_SamePairs_Iden(const std::vector<double> &x);
    double Cal_All_MEs_NoProjection_Iden(const std::vector<double> &x);
    double up() const { return 1.; };

private:
    string Ovl_filename = "Output/OvlME.dat";
    string Ham_filename = "Output/HamME.dat";
    string Saved_Ovl_filename = "Input/OvlME.dat";
    string Saved_Ham_filename = "Input/HamME.dat";
    string PairMatrixPath = "Output/PS_";
    string CheckMatrix = "Output/CheckOvlME.dat";
    string CheckHMatrix = "Output/CheckHamME.dat";
    double fxval = 1.e7; // record the sum value

    double DealTotalHamiltonianMatrix(ComplexNum *OvlME, ComplexNum *HamME);
    void SaveData_Iden(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME);
    void SaveData_SamePairs_Iden(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME);
    void SaveData_pn(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME);
    void SaveData_SamePairs_pn(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME);
    string GetMatrixFilename(int isospin, int nth_pair, int Order);
    double EigenValues(int dim, int sumEigenvalues, ComplexNum *Ovl, ComplexNum *Ham);
    double RealEigenValues(int dim, int sumEigenvalues, ComplexNum *Ovl, ComplexNum *Ham);
    void Read_Build_Matrix(int dim, ComplexNum *ele, ComplexNum *NewEle, string fileName);

    // basis constructed by different pairs
    // Variation without Angula momemtum projection
    void Calculate_ME_NoProjection_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME, ComplexNum *QpME, ComplexNum *QnME);
    void Calculate_ME_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME);
    ComplexNum Calculate_Vpn_MEs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn);
    ComplexNum Calculate_Vpn_MEs_pairty(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n);

    // basis constructed by same pairs
    void Calculate_ME_SamePairs_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME);
    // Variation without Angula momemtum projection
    void Calculate_ME_NoProjection_pn_SamePairs(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME, ComplexNum *QpME, ComplexNum *QnME);
    ComplexNum Calculate_Vpn_MEs_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn);
    ComplexNum Calculate_Vpn_MEs_pairty_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n);

    // half-closed
    void Calculate_ME_Iden(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME);
    void Calculate_ME_NoProjection_Iden(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME);
    void Calculate_ME_SamePairs_Iden(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME);
};

////////////////////////

class NPSM_VAP : public FCNBase
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    AngMomProjection *AngMomProj;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;
    NPSM_VAP(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj);
    ~NPSM_VAP();
    double operator()(const std::vector<double> &x) const;
    double up() const { return 1.; };

private:
};

class NPSM_V : public FCNBase
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;
    NPSM_V(ModelSpace &ms, Hamiltonian &Ham);
    ~NPSM_V();
    double operator()(const std::vector<double> &x) const;
    double up() const { return 1.; };

private:
};

//------------ half colsed nuclei --------------//
class Iden_BrokenJ_NoJproj : public FCNBase
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;
    Iden_BrokenJ_NoJproj(ModelSpace &ms, Hamiltonian &Ham);
    ~Iden_BrokenJ_NoJproj();
    double operator()(const std::vector<double> &x) const;
    double up() const { return 1.; };

private:
};

class Iden_VAP : public FCNBase
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    AngMomProjection *AngMomProj;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;
    Iden_VAP(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj);
    ~Iden_VAP();
    double operator()(const std::vector<double> &x) const;
    double up() const { return 1.; };

private:
};

// GCM
class NPSM_GCM_variation : public FCNBase
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;
    NPSM_GCM_variation(ModelSpace &ms, Hamiltonian &Ham);
    ~NPSM_GCM_variation();
    double operator()(const std::vector<double> &x) const;
    double up() const { return 1.; };
    CountEvaluationsGCM *GCMcount;

private:
};

class GCM_Projection
{
public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    AngMomProjection *AngMomProj;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;

    // methods
    GCM_Projection(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj);
    GCM_Projection(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj, std::vector<std::vector<double>> &record_paras);
    ~GCM_Projection();
    std::vector<double> DoCalculation(); // Do projection for GCM
    void ReadBasis(string Path);         // initial index and array // must run this
    int SelectBasis();                   // select orthogonal basis, return the number of taken basis
    void OptimizeConfigruation();        // Generate a orthogonal configruation; Orthogonal to one configuration
    void GnerateNewOrthogonalConfigruation();
    void Cal_angle_between_2configruations();
    void Cal_CosAs(); // return cos for all configurations

private:
    std::vector<double> E_calculated;              // record the energies
    std::vector<std::vector<double>> record_paras; // record the parameters

    double overlap_cosA(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
    double overlap_nlopt(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
    double Cal_cosAB_between_2configruations(const std::vector<double> &x, const std::vector<double> &y);
    double overlap_cosAB(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
};

class MCMC_NPSM
{
private:
    Ensemble sample;
    int nwalkers;
    int dim;
    bool constrained = true;
    double RandomGenerator(); // return a random double type number range from (0,1)
    std::vector<double> GetNormDistribution(int dim, double mean, double sigma);
    double CalNPSM_E(const std::vector<double> &x);
    double CalNPSM_MCMC_overlap(const std::vector<double> &x, const std::vector<double> &y);
    double CalNPSM_MCMC_overlap_ratio(const std::vector<double> &x, const std::vector<double> &y);
    double Objective_function_MCMC(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
    double Objective_E_function(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);

    std::vector<double> E_calculated;                // record the energies
    std::vector<double> Overlap_calculated;          // record the energies
    std::vector<std::vector<double>> Previous_paras; // record the parameters

public:
    ModelSpace *ms;
    Hamiltonian *Ham;
    MatrixIndex *MyIndex;
    MultiPairs *MP_stored_p, *MP_stored_n;

    MCMC_NPSM(ModelSpace &ms, Hamiltonian &Ham, int numWalkers, int dim, std::vector<std::vector<double>> init_positions);
    MCMC_NPSM(ModelSpace &ms, Hamiltonian &Ham, int numWalkers, int dim, char *file_name);
    ~MCMC_NPSM();

    // MCMC methods
    void run_Metropolis(int total_draws);
    void run_Metropolis_HeatUp(int total_draws);
    void run_ConstrainedMetropolis(int total_draws);

    void run_SearchNlopt();
    void run_NloptE_min();

    // for save data
    std::vector<std::vector<double>> load_data(char *file_name);
    std::vector<std::vector<double>> GetHistory(int index) { return this->sample.getWalkerHistroy(index); };
    int GetSavedSteps(int index) { return this->sample.getWalkerCopy(index).getSteps(); };
    double GetPHistory(int index, int step) { return this->sample.getWalkerCopy(index).Get_p_history(step); };
    std::vector<std::vector<double>> get_chain();
};

#endif
