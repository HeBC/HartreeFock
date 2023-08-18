#ifndef ModelSpace_h
#define ModelSpace_h 1

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <complex>
#include <map>

#define MKL_Complex16 std::complex<double>
using namespace std;
using namespace std::complex_literals;
typedef std::complex<double> ComplexNum;
#define sgn(x) ((x) % 2 ? -1 : 1)
#define Proton -1
#define Neutron 1

class Orbit
{
public:
    int n;
    int l;
    int j2;  // j * 2
    int tz2; // tz *2   Neutron tz = 1   Proton tz = -1
    int parity;
    double SPE; // single particle energy

    Orbit(){};
    ~Orbit(){};
    Orbit(int n, int l, int j2, int tz2, int parity, double SPE)
        : n(n), l(l), j2(j2), tz2(tz2), parity(parity), SPE(SPE){};
    Orbit(int n, int l, int j2, int tz2, int parity)
        : n(n), l(l), j2(j2), tz2(tz2), parity(parity), SPE(0.){};

    void SetSPE(double eps) { SPE = eps; };
};

class CollectivePairs
{
private:
    std::vector<int> index_i, index_j;

public:
    int J;
    int parity;

    // methods
    CollectivePairs(){};
    ~CollectivePairs(){};
    CollectivePairs(int J, int parity)
        : J(J), parity(parity){};
    int GetJ() { return J; };
    int GetParity() { return parity; };
    int GetNumberofNoncollectivePair() { return index_i.size(); };
    int GetIndex_i(int index) { return index_i[index]; }
    int GetIndex_j(int index) { return index_j[index]; }
    vector<int> GetVector_i() { return index_i; }
    vector<int> GetVector_j() { return index_j; }
    vector<int> *GetVectorPointer_i() { return &index_i; }
    vector<int> *GetVectorPointer_j() { return &index_j; }
};

class BrokenRotationalPairs
{
private:
    int para_index;
    int index_a, index_b; // index in M scheme matrix
    int a, b, ma, mb, ja, jb, J, M, parity;

public:
    BrokenRotationalPairs(){};
    ~BrokenRotationalPairs(){};
    BrokenRotationalPairs(int para_index, int index_a, int index_b, int a, int b, int ma, int mb, int ja, int jb, int J, int M, int parity)
        : para_index(para_index), index_a(index_a), index_b(index_b), a(a), b(b), ma(ma), mb(mb), ja(ja), jb(jb), J(J), M(M), parity(parity){};

    int Get_para_index() { return para_index; };
    int GetMscheme_index_a() { return index_a; };
    int GetMscheme_index_b() { return index_b; };
    int GetOrbitIndex_a() { return a; };
    int GetOrbitIndex_b() { return a; };
    int Get_2ja() { return ja; };
    int Get_2jb() { return jb; };
    int Get_2ma() { return ma; };
    int Get_2mb() { return mb; };
    int Get_J() { return J; };
    int Get_M() { return M; };
    int Get_parity() { return parity; };
};

class MSchemeMatrix // store information about M-scheme Matrix
{
public:
    std::vector<int> j, m, j_index; // j * 2, m * 2 and retrun index of j orbit
    std::vector<int> SPj;           // starting point of each orbit in Matrix
    ComplexNum *PairParity;         // for Parity projection
    std::vector<double> CGC_memory; // store the CG coefficient for M matrix
    std::map<std::array<int, 2>, int> CGC_lookup;

    // method
    MSchemeMatrix(){};
    ~MSchemeMatrix();
    int LookupStartingPoint(int Orbit_index) { return SPj.at(Orbit_index); };                               // look up the staring point in M matrix
    int LookupIndexInMSmatrix(int Orbit_index, int j, int m) { return SPj.at(Orbit_index) + (j - m) / 2; }; // look up the index in M matrix
    void Set_MScheme_Dim(int d);                                                                            // Set M scheme dimension
    int Get_MScheme_Dim() { return dim; };
    int Get_MScheme_Dim2() { return dim2; };
    int GetMschemeM_StartingPoint(int t, int m) { return CGC_lookup.at({t, m}); }
    int GetOrbitIndex(int index) { return j_index[index]; };
    int Get_2j(int index) { return j[index]; };
    int Get_2m(int index) { return m[index]; };
    std::vector<double> *GetCGC_prt() { return &CGC_memory; };
    double GetCGC(int t, int m, int i, int j); // Get CG coefficient
    ComplexNum *GetParityProjOperator_prt() { return PairParity; };

private:
    int dim, dim2;
};

class ModelSpace
{
public:
    vector<Orbit> Orbits_p, Orbits_n; // Single particle otbits

    // methods:
    ModelSpace()
    {
        CheckComplexDefinition();
    };
    ~ModelSpace(){};

    // Initial
    void InitialModelSpace_Iden();
    void InitialModelSpace_pn();
    void InitialModelSpace_HF();

    // HyperParameters
    void SetProtonPairNum(int N) { MS_N_p = N; };
    void SetNeutronPairNum(int N) { MS_N_n = N; };
    void SetProtonNum(int N) { MS_N_p = N; };
    void SetNeutronNum(int N) { MS_N_n = N; };
    void SetCoreProtonNum(int N) { pcore = N; };
    void SetCoreNeutronNum(int N) { ncore = N; };
    void SetTotalOrders(int input) { MSTotal_Order = input; };
    void SetNumberOfBasisSummed(int input) { MSSum_Num = input; };
    void SetNucleiMassA(int input) { MSNucleiMass = input; };
    void SetMassPowerFactor(double input) { MSmass_scaling = input; };
    void SetMassReferenceA(double input) { ReferenceAMassDep = input; };
    void SetAMProjected_J(int input) { MSTotal_J = input; };
    void SetAMProjected_K(int input) { MSTotal_K = input; };
    void SetAMProjected_M(int input) { MSTotal_M = input; };
    void SetProjected_parity(int input) { MSTotal_Parity = input; };
    void SetEnergyConstantShift(double input) { MSEnergyShift = input; };
    void SetMaxNumberOfIteration(int n) { MaxNumberOfIteration = n; };
    void SetRandomSeed(int r) { Random_Seed = r; };
    void SetGuassQuadMesh(int a, int b, int c);
    void SetBasisType(int type) { basis_type = type; };
    void SetPairType(int type) { pair_type = type; };
    void SetNumPara_p(int num) { PairStructurePara_num_p = num; };
    void SetNumPara_n(int num) { PairStructurePara_num_n = num; };
    void SetPrintDiagResult(bool tf) { Diag_Print = tf; };
    void SetGCMTemperature(double input) { Temperature = input; };
    void SetWalkerAmount(int num) { walker_amount = num; };
    void SetSavingWalkingHistory(bool tf) { SavingWalkingHistory = tf; };
    void SetSelectBasis(int tf) { DoSelectionForMCMC = tf; };
    void SetDoGCMprojection(bool tf) { DoGCMprojection = tf; }
    void SetReadingSP_RWMH(bool tf) { IsReadingSP = tf; };
    void SetOverlapMin(double tf) { check_Overlap_dependence = tf; };
    void SetEnergyTruncationGCM(double tf) { E_truncation_GCM = tf; };
    void SetShapeConstrained(bool tf) { Shape_constarined = tf; };
    void SetShapeQ(double q0, double q2)
    {
        shape_Q0 = q0;
        shape_Q2 = q2;
    };
    void Set_hw(double hw_input) { hw = hw_input; };
    void Set_MeshType(std::string input) { MeshType = input; };
    void Set_RefString(std::string input) { Reference_string = input; };
    void Set_ParticleNumberConstrained(bool tf) { ParticleNum_constarined = tf; };
    void SetTargetJz(double jz){ Jz_target = jz; };
    void SetTargetJx(double jx){ Jx_target = jx; };
    void Set_Jz_constraint(bool tf) { Is_Jz_constarined = tf; };
    void Set_Jx_constraint(bool tf) { Is_Jx_constarined = tf; };

    int GetProtonPairNum() { return MS_N_p; };
    int GetNeutronPairNum() { return MS_N_n; };
    int GetCoreProtonNum() { return pcore; };
    int GetCoreNeutronNum() { return ncore; };
    int GetProtonNum() { return MS_N_p; };
    int GetNeutronNum() { return MS_N_n; };
    int GetParticleNumber(int tz); // retrun particle number

    int GetPairNumber(int tz); // retrun pair number
    int GetTotalOrders() { return MSTotal_Order; };
    int GetNumberOfBasisSummed() { return MSSum_Num; };
    int GetNucleiMassA();
    double GetMassPowerFactor() { return MSmass_scaling; };
    int GetAMProjected_J() { return MSTotal_J; };
    int GetAMProjected_K() { return MSTotal_K; };
    int GetAMProjected_M() { return MSTotal_M; };
    int GetProjected_parity() { return MSTotal_Parity; };
    double GetEnergyConstantShift() { return MSEnergyShift; };
    int GetGQ_alpha() { return this->GQ_alpha; };
    int GetGQ_beta() { return this->GQ_beta; };
    int GetGQ_gamma() { return this->GQ_gamma; };
    int GetMaxNumberOfIteration() { return MaxNumberOfIteration; };
    int GetRandomSeed() { return Random_Seed; };
    int GetCoreMass();
    int GetBasisType() { return basis_type; };
    int GetPairType() { return pair_type; };
    int GetNumPara_p() { return PairStructurePara_num_p; };
    int GetNumPara_n() { return PairStructurePara_num_n; };
    bool IsPrintDiagResult() { return Diag_Print; };
    double GetGCMTemperature() { return Temperature; };
    int GetWalkerAmount() { return walker_amount; };
    bool IsSavingWalkingHistory() { return SavingWalkingHistory; };
    int DoesSelectBasis() { return DoSelectionForMCMC; };
    bool DoesRunGCMprojection() { return DoGCMprojection; };
    bool DoesReadingSP_RWMH() { return IsReadingSP; };
    double GetOverlapMin() { return check_Overlap_dependence; };
    double GetEnergyTruncationGCM() { return E_truncation_GCM; };
    bool GetIsShapeConstrained() { return Shape_constarined; };
    double GetShapeQ0() { return shape_Q0; };
    double GetShapeQ2() { return shape_Q2; };
    double GetShapeConstant() { return Sheape_Constant; };
    double Get_hw() { return hw; };
    double GetMassReferenceA() { return ReferenceAMassDep; };
    void GetAZfromString(std::string str, double &A, double &Z);
    std::string Get_MeshType() { return MeshType; };
    std::string Get_RefString() { return Reference_string; };
    bool Get_ParticleNumberConstrained() { return ParticleNum_constarined; };
    double GetTargetJz(){ return Jz_target; };
    double GetTargetJx(){ return Jx_target; };
    bool Get_Jz_constraint() { return Is_Jz_constarined; };
    bool Get_Jx_constraint() { return Is_Jx_constarined; };


    // print all parameters
    void PrintAllParameters_Iden();
    void PrintAllParameters_pn();
    void PrintAllParameters_pn_GCM();
    void PrintAllParameters_HF();
    void PrintAllParameters_HFB();

    // Orbits
    int FindOrbit(int Tz, int j);
    int FindOrbit(int Tz, int n, int l, int j);
    int GetOrbitsNumber(int isospin);
    int GetProtonOrbitsNum() { return Orbits_p.size(); };
    int GetNeutronOrbitsNum() { return Orbits_n.size(); };
    int GetProtonOrbits_parity(int index) { return Orbits_p[index].parity; };
    int GetNeutronOrbits_parity(int index) { return Orbits_n[index].parity; };
    int GetProtonOrbit_2j(int index) { return Orbits_p[index].j2; };
    int GetNeutronOrbit_2j(int index) { return Orbits_n[index].j2; };
    int GetOrbit_2j(int index, int isospin);
    int GetOrbit_2m(int index, int isospin);
    int GetProtonOrbit_l(int index) { return Orbits_p[index].l; };
    int GetNeutronOrbit_l(int index) { return Orbits_n[index].l; };
    int GetProtonOrbit_n(int index) { return Orbits_p[index].n; };
    int GetNeutronOrbit_n(int index) { return Orbits_n[index].n; };
    double GetProtonSPE(int index) { return Orbits_p[index].SPE; };
    double GetNeutronSPE(int index) { return Orbits_n[index].SPE; };
    Orbit &GetOrbit(int isospin, int i);

    // M-scheme
    void InitMSMatrix(int isospin);
    void InitMSMatrix_HF(int isospin);
    int Get_Proton_MScheme_dim() { return MSM_p.Get_MScheme_Dim(); };
    int Get_Proton_MScheme_dim2() { return MSM_p.Get_MScheme_Dim2(); };
    int Get_Neutron_MScheme_dim() { return MSM_n.Get_MScheme_Dim(); };
    int Get_Neutron_MScheme_dim2() { return MSM_n.Get_MScheme_Dim2(); };
    int Get_MScheme_dim(int tz);
    int Get_MScheme_dim2(int tz);
    int Get_ProtonOrbitIndexInMscheme(int index) { return MSM_p.GetOrbitIndex(index); };
    int Get_NeutronOrbitIndexInMscheme(int index) { return MSM_n.GetOrbitIndex(index); };
    int Get_OrbitIndex_Mscheme(int index, int isospin);
    double GetProtonCGC(int t, int m, int i, int j) { return MSM_p.GetCGC(t, m, i, j); };
    double GetNeutronCGC(int t, int m, int i, int j) { return MSM_n.GetCGC(t, m, i, j); };
    double Get_CGC(int isospin, int t, int m, int i, int j);
    int Get_CGC_StartPoint(int isospin, int t, int m);
    int Get_MSmatrix_2j(int isospin, int index);
    int Get_MSmatrix_2m(int isospin, int index);
    int LookupStartingPoint(int isospin, int index);
    MSchemeMatrix *GetMSmatrixPointer(int isospin);
    std::vector<double> *GetCGC_prt(int isospin);
    ComplexNum *GetParityProjOperator_prt(int isospin);
    int GetMSchemeNumberOfFreePara(int isospin);
    int GetMS_index_p(int orbit_index, int j, int m) { return MSM_p.LookupIndexInMSmatrix(orbit_index, j, m); };
    int GetMS_index_n(int orbit_index, int j, int m) { return MSM_n.LookupIndexInMSmatrix(orbit_index, j, m); };
    int Get2Jmax();

    // pair structure
    void InitCollectivePairs(int isospin);
    vector<CollectivePairs> GetCollectivePairVector(int isospin);
    vector<CollectivePairs> *GetCollectivePairVectorPointer(int isospin);
    int GetProtonCollectivePairNum() { return CollectivePair_p.size(); }
    int GetNeutronCollectivePairNum() { return CollectivePair_n.size(); }
    int GetCollectivePairNumber(int isospin);
    int GetTotal_NonCollectivePair_num_p() { return PairStructurePara_num_p; };
    int GetTotal_NonCollectivePair_num_n() { return PairStructurePara_num_n; };
    int Get_NonCollecitvePairNumber(int isospin);
    void Initial_BrokenJPairIndex(int isospin);
    vector<BrokenRotationalPairs> GetBrokenJPairVector(int isospin);
    bool IsJbrokenPair() { return Is_JBroken_pair; };
    void UsingJbrokenPair() { Is_JBroken_pair = true; };

private:
    int MS_N_p, MS_N_n;                                   // number of proton and nuetron
    int pcore = 0, ncore = 0;                             // number of particle in the core
    int PairStructurePara_num_p, PairStructurePara_num_n; // number of paramters
    int NumVarPara_p, NumVarPara_n;                       // number of parameters in variation procedure
    int MSTotal_Order;                                    // Total number of states
    int MSSum_Num;                                        // sum how many states in each cal
    int MSNucleiMass = 0;
    double MSmass_scaling = 0;
    double ReferenceAMassDep = 0; // for the mass dep
    int MSTotal_J;                // twice total J of final state
    int MSTotal_K;                // twice total K of final state
    int MSTotal_M;                // twice total M of final state
    int MSTotal_Parity;
    double MSEnergyShift = 0;
    double hw = 0.;         // frequency of the oscillator
    bool Diag_Print = true; // Turn off the Diagonalization output, 0 Turn off; 1 turn on
    bool SavingWalkingHistory = false;
    int DoSelectionForMCMC = 0; // 0 false, 1 without J projection, 2 with J projection
    bool DoGCMprojection = true;
    string Reference_string;

    int walker_amount = 0; // the number of walkers
    MSchemeMatrix MSM_p, MSM_n;
    vector<CollectivePairs> CollectivePair_p, CollectivePair_n; // pair Hierarchy
    vector<BrokenRotationalPairs> BrokenRPairs_p, BrokenRPairs_n;
    string VppFileName, VnnFileName, VpnFileName;
    int GQ_alpha, GQ_beta, GQ_gamma;
    int MaxNumberOfIteration;
    int Random_Seed;
    int basis_type;               // 0 different pairs; 1 identical pairs
    bool Is_JBroken_pair = false; // ture J borken pairs;  false J conserved pairs
    int pair_type;                // 0 J conserved pairs; 1 J borken pairs
    double Temperature = 10000;
    bool IsReadingSP = false;                // Read starting points for Random walking, false 0, true 1;
    double check_Overlap_dependence = 1.e-4; // the min eigenvalue of overlap matrix
    double E_truncation_GCM = 1000000;       // Energy truncation to pick up configurations
    bool Shape_constarined = false;          // if constrain Q0 and Q2
    double shape_Q0 = 0.;
    double shape_Q2 = 0.;
    double Sheape_Constant = 1.e+3;
    bool ParticleNum_constarined = false; // constaint the number of particle in HFB
    bool Is_Jz_constarined = false; // constaint Jz
    bool Is_Jx_constarined = false; // constaint Jx
    double Jz_target = 0.;
    double Jx_target = 0.;
    std::string MeshType = "";
    void CheckComplexDefinition();
};

#endif