#ifndef Hamiltonian_h
#define Hamiltonian_h 1

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cctype>

#include "ModelSpace.h"

// GSL
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>

using namespace std;

#define Read_FAILURE -100
#define Read_SUCCESSFUL 100

class HamiltonianElements
{
public:
  int Tz2 = -100; // Vnn Tz = 2 Vpp Tz = -2  Vpn = 0
  int j1;         // index of the orbits
  int j2;
  int j3;
  int j4;
  int J;
  double V;

  // methods
  HamiltonianElements(){};
  ~HamiltonianElements(){};
  HamiltonianElements(int Tz2, int j1, int j2, int j3, int j4, int J, double V);
};

class HamiltoaninColllectiveElements
{
public:
  std::vector<double> y_pair; // for identical pairing interaction
  int Sign;                   // effective sign
  int Parity;
  int J;

  // method
  HamiltoaninColllectiveElements(){};
  ~HamiltoaninColllectiveElements(){};
  HamiltoaninColllectiveElements(int J, int Parity, int Sign, std::vector<double> y_pair);
  std::vector<double> *Get_y_prt() { return &y_pair; };
  double GetY(int index) { return y_pair[index]; };
};

class OneBodyOperatorChannel
{
public:
  int orbit_a, orbit_b, t, Tz2; // (C_a^+ X \tilde C_b)^t, orbit index, t and 2tz
  // method
  OneBodyOperatorChannel(){};
  ~OneBodyOperatorChannel(){};
  OneBodyOperatorChannel(int orbit_a, int orbit_b, int t, int Tz2);

  // method
  int GetIndex_a() { return orbit_a; };
  int GetIndex_b() { return orbit_b; };
  int Get_t() { return t; };
  int Get_tz() { return Tz2; };
};

class OneBodyElement /// store one body operator, especially single particle energy
{
public:
  int orbit_a, orbit_b, t, Tz2; // (C_a^+ X \tilde C_b)^t, orbit index, t and 2tz
  double OBE = 0;               // one body elements
  // method
  OneBodyElement(){};
  ~OneBodyElement(){};

  OneBodyElement(int orbit_a, int orbit_b, int t, int Tz2, double OBE)
      : orbit_a(orbit_a), orbit_b(orbit_b), t(t), Tz2(Tz2), OBE(OBE){};

  // method
  int GetIndex_a() { return orbit_a; };
  int GetIndex_b() { return orbit_b; };
  int Get_t() { return t; };
  int Get_tz() { return Tz2; };
  double GetE() { return OBE; }
};

class Vpn_phCoupledElements
{
public:
  double Vpnvalue;
  int QpOBChannelindex;
  int QnOBChannelindex;
  int t; /// angular momentum

  // method
  Vpn_phCoupledElements(){};
  ~Vpn_phCoupledElements(){};
  Vpn_phCoupledElements(int QpOBChannelindex, int QnOBChannelindex, int t, double Vpnvalue)
      : QpOBChannelindex(QpOBChannelindex), QnOBChannelindex(QnOBChannelindex), t(t), Vpnvalue(Vpnvalue)
  {
  }

  // method
  int GetQpindex() { return QpOBChannelindex; };
  int GetQnindex() { return QnOBChannelindex; };
  double GetV() { return Vpnvalue; };
  int GeT_t() { return t; };
};

class QuadrupoleMEs
{
public:
  vector<int> Q0_list, Q2_list, Q_2_list;
  vector<int> orbit_a, orbit_b;
  vector<double> Q_MEs; // J scheme operator
  vector<double> Q2_MSMEs, Q0_MSMEs, Q_2_MSMEs;
  int Total_Number;

  // method
  QuadrupoleMEs(){};
  ~QuadrupoleMEs(){};
};

class SixteenPoleMEs
{
public:
  vector<int> Q0_list, Q1_list, Q_1_list, Q2_list, Q_2_list, Q3_list, Q_3_list, Q4_list, Q_4_list;
  vector<int> orbit_a, orbit_b;
  vector<double> Q_MEs; // J scheme operator
  vector<double> Q4_MSMEs, Q3_MSMEs, Q2_MSMEs, Q1_MSMEs, Q0_MSMEs, Q_1_MSMEs, Q_2_MSMEs, Q_3_MSMEs, Q_4_MSMEs;
  int Total_Number;

  // method
  SixteenPoleMEs(){};
  ~SixteenPoleMEs(){};
};

class MSOneBodyOperator
{
public:
  int MSorbit_a, MSorbit_b, Tz2; // (C_a^+ X \tilde C_b)^t, orbit index, and 2tz
  // method
  MSOneBodyOperator(){};
  ~MSOneBodyOperator(){};
  MSOneBodyOperator(int MSorbit_a, int MSorbit_b, int Tz2)
      : MSorbit_a(MSorbit_a), MSorbit_b(MSorbit_b), Tz2(Tz2){};

  // method
  int GetIndex_a() { return MSorbit_a; };
  int GetIndex_b() { return MSorbit_b; };
  int Get_tz() { return Tz2; };
};

class MschemeHamiltonian
{
public:
  // method

  MschemeHamiltonian(){};
  ~MschemeHamiltonian();

  void Initial(ModelSpace &ms);
  double Vpp(int a, int b, int c, int d);
  double Vnn(int a, int b, int c, int d);
  double Vpn(int a, int b, int c, int d);
  void add_Vpp(int a, int b, int c, int d, double V);
  void add_Vnn(int a, int b, int c, int d, double V);
  void add_Vpn(int a, int b, int c, int d, double V);
  void Set_Vpp(int a, int b, int c, int d, double V);
  void Set_Vnn(int a, int b, int c, int d, double V);
  void Set_Vpn(int a, int b, int c, int d, double V);
  double *GetVppPrt() { return ME_pp; };
  double *GetVnnPrt() { return ME_nn; };
  double *GetVpnPrt() { return ME_pn; };

  void InitialMSOB();

  std::vector<std::array<int, 4>> Hpp_index, Hnn_index, Hpn_index;
  std::vector<std::array<int, 2>> Hpn_OBindex;
  std::vector<MSOneBodyOperator> OB_p, OB_n;              // One body operator list
  std::vector<std::array<int, 2>> SPOindex_p, SPOindex_n; // the position of One body operator for SP. C^+_i C_i
  int NumOB_p, NumOB_n;                                   // the number of one body operator   a<b

private:
  ModelSpace *ms;
  int dim_p;
  int dim_n;
  double *ME_pp, *ME_nn, *ME_pn;
};

class Hamiltonian
{
public:
  ModelSpace *ms;
  string Vpp_filename, Vnn_filename, Vpn_filename;                  // file name for OSLO format files
  string snt_file;                                                  // snt file for kshell
  vector<HamiltonianElements> Vpp, Vnn, Vpn;                        // normal shell model interaction
  vector<OneBodyElement> OBEs_p, OBEs_n;                            // store one body operator SPE
  vector<HamiltoaninColllectiveElements> VCol_pp, VCol_nn, VCol_pn; // recasted collective pair interaction
  vector<OneBodyOperatorChannel> OBchannel_p, OBchannel_n;          // record the one body channels
  vector<Vpn_phCoupledElements> Vpn_PHcoupled;                      // Vpn for particle hole channel
  QuadrupoleMEs Q2MEs_p, Q2MEs_n;
  SixteenPoleMEs Q4MEs_p, Q4MEs_n;
  MschemeHamiltonian MSMEs; // MScheme Hamiltonian

  Hamiltonian(){};
  ~Hamiltonian();
  Hamiltonian(ModelSpace &ms);
  int GetVppNum() { return Vpp.size(); }; // return the non collective Vpp number
  int GetVpnNum() { return Vpn.size(); };
  int GetVnnNum() { return Vnn.size(); };
  int GetColVppNum() { return VCol_pp.size(); }; // return the non collective Vpp number
  int GetColVnnNum() { return VCol_nn.size(); };
  int GetColVpnNum() { return VCol_pn.size(); };
  string GetVppFilename() { return Vpp_filename; };
  string GetVpnFilename() { return Vpn_filename; };
  string GetVnnFilename() { return Vnn_filename; };
  string GetKshellSntFile() { return snt_file; };
  ModelSpace *GetModelSpace() { return ms; };
  void RemoveWhitespaceInFilename();
  void PrepareV_pnSystem_v1(); // After reading files, Initial Hamiltonian
  void PrepareV_Identical();   // After reading files, Initial Hamiltonian
  void Prepare_MschemeH();     // convert to m scheme Hamiltonian
  void Prepare_MschemeH_Unrestricted();
  void Prepare_MschemeH_Unrestricted_ForPhaffian();

  void Set_H_ColOrNot(bool tf) { H_CollectiveForm = tf; };
  bool H_IsCollective() { return H_CollectiveForm; }; // If H is collective return true
  bool ISIncludedHermitationElements() { return H_has_included_Hermitation; };
  int Get_CalTerms_pp() { return TotalCalterm_pp; };
  int Get_CalTerms_nn() { return TotalCalterm_nn; };
  vector<HamiltoaninColllectiveElements> *GetColMatrixEle_prt(int isospin); // Get collective interaction matrix elements
  void SetMassDep(bool tf) { IsMassDep = tf; };
  bool GetMassDep() { return this->IsMassDep; };

  // Vpn
  int Get_CalTerms_pn() { return TotalCalterm_pn; };
  int GetV_phCoupledChannelNum() { return Vpn_PHcoupled.size(); }; // return the number of ph coupled channel
  int FindOneBodyOperatorChannel(int a, int b, int t, int Tz);
  int FindVpnParticleHoleIndex(int Qp_index, int Qn_index);
  void TranformHpnToPHRepresentation();
  double GetVpn_phCoupledElements(int index) { return Vpn_PHcoupled[index].Vpnvalue; };
  bool IS_Vpn_ph_coupled() { return VpnISphcoupled; }; // if Vpn transform to particle-hole coupled form, return true
  int GetOneBodyOperatorNumber_p() { return OBchannel_p.size(); };
  int GetOneBodyOperatorNumber_n() { return OBchannel_n.size(); };
  int GetCalOBOperatorNumber_p() { return TotalCalOneBodyOperaotr_p; };
  int GetCalOBOperatorNumber_n() { return TotalCalOneBodyOperaotr_n; };
  OneBodyOperatorChannel GetOneBodyOperator(int isospin, int index);
  Vpn_phCoupledElements GetVpn_phcoupled_ME(int index) { return Vpn_PHcoupled[index]; };
  int GetAllowedMaxOneBody_t() { return OB_max_t; };

  // SPE
  void InitialSPEmatrix(int isospin, ModelSpace &ms); // Gnerate a matrix for single particle energy
  ComplexNum *GetSPEpointer(int isospin);
  double GetProtonSPE_Mscheme(int index_p) { return ms->GetProtonSPE(ms->Get_ProtonOrbitIndexInMscheme(index_p)); };
  double GetNeutronSPE_Mscheme(int index_n) { return ms->GetNeutronSPE(ms->Get_NeutronOrbitIndexInMscheme(index_n)); };

  // Print
  void PrintHamiltonianInfo_Iden();
  void PrintHamiltonianInfo_pn();
  void PrintNonCollectiveV(int tz); // print non-collective interaction matrix elements
  void Print_Vpn_ParticleHole();    // print particle hole representation
  void Print_MschemeHpp();          // print particle hole representation
  void Print_MschemeHnn();          // print particle hole representation
  void Print_MschemeHpn();          // print particle hole representation

  double HarmonicRadialIntegral(int isospin, int lamda, int orbit_a, int orbit_b);

private: // Private access specifier
  ComplexNum *SPEmatrix_p, *SPEmatrix_n;
  bool H_has_been_Normlized = false;
  bool SPE_malloc_p = false; // record for the matrix of s.p. energies
  bool SPE_malloc_n = false;
  bool H_has_included_Hermitation = false;
  bool H_CollectiveForm = false; // the Identical H is collective format.
  bool VpnISphcoupled = false;   // if Vpn transform to particle-hole coupled form, return true
  int TotalCalterm_pp, TotalCalterm_nn, TotalCalterm_pn;
  // Total number of one-body operator O^t_m  ->  \sum_{tm} 1
  int TotalCalOneBodyOperaotr_p, TotalCalOneBodyOperaotr_n;
  int OB_max_t, t_p_max, t_n_max; // max t for QQ format interaction, min(t_p, t_n)
  void NormalHamiltonian();       // The normal factor will be added to the elements
  void AddHermitation();          // add Hermitation MEs <cd|H|ab>
  void CalNumberInitial_Iden();
  void CalNumberInitial();
  void SetVpn_phCoupledElements(int index, double V) { Vpn_PHcoupled[index].Vpnvalue = V; };
  double Calculate_Q2(int a, int b, int tz);
  void Initial_Q2();
  double Calculate_Q3(int a, int b, int tz);
  double Calculate_Qt(int lamda, int a, int b, int tz);
  void Initial_Q4();

  double USDmassScalingFactor(double mass, double powerf, double ReferenceA = 18.); //(18/A)^P
  double GetMassDependentFactor();
  int GetCoreMass();
  bool IsMassDep = false;
};

#endif