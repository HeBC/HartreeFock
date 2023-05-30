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
          Code is far away from bugs
          with the Buddha protecting
*/

#ifndef HFbasis_h
#define HFbasis_h 1

#include <cstring>
#include "mkl.h"
#include "ModelSpace.h"
#include "AngMom.h"

class HFbasis
{
private:
  ModelSpace *ms;
  AngMomProjection *AMJ;
  ComplexNum *Basis_complex; // For projection
  const double *Basis_double;
  int N, dim, isospin, vector_dim = 1;
  bool Malloced_Memory = false;
  bool Is_basis_complex = false;

public:
  // structor
  HFbasis(){};
  ~HFbasis();
  HFbasis(ModelSpace &ms, int isospin);
  HFbasis(ModelSpace &ms, AngMomProjection &AMJ, int isospin);

  // method
  void SetArrayPrt(const double *Array_double);
  void SetArrayPrt(const double *Array_double, int inner_dim);
  void SetArrayPrt(ComplexNum *Array_ComplexNum);
  ComplexNum *GetArrayPointerComplex() { return Basis_complex; };
  const double *GetArrayPointerDouble() { return Basis_double; };
  ComplexNum *GetArrayPointerComplex(int ith) { return Basis_complex + ith * dim; };
  const double *GetArrayPointerDouble(int ith) { return Basis_double + ith * dim; };

  void MallocMemoryComplex();
  void FreeMemory();
  int GetDim() { return dim; };
  int GetParticleNumber() { return N; };
  int GetInnerDim() { return vector_dim; };
  void RotatedOperator(int alpha, int beta, int gamma);
  void ZeroOperatorStructure(ComplexNum *ystruc);
  void MatrixCope(ComplexNum *destination, const ComplexNum *source, int number);
  void MatrixCope(ComplexNum *destination, const double *source, int number);
  int GetTotoalDim() { return dim * N; };

  // debug
  void PrintAllParameters_Double();
  void PrintAllParameters_Complex();
};

class PNbasis
{
private:
  ModelSpace *ms;
  AngMomProjection *AMJ;
  bool Is_basis_complex = false;
  bool InnerVectorParamters = false;
  HFbasis *basis_p = nullptr, *basis_n = nullptr;

public:
  // structor
  PNbasis(){};
  ~PNbasis();
  PNbasis(ModelSpace &ms);
  PNbasis(ModelSpace &ms, AngMomProjection &AMJ);
  PNbasis(PNbasis &anotherBasis);

  // Copy basis
  PNbasis &operator=(PNbasis &rhs);

  // method
  void SetBaiss(HFbasis &inputbasis_p, HFbasis &inputbasis_n);
  void SetArray(const double *Array_p, const double *Array_n);
  void SetArray(const double *Array_p, int inner_dim_p, const double *Array_n, int inner_dim_n);
  void SetArray(ComplexNum *Array_p, ComplexNum *Array_n);
  const double *AccessArrayD_p() { return basis_p->GetArrayPointerDouble(); };
  const double *AccessArrayD_n() { return basis_n->GetArrayPointerDouble(); };
  const ComplexNum *AccessArrayC_p() { return basis_p->GetArrayPointerComplex(); };
  const ComplexNum *AccessArrayC_n() { return basis_n->GetArrayPointerComplex(); };
  HFbasis *GetProtonPrt() { return basis_p; };
  HFbasis *GetNeutronPrt() { return basis_n; };
  int GetProntonInnerDim() { return basis_p->GetInnerDim(); };
  int GetNeutronInnerDim() { return basis_p->GetInnerDim(); };
  void RotatedOperator(int alpha, int beta, int gamma);
  void FullBasis(const std::vector<double> para_vector);
  int GetTotalDim() { return basis_p->GetTotoalDim() + basis_n->GetTotoalDim(); };
  int GetBasis_p_Dim() { return basis_p->GetTotoalDim(); };
  int GetBasis_n_Dim() { return basis_n->GetTotoalDim(); };

  // deprecated
  bool HaveInnerStructure() { return InnerVectorParamters; }
};

#endif
