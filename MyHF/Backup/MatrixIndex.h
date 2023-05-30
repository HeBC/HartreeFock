#ifndef MatrixIndex_h
#define MatrixIndex_h 1

#include "ModelSpace.h"
#include "Hamiltonian.h"
#include "AngMom.h"
#include "ReadWriteFiles.h"

class MatrixIndex
{
private:
    void InitializeHindex(Hamiltonian &Ham);
    void InitializeHindex_Iden(Hamiltonian &Ham);
    ModelSpace *ms;
    AngMomProjection *AMproj;
    Hamiltonian *Ham;

public:
    int ME_total;
    int Ovl_total;
    int *MEindex_i, *MEindex_j; // index of ME_{ij}
    int *OvlInd_R, *OvlInd_ME;  // index of rotation and ME_{ij}
    int *Hp_index, *Hp_index_M; // Hp_index is the index of the Hp, Hp_index_M record the M
    int *Hn_index, *Hn_index_M;
    int *Hpn_Hindex, *Hpn_m;
    int *QpListSP, *QnListSP;
    int num_cal_Vpp, num_cal_Vnn, num_cal_Qp, num_cal_Qn;
    MatrixIndex();
    MatrixIndex(ModelSpace &ms, AngMomProjection &AMproj, Hamiltonian &Ham);                     // with angular momentum protjection
    MatrixIndex(ModelSpace &ms, Hamiltonian &Ham);                                               // without J protjection
    MatrixIndex(ModelSpace &ms, Hamiltonian &Ham, int HalfColsed);                               // for half-closed NPSM without projection
    MatrixIndex(ModelSpace &ms, AngMomProjection &AMproj, Hamiltonian &Ham, bool GCMprojection); // GCM projection
    ~MatrixIndex();

    // method
    int GetQpIndex(int Qindex, int t, int m) { return QpListSP[Qindex] + t + m; };
    int GetQnIndex(int Qindex, int t, int m) { return QnListSP[Qindex] + t + m; };
    void GetQpIndex(int Qtm_index, int &Qt_index, int &t, int &m);
    void GetQnIndex(int Qtm_index, int &Qt_index, int &t, int &m);
};



#endif