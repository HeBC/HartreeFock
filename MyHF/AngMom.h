
#ifndef AngMom_hh
#define AngMom_hh 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include "ModelSpace.h"

#define sgn(x) ((x) % 2 ? -1 : 1)

/// Angular momentum projection procedure
#define PI 3.1415926535898

class QuadratureClass // store information about mesh grid
{
private:
    int meshNum;             // number of mesh point
    double *mesh_x, *mesh_w; // Abscissa and weight
public:
    QuadratureClass(){};
    ~QuadratureClass();
    void MallocMemory();
    double GetX(int index) { return mesh_x[index]; };
    double GetWeight(int index) { return mesh_w[index]; };
    int GetTotalNumber() { return meshNum; };
    double *GetXpointer() { return mesh_x; };
    double *GetWeightPointer() { return mesh_w; };
    int *GetNumberPointer() { return &meshNum; };
    void SetNumber(int num) { meshNum = num; };
};

class AngMomProjection
{
public:
    // methods
    AngMomProjection(){};
    AngMomProjection(ModelSpace &ms)
        : ms(&ms){};
    ~AngMomProjection();

    /// initial Projection
    void InitInt_pn();
    void InitInt_Iden();
    void InitInt_HF_Projection();

    //////////////////////
    void PrintInfo();
    int GetIndex_Alpha(int index) { return GasussQuadMap[3 * index]; };
    int GetIndex_Beta(int index) { return GasussQuadMap[3 * index + 1]; };
    int GetIndex_Gamma(int index) { return GasussQuadMap[3 * index + 2]; };
    double GetAlpha_x(int index) { return GQAlpha.GetX(index); };
    double GetAlpha_w(int index) { return GQAlpha.GetWeight(index); };
    double GetBeta_x(int index) { return GQBeta.GetX(index); };
    double GetBeta_w(int index) { return GQBeta.GetWeight(index); };
    double GetGamma_x(int index) { return GQGamma.GetX(index); };
    double GetGamma_w(int index) { return GQGamma.GetWeight(index); };
    double GetWigner_d_beta(int isospin, int beta, int i, int j);
    double *GetWigner_d_prt(int isospin, int beta);
    int GetTotalMeshPoints() { return GQdim; };
    int GetMeshDimensionIn(int type); // type = 0 alpha, 1 beta, 2 gamma
    ComplexNum GuassQuad_weight(int alpha, int beta, int gamma);
    ComplexNum LinearAlgebra_weight(int alpha, int gamma);
    /// Testing Code
    void PrintMatrix_p();

private:
    int GQdim; // total dimension mesh_a * mesh_b * mesh_c
    std::vector<int> GasussQuadMap;
    QuadratureClass GQAlpha, GQBeta, GQGamma;
    double *RotatePairs_p, *RotatePairs_n; // the wigner D function for rotating pairs
    ComplexNum *RotateSPO_p, *RotateSPO_n; // the wigner D function for rotating Single particle operator (SPO)
    std::vector<int> RP_StartPoint_p, RP_StartPoint_n;
    double *WDTab; // Wigner D function
    string FileName_ax, FileName_bx, FileName_cx, FileName_aw, FileName_bw, FileName_cw;
    ModelSpace *ms;

    void UpdateFilenames();
    void ReadMesh();
    void Generate_GQ_Mesh(QuadratureClass &QCprt, std::string type); /// generate Gauss quadrature
    void Generate_LA_Mesh(QuadratureClass &QCprt, std::string type); /// generate linear algebra
    bool initial_proton = false;
    bool initial_neutron = false;
    void InitializeBetaFuncs();
    void InitializeMatrix(int isospin); // for NPSM
};

/// namespace Angular momentum
namespace AngMom
{
#define AngMomMax(x, y) ((x) > (y) ? x : y)
#define AngMomMin(x, y) ((x) < (y) ? x : y)
#define logdel(x, y, z) (sqg[x + y - z] + sqg[x - y + z] + sqg[y + z - x] - sqg[x + y + z + 2])

    double threej(double j1, double j2, double j3, double m1, double m2, double m3);
    double cgc(double j3, double m3, double j1, double m1, double j2, double m2);
    void init_sixj();                                      // Before using sixJ and U. please call init_sixj first!
    double sixJ(int a, int b, int e, int d, int c, int f); /* in units of 2*j */
    double U(int a, int b, int c, int d, int e, int f);
    double Wigner_d(int j, int m1, int m2, double beta); // Wigner small d function
    ComplexNum Wigner_D(int j, int m1, int m2, double alpha, double beta, double gamma); // Wigner small d function
    double Factorial(int N);
};

#endif