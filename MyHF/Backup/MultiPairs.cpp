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

#include "MultiPairs.h"
#include "ReadWriteFiles.h"
#include "NPSMCommutator.h"
using namespace NPSMCommutator;

MultiPairs::MultiPairs(ModelSpace &ms, int isospin)
    : ms(&ms), isospin(isospin)
{
    N = ms.GetPairNumber(isospin);
    dim2 = ms.Get_MScheme_dim2(isospin);
    dim = ms.Get_MScheme_dim(isospin);
    MallocMemory(&Basis);
}

MultiPairs::MultiPairs(int N, ModelSpace &ms, int isospin)
    : ms(&ms), N(N), isospin(isospin)
{
    dim2 = ms.Get_MScheme_dim2(isospin);
    dim = ms.Get_MScheme_dim(isospin);
    MallocMemory(&Basis);
}

MultiPairs::MultiPairs(ModelSpace *ms, AngMomProjection *prt, int isospin)
    : ms(ms), isospin(isospin), AMProjector(prt)
{
    N = ms->GetPairNumber(isospin);
    dim2 = ms->Get_MScheme_dim2(isospin);
    dim = ms->Get_MScheme_dim(isospin);
    MallocMemory(&Basis);
}

MultiPairs::MultiPairs(MultiPairs &anotherMP)
{
    ms = anotherMP.ms;
    isospin = anotherMP.isospin;
    AMProjector = anotherMP.AMProjector;
    N = anotherMP.N;
    dim2 = ms->Get_MScheme_dim2(isospin);
    dim = ms->Get_MScheme_dim(isospin);
    MallocMemory(&Basis);
    MatrixCope(this->GetPointer(), anotherMP.GetPointer());
}

void MultiPairs::MallocMemory(ComplexNum ***pt)
{
    if (Malloced_Memory == false)
    {
        // std::cout << " alloc pair number: " << N << std::endl;
        *pt = (ComplexNum **)mkl_malloc((N + 1) * sizeof(ComplexNum *), 64);
        for (int i = 0; i <= N; i++)
        {
            // std::cout<< " alloc "<<N <<" " << dim2 <<std::endl;
            (*pt)[i] = (ComplexNum *)mkl_malloc((dim2) * sizeof(ComplexNum), 64);
        }
        Malloced_Memory = true;
    }
}

MultiPairs::~MultiPairs()
{
    FreeMemory();
}

void MultiPairs::FreeMemory()
{
    if (Malloced_Memory == true)
    {
        // std::cout << " free " << N << std::endl;
        for (int i = 0; i <= N; i++)
            mkl_free(Basis[i]);
        mkl_free(Basis);
    }
}

void MultiPairs::MatrixCope(ComplexNum **Matrix_a, ComplexNum **Matrix_b)
{ // a = b // copy all matrix elements matrix[0][] is redundant
    int dim = this->GetDim();
    int N = this->GetPairNumber();
    for (size_t i = 1; i <= N; i++)
        cblas_zcopy(dim * dim, Matrix_b[i], 1, Matrix_a[i], 1); // b -> a
    return;
}

void MultiPairs::MatrixCope(ComplexNum *Matrix_a, ComplexNum *Matrix_b)
{ // a = b // copy all matrix elements matrix[0][] is redundant
    int dim = this->GetDim();
    cblas_zcopy(dim * dim, Matrix_b, 1, Matrix_a, 1); // b -> a
    return;
}

void MultiPairs::VectorDotVector(int num, ComplexNum *Matrix_a, ComplexNum *Matrix_b, ComplexNum *Y) // Y = a*b
{
    vzMul(num, Matrix_a, Matrix_b, Y);
    return;
}

void MultiPairs::ZeroPairStructure()
{
    for (size_t i = 0; i <= N; i++)
        memset(this->GetPointer(i), 0, sizeof(double) * 2 * this->GetDim2());
}

void MultiPairs::ZeroPairStructure(int i)
{
    memset(this->GetPointer(i), 0, sizeof(double) * 2 * this->GetDim2());
}

void MultiPairs::ZeroPairStructure(ComplexNum *ystruc)
{
    memset(ystruc, 0, sizeof(double) * 2 * this->GetDim2());
}

void MultiPairs::Build_Basis_QRdecomp_Diff_Iden(const std::vector<double> &x)
{
    int i, j, t, totParity, m;
    int i1, j1, SP, loopN, Count;
    double *Pair_QRdecom, *QRtau, *tempPairY_p;
    int JschemeDim = ms->GetProtonOrbitsNum();
    int PairStr_num_p = ms->GetTotal_NonCollectivePair_num_p();
    int PairHierarchyNum_p = ms->GetProtonCollectivePairNum();

    tempPairY_p = (double *)mkl_malloc((JschemeDim * JschemeDim) * sizeof(double), 64);
    Pair_QRdecom = (double *)mkl_malloc((PairStr_num_p * N) * sizeof(double), 64);
    QRtau = (double *)mkl_malloc((N) * sizeof(double), 64);
    std::copy(x.begin(), x.begin() + (PairStr_num_p * N), Pair_QRdecom);
    // QR decomposition, generate Orthogonal basis
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, PairStr_num_p, N, Pair_QRdecom, PairStr_num_p, QRtau);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, PairStr_num_p, N, N, Pair_QRdecom, PairStr_num_p, QRtau);
    /// generate proton structure
    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    for (loopN = 1; loopN <= N; loopN++)
    {
        SP = PairStr_num_p * (loopN - 1);
        ComplexNum *ystrc_prt = this->GetPointer(loopN);
        Count = 0;
        for (t = 0; t < PairHierarchyNum_p; t++) // loop all collective pairs
        {
            vector<CollectivePairs> ColPair = ms->GetCollectivePairVector(isospin);
            int J = ColPair[t].GetJ();
            int PairParity = ColPair[t].GetParity();
            vector<int> index_i = ColPair[t].GetVector_i();
            vector<int> index_j = ColPair[t].GetVector_j();
            memset(tempPairY_p, 0, sizeof(double) * JschemeDim * JschemeDim);
            /// J-scheme pair structure
            for (j = 0; j < ColPair[t].GetNumberofNoncollectivePair(); j++)
            {
                if (index_i[j] != index_j[j])
                {
                    tempPairY_p[index_i[j] * JschemeDim + index_j[j]] = 0.5 * Pair_QRdecom[SP + Count];
                    int phase = sgn(1 + J + (ms->GetProtonOrbit_2j(index_i[j]) + ms->GetProtonOrbit_2j(index_j[j])) / 2);
                    tempPairY_p[index_i[j] * JschemeDim + index_j[j]] = 0.5 * Pair_QRdecom[SP + Count] * phase;
                }
                else
                {
                    tempPairY_p[index_i[j] * JschemeDim + index_j[j]] = Pair_QRdecom[SP + Count];
                }
                Count++;
            }

            /// M-scheme pair structure
            for (m = -J; m <= J; m++)
            {
                for (i1 = 0; i1 < dim; i1++)
                {
                    for (j1 = i1; j1 < dim; j1++)
                    { // 0 => i1
                        double cgc = ms->GetProtonCGC(J, m, i1, j1);
                        ystrc_prt[i1 * dim + j1] += cgc * tempPairY_p[ms->Get_ProtonOrbitIndexInMscheme(i1) * JschemeDim + ms->Get_ProtonOrbitIndexInMscheme(j1)];
                    }
                }
            }
        }
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = 0; j1 < i1; j1++)
            {
                ystrc_prt[i1 * dim + j1] = -ystrc_prt[j1 * dim + i1];
            }
        }
    }
    // Free Orthogonal basis
    mkl_free(tempPairY_p);
    mkl_free(Pair_QRdecom);
    mkl_free(QRtau);

    /// generate neutron structure
    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////

    return;
}

void MultiPairs::Build_Basis_QRdecomp_Diff(int isospin, const std::vector<double> &x)
{
    int i, j, t, totParity, m;
    int i1, j1, SP, loopN, Count;
    double *Pair_QRdecom, *QRtau, *tempPairY;
    int JschemeDim = ms->GetOrbitsNumber(isospin);
    int PairStr_num = ms->Get_NonCollecitvePairNumber(isospin);
    int PairHierarchyNum = ms->GetCollectivePairNumber(isospin);

    tempPairY = (double *)mkl_malloc((JschemeDim * JschemeDim) * sizeof(double), 64);
    Pair_QRdecom = (double *)mkl_malloc((PairStr_num * N) * sizeof(double), 64);
    QRtau = (double *)mkl_malloc((N) * sizeof(double), 64);
    int IsospinSP = 0;
    if (isospin == Neutron)
    {
        IsospinSP = ms->GetTotal_NonCollectivePair_num_p();
    }
    std::copy(x.begin() + IsospinSP, x.begin() + (PairStr_num * N) + IsospinSP, Pair_QRdecom);
    // QR decomposition, generate Orthogonal basis
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, PairStr_num, N, Pair_QRdecom, PairStr_num, QRtau);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, PairStr_num, N, N, Pair_QRdecom, PairStr_num, QRtau);
    /// generate proton structure
    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    for (loopN = 1; loopN <= N; loopN++)
    {
        SP = PairStr_num * (loopN - 1);
        ComplexNum *ystrc_prt = this->GetPointer(loopN);
        Count = 0;
        for (t = 0; t < PairHierarchyNum; t++) // loop all collective pairs
        {
            vector<CollectivePairs> ColPair = ms->GetCollectivePairVector(isospin);
            int J = ColPair[t].GetJ();
            int PairParity = ColPair[t].GetParity();
            vector<int> index_i = ColPair[t].GetVector_i();
            vector<int> index_j = ColPair[t].GetVector_j();
            memset(tempPairY, 0, sizeof(double) * JschemeDim * JschemeDim);
            /// J-scheme pair structure
            for (j = 0; j < ColPair[t].GetNumberofNoncollectivePair(); j++)
            {
                if (index_i[j] != index_j[j])
                {
                    tempPairY[index_i[j] * JschemeDim + index_j[j]] = 0.5 * Pair_QRdecom[SP + Count];
                    int phase = sgn(1 + J + (ms->GetOrbit_2j(index_i[j], isospin) + ms->GetOrbit_2j(index_j[j], isospin)) / 2);
                    tempPairY[index_i[j] * JschemeDim + index_j[j]] = 0.5 * Pair_QRdecom[SP + Count] * phase;
                }
                else
                {
                    tempPairY[index_i[j] * JschemeDim + index_j[j]] = Pair_QRdecom[SP + Count];
                }
                Count++;
            }

            /// M-scheme pair structure
            for (m = -J; m <= J; m++)
            {
                for (i1 = 0; i1 < dim; i1++)
                {
                    for (j1 = i1; j1 < dim; j1++)
                    { // 0 => i1
                        double cgc = ms->Get_CGC(isospin, J, m, i1, j1);
                        ystrc_prt[i1 * dim + j1] += cgc * tempPairY[ms->Get_OrbitIndex_Mscheme(i1, isospin) * JschemeDim + ms->Get_OrbitIndex_Mscheme(j1, isospin)];
                    }
                }
            }
        }
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = 0; j1 < i1; j1++)
            {
                ystrc_prt[i1 * dim + j1] = -ystrc_prt[j1 * dim + i1];
            }
        }
    }
    // Free Orthogonal basis
    mkl_free(tempPairY);
    mkl_free(Pair_QRdecom);
    mkl_free(QRtau);
    return;
}

void MultiPairs::Build_Basis_Diff_Iden(const std::vector<double> &x)
{
    int i, j, t, totParity, m;
    int i1, j1, SP, loopN, Count;

    int JschemeDim = ms->GetProtonOrbitsNum();
    int PairStr_num_p = ms->GetTotal_NonCollectivePair_num_p();
    int PairHierarchyNum_p = ms->GetProtonCollectivePairNum();
    std::vector<double> tempPairY_p;
    tempPairY_p.resize(JschemeDim * JschemeDim);

    double NormalFactor = 0;
    this->ZeroPairStructure();
    for (loopN = 1; loopN <= N; loopN++)
    {
        SP = PairStr_num_p * (loopN - 1);
        ///  normal   <P0 | P0 > = 1  ///
        NormalFactor = 0;
        for (i = SP; i < SP + PairStr_num_p; i++)
        {
            NormalFactor += x[i] * x[i];
        }
        NormalFactor = std::sqrt(2 * NormalFactor);
        /////////////////////////////////
        ComplexNum *ystrc_prt = this->GetPointer(loopN);
        Count = 0;
        for (t = 0; t < PairHierarchyNum_p; t++) // loop all collective pairs
        {
            vector<CollectivePairs> ColPair = ms->GetCollectivePairVector(isospin);
            int J = ColPair[t].GetJ();
            int PairParity = ColPair[t].GetParity();
            vector<int> index_i = ColPair[t].GetVector_i();
            vector<int> index_j = ColPair[t].GetVector_j();
            memset(&tempPairY_p[0], 0, sizeof(double) * tempPairY_p.size());

            /// J-scheme pair structure
            for (j = 0; j < ColPair[t].GetNumberofNoncollectivePair(); j++)
            {
                // std::cout<<t << J << "  " << SP + Count <<"  " << x[SP + Count] << std::endl;
                if (index_i[j] != index_j[j])
                {
                    tempPairY_p[index_i[j] * JschemeDim + index_j[j]] = 0.5 * x[SP + Count] / NormalFactor;
                    int phase = sgn(1 + J + (ms->GetProtonOrbit_2j(index_i[j]) + ms->GetProtonOrbit_2j(index_j[j])) / 2);
                    tempPairY_p[index_j[j] * JschemeDim + index_i[j]] = 0.5 * x[SP + Count] * phase / NormalFactor;
                }
                else
                {
                    tempPairY_p[index_i[j] * JschemeDim + index_j[j]] = x[SP + Count] / NormalFactor;
                }
                Count++;
            }
            /// M-scheme pair structure
            for (m = -J; m <= J; m++)
            {
                for (i1 = 0; i1 < dim; i1++)
                {
                    for (j1 = i1 + 1; j1 < dim; j1++)
                    { // 0 => i1
                        double cgc = ms->GetProtonCGC(J, m, i1, j1);
                        ystrc_prt[i1 * dim + j1] += cgc * tempPairY_p[ms->Get_ProtonOrbitIndexInMscheme(i1) * JschemeDim + ms->Get_ProtonOrbitIndexInMscheme(j1)];
                    }
                }
            }
        }
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = 0; j1 < i1; j1++)
            {
                ystrc_prt[i1 * dim + j1] = -ystrc_prt[j1 * dim + i1];
                // std::cout << i1 << "  " << j1 << " " << ystrc_prt[i1 * dim + j1].real() << std::endl;
            }
        }
    }

    return;
}

void MultiPairs::Build_Basis_Diff(int isospin, const std::vector<double> &x)
{
    int i, j, t, totParity, m;
    int i1, j1, SP, loopN, Count;
    int JschemeDim = ms->GetOrbitsNumber(isospin);
    int PairStr_num = ms->Get_NonCollecitvePairNumber(isospin);
    int PairHierarchyNum = ms->GetCollectivePairNumber(isospin);

    std::vector<double> tempPairY;
    tempPairY.resize(JschemeDim * JschemeDim);
    std::vector<double> *CGC_prt = ms->GetCGC_prt(isospin);

    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    int IsospinSP = 0;
    if (isospin == Neutron)
    {
        IsospinSP = ms->GetTotal_NonCollectivePair_num_p() * ms->GetProtonPairNum();
    }

    double NormalFactor = 0;
    for (loopN = 1; loopN <= N; loopN++)
    {
        SP = PairStr_num * (loopN - 1) + IsospinSP;

        NormalFactor = 0;
        for (i = SP; i < SP + PairStr_num; i++)
        {
            NormalFactor += x[i] * x[i];
        }
        NormalFactor = std::sqrt(NormalFactor);
        ComplexNum *ystrc_prt = this->GetPointer(loopN);
        Count = 0;
        for (t = 0; t < PairHierarchyNum; t++) // loop all collective pairs
        {
            vector<CollectivePairs> ColPair = ms->GetCollectivePairVector(isospin);
            int J = ColPair[t].GetJ();
            int PairParity = ColPair[t].GetParity();
            vector<int> index_i = ColPair[t].GetVector_i();
            vector<int> index_j = ColPair[t].GetVector_j();
            memset(&tempPairY[0], 0, sizeof(tempPairY[0]) * tempPairY.size());
            /// J-scheme pair structure
            for (j = 0; j < ColPair[t].GetNumberofNoncollectivePair(); j++)
            {
                if (index_i[j] != index_j[j])
                {
                    tempPairY[index_i[j] * JschemeDim + index_j[j]] = 0.5 * x[SP + Count] / NormalFactor;
                    int phase = sgn(1 + J + (ms->GetOrbit_2j(index_i[j], isospin) + ms->GetOrbit_2j(index_j[j], isospin)) / 2);
                    tempPairY[index_j[j] * JschemeDim + index_i[j]] = 0.5 * x[SP + Count] * phase / NormalFactor;
                }
                else
                {
                    tempPairY[index_i[j] * JschemeDim + index_j[j]] = x[SP + Count] / NormalFactor;
                }
                Count++;
            }

            /// M-scheme pair structure
            for (m = -J; m <= J; m++)
            {
                int CGC_SP = ms->Get_CGC_StartPoint(isospin, J, m);
                for (i1 = 0; i1 < dim; i1++)
                {
                    for (j1 = i1 + 1; j1 < dim; j1++) // Pauli blocking effects
                    {                                 // 0 => i1
                        double cgc = (*CGC_prt)[CGC_SP + i1 * dim + j1];
                        ystrc_prt[i1 * dim + j1] += cgc * tempPairY[ms->Get_OrbitIndex_Mscheme(i1, isospin) * JschemeDim + ms->Get_OrbitIndex_Mscheme(j1, isospin)];
                    }
                }
            }
        }
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = 0; j1 < i1; j1++)
            {
                ystrc_prt[i1 * dim + j1] = -ystrc_prt[j1 * dim + i1];
            }
        }
    }
    return;
}

void MultiPairs::Build_Basis_SamePairs(int isospin, const std::vector<double> &x)
{
    int i, j, t, totParity, m;
    int i1, j1, SP, loopN, Count;
    int JschemeDim = ms->GetOrbitsNumber(isospin);
    int PairStr_num = ms->Get_NonCollecitvePairNumber(isospin);
    int PairHierarchyNum = ms->GetCollectivePairNumber(isospin);

    std::vector<double> tempPairY;
    tempPairY.resize(JschemeDim * JschemeDim);
    std::vector<double> *CGC_prt = ms->GetCGC_prt(isospin);

    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    int IsospinSP = 0;
    if (isospin == Neutron)
    {
        IsospinSP = PairStr_num;
    }

    double NormalFactor = 0;
    // for (loopN = 1; loopN <= N; loopN++)
    //{
    loopN = 1;
    SP = IsospinSP;

    NormalFactor = 0;
    for (i = SP; i < SP + PairStr_num; i++)
    {
        NormalFactor += x[i] * x[i];
    }
    NormalFactor = std::sqrt(NormalFactor);
    ComplexNum *ystrc_prt = this->GetPointer(loopN);
    Count = 0;
    for (t = 0; t < PairHierarchyNum; t++) // loop all collective pairs
    {
        vector<CollectivePairs> ColPair = ms->GetCollectivePairVector(isospin);
        int J = ColPair[t].GetJ();
        int PairParity = ColPair[t].GetParity();
        vector<int> index_i = ColPair[t].GetVector_i();
        vector<int> index_j = ColPair[t].GetVector_j();
        memset(&tempPairY[0], 0, sizeof(tempPairY[0]) * tempPairY.size());
        /// J-scheme pair structure
        for (j = 0; j < ColPair[t].GetNumberofNoncollectivePair(); j++)
        {
            if (index_i[j] != index_j[j])
            {
                tempPairY[index_i[j] * JschemeDim + index_j[j]] = 0.5 * x[SP + Count] / NormalFactor;
                int phase = sgn(1 + J + (ms->GetOrbit_2j(index_i[j], isospin) + ms->GetOrbit_2j(index_j[j], isospin)) / 2);
                tempPairY[index_j[j] * JschemeDim + index_i[j]] = 0.5 * x[SP + Count] * phase / NormalFactor;
            }
            else
            {
                tempPairY[index_i[j] * JschemeDim + index_j[j]] = x[SP + Count] / NormalFactor;
            }
            Count++;
        }

        /// M-scheme pair structure
        for (m = -J; m <= J; m++)
        {
            int CGC_SP = ms->Get_CGC_StartPoint(isospin, J, m);
            for (i1 = 0; i1 < dim; i1++)
            {
                for (j1 = i1 + 1; j1 < dim; j1++) // Pauli blocking effects
                {                                 // 0 => i1
                    double cgc = (*CGC_prt)[CGC_SP + i1 * dim + j1];
                    ystrc_prt[i1 * dim + j1] += cgc * tempPairY[ms->Get_OrbitIndex_Mscheme(i1, isospin) * JschemeDim + ms->Get_OrbitIndex_Mscheme(j1, isospin)];
                }
            }
        }
    }
    for (i1 = 0; i1 < dim; i1++)
    {
        for (j1 = 0; j1 < i1; j1++)
        {
            ystrc_prt[i1 * dim + j1] = -ystrc_prt[j1 * dim + i1];
        }
    }
    //}
    // copy Identical pairs
    for (loopN = 2; loopN <= N; loopN++)
        MatrixCope(this->GetPointer(loopN), ystrc_prt);
    return;
}

void MultiPairs::Build_Basis_MSchemePairs_SamePairs(int isospin, const std::vector<double> &x)
{
    int i, i1, j1, SP, loopN, Count;
    int PairStr_num = ms->GetMSchemeNumberOfFreePara(isospin);

    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    int IsospinSP = 0;
    if (isospin == Neutron)
    {
        IsospinSP = ms->GetMSchemeNumberOfFreePara(Proton);
    }

    double NormalFactor = 0;
    loopN = 1;
    SP = IsospinSP;

    NormalFactor = 0;
    for (i = SP; i < SP + PairStr_num; i++)
    {
        NormalFactor += x[i] * x[i];
    }
    NormalFactor = std::sqrt(NormalFactor);
    ComplexNum *ystrc_prt = this->GetPointer(loopN);
    Count = 0;
    for (i1 = 0; i1 < dim; i1++)
    {
        for (j1 = 0; j1 < i1; j1++)
        {
            ystrc_prt[i1 * dim + j1] = -x[Count + SP] / NormalFactor;
            ystrc_prt[j1 * dim + i1] = x[Count + SP] / NormalFactor;
            Count++;
        }
    }
    // copy Identical pairs
    for (loopN = 2; loopN <= N; loopN++)
        MatrixCope(this->GetPointer(loopN), ystrc_prt);
    return;
}

void MultiPairs::Build_Basis_MSchemePairs_Diff(int isospin, const std::vector<double> &x)
{
    int i, i1, j1, SP, loopN, Count;
    int PairStr_num = ms->GetMSchemeNumberOfFreePara(isospin);

    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    int IsospinSP = 0;
    if (isospin == Neutron)
    {
        IsospinSP = ms->GetMSchemeNumberOfFreePara(Proton) * ms->GetProtonPairNum();
    }

    double NormalFactor = 0;
    for (loopN = 1; loopN <= N; loopN++)
    {
        SP = PairStr_num * (loopN - 1) + IsospinSP;

        NormalFactor = 0;
        for (i = SP; i < SP + PairStr_num; i++)
        {
            NormalFactor += x[i] * x[i];
        }
        std::cout<< NormalFactor <<std::endl;
        ComplexNum *ystrc_prt = this->GetPointer(loopN);
        Count = 0;
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = 0; j1 < i1; j1++)
            {
                ystrc_prt[i1 * dim + j1] = -x[Count + SP] / NormalFactor;
                ystrc_prt[j1 * dim + i1] = x[Count + SP] / NormalFactor;
            }
        }
    }
    return;
}

void MultiPairs::Build_Basis_BorkenJSchemePairs_SamePairs(int isospin, const std::vector<double> &x)
{
    int i, j, t, totParity, J, M, ma, mb;
    int i1, j1, SP, loopN, Count;
    int PairHierarchyNum = ms->GetCollectivePairNumber(isospin);
    int PairStr_num = ms->GetMSchemeNumberOfFreePara(isospin);

    vector<BrokenRotationalPairs> tempIndex = ms->GetBrokenJPairVector(isospin);
    std::vector<double> *CGC_prt = ms->GetCGC_prt(isospin);

    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    int IsospinSP = 0;
    if (isospin == Neutron)
    {
        IsospinSP = ms->GetMSchemeNumberOfFreePara(Proton);
    }

    double NormalFactor = 0;
    loopN = 1;
    SP = IsospinSP;

    NormalFactor = 0;
    for (i = SP; i < SP + PairStr_num; i++)
    {
        NormalFactor += x[i] * x[i];
    }
    NormalFactor = std::sqrt(NormalFactor);
    ComplexNum *ystrc_prt = this->GetPointer(loopN);
    for (t = 0; t < tempIndex.size(); t++)
    {
        i1 = tempIndex[t].GetMscheme_index_a();
        j1 = tempIndex[t].GetMscheme_index_b();
        J = tempIndex[t].Get_J();
        M = tempIndex[t].Get_M();
        Count = tempIndex[t].Get_para_index();
        // std::cout << i1 << "  " << j1 << " " << tempIndex.size() << std::endl;
        /*if ( i1 > j1 )
        {
            std::cout<< "Pair index error! "<< i1 << "  "<< j1 <<std::endl;
        }*/
        int CGC_SP = ms->Get_CGC_StartPoint(isospin, J, M);
        double cgc = (*CGC_prt)[CGC_SP + i1 * dim + j1];
        ystrc_prt[i1 * dim + j1] += cgc * x[SP + Count] / NormalFactor;
    }
    for (i1 = 0; i1 < dim; i1++)
    {
        for (j1 = 0; j1 < i1; j1++)
        {
            ystrc_prt[i1 * dim + j1] = -ystrc_prt[j1 * dim + i1];
        }
    }
    //}
    // copy Identical pairs
    for (loopN = 2; loopN <= N; loopN++)
        MatrixCope(this->GetPointer(loopN), ystrc_prt);
    return;
}

void MultiPairs::Build_Basis_BorkenJSchemePairs_Diff(int isospin, const std::vector<double> &x)
{
    int i, j, t, totParity, J, M, ma, mb;
    int i1, j1, SP, loopN, Count;
    int PairHierarchyNum = ms->GetCollectivePairNumber(isospin);
    int PairStr_num = ms->GetMSchemeNumberOfFreePara(isospin);
    //int N = ms->GetPairNumber(isospin);

    vector<BrokenRotationalPairs> tempIndex = ms->GetBrokenJPairVector(isospin);
    std::vector<double> *CGC_prt = ms->GetCGC_prt(isospin);

    /////////////////////////////////
    ///  normal   <P0 | P0 > = 1  ///
    /////////////////////////////////
    this->ZeroPairStructure();
    int IsospinSP = 0;
    if (isospin == Neutron)
    {
        IsospinSP = ms->GetMSchemeNumberOfFreePara(Proton) * ms->GetProtonPairNum();
    }
    double NormalFactor = 0;
    for (loopN = 1; loopN <= N; loopN++)
    {
        SP = PairStr_num * (loopN - 1) + IsospinSP;
        NormalFactor = 0;
        for (i = SP; i < SP + PairStr_num; i++)
        {
            NormalFactor += x[i] * x[i];
        }
        NormalFactor = std::sqrt(NormalFactor);
        ComplexNum *ystrc_prt = this->GetPointer(loopN);
        for (t = 0; t < tempIndex.size(); t++)
        {
            i1 = tempIndex[t].GetMscheme_index_a();
            j1 = tempIndex[t].GetMscheme_index_b();
            J = tempIndex[t].Get_J();
            M = tempIndex[t].Get_M();
            Count = tempIndex[t].Get_para_index();
            //std::cout << i1 << "  " << j1 << " " << x[SP + Count] << "  " << tempIndex.size() << std::endl;
            /*if ( i1 > j1 )
            {
                std::cout<< "Pair index error! "<< i1 << "  "<< j1 <<std::endl;
            }*/
            int CGC_SP = ms->Get_CGC_StartPoint(isospin, J, M);
            double cgc = (*CGC_prt)[CGC_SP + i1 * dim + j1];
            ystrc_prt[i1 * dim + j1] += cgc * x[SP + Count] / NormalFactor;
        }
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = 0; j1 < i1; j1++)
            {
                ystrc_prt[i1 * dim + j1] = -ystrc_prt[j1 * dim + i1];
            }
        }
    }
    return;
}

string MultiPairs::GetMatrixFilename(int nth_pair, int Order)
{
    if (isospin == Proton)
    {
        return ReadMatrixPath_p + to_string(nth_pair) + "_" + to_string(Order) + "_p.dat";
    }
    else
    {
        return ReadMatrixPath_n + to_string(nth_pair) + "_" + to_string(Order) + "_n.dat";
    }
}

void MultiPairs::ReadPairs(int order)
{
    ReadWriteFiles rw;
    for (size_t i = 1; i <= N; i++)
    {
        rw.MPI_ReadMatrix(dim, this->GetPointer(i), this->GetMatrixFilename(i, order));
    }
}

void MultiPairs::RotatedPairs(int alpha, int beta, int gamma)
{
    int i1, j1, k1, k2;
    ComplexNum factor1, factor2, expPart;
    ComplexNum *ystruct, *sourceStruc;
    double *rotateMatrix;
    rotateMatrix = AMProjector->GetWignerDFunc_pair_prt(isospin, beta);
    ystruct = (ComplexNum *)mkl_malloc(GetDim2() * sizeof(ComplexNum), 64);
    for (size_t i = 1; i <= N; i++)
    {
        sourceStruc = this->GetPointer(i);
        ZeroPairStructure(ystruct); /// Zero
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = i1; j1 < dim; j1++)
            { // 0 => i1
                expPart = -1.i * (AMProjector->GetAlpha_x(alpha) * (ms->Get_MSmatrix_2m(isospin, i1) + ms->Get_MSmatrix_2m(isospin, j1)) * PI);
                factor1 = std::exp(expPart); // 2 Pi and mi1 and mj1 are twiced
                for (k1 = 0; k1 < dim; k1++)
                {
                    for (k2 = 0; k2 < dim; k2++)
                    {
                        expPart = -1.i * ((ms->Get_MSmatrix_2m(isospin, k1) + ms->Get_MSmatrix_2m(isospin, k2)) * AMProjector->GetGamma_x(gamma) * PI);
                        factor2 = std::exp(expPart); // 2 Pi and mk1 and mk2 are twiced
                        ystruct[i1 * dim + j1] += factor1 * factor2 * rotateMatrix[i1 * dim + k1] * (sourceStruc)[k1 * dim + k2] * rotateMatrix[j1 * dim + k2];
                    }
                }
            }
        }
        for (i1 = 0; i1 < dim; i1++)
        {
            for (j1 = 0; j1 < i1; j1++)
            { // msm_info_p.dim => i1
                (ystruct)[i1 * dim + j1] = -(ystruct)[j1 * dim + i1];
            }
        }
        NPSMCommutator::MatrixCope(sourceStruc, ystruct, GetDim2());
    }
    mkl_free(ystruct);
}

// Calculate \Pi^+ A \Pi
void MultiPairs::ParityProjection()
{
    ComplexNum *ystruct, *sourceStruc;
    ComplexNum *ProjectionMatrix;
    ProjectionMatrix = ms->GetParityProjOperator_prt(isospin);
    ystruct = (ComplexNum *)mkl_malloc(GetDim2() * sizeof(ComplexNum), 64);
    for (size_t i = 1; i <= N; i++)
    {
        sourceStruc = this->GetPointer(i);
        ZeroPairStructure(ystruct); /// Zero
        VectorDotVector(dim2, sourceStruc, ProjectionMatrix, ystruct);
        NPSMCommutator::MatrixCope(sourceStruc, ystruct, GetDim2());
    }
    mkl_free(ystruct);
}

void MultiPairs::PrintAllParameters()
{
    std::cout << "Total number of pairs: " << this->N << std::endl;
    std::cout << "Isospin: " << this->isospin << "    dim: " << this->dim << std::endl;
    for (size_t i = 1; i <= this->N; i++)
    {
        for (size_t j = 0; j < dim * dim; j++)
        {
            std::cout << " N-i  " << i << "-" << j << "    " << Basis[i][j] << std::endl;
        }
    }
    //exit(0);
    return;
}

MultiPairs &MultiPairs::operator=(MultiPairs &rhs)
{
    this->ms = rhs.ms;
    this->isospin = rhs.isospin;
    this->AMProjector = rhs.AMProjector;
    this->N = rhs.N;
    this->dim2 = ms->Get_MScheme_dim2(isospin);
    this->dim = ms->Get_MScheme_dim(isospin);
    MallocMemory(&Basis);
    MatrixCope(this->GetPointer(), rhs.GetPointer());
    return *this;
}
