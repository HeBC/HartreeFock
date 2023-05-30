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

#include "NPSMCommutator.h"

namespace NPSMCommutator
{
    ComplexNum PairPairContractor(MultiPairs &ytabL, int index_L, MultiPairs &ytabR, int index_R)
    {
        int i, j;
        int dim = ytabL.GetDim();
        ComplexNum value, **matrix_L, **matrix_R;
        matrix_L = ytabL.GetPointer();
        matrix_R = ytabR.GetPointer();
        value = 0;
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                value += matrix_L[index_L][i * dim + j] * matrix_R[index_R][j * dim + i];
        return -2. * value;
    }

    ComplexNum PairPairContractor(ComplexNum *matrix_L, ComplexNum *matrix_R, int dim)
    {
        int i, j;
        ComplexNum value;
        value = 0;
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                value += matrix_L[i * dim + j] * matrix_R[j * dim + i];
        return -2. * value;
    }

    void Transpose(ComplexNum *a, ComplexNum *b, int dim, double factor)
    { // b := alpha*op(a)
        ComplexNum alpha(factor, 0);
        mkl_zomatcopy('R', 'T', dim, dim, alpha, a, dim, b, dim);
        return;
    }

    void MatrixCope(MultiPairs &ytab_a, int index_a, MultiPairs &ytab_b, int index_b) // copy matrix index_a from index_b
    {                                                                                 // a = b
        int dim = ytab_a.GetDim();
        cblas_zcopy(dim * dim, ytab_b.GetPointer(index_b), 1, ytab_a.GetPointer(index_a), 1); // b -> a
        return;
    }

    void MatrixCope(MultiPairs &ytab_a, MultiPairs &ytab_b)
    { // a = b // copy all matrix elements matrix[0][] is redundant
        int dim = ytab_a.GetDim();
        int N = ytab_a.GetPairNumber();
        for (size_t i = 1; i <= N; i++)
            cblas_zcopy(dim * dim, ytab_b.GetPointer(i), 1, ytab_a.GetPointer(i), 1); // b -> a
        return;
    }

    void MatrixCope(int N, MultiPairs &ytab_a, MultiPairs &ytab_b)
    { // a = b // copy all matrix elements matrix[0][] is redundant
        int dim = ytab_a.GetDim();
        for (size_t i = 1; i <= N; i++)
            cblas_zcopy(dim * dim, ytab_b.GetPointer(i), 1, ytab_a.GetPointer(i), 1); // b -> a
        return;
    }

    void MatrixCope(MultiPairs &ytab_a, int Start_a, int End_a, MultiPairs &ytab_b, int Start_b)
    { // a = b // copy matrix elements from Start_a to End_a, matrix[0][] is redundant
        int dim = ytab_a.GetDim();
        int N = (End_a - Start_a + 1);
        for (size_t i = 0; i < N; i++)
            cblas_zcopy(dim * dim, ytab_b.GetPointer(Start_b + i), 1, ytab_a.GetPointer(Start_a + i), 1); // b -> a
        return;
    }

    void MatrixCope(ComplexNum *matrix_a, const ComplexNum *matrix_b, int number)
    {                                                  // a = b // copy matrix elements from Start_a to End_a, matrix[0][] is redundant
        cblas_zcopy(number, matrix_b, 1, matrix_a, 1); // b -> a
        return;
    }

    void MKLMatrixMultiply(ComplexNum *a, const ComplexNum *b, ComplexNum *c, int dim, double factor)
    {
        ComplexNum alpha(factor, 0), beta(0, 0);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha, a, dim, b, dim, &beta, c, dim);
        return;
    }

    void old_DoubleCommutator(ComplexNum *a, ComplexNum *b, ComplexNum *c, ComplexNum *p4, int dim) // p1 p2 p3 p4
    {
        int i, j, k;
        ComplexNum temp1[dim * dim];
        for (i = 0; i < dim; i++)
        {
            for (j = 0; j < dim; j++)
            {
                temp1[i * dim + j] = 0;
                for (k = 0; k < dim; k++)
                {
                    temp1[i * dim + j] += a[i * dim + k] * b[k * dim + j];
                }
            }
        }
        for (i = 0; i < dim; i++)
        {
            for (j = 0; j < dim; j++)
            {
                p4[i * dim + j] = 0;
                for (k = 0; k < dim; k++)
                {
                    p4[i * dim + j] += 4. * temp1[i * dim + k] * c[k * dim + j];
                }
            }
        }

        for (i = 0; i < dim; i++)
        {
            for (j = i; j < dim; j++)
            {
                p4[i * dim + j] -= p4[j * dim + i];
            }
        }
        for (i = 0; i < dim; i++)
        {
            for (j = 0; j < i; j++)
            {
                p4[i * dim + j] = -p4[j * dim + i];
            }
        }
        return;
    }

    void DoubleCommutator(ComplexNum *a, ComplexNum *b, ComplexNum *c, ComplexNum *p4, int dim) // p1 p2 p3 p4
    {
        int i, j;
        ComplexNum *temp1 = new ComplexNum[dim * dim];
        // memset(temp1, 0, dim * dim * 2 * sizeof(double));
        MKLMatrixMultiply(a, b, temp1, dim, 1.);
        // memset(p4, 0, dim * dim * 2 * sizeof(double));
        MKLMatrixMultiply(temp1, c, p4, dim, 4.);
        delete[] temp1;
        for (i = 0; i < dim; i++)
        {
            for (j = i; j < dim; j++)
            {
                p4[i * dim + j] -= p4[j * dim + i];
            }
        }
        for (i = 0; i < dim; i++)
        {
            for (j = 0; j < i; j++)
            {
                p4[i * dim + j] = -p4[j * dim + i];
            }
        }
        return;
    }

    void MKLDoubleCommutator(ComplexNum *a, ComplexNum *b, ComplexNum *c, ComplexNum *p4, int dim) // p1 p2 p3 p4
    {
        int i, j;
        ComplexNum temp1[dim * dim];
        ComplexNum temp2[dim * dim];
        MKLMatrixMultiply(a, b, temp1, dim, 1.);
        MKLMatrixMultiply(temp1, c, temp2, dim, 4.);
        Transpose(temp2, temp1, dim, -1.);
        vzAdd(dim * dim, temp1, temp2, p4);
        return;
    }

    void PairOBOCommutator(ComplexNum *a, ComplexNum *Q, ComplexNum *b, int dim)
    {
        int i, j;
        MKLMatrixMultiply(a, Q, b, dim, 1.);
        for (i = 0; i < dim; i++)
        {
            for (j = i; j < dim; j++)
            {
                b[i * dim + j] -= b[j * dim + i];
            }
        }
        for (i = 0; i < dim; i++)
        {
            for (j = 0; j < i; j++)
            {
                b[i * dim + j] = -b[j * dim + i];
            }
        }
        return;
    }

    void MKLPairOBOCommutator(ComplexNum *a, const ComplexNum *Q, ComplexNum *b, int dim)
    {
        int i, j;
        ComplexNum temp1[dim * dim];
        MKLMatrixMultiply(a, Q, temp1, dim, 1.);
        ComplexNum temp2[dim * dim];
        Transpose(temp1, temp2, dim, -1.);
        vzAdd(dim * dim, temp1, temp2, b);
        return;
    }

    void MKLPairOBOCommutator_antiSym(ComplexNum *a, const ComplexNum *Q, ComplexNum *b, int dim)
    {
        int i, j;
        ComplexNum temp1[dim * dim];
        MKLMatrixMultiply(a, Q, temp1, dim, 1.);
        ComplexNum temp2[dim * dim];
        Transpose(temp1, temp2, dim, -1.);
        vzAdd(dim * dim, temp1, temp2, b);

        for (i = 0; i < dim; i++)
        {
            for (j = 0; j < i; j++)
            {
                if (std::abs(b[i * dim + j] + b[j * dim + i]) > 1.e-6)
                {
                    std::cout << i << "  " << j << "  " << b[i * dim + j] << "  " << b[j * dim + i] << std::endl;
                }
            }
        }
        return;
    }

    ComplexNum Cal_Overlap(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR)
    {
        int isospin = ytabL.GetIsospin();
        int dim = ytabL.GetDim();
        ///////////////////////////////////////
        /////////     N = 1   term    /////////
        ///////////////////////////////////////
        if (N == 1)
        {
            return PairPairContractor(ytabL.GetPointer(N), ytabR.GetPointer(N), dim);
        }
        ///////////////////////////////////////
        /////////   the first term    /////////
        ///////////////////////////////////////
        int k, i, j;
        ComplexNum sum, temp;
        MultiPairs ytab_new(N, ms, isospin);
        sum = 0.;
        //////////////////////////////////////
        MatrixCope(ytab_new, ytabL); // copy all matrix
        for (k = N; k > 0; k--)      // here k indicate contracting k-th term
        {
            MatrixCope(ytab_new, k, N - 1, ytabL, k + 1);
            // temp = PairPairContractor(ytabL, k, ytabR, N);
            temp = PairPairContractor(ytabL.GetPointer(k), ytabR.GetPointer(N), dim);
            sum += temp * Cal_Overlap(N - 1, ms, ytab_new, ytabR);
        }
        ///////////////////////////////////////
        ////////   the second term     ////////
        ///////////////////////////////////////
        for (k = N; k > 1; k--) // index k
        {
            MatrixCope(ytab_new, k, N - 1, ytabL, k + 1); // inital pairs k <-> N-1
            for (i = k - 1; i > 0; i--)                   // index i
            {
                MatrixCope(ytab_new, 1, k - 1, ytabL, 1); // inital pairs 1 <-> k
                // DoubleCommutator(ytabL.GetPointer(k), ytabR.GetPointer(N), ytabL.GetPointer(i), ytab_new.GetPointer(i), dim);
                MKLDoubleCommutator(ytabL.GetPointer(k), ytabR.GetPointer(N), ytabL.GetPointer(i), ytab_new.GetPointer(i), dim);
                // old_DoubleCommutator(ytabL.GetPointer(k), ytabR.GetPointer(N), ytabL.GetPointer(i), ytab_new.GetPointer(i), dim);
                sum += Cal_Overlap(N - 1, ms, ytab_new, ytabR);
            } // index i
        }     // index k
        // ytab_new.FreeMemory();
        return sum;
    }

    void GeneratePairOperator_p_Anitsym(ModelSpace *ms, int isospin, int J, int m, int a, int b, ComplexNum *V_A) /// (C_a X C_b)^J_M = C^{JM}_{ja ma jb mb} Ca Cb
    {
        int i, j, ja, jb;
        int SPja, SPjb;
        int ma, mb;
        double factor1 = 1.; // anti-syme
        double factor2 = 0;
        int dim = ms->Get_MScheme_dim(isospin);
        MSchemeMatrix *MSMprt = ms->GetMSmatrixPointer(isospin);
        ja = ms->GetOrbit_2j(a, isospin);
        jb = ms->GetOrbit_2j(b, isospin);
        if (a != b)
        {
            factor1 = 1;
            factor2 = -1 * sgn(J + (ja + jb) / 2);
        }
        memset(V_A, 0, dim * dim * 2 * sizeof(double));
        SPja = ms->LookupStartingPoint(isospin, a);
        SPjb = ms->LookupStartingPoint(isospin, b);
        for (i = SPja; i <= SPja + ja; i += 1)
        {
            for (j = SPjb; j <= SPjb + jb; j += 1)
            {
                if (MSMprt->m[i] + MSMprt->m[j] == 2 * m)
                {
                    V_A[i * dim + j] = factor1 * ms->Get_CGC(isospin, J, m, i, j);
                    if (a != b)
                        V_A[j * dim + i] = factor2 * ms->Get_CGC(isospin, J, m, j, i);
                }
            }
        }
    }

    ComplexNum Prepare_pairingME(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        ComplexNum *A_dagger, *A_annihilation; // for interaction matrix
        ComplexNum res;
        A_dagger = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        A_annihilation = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        int a, b, c, d, J;
        double TBME;
        a = Ham.Vpp[Hindex].j1;
        b = Ham.Vpp[Hindex].j2;
        c = Ham.Vpp[Hindex].j3;
        d = Ham.Vpp[Hindex].j4;
        J = Ham.Vpp[Hindex].J;
        TBME = Ham.Vpp[Hindex].V;
        GeneratePairOperator_p_Anitsym(&ms, isospin, J, m, a, b, A_dagger);
        GeneratePairOperator_p_Anitsym(&ms, isospin, J, m, c, d, A_annihilation);

        // Cal ME
        res = TBME * CalPairingME(N, ms, ytabL, ytabR, A_dagger, A_annihilation);
        //////////////////////////////////////////
        mkl_free(A_dagger);
        mkl_free(A_annihilation);
        return res;
    }

    ComplexNum Prepare_pairingME_SamePairs(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        ComplexNum *A_dagger, *A_annihilation; // for interaction matrix
        ComplexNum res;
        A_dagger = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        A_annihilation = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        int a, b, c, d, J;
        double TBME;
        a = Ham.Vpp[Hindex].j1;
        b = Ham.Vpp[Hindex].j2;
        c = Ham.Vpp[Hindex].j3;
        d = Ham.Vpp[Hindex].j4;
        J = Ham.Vpp[Hindex].J;
        TBME = Ham.Vpp[Hindex].V;
        GeneratePairOperator_p_Anitsym(&ms, isospin, J, m, a, b, A_dagger);
        GeneratePairOperator_p_Anitsym(&ms, isospin, J, m, c, d, A_annihilation);

        // Cal ME
        res = TBME * CalPairingME_SamePairs(N, ms, ytabL, ytabR, A_dagger, A_annihilation);
        //////////////////////////////////////////
        mkl_free(A_dagger);
        mkl_free(A_annihilation);
        return res;
    }

    ComplexNum CalPairingME(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, ComplexNum *A_dagger, ComplexNum *A_annihilation)
    {
        ComplexNum sum, temp;
        int k, i, j;
        int isospin = ytabL.GetIsospin();
        int dim = ytabL.GetDim();
        sum = 0;
        MultiPairs ytab_new(N, ms, isospin);
        ///////////////////////////////////////
        /////////   the first term    /////////
        ///////////////////////////////////////
        MatrixCope(ytab_new, ytabL); // copy all matrix
        for (k = N; k > 0; k--)
        {
            temp = PairPairContractor(ytabL.GetPointer(k), A_dagger, dim);
            MatrixCope(ytab_new.GetPointer(k), A_annihilation, dim * dim);
            MatrixCope(ytab_new, k + 1, N, ytabL, k + 1);
            sum += temp * Cal_Overlap(N, ms, ytab_new, ytabR);
        }
        ///////////////////////////////////////
        ////////   the second term     ////////
        ///////////////////////////////////////
        MatrixCope(ytab_new, ytabL); // copy all matrix
        for (k = N; k > 1; k--)      // index k
        {
            if (k != N) // recover k+1 pair
            {
                MatrixCope(ytab_new.GetPointer(k + 1), ytabL.GetPointer(k + 1), dim * dim);
            }
            MatrixCope(ytab_new.GetPointer(k), A_annihilation, dim * dim);
            for (i = k - 1; i > 0; i--) // index i
            {
                MatrixCope(ytab_new, 1, k - 1, ytabL, 1);
                // DoubleCommutator(ytabL.GetPointer()[k], A_dagger, ytabL.GetPointer()[i], ytab_new.GetPointer()[i], dim);
                MKLDoubleCommutator(ytabL.GetPointer(k), A_dagger, ytabL.GetPointer(i), ytab_new.GetPointer(i), dim);
                sum += Cal_Overlap(N, ms, ytab_new, ytabR);
            } // i
        }     // k
        // ytab_new.FreeMemory();
        return sum;
    }

    ComplexNum CalPairingME_SamePairs(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, ComplexNum *A_dagger, ComplexNum *A_annihilation)
    {
        ComplexNum sum, temp;
        int k, i, j;
        int isospin = ytabL.GetIsospin();
        int dim = ytabL.GetDim();
        sum = 0;
        MultiPairs ytab_new(N, ms, isospin);
        ///////////////////////////////////////
        /////////   the first term    /////////
        ///////////////////////////////////////
        MatrixCope(ytab_new, ytabL); // copy all matrix
        temp = PairPairContractor(ytabL.GetPointer(N), A_dagger, dim);
        MatrixCope(ytab_new.GetPointer(N), A_annihilation, dim * dim);
        sum += temp * Cal_Overlap(N, ms, ytab_new, ytabR) * (double)N;

        ///////////////////////////////////////
        ////////   the second term     ////////
        ///////////////////////////////////////
        // MatrixCope(ytab_new, ytabL); // copy all matrix
        // MatrixCope(ytab_new.GetPointer(N), A_annihilation, dim * dim);
        if (N > 1)
        {
            // DoubleCommutator(ytabL.GetPointer()[N], A_dagger, ytabL.GetPointer()[N-1], ytab_new.GetPointer()[N-1], dim);
            MKLDoubleCommutator(ytabL.GetPointer(N), A_dagger, ytabL.GetPointer(N - 1), ytab_new.GetPointer(N - 1), dim);
            sum += 0.5 * N * (N - 1) * Cal_Overlap(N, ms, ytab_new, ytabR);
        }
        return sum;
    }

    ComplexNum CalOneBodyOperator(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, const ComplexNum *OB_operator)
    {
        int k, j;
        ComplexNum sum = 0;
        int isospin = ytabL.GetIsospin();
        int dim = ytabL.GetDim();
        MultiPairs ytab_new(N, ms, isospin);
        MatrixCope(ytab_new, ytabL); // copy all matrix
        for (k = N; k >= 1; k--)
        {
            if (k != N)
            {
                MatrixCope(ytab_new.GetPointer(k + 1), ytabL.GetPointer(k + 1), dim * dim);
            }
            // PairOBOCommutator(ytabL.GetPointer()[k], OB_operator, ytab_new.GetPointer()[k], dim);
            MKLPairOBOCommutator(ytabL.GetPointer(k), OB_operator, ytab_new.GetPointer(k), dim);
            // MKLPairOBOCommutator_antiSym(ytabL.GetPointer(k), OB_operator, ytab_new.GetPointer(k), dim);
            sum += Cal_Overlap(N, ms, ytab_new, ytabR);
        }
        return sum;
    }

    ComplexNum CalOneBodyOperator_SamePairs(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, const ComplexNum *OB_operator)
    {
        int k, j;
        ComplexNum sum = 0;
        int isospin = ytabL.GetIsospin();
        int dim = ytabL.GetDim();
        MultiPairs ytab_new(N, ms, isospin);
        MatrixCope(ytab_new, ytabL); // copy all matrix
        // PairOBOCommutator(ytabL.GetPointer()[k], OB_operator, ytab_new.GetPointer()[k], dim);
        MKLPairOBOCommutator(ytabL.GetPointer(N), OB_operator, ytab_new.GetPointer(N), dim);
        // MKLPairOBOCommutator_antiSym(ytabL.GetPointer(k), OB_operator, ytab_new.GetPointer(k), dim);
        sum += Cal_Overlap(N, ms, ytab_new, ytabR) * (double)N;
        return sum;
    }

    void FullAAMatrix_p(vector<HamiltoaninColllectiveElements> *Hele, ModelSpace &ms, ComplexNum *ColPair_A, int isospin, int Hindex, int m, int dim)
    {
        int i, j, t;
        int JschemeDim = ms.GetOrbitsNumber(isospin);
        std::vector<double> *CGC_prt = ms.GetCGC_prt(isospin);
        std::vector<double> *Y_pair = Hele->at(Hindex).Get_y_prt();
        t = Hele->at(Hindex).J;
        int SP = ms.Get_CGC_StartPoint(isospin, t, m);
        memset(ColPair_A, 0, sizeof(double) * 2 * dim * dim);
        for (i = 0; i < dim; i++)
        {
            for (j = 0; j < dim; j++)
            {
                double CGC = (*CGC_prt)[SP + i * dim + j];
                double Ypair = (*Y_pair)[ms.Get_ProtonOrbitIndexInMscheme(i) * JschemeDim + ms.Get_ProtonOrbitIndexInMscheme(j)];
                ColPair_A[i * dim + j] += CGC * Ypair;
            }
        }
    }

    ComplexNum Prepare_Collective_pairingME(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        ComplexNum *A_dagger, *A_annihilation; // for interaction matrix
        ComplexNum res;
        vector<HamiltoaninColllectiveElements> *HME = Ham.GetColMatrixEle_prt(isospin);
        A_dagger = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        A_annihilation = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        FullAAMatrix_p(HME, ms, A_dagger, isospin, Hindex, m, dim);
        MatrixCope(A_annihilation, A_dagger, dim2);
        // Cal ME
        res = HME->at(Hindex).Sign;
        res *= CalPairingME(N, ms, ytabL, ytabR, A_dagger, A_annihilation);
        //////////////////////////////////////////
        mkl_free(A_dagger);
        mkl_free(A_annihilation);
        return res;
    }

    ComplexNum Prepare_Collective_pairingME_SamePairs(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        ComplexNum *A_dagger, *A_annihilation; // for interaction matrix
        ComplexNum res;
        vector<HamiltoaninColllectiveElements> *HME = Ham.GetColMatrixEle_prt(isospin);
        A_dagger = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        A_annihilation = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        FullAAMatrix_p(HME, ms, A_dagger, isospin, Hindex, m, dim);
        MatrixCope(A_annihilation, A_dagger, dim2);
        // Cal ME
        res = HME->at(Hindex).Sign;
        res *= CalPairingME_SamePairs(N, ms, ytabL, ytabR, A_dagger, A_annihilation);
        //////////////////////////////////////////
        mkl_free(A_dagger);
        mkl_free(A_annihilation);
        return res;
    }

    void GenerateQO(ModelSpace *ms, int isospin, int t, int m, int a, int b, ComplexNum *Q_operator)
    {

        int i, j, ja, jb;
        int SPja, SPjb;
        int ma, mb;
        int dim = ms->Get_MScheme_dim(isospin);
        double factor, cgc;
        MSchemeMatrix *MSMprt = ms->GetMSmatrixPointer(isospin);
        ja = ms->GetOrbit_2j(a, isospin);
        jb = ms->GetOrbit_2j(b, isospin);
        memset(Q_operator, 0, dim * dim * 2 * sizeof(double));
        SPja = ms->LookupStartingPoint(isospin, a);
        SPjb = ms->LookupStartingPoint(isospin, b);
        for (i = SPja; i <= SPja + ja; i += 1)
        {
            for (j = SPjb; j <= SPjb + jb; j += 1)
            {
                if (MSMprt->m[i] - MSMprt->m[j] == 2 * m)
                {
                    factor = sgn((jb + MSMprt->m[j]) / 2);
                    cgc = ms->Get_CGC(isospin, t, m, i, 2 * SPjb + jb - j);
                    // cgc = AngMom::cgc(t * 1., m * 1., ja / 2., MSMprt->m[i] / 2., jb / 2., -MSMprt->m[j] / 2.);
                    Q_operator[i * dim + j] = factor * cgc;
                }
            }
        }
    }

    ComplexNum Prepare_OB_ME(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        int index_a, index_b, t;
        ComplexNum *Q_operator; // for interaction matrix
        ComplexNum res;

        OneBodyOperatorChannel OP = Ham.GetOneBodyOperator(isospin, Hindex);
        index_a = OP.GetIndex_a();
        index_b = OP.GetIndex_b();
        t = OP.Get_t();
        Q_operator = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        GenerateQO(&ms, isospin, t, m, index_a, index_b, Q_operator);

        // Cal ME
        res = CalOneBodyOperator(N, ms, ytabL, ytabR, Q_operator);
        //////////////////////////////////////////
        mkl_free(Q_operator);
        return res;
    }

    ComplexNum Prepare_OB_ME_SamePairs(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        int index_a, index_b, t;
        ComplexNum *Q_operator; // for interaction matrix
        ComplexNum res;

        OneBodyOperatorChannel OP = Ham.GetOneBodyOperator(isospin, Hindex);
        index_a = OP.GetIndex_a();
        index_b = OP.GetIndex_b();
        t = OP.Get_t();
        Q_operator = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        GenerateQO(&ms, isospin, t, m, index_a, index_b, Q_operator);

        // Cal ME
        res = CalOneBodyOperator_SamePairs(N, ms, ytabL, ytabR, Q_operator);
        //////////////////////////////////////////
        mkl_free(Q_operator);
        return res;
    }
    //////////////////////////////////////////////////////// For parity projection
    ComplexNum Cal_Overlap_parity(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity)
    {
        ComplexNum res;
        res = Cal_Overlap(N, ms, ytabL, ytabR);
        res += ms.GetProjected_parity() * Cal_Overlap(N, ms, ytabL, ytabR_parity);
        return res * 0.5;
    }

    ComplexNum CalOneBodyOperator_parity(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, ComplexNum *OB_operator)
    {
        ComplexNum res;
        res = CalOneBodyOperator(N, ms, ytabL, ytabR, OB_operator);
        res += ms.GetProjected_parity() * CalOneBodyOperator(N, ms, ytabL, ytabR_parity, OB_operator);
        return res * 0.5;
    }

    ComplexNum CalOneBodyOperator_parity_SamePairs(int N, ModelSpace &ms, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, ComplexNum *OB_operator)
    {
        ComplexNum res;
        res = CalOneBodyOperator_SamePairs(N, ms, ytabL, ytabR, OB_operator);
        res += ms.GetProjected_parity() * CalOneBodyOperator_SamePairs(N, ms, ytabL, ytabR_parity, OB_operator);
        return res * 0.5;
    }

    ComplexNum Prepare_Collective_pairingME_parity(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        ComplexNum *A_dagger, *A_annihilation; // for interaction matrix
        ComplexNum res, Sign;
        vector<HamiltoaninColllectiveElements> *HME = Ham.GetColMatrixEle_prt(isospin);
        A_dagger = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        A_annihilation = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        FullAAMatrix_p(HME, ms, A_dagger, isospin, Hindex, m, dim);
        MatrixCope(A_annihilation, A_dagger, dim2);
        // Cal ME
        Sign = HME->at(Hindex).Sign;
        res = CalPairingME(N, ms, ytabL, ytabR, A_dagger, A_annihilation);
        res += ms.GetProjected_parity() * CalPairingME(N, ms, ytabL, ytabR_parity, A_dagger, A_annihilation);
        //////////////////////////////////////////
        mkl_free(A_dagger);
        mkl_free(A_annihilation);
        return res * 0.5 * Sign;
    }

    ComplexNum Prepare_pairingME_parity(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, int Hindex, int m)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        ComplexNum *A_dagger, *A_annihilation; // for interaction matrix
        ComplexNum res;
        A_dagger = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        A_annihilation = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        int a, b, c, d, J;
        double TBME;
        a = Ham.Vpp[Hindex].j1;
        b = Ham.Vpp[Hindex].j2;
        c = Ham.Vpp[Hindex].j3;
        d = Ham.Vpp[Hindex].j4;
        J = Ham.Vpp[Hindex].J;
        TBME = Ham.Vpp[Hindex].V;
        GeneratePairOperator_p_Anitsym(&ms, isospin, J, m, a, b, A_dagger);
        GeneratePairOperator_p_Anitsym(&ms, isospin, J, m, c, d, A_annihilation);

        // Cal ME
        res = CalPairingME(N, ms, ytabL, ytabR, A_dagger, A_annihilation);
        res += ms.GetProjected_parity() * CalPairingME(N, ms, ytabL, ytabR, A_dagger, A_annihilation);
        //////////////////////////////////////////
        mkl_free(A_dagger);
        mkl_free(A_annihilation);
        return 0.5 * res * TBME;
    }

    void Prepare_OB_ME_parity(int N, ModelSpace &ms, Hamiltonian &Ham, MultiPairs &ytabL, MultiPairs &ytabR, MultiPairs &ytabR_parity, int Hindex, int m, ComplexNum *res)
    {
        int isospin = ytabL.GetIsospin();
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        int index_a, index_b, t;
        ComplexNum *Q_operator; // for interaction matrix

        OneBodyOperatorChannel OP = Ham.GetOneBodyOperator(isospin, Hindex);
        index_a = OP.GetIndex_a();
        index_b = OP.GetIndex_b();
        t = OP.Get_t();
        Q_operator = (ComplexNum *)mkl_malloc(dim2 * sizeof(ComplexNum), 64);
        GenerateQO(&ms, isospin, t, m, index_a, index_b, Q_operator);

        // Cal ME
        res[0] = CalOneBodyOperator(N, ms, ytabL, ytabR, Q_operator);
        res[1] = CalOneBodyOperator(N, ms, ytabL, ytabR_parity, Q_operator);
        //////////////////////////////////////////
        mkl_free(Q_operator);
    }

}
