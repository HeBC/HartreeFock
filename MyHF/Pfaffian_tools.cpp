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

#include "Pfaffian_tools.h"
namespace HF_Pfaffian_Tools
{
    ///////////////////// Matrix elements ///////////////////////
    // Overlap
    void S_matrix_overlap(HFbasis &Bra, HFbasis &Ket, double *M)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N2 = 2 * N;
        const double *BraMatrix, *KetMatrix;
        double value;
        BraMatrix = Bra.GetArrayPointerDouble();
        KetMatrix = Ket.GetArrayPointerDouble();
        memset(M, 0, N2 * N2 * sizeof(double));
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                value = cblas_ddot(dim, BraMatrix + (N - i - 1) * dim, 1, KetMatrix + j * dim, 1);
                M[i * N2 + j + N] = value;
                M[(j + N) * N2 + i] = -value;
            }
        }
    }

    void S_matrix_overlap_complex(HFbasis &Bra, HFbasis &Ket, ComplexNum *M)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N2 = 2 * N;
        const ComplexNum *BraMatrix, *KetMatrix;
        ComplexNum value;
        BraMatrix = Bra.GetArrayPointerComplex();
        KetMatrix = Ket.GetArrayPointerComplex();
        memset(M, 0, N2 * N2 * 2 * sizeof(double));
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                // value = cblas_ddot(dim, BraMatrix + (N - i - 1) * dim, 1, KetMatrix + j * dim, 1);
                // res = \sum_i  xi * yi
                cblas_zdotu_sub(dim, BraMatrix + (N - i - 1) * dim, 1, KetMatrix + j * dim, 1, &value);
                M[i * N2 + j + N] = value;
                M[(j + N) * N2 + i] = -value;
            }
        }
    }

    void S_matrix_overlap(HFbasis &Bra, HFbasis &Ket, double *M, int inner_dim)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N2 = 2 * N;
        const double *BraMatrix, *KetMatrix;
        double value;
        BraMatrix = Bra.GetArrayPointerDouble();
        KetMatrix = Ket.GetArrayPointerDouble();
        memset(M, 0, N2 * N2 * sizeof(double));
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                value = cblas_ddot(dim * inner_dim, BraMatrix + (N - i - 1) * dim * inner_dim, 1, KetMatrix + j * dim * inner_dim, 1);
                M[i * N2 + j + N] = value;
                M[(j + N) * N2 + i] = -value;
            }
        }
    }

    // M matrix for Onebody operator <a|O_ij|b> = <a| C^+_i C_j |b>
    // the overlap part should be evaluated separately
    void S_matrix_OneBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N22 = 2 * N + 2;
        const double *BraMatrix, *KetMatrix;
        double value;
        BraMatrix = Bra.GetArrayPointerDouble();
        KetMatrix = Ket.GetArrayPointerDouble();
        for (size_t a = 0; a < N; a++)
        {
            /// C^+_i
            value = BraMatrix[(N - a - 1) * dim + i];
            M[a * N22 + N] = value;
            M[N * N22 + a] = -value;
            // std::cout << value << std::endl;
            ///_________________________
            /// C_j
            value = KetMatrix[a * dim + j];
            M[(N + 1) * N22 + a + N + 2] = value;
            M[(a + N + 2) * N22 + N + 1] = -value;
            // std::cout << value << std::endl;
        }
    }

    void S_matrix_OneBody_MEs_complex(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, int i, int j)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N22 = 2 * N + 2;
        const ComplexNum *BraMatrix, *KetMatrix;
        ComplexNum value;
        BraMatrix = Bra.GetArrayPointerComplex();
        KetMatrix = Ket.GetArrayPointerComplex();
        for (size_t a = 0; a < N; a++)
        {
            /// C^+_i
            value = BraMatrix[(N - a - 1) * dim + i];
            M[a * N22 + N] = value;
            M[N * N22 + a] = -value;
            // std::cout << value << std::endl;
            ///_________________________
            /// C_j
            value = KetMatrix[a * dim + j];
            M[(N + 1) * N22 + a + N + 2] = value;
            M[(a + N + 2) * N22 + N + 1] = -value;
            // std::cout << value << std::endl;
        }
    }

    void S_matrix_OneBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j, int inner_dim)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N22 = 2 * N + 2;
        const double *BraMatrix, *KetMatrix;
        double value;
        BraMatrix = Bra.GetArrayPointerDouble();
        KetMatrix = Ket.GetArrayPointerDouble();
        for (size_t a = 0; a < N; a++)
        {
            /// C^+_i
            value = BraMatrix[(N - a - 1) * dim * inner_dim + i * inner_dim];
            M[a * N22 + N] = value;
            M[N * N22 + a] = -value;
            // std::cout << value << std::endl;
            ///_________________________
            /// C_j
            value = KetMatrix[a * dim * inner_dim + j * inner_dim];
            M[(N + 1) * N22 + a + N + 2] = value;
            M[(a + N + 2) * N22 + N + 1] = -value;
            // std::cout << value << std::endl;
        }
    }

    // full overlap for S matrix of OneBody operator
    void S_matrix_FullOvl_OneBody(HFbasis &Bra, HFbasis &Ket, double *M, double *M_ovl)
    {
        int N = Bra.GetParticleNumber();
        int Np2 = N + 2;
        int N2 = 2 * N;
        int N22 = 2 * N + 2;
        for (size_t a = 0; a < N; a++)
        {
            for (size_t b = 0; b < N; b++)
            {
                M[a * N22 + b + Np2] = M_ovl[a * N2 + b + N];
                M[(b + Np2) * N22 + a] = M_ovl[(b + N) * N2 + a];
            }
        }
    }

    void S_matrix_FullOvl_OneBody(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, ComplexNum *M_ovl)
    {
        int N = Bra.GetParticleNumber();
        int Np2 = N + 2;
        int N2 = 2 * N;
        int N22 = 2 * N + 2;
        for (size_t a = 0; a < N; a++)
        {
            for (size_t b = 0; b < N; b++)
            {
                M[a * N22 + b + Np2] = M_ovl[a * N2 + b + N];
                M[(b + Np2) * N22 + a] = M_ovl[(b + N) * N2 + a];
            }
        }
    }
    // End OneBody

    // S matrix for Twobody operator <a|C^+_i C^+_j C_k C_l|b>
    // the overlap part should be evaluated separately
    void S_matrix_TwoBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j, int k, int l)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N24 = 2 * N + 4;
        const double *BraMatrix, *KetMatrix;
        double value;
        BraMatrix = Bra.GetArrayPointerDouble();
        KetMatrix = Ket.GetArrayPointerDouble();
        for (size_t a = 0; a < N; a++)
        {
            /// C^+_i
            value = BraMatrix[(N - a - 1) * dim + i];
            M[a * N24 + N] = value;
            M[N * N24 + a] = -value;
            ///_________________________
            /// C^+_j
            value = BraMatrix[(N - a - 1) * dim + j];
            M[a * N24 + N + 1] = value;
            M[(N + 1) * N24 + a] = -value;
            ///_________________________
            /// C_k
            value = KetMatrix[a * dim + k];
            M[(N + 2) * N24 + a + N + 4] = value;
            M[(a + N + 4) * N24 + N + 2] = -value;
            ///_________________________
            /// C_l
            value = KetMatrix[a * dim + l];
            M[(N + 3) * N24 + a + N + 4] = value;
            M[(a + N + 4) * N24 + N + 3] = -value;
        }
    }

    void S_matrix_TwoBody_MEs_complex(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, int i, int j, int k, int l)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N24 = 2 * N + 4;
        const ComplexNum *BraMatrix, *KetMatrix;
        ComplexNum value;
        BraMatrix = Bra.GetArrayPointerComplex();
        KetMatrix = Ket.GetArrayPointerComplex();
        for (size_t a = 0; a < N; a++)
        {
            /// C^+_i
            value = BraMatrix[(N - a - 1) * dim + i];
            M[a * N24 + N] = value;
            M[N * N24 + a] = -value;
            /// std::cout<< value << std::endl;
            ///_________________________
            /// C^+_j
            value = BraMatrix[(N - a - 1) * dim + j];
            M[a * N24 + N + 1] = value;
            M[(N + 1) * N24 + a] = -value;
            ///_________________________
            /// C_k
            value = KetMatrix[a * dim + k];
            M[(N + 2) * N24 + a + N + 4] = value;
            M[(a + N + 4) * N24 + N + 2] = -value;
            ///_________________________
            /// C_l
            value = KetMatrix[a * dim + l];
            M[(N + 3) * N24 + a + N + 4] = value;
            M[(a + N + 4) * N24 + N + 3] = -value;
        }
    }

    void S_matrix_TwoBody_MEs(HFbasis &Bra, HFbasis &Ket, double *M, int i, int j, int k, int l, int inner_dim)
    {
        int dim = Bra.GetDim();
        int N = Bra.GetParticleNumber();
        int N24 = 2 * N + 4;
        const double *BraMatrix, *KetMatrix;
        double value;
        BraMatrix = Bra.GetArrayPointerDouble();
        KetMatrix = Ket.GetArrayPointerDouble();
        for (size_t a = 0; a < N; a++)
        {
            /// C^+_i
            value = BraMatrix[(N - a - 1) * dim * inner_dim + i * inner_dim];
            M[a * N24 + N] = value;
            M[N * N24 + a] = -value;
            ///_________________________
            /// C^+_j
            value = BraMatrix[(N - a - 1) * dim * inner_dim + j * inner_dim];
            M[a * N24 + N + 1] = value;
            M[(N + 1) * N24 + a] = -value;
            ///_________________________
            /// C_k
            value = KetMatrix[a * dim * inner_dim + k * inner_dim];
            M[(N + 2) * N24 + a + N + 4] = value;
            M[(a + N + 4) * N24 + N + 2] = -value;
            ///_________________________
            /// C_l
            value = KetMatrix[a * dim * inner_dim + l * inner_dim];
            M[(N + 3) * N24 + a + N + 4] = value;
            M[(a + N + 4) * N24 + N + 3] = -value;
        }
    }

    // S matrix for TwoBody operator
    // the array M should be zero first!
    void S_matrix_FullOvl_TwoBody(HFbasis &Bra, HFbasis &Ket, double *M, double *M_ovl)
    {
        int N = Bra.GetParticleNumber();
        int Np4 = N + 4;
        int N2 = 2 * N;
        int N24 = 2 * N + 4;
        double value;
        // memset(M, 0, N22 * N22 * sizeof(double));
        for (size_t a = 0; a < N; a++)
        {
            for (size_t b = 0; b < N; b++)
            {
                M[a * N24 + b + Np4] = M_ovl[a * N2 + b + N];
                M[(b + Np4) * N24 + a] = M_ovl[(b + N) * N2 + a];
            }
        }
    }

    void S_matrix_FullOvl_TwoBody(HFbasis &Bra, HFbasis &Ket, ComplexNum *M, ComplexNum *M_ovl)
    {
        int N = Bra.GetParticleNumber();
        int Np4 = N + 4;
        int N2 = 2 * N;
        int N24 = 2 * N + 4;
        ComplexNum value;
        // memset(M, 0, N22 * N22 * sizeof(double));
        for (size_t a = 0; a < N; a++)
        {
            for (size_t b = 0; b < N; b++)
            {
                M[a * N24 + b + Np4] = M_ovl[a * N2 + b + N];
                M[(b + Np4) * N24 + a] = M_ovl[(b + N) * N2 + a];
            }
        }
    }
    // End Twobody

    ///////////////////// Pfaffian codes ///////////////////////
    // out-place Pfaffian
    // return the pfaffian value of the matrix S, N is the dim of S
    // J.R. Stembridge, Adv. Math. 83 (1990) 96.
    //       https://www.sciencedirect.com/science/article/pii/0001870890900704
    // M. Ishikawa, M. Wakayama, J. Comb. Theory, Ser. A 88 (1999) 136.
    //       https://www.sciencedirect.com/science/article/pii/S0097316599929886
    double PFAFFIAN(int N, const double *Smatrix)
    {
        double SH = 1.0;
        double PF;
        double *X, *S, *A;
        A = (double *)mkl_malloc((N * N) * sizeof(double), 64);
        std::memcpy(A, Smatrix, (N * N) * sizeof(double)); // copy the contents of src to dest
        X = (double *)mkl_malloc((N) * sizeof(double), 64);
        S = (double *)mkl_malloc((N) * sizeof(double), 64);
        for (int I = 0; I < N - 2; I += 2)
        {
            int JINI = I + 1;
            SH = -SH;
            memset(X, 0, N * sizeof(double));
            // X[I] = 0.;
            double XM = 0.0;
            for (int J = JINI; J < N; J++)
            {
                X[J] = A[I * N + J];
                XM += X[J] * X[J];
            }

            if (XM >= 1e-10)
            {
                XM = sqrt(XM);
                double ARG;
                if (abs(X[JINI]) < 1.e-13)
                {
                    ARG = 1.0;
                }
                else
                {
                    ARG = X[JINI] / abs(X[JINI]);
                }

                // X -> U vector
                if (abs(X[JINI] + ARG * XM) > abs(X[JINI] - ARG * XM))
                {
                    X[JINI] = X[JINI] + ARG * XM;
                }
                else
                {
                    X[JINI] = X[JINI] - ARG * XM;
                }

                double UM = 0.0;
                for (int J = JINI; J < N; J++)
                {
                    UM += X[J] * X[J];
                }
                UM = UM / 2.0;

                memset(S, 0, N * sizeof(double));
                for (int J = I; J < N; J++)
                {
                    S[J] = 0.0;
                    for (int K = I; K < N; K++)
                    {
                        S[J] += A[K * N + J] * X[K];
                    }
                    S[J] = S[J] / UM;
                }

                for (int J = I; J < N; J++)
                {
                    for (int K = J + 1; K < N; K++)
                    {
                        A[J * N + K] += S[J] * X[K] - X[J] * S[K];
                        A[K * N + J] = -A[J * N + K];
                    }
                }
            }
        }
        PF = 1.0;
        for (int I = 0; I < N; I += 2)
        {
            PF = PF * A[I * N + I + 1];
            if (abs(PF) < 1e-200)
            {
                PF = PF / 1e-200;
            }
            if (abs(PF) > 1e200)
            {
                PF = PF / 1e200;
            }
        }
        PF = PF * SH;
        mkl_free(X);
        mkl_free(A);
        mkl_free(S);
        return PF;
    }

    // This code is found from Dr. Gao's paper
    // more detail can be found in the Supplementary material at
    //       http://dx.doi.org/10.1016/j.physletb.2014.05.045.
    // Qing-Li Hu, Zao-Chun Gao ∗, Y.S. Chen, Physics Letters B 734 (2014) 162–166
    //       https://www.sciencedirect.com/science/article/pii/S0370269314003542?via%3Dihub
    ComplexNum PFAFFIAN(int N, const ComplexNum *Smatrix)
    {
        ComplexNum SH = 1.0;
        ComplexNum PF, ARG;
        ComplexNum *X, *S, *A;
        A = (ComplexNum *)mkl_malloc((N * N) * sizeof(ComplexNum), 64);
        std::memcpy(A, Smatrix, (N * N) * 2 * sizeof(double)); // copy the contents of src to dest
        X = (ComplexNum *)mkl_malloc((N) * sizeof(ComplexNum), 64);
        S = (ComplexNum *)mkl_malloc((N) * sizeof(ComplexNum), 64);
        memset(X, 0, N * 2 * sizeof(double));
        memset(S, 0, N * 2 * sizeof(double));
        for (int I = 0; I < N - 2; I += 2)
        {
            int JINI = I + 1;
            SH = -SH;
            memset(X, 0, N * 2 * sizeof(double));
            double XM = 0.0;
            for (int J = JINI; J < N; J++)
            {
                X[J] = A[I * N + J];
                XM += (X[J] * std::conj(X[J])).real();
            }
            if (XM < 1.0e-10)
                break;
            XM = std::sqrt(XM);
            if (std::abs(X[JINI]) < 1.e-13)
            {
                ARG = 1.0;
            }
            else
            {
                ARG = X[JINI] / std::abs(X[JINI]);
            }

            // X -> U vector
            if (std::abs(X[JINI] + ARG * XM) > std::abs(X[JINI] - ARG * XM))
            {
                X[JINI] = X[JINI] + ARG * XM;
            }
            else
            {
                X[JINI] = X[JINI] - ARG * XM;
            }

            double UM = 0.0;
            for (int J = JINI; J < N; ++J)
            {
                UM += (X[J] * std::conj(X[J])).real();
            }
            UM /= 2.0;

            memset(S, 0, N * 2 * sizeof(double));
            for (int J = I; J < N; J++)
            {
                S[J] = 0.0;
                for (int K = I; K < N; K++)
                {
                    S[J] += A[K * N + J] * std::conj(X[K]);
                }
                S[J] = S[J] / UM;
            }

            for (int J = I; J < N; J++)
            {
                for (int K = J + 1; K < N; K++)
                {
                    A[J * N + K] += S[J] * X[K] - X[J] * S[K];
                    A[K * N + J] = -A[J * N + K];
                }
            }
        }

        // 	THE PFAFFIAN VALUE
        PF = 1.0;
        for (int I = 0; I < N; I += 2)
        {
            PF = PF * A[I * N + I + 1];
            if (std::abs(PF) < 1e-200)
            {
                PF = PF / 1e-200;
            }
            if (std::abs(PF) > 1e200)
            {
                PF = PF / 1e200;
            }
        }
        PF = PF * SH;
        mkl_free(X);
        mkl_free(A);
        mkl_free(S);
        return PF;
    }

    ///////////////////// HF calculations ///////////////////////
    ////// HF kernels
    double CalHFKernels(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();

        double OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP, Hamiltonian;

        // evaluate overlap
        double *Ovl_p = (double *)mkl_malloc((4 * N_p * N_p) * sizeof(double), 64);
        double *Ovl_n = (double *)mkl_malloc((4 * N_n * N_n) * sizeof(double), 64);
        S_matrix_overlap(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);
        if (N_p == 0)
        {
            OvlME_p = 1.;
        }
        if (N_n == 0)
        {
            OvlME_n = 1.;
        }

        // evaluate Two body Hamiltonian MEs Vpp and Vnn
        MschemeHamiltonian &MSMEs = Ham.MSMEs;
        double *TB_p = (double *)mkl_malloc(((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double), 64);
        memset(TB_p, 0, ((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double));
        HME_p = 0;
        S_matrix_FullOvl_TwoBody(*Bra_p, *Ket_p, TB_p, Ovl_p);
        for (size_t i = 0; i < MSMEs.Hpp_index.size(); i++)
        {
            int *index = &MSMEs.Hpp_index[i][0];
            S_matrix_TwoBody_MEs(*Bra_p, *Ket_p, TB_p, index[0], index[1], index[2], index[3]);
            HME_p += MSMEs.Vpp(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_p + 4, TB_p) * OvlME_n;
        }
        // free memory
        mkl_free(TB_p);

        double *TB_n = (double *)mkl_malloc(((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(double), 64);
        memset(TB_n, 0, ((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(double));
        HME_n = 0;
        S_matrix_FullOvl_TwoBody(*Bra_n, *Ket_n, TB_n, Ovl_n);
        for (size_t i = 0; i < MSMEs.Hnn_index.size(); i++)
        {
            int *index = &MSMEs.Hnn_index[i][0];
            S_matrix_TwoBody_MEs(*Bra_n, *Ket_n, TB_n, index[0], index[1], index[2], index[3]);
            HME_n += MSMEs.Vnn(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_n + 4, TB_n) * OvlME_p;
        }
        // free memory
        mkl_free(TB_n);

        // evaluate Vpn and SP
        // Proton one body operator
        double *OB_p = (double *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double), 64);
        double *OBoperator_p = (double *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(double), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b());
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // Neutron one body operator
        double *OB_n = (double *)mkl_malloc(((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double), 64);
        double *OBoperator_n = (double *)mkl_malloc((Ham.MSMEs.NumOB_n) * sizeof(double), 64);
        memset(OB_n, 0, ((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_n, *Ket_n, OB_n, Ovl_n);
        std::vector<MSOneBodyOperator> &OBprt_n = Ham.MSMEs.OB_n;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_n; i++)
        {
            S_matrix_OneBody_MEs(*Bra_n, *Ket_n, OB_n, OBprt_n[i].GetIndex_a(), OBprt_n[i].GetIndex_b());
            OBoperator_n[i] = PFAFFIAN(2 * N_n + 2, OB_n);
        }
        // free memory
        mkl_free(Ovl_n);
        mkl_free(OB_n);

        // bulid Vpn MEs
        std::vector<std::array<int, 2>> &Hpn_OBindex = Ham.MSMEs.Hpn_OBindex;
        int a, b, c, d, index_p, index_n;
        HME_pn = 0;
        for (size_t i = 0; i < Hpn_OBindex.size(); i++)
        {
            index_p = Hpn_OBindex[i][0];
            index_n = Hpn_OBindex[i][1];
            a = OBprt_p[index_p].GetIndex_a();
            b = OBprt_p[index_p].GetIndex_b();
            c = OBprt_n[index_n].GetIndex_a();
            d = OBprt_n[index_n].GetIndex_b();
            HME_pn += MSMEs.Vpn(a, b, c, d) * OBoperator_p[index_p] * OBoperator_n[index_n];
        }

        // bulid SP MEs
        std::vector<std::array<int, 2>> &SPOindex_p = Ham.MSMEs.SPOindex_p;
        std::vector<std::array<int, 2>> &SPOindex_n = Ham.MSMEs.SPOindex_n;
        H_SP = 0;
        for (size_t i = 0; i < SPOindex_p.size(); i++)
        {
            a = SPOindex_p[i][0]; /// One body operator index
            b = SPOindex_p[i][1]; /// j orbit index
            H_SP += Ham.ms->GetProtonSPE(b) * OBoperator_p[a] * OvlME_n;
            // std::cout<< "  "  << i << "  " << Ham.ms->GetProtonSPE(b) << "  " << OBoperator_p[a]   << std::endl;
        }
        for (size_t i = 0; i < SPOindex_n.size(); i++)
        {
            a = SPOindex_n[i][0]; /// One body operator index
            b = SPOindex_n[i][1]; /// j orbit index
            H_SP += Ham.ms->GetNeutronSPE(b) * OBoperator_n[a] * OvlME_p;
        }

        Hamiltonian = (H_SP + HME_p + HME_n + HME_pn) / (OvlME_p * OvlME_n);
        // std::cout<< "  " << "  "<< Hamiltonian << "  " << H_SP << "  " << HME_p << "  " << HME_n << "  " << HME_pn << "  " << OvlME_p <<"  "<< OvlME_n << std::endl;
        // std::cout << H_SP << "  " << HME_p << "  " << HME_n << "  " << HME_pn << "  " << (OvlME_p * OvlME_n) << std::endl;

        double Qud0, Qud2, Qud_2;
        if (Ham.ms->GetIsShapeConstrained())
        {
            Qud0 = 0;
            Qud2 = 0;
            Qud_2 = 0;
            ////-----  Qp
            for (int i = 0; i < Ham.Q2MEs_p.Q0_list.size(); i++)
            {
                Qud0 += OvlME_n * Ham.Q2MEs_p.Q0_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q0_list[i]];
            }
            for (int i = 0; i < Ham.Q2MEs_p.Q2_list.size(); i++)
            {
                Qud2 += OvlME_n * Ham.Q2MEs_p.Q2_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q2_list[i]];
            }
            for (int i = 0; i < Ham.Q2MEs_p.Q_2_list.size(); i++)
            {
                Qud_2 += OvlME_n * Ham.Q2MEs_p.Q_2_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q_2_list[i]];
            }
            ////-----  Qn
            for (int i = 0; i < Ham.Q2MEs_n.Q0_list.size(); i++)
            {
                Qud0 += OvlME_p * Ham.Q2MEs_n.Q0_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q0_list[i]];
            }
            for (int i = 0; i < Ham.Q2MEs_n.Q2_list.size(); i++)
            {
                Qud2 += OvlME_p * Ham.Q2MEs_n.Q2_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q2_list[i]];
            }
            for (int i = 0; i < Ham.Q2MEs_n.Q_2_list.size(); i++)
            {
                Qud_2 += OvlME_p * Ham.Q2MEs_n.Q_2_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q_2_list[i]];
            }

            //-------- add restruction
            // std::cout << Hamiltonian << "  " << Qud0 << "  " << Qud2 << "  " << Qud_2 << std::endl;
            Hamiltonian += Ham.ms->GetShapeConstant() * pow(Qud0 / (OvlME_p * OvlME_n) - Ham.ms->GetShapeQ0(), 2);
            Hamiltonian += Ham.ms->GetShapeConstant() * pow(Qud2 / (OvlME_p * OvlME_n) - Ham.ms->GetShapeQ2(), 2);
            Hamiltonian += Ham.ms->GetShapeConstant() * pow(Qud_2 / (OvlME_p * OvlME_n) - Ham.ms->GetShapeQ2(), 2);
        }

        // free memory
        mkl_free(OBoperator_p);
        mkl_free(OBoperator_n);
        return Hamiltonian;
    }

    ComplexNum CalHFKernels_Complex(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket)
    {
        ComplexNum OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP, Hamiltonian;
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();

        // evaluate overlap
        ComplexNum *Ovl_p = (ComplexNum *)mkl_malloc((4 * N_p * N_p) * sizeof(ComplexNum), 64);
        ComplexNum *Ovl_n = (ComplexNum *)mkl_malloc((4 * N_n * N_n) * sizeof(ComplexNum), 64);
        S_matrix_overlap_complex(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap_complex(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);
        if (N_p == 0)
        {
            OvlME_p = 1.;
        }
        if (N_n == 0)
        {
            OvlME_n = 1.;
        }

        // evaluate Two body Hamiltonian MEs Vpp and Vnn
        MschemeHamiltonian &MSMEs = Ham.MSMEs;
        ComplexNum *TB_p = (ComplexNum *)mkl_malloc(((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(ComplexNum), 64);
        memset(TB_p, 0, ((2 * N_p + 4) * (2 * N_p + 4)) * 2 * sizeof(double));
        HME_p = 0;
        S_matrix_FullOvl_TwoBody(*Bra_p, *Ket_p, TB_p, Ovl_p);
        for (size_t i = 0; i < MSMEs.Hpp_index.size(); i++)
        {
            int *index = &MSMEs.Hpp_index[i][0];
            S_matrix_TwoBody_MEs_complex(*Bra_p, *Ket_p, TB_p, index[0], index[1], index[2], index[3]);
            HME_p += MSMEs.Vpp(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_p + 4, TB_p) * OvlME_n;
        }
        // free memory
        mkl_free(TB_p);

        ComplexNum *TB_n = (ComplexNum *)mkl_malloc(((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(ComplexNum), 64);
        memset(TB_n, 0, ((2 * N_n + 4) * (2 * N_n + 4)) * 2 * sizeof(double));
        HME_n = 0;
        S_matrix_FullOvl_TwoBody(*Bra_n, *Ket_n, TB_n, Ovl_n);
        for (size_t i = 0; i < MSMEs.Hnn_index.size(); i++)
        {
            int *index = &MSMEs.Hnn_index[i][0];
            S_matrix_TwoBody_MEs_complex(*Bra_n, *Ket_n, TB_n, index[0], index[1], index[2], index[3]);
            HME_n += MSMEs.Vnn(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_n + 4, TB_n) * OvlME_p;
        }
        // free memory
        mkl_free(TB_n);

        // evaluate Vpn and SP
        // Proton one body operator
        ComplexNum *OB_p = (ComplexNum *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(ComplexNum), 64);
        ComplexNum *OBoperator_p = (ComplexNum *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(ComplexNum), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * 2 * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs_complex(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b());
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // Neutron one body operator
        ComplexNum *OB_n = (ComplexNum *)mkl_malloc(((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(ComplexNum), 64);
        ComplexNum *OBoperator_n = (ComplexNum *)mkl_malloc((Ham.MSMEs.NumOB_n) * sizeof(ComplexNum), 64);
        memset(OB_n, 0, ((2 * N_n + 2) * (2 * N_n + 2)) * 2 * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_n, *Ket_n, OB_n, Ovl_n);
        std::vector<MSOneBodyOperator> &OBprt_n = Ham.MSMEs.OB_n;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_n; i++)
        {
            S_matrix_OneBody_MEs_complex(*Bra_n, *Ket_n, OB_n, OBprt_n[i].GetIndex_a(), OBprt_n[i].GetIndex_b());
            OBoperator_n[i] = PFAFFIAN(2 * N_n + 2, OB_n);
        }
        // free memory
        mkl_free(Ovl_n);
        mkl_free(OB_n);

        // bulid Vpn MEs
        std::vector<std::array<int, 2>> &Hpn_OBindex = Ham.MSMEs.Hpn_OBindex;
        int a, b, c, d, index_p, index_n;
        HME_pn = 0;
        for (size_t i = 0; i < Hpn_OBindex.size(); i++)
        {
            index_p = Hpn_OBindex[i][0];
            index_n = Hpn_OBindex[i][1];
            a = OBprt_p[index_p].GetIndex_a();
            b = OBprt_p[index_p].GetIndex_b();
            c = OBprt_n[index_n].GetIndex_a();
            d = OBprt_n[index_n].GetIndex_b();
            HME_pn += MSMEs.Vpn(a, b, c, d) * OBoperator_p[index_p] * OBoperator_n[index_n];
        }

        // bulid SP MEs
        std::vector<std::array<int, 2>> &SPOindex_p = Ham.MSMEs.SPOindex_p;
        std::vector<std::array<int, 2>> &SPOindex_n = Ham.MSMEs.SPOindex_n;
        H_SP = 0;
        for (size_t i = 0; i < SPOindex_p.size(); i++)
        {
            a = SPOindex_p[i][0]; /// One body operator index
            b = SPOindex_p[i][1]; /// j orbit index
            H_SP += Ham.ms->GetProtonSPE(b) * OBoperator_p[a] * OvlME_n;
        }
        for (size_t i = 0; i < SPOindex_n.size(); i++)
        {
            a = SPOindex_n[i][0]; /// One body operator index
            b = SPOindex_n[i][1]; /// j orbit index
            H_SP += Ham.ms->GetNeutronSPE(b) * OBoperator_n[a] * OvlME_p;
        }

        Hamiltonian = (H_SP + HME_p + HME_n + HME_pn) / (OvlME_p * OvlME_n);
        // std::cout << H_SP << "  " << HME_p <<  "  " << HME_n <<"  " << HME_pn << "  " << (OvlME_p ) << ( OvlME_n) << std::endl;
        //  free memory
        mkl_free(OBoperator_p);
        mkl_free(OBoperator_n);
        return Hamiltonian;
    }

    void CalHFKernels_Complex(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket, ComplexNum &HamME, ComplexNum &OvlME)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();
        ComplexNum OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP, Hamiltonian;

        // evaluate overlap
        ComplexNum *Ovl_p = (ComplexNum *)mkl_malloc((4 * N_p * N_p) * sizeof(ComplexNum), 64);
        ComplexNum *Ovl_n = (ComplexNum *)mkl_malloc((4 * N_n * N_n) * sizeof(ComplexNum), 64);
        S_matrix_overlap_complex(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap_complex(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);
        if (N_p == 0)
        {
            OvlME_p = 1.;
        }
        if (N_n == 0)
        {
            OvlME_n = 1.;
        }
        OvlME = (OvlME_p * OvlME_n);

        // evaluate Two body Hamiltonian MEs Vpp and Vnn
        MschemeHamiltonian &MSMEs = Ham.MSMEs;
        ComplexNum *TB_p = (ComplexNum *)mkl_malloc(((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(ComplexNum), 64);
        memset(TB_p, 0, ((2 * N_p + 4) * (2 * N_p + 4)) * 2 * sizeof(double));
        HME_p = 0;
        S_matrix_FullOvl_TwoBody(*Bra_p, *Ket_p, TB_p, Ovl_p);
        for (size_t i = 0; i < MSMEs.Hpp_index.size(); i++)
        {
            int *index = &MSMEs.Hpp_index[i][0];
            S_matrix_TwoBody_MEs_complex(*Bra_p, *Ket_p, TB_p, index[0], index[1], index[2], index[3]);
            HME_p += MSMEs.Vpp(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_p + 4, TB_p) * OvlME_n;
        }
        // free memory
        mkl_free(TB_p);

        ComplexNum *TB_n = (ComplexNum *)mkl_malloc(((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(ComplexNum), 64);
        memset(TB_n, 0, ((2 * N_n + 4) * (2 * N_n + 4)) * 2 * sizeof(double));
        HME_n = 0;
        S_matrix_FullOvl_TwoBody(*Bra_n, *Ket_n, TB_n, Ovl_n);
        for (size_t i = 0; i < MSMEs.Hnn_index.size(); i++)
        {
            int *index = &MSMEs.Hnn_index[i][0];
            S_matrix_TwoBody_MEs_complex(*Bra_n, *Ket_n, TB_n, index[0], index[1], index[2], index[3]);
            HME_n += MSMEs.Vnn(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_n + 4, TB_n) * OvlME_p;
        }
        // free memory
        mkl_free(TB_n);

        // evaluate Vpn and SP
        // Proton one body operator
        ComplexNum *OB_p = (ComplexNum *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(ComplexNum), 64);
        ComplexNum *OBoperator_p = (ComplexNum *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(ComplexNum), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * 2 * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs_complex(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b());
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // Neutron one body operator
        ComplexNum *OB_n = (ComplexNum *)mkl_malloc(((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(ComplexNum), 64);
        ComplexNum *OBoperator_n = (ComplexNum *)mkl_malloc((Ham.MSMEs.NumOB_n) * sizeof(ComplexNum), 64);
        memset(OB_n, 0, ((2 * N_n + 2) * (2 * N_n + 2)) * 2 * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_n, *Ket_n, OB_n, Ovl_n);
        std::vector<MSOneBodyOperator> &OBprt_n = Ham.MSMEs.OB_n;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_n; i++)
        {
            S_matrix_OneBody_MEs_complex(*Bra_n, *Ket_n, OB_n, OBprt_n[i].GetIndex_a(), OBprt_n[i].GetIndex_b());
            OBoperator_n[i] = PFAFFIAN(2 * N_n + 2, OB_n);
        }
        // free memory
        mkl_free(Ovl_n);
        mkl_free(OB_n);

        // bulid Vpn MEs
        std::vector<std::array<int, 2>> &Hpn_OBindex = Ham.MSMEs.Hpn_OBindex;
        int a, b, c, d, index_p, index_n;
        HME_pn = 0;
        for (size_t i = 0; i < Hpn_OBindex.size(); i++)
        {
            index_p = Hpn_OBindex[i][0];
            index_n = Hpn_OBindex[i][1];
            a = OBprt_p[index_p].GetIndex_a();
            b = OBprt_p[index_p].GetIndex_b();
            c = OBprt_n[index_n].GetIndex_a();
            d = OBprt_n[index_n].GetIndex_b();
            HME_pn += MSMEs.Vpn(a, b, c, d) * OBoperator_p[index_p] * OBoperator_n[index_n];
        }

        // bulid SP MEs
        std::vector<std::array<int, 2>> &SPOindex_p = Ham.MSMEs.SPOindex_p;
        std::vector<std::array<int, 2>> &SPOindex_n = Ham.MSMEs.SPOindex_n;
        H_SP = 0;
        for (size_t i = 0; i < SPOindex_p.size(); i++)
        {
            a = SPOindex_p[i][0]; /// One body operator index
            b = SPOindex_p[i][1]; /// j orbit index
            H_SP += Ham.ms->GetProtonSPE(b) * OBoperator_p[a] * OvlME_n;
        }
        for (size_t i = 0; i < SPOindex_n.size(); i++)
        {
            a = SPOindex_n[i][0]; /// One body operator index
            b = SPOindex_n[i][1]; /// j orbit index
            H_SP += Ham.ms->GetNeutronSPE(b) * OBoperator_n[a] * OvlME_p;
        }

        Hamiltonian = (H_SP + HME_p + HME_n + HME_pn) / (OvlME_p * OvlME_n);
        // std::cout << H_SP << "  " << HME_p << "  " << HME_n << "  " << HME_pn << "  " << OvlME_p  <<"  " << OvlME_n << std::endl;

        // free memory
        mkl_free(OBoperator_p);
        mkl_free(OBoperator_n);
        HamME = (H_SP + HME_p + HME_n + HME_pn);
        return;
    }

    double CalHFKernels_halfClosed(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();

        double OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP, Hamiltonian;

        // evaluate overlap
        double *Ovl_p = (double *)mkl_malloc((4 * N_p * N_p) * sizeof(double), 64);
        double *Ovl_n = (double *)mkl_malloc((4 * N_n * N_n) * sizeof(double), 64);
        S_matrix_overlap(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);

        // evaluate Two body Hamiltonian MEs Vpp and Vnn
        MschemeHamiltonian &MSMEs = Ham.MSMEs;
        double *TB_p = (double *)mkl_malloc(((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double), 64);
        memset(TB_p, 0, ((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double));
        HME_p = 0;
        S_matrix_FullOvl_TwoBody(*Bra_p, *Ket_p, TB_p, Ovl_p);
        for (size_t i = 0; i < MSMEs.Hpp_index.size(); i++)
        {
            int *index = &MSMEs.Hpp_index[i][0];
            S_matrix_TwoBody_MEs(*Bra_p, *Ket_p, TB_p, index[0], index[1], index[2], index[3]);
            double FAvalue = PFAFFIAN(2 * N_p + 4, TB_p);
            HME_p += MSMEs.Vpp(index[0], index[1], index[2], index[3]) * FAvalue;
            // std::cout << i << "  " <<index[0]<< index[1]<< index[2]<< index[3] <<"  " << HME_p << "  " << MSMEs.Vpp(index[0], index[1], index[2], index[3]) << "  " <<  FAvalue<< std::endl;
        }
        // free memory
        mkl_free(TB_p);

        // evaluate Vpn and SP
        // Proton one body operator
        double *OB_p = (double *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double), 64);
        double *OBoperator_p = (double *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(double), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b());
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // bulid SP MEs
        std::vector<std::array<int, 2>> &SPOindex_p = Ham.MSMEs.SPOindex_p;
        std::vector<std::array<int, 2>> &SPOindex_n = Ham.MSMEs.SPOindex_n;
        H_SP = 0;
        for (size_t i = 0; i < SPOindex_p.size(); i++)
        {
            int a, b;
            a = SPOindex_p[i][0]; /// One body operator index
            b = SPOindex_p[i][1]; /// j orbit index
            H_SP += Ham.ms->GetProtonSPE(b) * OBoperator_p[a];
        }

        Hamiltonian = (H_SP + HME_p) / (OvlME_p);

        // free memory
        mkl_free(OBoperator_p);

        return Hamiltonian;
    }

    // non-normalized
    double CalHF_Hamiltonian(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();

        double OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP, Hamiltonian;

        // evaluate overlap
        double *Ovl_p = (double *)mkl_malloc((4 * N_p * N_p) * sizeof(double), 64);
        double *Ovl_n = (double *)mkl_malloc((4 * N_n * N_n) * sizeof(double), 64);
        S_matrix_overlap(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);

        // evaluate Two body Hamiltonian MEs Vpp and Vnn
        MschemeHamiltonian &MSMEs = Ham.MSMEs;
        double *TB_p = (double *)mkl_malloc(((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double), 64);
        memset(TB_p, 0, ((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double));
        HME_p = 0;
        S_matrix_FullOvl_TwoBody(*Bra_p, *Ket_p, TB_p, Ovl_p);
        for (size_t i = 0; i < MSMEs.Hpp_index.size(); i++)
        {
            int *index = &MSMEs.Hpp_index[i][0];
            S_matrix_TwoBody_MEs(*Bra_p, *Ket_p, TB_p, index[0], index[1], index[2], index[3]);
            HME_p += MSMEs.Vpp(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_p + 4, TB_p) * OvlME_n;
        }
        // free memory
        mkl_free(TB_p);

        double *TB_n = (double *)mkl_malloc(((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(double), 64);
        memset(TB_n, 0, ((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(double));
        HME_n = 0;
        S_matrix_FullOvl_TwoBody(*Bra_n, *Ket_n, TB_n, Ovl_n);
        for (size_t i = 0; i < MSMEs.Hnn_index.size(); i++)
        {
            int *index = &MSMEs.Hnn_index[i][0];
            S_matrix_TwoBody_MEs(*Bra_n, *Ket_n, TB_n, index[0], index[1], index[2], index[3]);
            HME_n += MSMEs.Vnn(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_n + 4, TB_n) * OvlME_p;
        }
        // free memory
        mkl_free(TB_n);

        // evaluate Vpn and SP
        // Proton one body operator
        double *OB_p = (double *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double), 64);
        double *OBoperator_p = (double *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(double), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b());
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // Neutron one body operator
        double *OB_n = (double *)mkl_malloc(((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double), 64);
        double *OBoperator_n = (double *)mkl_malloc((Ham.MSMEs.NumOB_n) * sizeof(double), 64);
        memset(OB_n, 0, ((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_n, *Ket_n, OB_n, Ovl_n);
        std::vector<MSOneBodyOperator> &OBprt_n = Ham.MSMEs.OB_n;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_n; i++)
        {
            S_matrix_OneBody_MEs(*Bra_n, *Ket_n, OB_n, OBprt_n[i].GetIndex_a(), OBprt_n[i].GetIndex_b());
            OBoperator_n[i] = PFAFFIAN(2 * N_n + 2, OB_n);
        }
        // free memory
        mkl_free(Ovl_n);
        mkl_free(OB_n);

        // bulid Vpn MEs
        std::vector<std::array<int, 2>> &Hpn_OBindex = Ham.MSMEs.Hpn_OBindex;
        int a, b, c, d, index_p, index_n;
        HME_pn = 0;
        for (size_t i = 0; i < Hpn_OBindex.size(); i++)
        {
            index_p = Hpn_OBindex[i][0];
            index_n = Hpn_OBindex[i][1];
            a = OBprt_p[index_p].GetIndex_a();
            b = OBprt_p[index_p].GetIndex_b();
            c = OBprt_n[index_n].GetIndex_a();
            d = OBprt_n[index_n].GetIndex_b();
            HME_pn += MSMEs.Vpn(a, b, c, d) * OBoperator_p[index_p] * OBoperator_n[index_n];
        }

        // bulid SP MEs
        std::vector<std::array<int, 2>> &SPOindex_p = Ham.MSMEs.SPOindex_p;
        std::vector<std::array<int, 2>> &SPOindex_n = Ham.MSMEs.SPOindex_n;
        H_SP = 0;
        for (size_t i = 0; i < SPOindex_p.size(); i++)
        {
            a = SPOindex_p[i][0]; /// One body operator index
            b = SPOindex_p[i][1]; /// j orbit index
            H_SP += Ham.ms->GetProtonSPE(b) * OBoperator_p[a] * OvlME_n;
        }
        for (size_t i = 0; i < SPOindex_n.size(); i++)
        {
            a = SPOindex_n[i][0]; /// One body operator index
            b = SPOindex_n[i][1]; /// j orbit index
            H_SP += Ham.ms->GetNeutronSPE(b) * OBoperator_n[a] * OvlME_p;
        }

        Hamiltonian = (H_SP + HME_p + HME_n + HME_pn);

        // free memory
        mkl_free(OBoperator_p);
        mkl_free(OBoperator_n);
        return Hamiltonian;
    }

    // the reuslt is not nomalized
    void CalHF_Q2(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket, double *Q_array)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();

        double OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP, Hamiltonian;

        // evaluate overlap
        double *Ovl_p = (double *)mkl_malloc((4 * N_p * N_p) * sizeof(double), 64);
        double *Ovl_n = (double *)mkl_malloc((4 * N_n * N_n) * sizeof(double), 64);
        S_matrix_overlap(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);

        // Proton one body operator
        double *OB_p = (double *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double), 64);
        double *OBoperator_p = (double *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(double), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b());
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // Neutron one body operator
        double *OB_n = (double *)mkl_malloc(((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double), 64);
        double *OBoperator_n = (double *)mkl_malloc((Ham.MSMEs.NumOB_n) * sizeof(double), 64);
        memset(OB_n, 0, ((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_n, *Ket_n, OB_n, Ovl_n);
        std::vector<MSOneBodyOperator> &OBprt_n = Ham.MSMEs.OB_n;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_n; i++)
        {
            S_matrix_OneBody_MEs(*Bra_n, *Ket_n, OB_n, OBprt_n[i].GetIndex_a(), OBprt_n[i].GetIndex_b());
            OBoperator_n[i] = PFAFFIAN(2 * N_n + 2, OB_n);
        }
        // free memory
        mkl_free(Ovl_n);
        mkl_free(OB_n);

        double Qud0, Qud2, Qud_2;
        Qud0 = 0;
        Qud2 = 0;
        Qud_2 = 0;
        ////-----  Qp
        for (int i = 0; i < Ham.Q2MEs_p.Q0_list.size(); i++)
        {
            Qud0 += OvlME_n * Ham.Q2MEs_p.Q0_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q0_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_p.Q2_list.size(); i++)
        {
            Qud2 += OvlME_n * Ham.Q2MEs_p.Q2_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q2_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_p.Q_2_list.size(); i++)
        {
            Qud_2 += OvlME_n * Ham.Q2MEs_p.Q_2_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q_2_list[i]];
        }
        ////-----  Qn
        for (int i = 0; i < Ham.Q2MEs_n.Q0_list.size(); i++)
        {
            Qud0 += OvlME_p * Ham.Q2MEs_n.Q0_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q0_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_n.Q2_list.size(); i++)
        {
            Qud2 += OvlME_p * Ham.Q2MEs_n.Q2_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q2_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_n.Q_2_list.size(); i++)
        {
            Qud_2 += OvlME_p * Ham.Q2MEs_n.Q_2_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q_2_list[i]];
        }

        Q_array[0] = Qud0;
        Q_array[1] = Qud2;
        Q_array[2] = Qud_2;
        Q_array[3] = Ham.ms->GetShapeConstant() * (Qud0 / (OvlME_p * OvlME_n) - Ham.ms->GetShapeQ0());
        Q_array[4] = Ham.ms->GetShapeConstant() * (Qud2 / (OvlME_p * OvlME_n) - Ham.ms->GetShapeQ2());
        Q_array[5] = Ham.ms->GetShapeConstant() * (Qud_2 / (OvlME_p * OvlME_n) - Ham.ms->GetShapeQ2());

        // free memory
        mkl_free(OBoperator_p);
        mkl_free(OBoperator_n);
        return;
    }

    // Qcosγ =⟨Ψ|√16 π/5 r^2 / b^2 Y20|Ψ⟩
    // Qsinγ =⟨Ψ|√16 π/5 r^2 / b^2 1/√2 (Y22 +Y2−2)|Ψ⟩

    // Q0 = 3 /2π Sqrt( 4π / 5 ) <r^2> β cosγ
    // Q2 = 3 /2π Sqrt( 4π / 5 ) <r^2> β/√2 sinγ
    // the r^2 is given by ⟨Ψ| r^2 |Ψ⟩
    // tanγ = √2 Q2 / Q0
    // β = Q0 * 2π Sqrt( 4π / 5 ) / 3 / <r^2> / cosγ
    // see more in PHYSICAL REVIEW C, VOLUME 61, 034303
    // Shape parameters (in unit of b^2) (b = 1.005 A^{1/6} fm)
    void CalHFShape(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();
        double OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP, Hamiltonian;

        // evaluate overlap
        double *Ovl_p = (double *)mkl_malloc((4 * N_p * N_p) * sizeof(double), 64);
        double *Ovl_n = (double *)mkl_malloc((4 * N_n * N_n) * sizeof(double), 64);
        S_matrix_overlap(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);

        // evaluate one body operator
        // Proton one body operator
        double *OB_p = (double *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double), 64);
        double *OBoperator_p = (double *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(double), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b());
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // Neutron one body operator
        double *OB_n = (double *)mkl_malloc(((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double), 64);
        double *OBoperator_n = (double *)mkl_malloc((Ham.MSMEs.NumOB_n) * sizeof(double), 64);
        memset(OB_n, 0, ((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_n, *Ket_n, OB_n, Ovl_n);
        std::vector<MSOneBodyOperator> &OBprt_n = Ham.MSMEs.OB_n;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_n; i++)
        {
            S_matrix_OneBody_MEs(*Bra_n, *Ket_n, OB_n, OBprt_n[i].GetIndex_a(), OBprt_n[i].GetIndex_b());
            OBoperator_n[i] = PFAFFIAN(2 * N_n + 2, OB_n);
        }
        // free memory
        mkl_free(Ovl_n);
        mkl_free(OB_n);

        double Qud0, Qud2, Qud_2;
        Qud0 = 0;
        Qud2 = 0;
        Qud_2 = 0;
        ////-----  Qp
        for (int i = 0; i < Ham.Q2MEs_p.Q0_list.size(); i++)
        {
            Qud0 += OvlME_n * Ham.Q2MEs_p.Q0_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q0_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_p.Q2_list.size(); i++)
        {
            Qud2 += OvlME_n * Ham.Q2MEs_p.Q2_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q2_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_p.Q_2_list.size(); i++)
        {
            Qud_2 += OvlME_n * Ham.Q2MEs_p.Q_2_MSMEs[i] * OBoperator_p[Ham.Q2MEs_p.Q_2_list[i]];
        }
        ////-----  Qn
        for (int i = 0; i < Ham.Q2MEs_n.Q0_list.size(); i++)
        {
            Qud0 += OvlME_p * Ham.Q2MEs_n.Q0_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q0_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_n.Q2_list.size(); i++)
        {
            Qud2 += OvlME_p * Ham.Q2MEs_n.Q2_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q2_list[i]];
        }
        for (int i = 0; i < Ham.Q2MEs_n.Q_2_list.size(); i++)
        {
            Qud_2 += OvlME_p * Ham.Q2MEs_n.Q_2_MSMEs[i] * OBoperator_n[Ham.Q2MEs_n.Q_2_list[i]];
        }

        // bulid r^2 MEs
        double R_2 = 0;
        int a, b;
        std::vector<std::array<int, 2>> &SPOindex_p = Ham.MSMEs.SPOindex_p;
        std::vector<std::array<int, 2>> &SPOindex_n = Ham.MSMEs.SPOindex_n;
        for (size_t i = 0; i < SPOindex_p.size(); i++)
        {
            a = SPOindex_p[i][0]; /// One body operator index
            b = SPOindex_p[i][1]; /// j orbit index
            R_2 += Ham.HarmonicRadialIntegral(Proton, 2, b, b) * OBoperator_p[a] * OvlME_n;
        }
        for (size_t i = 0; i < SPOindex_n.size(); i++)
        {
            a = SPOindex_n[i][0]; /// One body operator index
            b = SPOindex_n[i][1]; /// j orbit index
            R_2 += Ham.HarmonicRadialIntegral(Neutron, 2, b, b) * OBoperator_n[a] * OvlME_p;
        }
        R_2 /= (OvlME_n * OvlME_p);

        // free memory
        mkl_free(OBoperator_p);
        mkl_free(OBoperator_n);

        //-------- shape restruction
        std::cout << "  Deformation parameters (in unit of b^2) (b = 1.005 A^{1/6} fm): " << std::endl;
        std::cout << "      Q0: " << std::setw(7) << std::setfill(' ') << std::fixed << std::setprecision(4) << Qud0 / (OvlME_p * OvlME_n) << "     Q2: " << Qud2 / (OvlME_p * OvlME_n) << "     Q-2: " << Qud_2 / (OvlME_p * OvlME_n) << std::endl;

        double gamma = sqrt(2.) * Qud2 / Qud0;
        if (std::abs(Qud2) < 1.e-5 and std::abs(Qud0) < 1.e-5)
        {
            gamma = 0.; // gamma in unit of degree
        }
        else
            gamma = std::atan(gamma) * 180.0 / M_PI; // gamma in unit of degree
        // double beta =  Qud0 / (OvlME_p * OvlME_n) / ( sqrt(16. * M_PI / 5.) * std::cos(gamma * M_PI / 180.));
        double beta = 2. * M_PI * Qud0 / (OvlME_p * OvlME_n) / R_2 / (3. * sqrt(4. * M_PI / 5.) * std::cos(gamma * M_PI / 180.));

        std::cout << "\033[31m      Beta and Gamma are obtained in Lab. frame! \033[0m" << std::endl;
        std::cout << "      Beta: " << beta << "     Gamma: " << gamma << "   (deg) " << std::endl;
        return;
    }

    double CalHFOverlap(PNbasis &Bra, PNbasis &Ket)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Bra_p->GetParticleNumber();
        int N_n = Bra_n->GetParticleNumber();
        double OvlME_p, OvlME_n;
        // evaluate overlap
        double *Ovl_p = (double *)mkl_malloc((4 * N_p * N_p) * sizeof(double), 64);
        double *Ovl_n = (double *)mkl_malloc((4 * N_n * N_n) * sizeof(double), 64);
        S_matrix_overlap(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);
        mkl_free(Ovl_p);
        mkl_free(Ovl_n);
        return OvlME_p * OvlME_n;
    }

    ComplexNum CalHFOverlap_Complex(PNbasis &Bra, PNbasis &Ket)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Bra_p->GetParticleNumber();
        int N_n = Bra_n->GetParticleNumber();
        ComplexNum OvlME_p, OvlME_n;

        // evaluate overlap
        ComplexNum *Ovl_p = (ComplexNum *)mkl_malloc((4 * N_p * N_p) * sizeof(ComplexNum), 64);
        ComplexNum *Ovl_n = (ComplexNum *)mkl_malloc((4 * N_n * N_n) * sizeof(ComplexNum), 64);
        S_matrix_overlap_complex(*Bra_p, *Ket_p, Ovl_p);
        S_matrix_overlap_complex(*Bra_n, *Ket_n, Ovl_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);
        mkl_free(Ovl_p);
        mkl_free(Ovl_n);
        if (N_p == 0)
        {
            return (OvlME_n);
        }
        else if (N_n == 0)
        {
            return (OvlME_p);
        }
        else
        {
            return (OvlME_p * OvlME_n);
        }
    }

    double CalHFKernels_Advanced(Hamiltonian &Ham, PNbasis &Bra, PNbasis &Ket)
    {
        HFbasis *Bra_p = Bra.GetProtonPrt();
        HFbasis *Bra_n = Bra.GetNeutronPrt();
        HFbasis *Ket_p = Ket.GetProtonPrt();
        HFbasis *Ket_n = Ket.GetNeutronPrt();
        int N_p = Ham.ms->GetProtonNum();
        int N_n = Ham.ms->GetNeutronNum();
        int inner_dim_p = Bra.GetProntonInnerDim();
        int inner_dim_n = Bra.GetNeutronInnerDim();
        double OvlME_p, OvlME_n, HME_p, HME_n, HME_pn, H_SP;

        // evaluate overlap
        double *Ovl_p = (double *)mkl_malloc((4 * N_p * N_p) * sizeof(double), 64);
        double *Ovl_n = (double *)mkl_malloc((4 * N_n * N_n) * sizeof(double), 64);
        S_matrix_overlap(*Bra_p, *Ket_p, Ovl_p, inner_dim_p);
        S_matrix_overlap(*Bra_n, *Ket_n, Ovl_n, inner_dim_n);
        OvlME_p = PFAFFIAN(2 * N_p, Ovl_p);
        OvlME_n = PFAFFIAN(2 * N_n, Ovl_n);

        // evaluate Two body Hamiltonian MEs Vpp and Vnn
        MschemeHamiltonian &MSMEs = Ham.MSMEs;
        double *TB_p = (double *)mkl_malloc(((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double), 64);
        memset(TB_p, 0, ((2 * N_p + 4) * (2 * N_p + 4)) * sizeof(double));
        HME_p = 0;
        S_matrix_FullOvl_TwoBody(*Bra_p, *Ket_p, TB_p, Ovl_p);
        for (size_t i = 0; i < MSMEs.Hpp_index.size(); i++)
        {
            int *index = &MSMEs.Hpp_index[i][0];
            S_matrix_TwoBody_MEs(*Bra_p, *Ket_p, TB_p, index[0], index[1], index[2], index[3], inner_dim_p);
            HME_p += MSMEs.Vpp(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_p + 4, TB_p) * OvlME_n;
        }
        // free memory
        mkl_free(TB_p);

        double *TB_n = (double *)mkl_malloc(((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(double), 64);
        memset(TB_n, 0, ((2 * N_n + 4) * (2 * N_n + 4)) * sizeof(double));
        HME_n = 0;
        S_matrix_FullOvl_TwoBody(*Bra_n, *Ket_n, TB_n, Ovl_n);
        for (size_t i = 0; i < MSMEs.Hnn_index.size(); i++)
        {
            int *index = &MSMEs.Hnn_index[i][0];
            S_matrix_TwoBody_MEs(*Bra_n, *Ket_n, TB_n, index[0], index[1], index[2], index[3], inner_dim_n);
            HME_n += MSMEs.Vnn(index[0], index[1], index[2], index[3]) * PFAFFIAN(2 * N_n + 4, TB_n) * OvlME_p;
        }
        // free memory
        mkl_free(TB_n);

        // evaluate Vpn and SP
        // Proton one body operator
        double *OB_p = (double *)mkl_malloc(((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double), 64);
        double *OBoperator_p = (double *)mkl_malloc((Ham.MSMEs.NumOB_p) * sizeof(double), 64);
        memset(OB_p, 0, ((2 * N_p + 2) * (2 * N_p + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_p, *Ket_p, OB_p, Ovl_p);
        std::vector<MSOneBodyOperator> &OBprt_p = Ham.MSMEs.OB_p;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_p; i++)
        {
            S_matrix_OneBody_MEs(*Bra_p, *Ket_p, OB_p, OBprt_p[i].GetIndex_a(), OBprt_p[i].GetIndex_b(), inner_dim_p);
            OBoperator_p[i] = PFAFFIAN(2 * N_p + 2, OB_p);
        }
        // free memory
        mkl_free(Ovl_p);
        mkl_free(OB_p);

        // Neutron one body operator
        double *OB_n = (double *)mkl_malloc(((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double), 64);
        double *OBoperator_n = (double *)mkl_malloc((Ham.MSMEs.NumOB_n) * sizeof(double), 64);
        memset(OB_n, 0, ((2 * N_n + 2) * (2 * N_n + 2)) * sizeof(double));
        S_matrix_FullOvl_OneBody(*Bra_n, *Ket_n, OB_n, Ovl_n);
        std::vector<MSOneBodyOperator> &OBprt_n = Ham.MSMEs.OB_n;
        for (size_t i = 0; i < Ham.MSMEs.NumOB_n; i++)
        {
            S_matrix_OneBody_MEs(*Bra_n, *Ket_n, OB_n, OBprt_n[i].GetIndex_a(), OBprt_n[i].GetIndex_b(), inner_dim_n);
            OBoperator_n[i] = PFAFFIAN(2 * N_n + 2, OB_n);
        }
        // free memory
        mkl_free(Ovl_n);
        mkl_free(OB_n);

        // bulid Vpn MEs
        std::vector<std::array<int, 2>> &Hpn_OBindex = Ham.MSMEs.Hpn_OBindex;
        int a, b, c, d, index_p, index_n;
        HME_pn = 0;
        for (size_t i = 0; i < Hpn_OBindex.size(); i++)
        {
            index_p = Hpn_OBindex[i][0];
            index_n = Hpn_OBindex[i][1];
            a = OBprt_p[index_p].GetIndex_a();
            b = OBprt_p[index_p].GetIndex_b();
            c = OBprt_n[index_n].GetIndex_a();
            d = OBprt_n[index_n].GetIndex_b();
            HME_pn += MSMEs.Vpn(a, b, c, d) * OBoperator_p[index_p] * OBoperator_n[index_n];
        }

        // bulid SP MEs
        std::vector<std::array<int, 2>> &SPOindex_p = Ham.MSMEs.SPOindex_p;
        std::vector<std::array<int, 2>> &SPOindex_n = Ham.MSMEs.SPOindex_n;
        H_SP = 0;
        for (size_t i = 0; i < SPOindex_p.size(); i++)
        {
            a = SPOindex_p[i][0]; /// One body operator index
            b = SPOindex_p[i][1]; /// j orbit index
            H_SP += Ham.ms->GetProtonSPE(b) * OBoperator_p[a] * OvlME_n;
        }
        for (size_t i = 0; i < SPOindex_n.size(); i++)
        {
            a = SPOindex_n[i][0]; /// One body operator index
            b = SPOindex_n[i][1]; /// j orbit index
            H_SP += Ham.ms->GetNeutronSPE(b) * OBoperator_n[a] * OvlME_p;
        }

        // free memory
        mkl_free(OBoperator_p);
        mkl_free(OBoperator_n);
        std::cout << (H_SP + HME_p + HME_n + HME_pn) << "  " << (OvlME_p * OvlME_n) << " " << (H_SP + HME_p + HME_n + HME_pn) / (OvlME_p * OvlME_n) << std::endl;
        return (H_SP + HME_p + HME_n + HME_pn) / (OvlME_p * OvlME_n);
    }

    void normalizeBais(double *x, int N, int dim, int inner_dim)
    {
        for (size_t i = 0; i < N; i++)
        {
            double norm = cblas_ddot(dim * inner_dim, x + i * dim * inner_dim, 1, x + i * dim * inner_dim, 1);
            cblas_dscal(dim * inner_dim, 1. / sqrt(norm), x + i * dim * inner_dim, 1);
        }
    }
}

ComplexNum Pfaffian_naive(int dim, const ComplexNum *in)
{
    //  calculate the pfaffian of skew-symmetric matrix
    int n = dim / 2;
    std::vector<ComplexNum> tmp(dim * dim);
    cblas_zcopy(dim * dim, in, 1, tmp.data(), 1);

    ComplexNum PF = 1.0;
    ComplexNum fac;

    for (int i = 0; i < 2 * n; i = i + 2)
    {
        for (int j = i + 2; j < 2 * n; j++)
        {
            fac = -tmp[i * dim + j] / tmp[i * dim + (i + 1)];
            for (int k = i + 1; k < 2 * n; k++)
            {
                tmp[k * dim + j] = tmp[k * dim + j] + fac * tmp[k * dim + i + 1];
                tmp[j * dim + k] = tmp[j* dim + k] + fac * tmp[ (i + 1) * dim + k];
            }
        }
        PF = PF * tmp[i * dim + i + 1];
    }
    return PF;
}

void Check_matrix(int dim, ComplexNum *Matrix)
{
    std::cout << "   Chcecking matrix:  dim: " << dim << std::endl;
    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << Matrix[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

ComplexNum Determinant(int dim, ComplexNum *Matrix)
{
    MKL_INT n = dim; // Size of the matrix
    MKL_INT ipiv[n];
    MKL_INT info;

    std::vector<ComplexNum> MatrixCopy(dim * dim);
    cblas_zcopy(dim * dim, Matrix, 1, MatrixCopy.data(), 1);

    // Perform LU factorization using LAPACKE_zgetrf
    info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, MatrixCopy.data(), n, ipiv);

    if (info != 0)
    {
        std::cerr << "LAPACKE_zgetrf in Determinant function failed with code " << info << std::endl;
        exit(0);
    }

    // Calculate the determinant using the LU factors
    std::complex<double> determinant = 1.0;
    for (MKL_INT i = 0; i < n; ++i)
    {
        determinant *= MatrixCopy[i * n + i];
    }

    // Output the determinant
    // std::cout << "Determinant of the complex matrix: " << determinant.real() << "+" << determinant.imag() << "i" << std::endl;
    return determinant;
}

namespace HFB_Pfaffian_Tools
{
    //------------------------------------------------------------//
    //                     HFB calculations                       //
    //------------------------------------------------------------//

    // Overlap
    // see more in PHYSICAL REVIEW C 84, 014307 (2011) Eq.(4)
    // <φ0|φ1> = (−1)^N(N+1)/2 pf(M), where M is a 2N X 2N skew-symmetric matrix
    // M = [ M1   -I   ]
    //     [  I   -M0* ]
    // where M1 and M0 come from HFB wave function |φ1> = exp( 1/2 sum_kk' M1_kk' a^+_k a^+_k' ) |0>
    // M1_kk' = (V U^-1)^*
    ComplexNum Compute_Overlap(int N, ComplexNum *M0, ComplexNum *M1)
    {
        int dim2 = 2 * N;
        ComplexNum NORM = 0.;
        std::vector<ComplexNum> MatrixM(dim2 * dim2, 0.);
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                MatrixM[i * dim2 + j] = M1[i * N + j];
                MatrixM[(i + N) * dim2 + (j + N)] = -std::conj(M0[i * N + j]);
            }
            MatrixM[(i + N) * dim2 + i] = 1.;
            MatrixM[i * dim2 + (i + N)] = -1.;
        }
        NORM = HF_Pfaffian_Tools::PFAFFIAN(dim2, MatrixM.data());
        return sgn(N * (N + 1) / 2) * 1. * NORM;
    }

}
