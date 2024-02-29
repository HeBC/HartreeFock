#include <iostream>
#include <string>
#include <vector>
#include <nlopt.h>

#include <cstdlib>
#include <ctime>

#include <mkl.h>
using namespace std;

#include "ReadWriteFiles.h"
// #include "CalNPSM.h"
#include "HFbasis.h"

#include "Pfaffian_tools.h"
using namespace HF_Pfaffian_Tools;

struct MyData
{
    PNbasis *Bra, *Ket;
    Hamiltonian *Hinput;
};

double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    MyData *params = reinterpret_cast<MyData *>(my_func_data);
    double *prt_p, *prt_n;
    int N_p = params->Hinput->ms->GetProtonNum();
    int N_n = params->Hinput->ms->GetNeutronNum();
    int dim_p = params->Hinput->ms->Get_MScheme_dim(Proton);
    int dim_n = params->Hinput->ms->Get_MScheme_dim(Neutron);
    double *prt = (double *)mkl_malloc((N_p * dim_p + N_n * dim_n) * sizeof(double), 64);
    prt_p = prt;
    prt_n = prt + N_p * dim_p;
    std::memcpy(prt, x, (N_p * dim_p + N_n * dim_n) * sizeof(double)); // copy the contents of src to dest
    // normalizeBais(prt_p, N_p, dim_p, 1);
    // normalizeBais(prt_n, N_n, dim_n, 1);
    params->Bra->SetArray(prt_p, prt_n);
    params->Ket->SetArray(prt_p, prt_n);
    double E = CalHFKernels(*(params->Hinput), *(params->Bra), *(params->Ket));
    mkl_free(prt);
    printf("  E:  %f \n", E);
    return E;
}

double myfuncGradient(unsigned n, const double *x, double *grad, void *my_func_data)
{
    MyData *params = reinterpret_cast<MyData *>(my_func_data);
    double *prt_p, *prt_n;
    double Qconstraint[6];
    int N_p = params->Hinput->ms->GetProtonNum();
    int N_n = params->Hinput->ms->GetNeutronNum();
    int dim_p = params->Hinput->ms->Get_MScheme_dim(Proton);
    int dim_n = params->Hinput->ms->Get_MScheme_dim(Neutron);
    double *prt = (double *)mkl_malloc((N_p * dim_p + N_n * dim_n) * sizeof(double), 64);

    prt_p = prt;
    prt_n = prt + N_p * dim_p;
    std::memcpy(prt, x, (N_p * dim_p + N_n * dim_n) * sizeof(double)); // copy the contents of src to dest
    normalizeBais(prt_p, N_p, dim_p, 1);
    normalizeBais(prt_n, N_n, dim_n, 1);

    params->Bra->SetArray(prt_p, prt_n);
    params->Ket->SetArray(prt_p, prt_n);
    double E = CalHFKernels(*(params->Hinput), *(params->Bra), *(params->Ket));
    double overlap = CalHFOverlap(*(params->Bra), *(params->Ket));
    if (params->Hinput->ms->GetIsShapeConstrained())
    {
        CalHF_Q2(*(params->Hinput), *(params->Bra), *(params->Ket), Qconstraint);
    }
    std::cout << E << std::endl;

    //-------------------------------------------------
    double *prt_grad_p, *prt_grad_n;
    double *prt_grad = (double *)mkl_malloc((n) * sizeof(double), 64);
    for (size_t i = 0; i < n; i++)
    {
        grad[i] = 0;
        std::memcpy(prt_grad, prt, n * sizeof(double)); // copy the contents of src to dest
        if (i < N_p * dim_p)                            // proton
        {
            int m = i / dim_p; // derivative on the m particle
            double norm = cblas_ddot(dim_p, x + m * dim_p, 1, x + m * dim_p, 1);
            cblas_dscal(dim_p, -x[i] / norm, prt_grad + m * dim_p, 1);
            prt_grad[i] += 1. / sqrt(norm) * sqrt(N_p);

            // memset(prt_grad + m * dim_p, 0, dim_p * sizeof(double));
            // prt_grad[i] = 1.;
        }
        else
        {
            int m = (i - N_p * dim_p) / dim_n; // derivative on the m particle
            double norm = cblas_ddot(dim_n, x + N_p * dim_p + m * dim_n, 1, x + N_p * dim_p + m * dim_n, 1);
            cblas_dscal(dim_n, -x[i] / norm, prt_grad + N_p * dim_p + m * dim_n, 1);
            prt_grad[i] += 1. / sqrt(norm) * sqrt(N_n);

            // memset(prt_grad + N_p * dim_p + m * dim_n, 0, dim_n * sizeof(double));
            // prt_grad[i] = 1.;
        }

        prt_grad_p = prt_grad;
        prt_grad_n = prt_grad + N_p * dim_p;
        params->Bra->SetArray(prt_p, prt_n);
        params->Ket->SetArray(prt_grad_p, prt_grad_n);
        grad[i] += (CalHF_Hamiltonian(*(params->Hinput), *(params->Bra), *(params->Ket)) + CalHF_Hamiltonian(*(params->Hinput), *(params->Ket), *(params->Bra))) / overlap;
        double overlap1 = CalHFOverlap(*(params->Bra), *(params->Ket));
        grad[i] -= 2 * overlap1 / overlap * E;

        //  Q constrant
        if (params->Hinput->ms->GetIsShapeConstrained())
        {
            double Qconstraint1[6];
            CalHF_Q2(*(params->Hinput), *(params->Bra), *(params->Ket), Qconstraint1);
            grad[i] += 2 * Qconstraint1[0] / overlap * Qconstraint[3];
            grad[i] += 2 * Qconstraint1[1] / overlap * Qconstraint[4];
            grad[i] += 2 * Qconstraint1[2] / overlap * Qconstraint[5];

            CalHF_Q2(*(params->Hinput), *(params->Ket), *(params->Bra), Qconstraint1);
            grad[i] += 2 * Qconstraint1[0] / overlap * Qconstraint[3];
            grad[i] += 2 * Qconstraint1[1] / overlap * Qconstraint[4];
            grad[i] += 2 * Qconstraint1[2] / overlap * Qconstraint[5];

            grad[i] -= 4 * overlap1 / overlap * Qconstraint[0] / overlap * Qconstraint[3];
            grad[i] -= 4 * overlap1 / overlap * Qconstraint[1] / overlap * Qconstraint[4];
            grad[i] -= 4 * overlap1 / overlap * Qconstraint[2] / overlap * Qconstraint[5];
        }
    }
    mkl_free(prt_grad);
    mkl_free(prt);
    return E;
}

void ShapeParameters(const double *x, void *my_func_data)
{
    MyData *params = reinterpret_cast<MyData *>(my_func_data);
    // double *prt_p, *prt_n;
    double *prt_p, *prt_n;
    int N_p = params->Hinput->ms->GetProtonNum();
    int N_n = params->Hinput->ms->GetNeutronNum();
    int dim_p = params->Hinput->ms->Get_MScheme_dim(Proton);
    int dim_n = params->Hinput->ms->Get_MScheme_dim(Neutron);
    double *prt = (double *)mkl_malloc((N_p * dim_p + N_n * dim_n) * sizeof(double), 64);
    prt_p = prt;
    prt_n = prt + N_p * dim_p;
    std::memcpy(prt, x, (N_p * dim_p + N_n * dim_n) * sizeof(double)); // copy the contents of src to dest
    normalizeBais(prt_p, N_p, dim_p, 1);
    normalizeBais(prt_n, N_n, dim_n, 1);
    params->Bra->SetArray(prt_p, prt_n);
    params->Ket->SetArray(prt_p, prt_n);
    CalHFShape(*(params->Hinput), *(params->Bra), *(params->Ket));
    mkl_free(prt);
    return;
}

double myfunc_Standard(unsigned n, const double *x, double *grad, void *my_func_data)
{
    MyData *params = reinterpret_cast<MyData *>(my_func_data);
    // double *prt_p, *prt_n;
    double *prt_p;
    double *prt_n;
    int N_p = params->Hinput->ms->GetProtonNum();
    int N_n = params->Hinput->ms->GetNeutronNum();
    int dim_p = params->Hinput->ms->Get_MScheme_dim(Proton);
    int dim_n = params->Hinput->ms->Get_MScheme_dim(Neutron);
    int inner_dim_p = params->Bra->GetProntonInnerDim();
    int inner_dim_n = params->Bra->GetNeutronInnerDim();

    double *prt = (double *)mkl_malloc((N_p * dim_p * inner_dim_p + N_n * dim_n * inner_dim_n) * sizeof(double), 64);
    prt_p = prt;
    prt_n = prt + N_p * dim_p * inner_dim_p;
    std::memcpy(prt, x, (N_p * dim_p * inner_dim_p + N_n * dim_n * inner_dim_n) * sizeof(double)); // copy the contents of src to dest
    normalizeBais(prt_p, N_p, dim_p, inner_dim_p);
    normalizeBais(prt_n, N_n, dim_n, inner_dim_n);
    params->Bra->SetArray(prt_p, inner_dim_p, prt_n, inner_dim_n);
    params->Ket->SetArray(prt_p, inner_dim_p, prt_n, inner_dim_n);
    double E = CalHFKernels_Advanced(*(params->Hinput), *(params->Bra), *(params->Ket));
    mkl_free(prt);
    printf("%f \n", E);
    return E;
}

double myfunc_advanced(unsigned n, const double *x, double *grad, void *my_func_data)
{
    MyData *params = reinterpret_cast<MyData *>(my_func_data);
    double *prt_p;
    double *prt_n;
    int N_p = params->Hinput->ms->GetProtonNum();
    int N_n = params->Hinput->ms->GetNeutronNum();
    int dim_p = params->Hinput->ms->Get_MScheme_dim(Proton);
    int dim_n = params->Hinput->ms->Get_MScheme_dim(Neutron);
    int inner_dim_p = params->Bra->GetProntonInnerDim();
    int inner_dim_n = params->Bra->GetNeutronInnerDim();

    double *prt = (double *)mkl_malloc((N_p * dim_p * inner_dim_p + N_n * dim_n * inner_dim_n) * sizeof(double), 64);
    prt_p = prt;
    prt_n = prt + N_p * dim_p * inner_dim_p;
    for (size_t i = 0; i < N_p / 2; i++)
    {
        std::memcpy(prt_p + i * dim_p * inner_dim_p * 2, x, (dim_p * inner_dim_p * 2) * sizeof(double)); // copy the contents of src to dest
    }

    for (size_t i = 0; i < N_n / 2; i++)
    {
        std::memcpy(prt_n + i * dim_n * inner_dim_n * 2, x + dim_p * inner_dim_p * 2, (dim_n * inner_dim_n * 2) * sizeof(double)); // copy the contents of src to dest
    }

    normalizeBais(prt_p, N_p, dim_p, inner_dim_p);
    normalizeBais(prt_n, N_n, dim_n, inner_dim_n);
    params->Bra->SetArray(prt_p, inner_dim_p, prt_n, inner_dim_n);
    params->Ket->SetArray(prt_p, inner_dim_p, prt_n, inner_dim_n);
    double E = CalHFKernels_Advanced(*(params->Hinput), *(params->Bra), *(params->Ket));
    mkl_free(prt);
    printf("%f \n", E);
    return E;
}

int main(int argc, char *argv[])
{
    int myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ///------- Initial NPSM --------///

    ReadWriteFiles rw;
    ModelSpace MS;
    Hamiltonian Hinput(MS);
    // OSLO format interaction
    /*
    rw.ReadInputInfo_pnSystem_GCM("InputFile_OSLO.dat", MS, Hinput);
    MS.InitialModelSpace_pn();
    Hinput.SetMassDep(true);
    rw.Read_InteractionFile_Mscheme(Hinput);
    // rw.Read_InteractionFile_Mscheme_Unrestricted_ForPhaffian(Hinput);
    */

    // Kshell format interaction
    rw.ReadInput_HF("Input_HF.txt", MS, Hinput);
    MS.InitialModelSpace_HF();
    Hinput.Prepare_MschemeH();
    // Hinput.Prepare_MschemeH_Unrestricted_ForPhaffian();

    if (myid == 0)
    {
        MS.PrintAllParameters_HF();
        Hinput.PrintHamiltonianInfo_pn();
        // AngMomProj.PrintInfo();
    }

    int N_p = MS.GetProtonNum();
    int N_n = MS.GetNeutronNum();
    int dim_p = MS.Get_MScheme_dim(Proton);
    int dim_n = MS.Get_MScheme_dim(Neutron);
    double *prt = (double *)mkl_malloc((N_p * dim_p + N_n * dim_n) * sizeof(double), 64);
    //________________________________________
    // srand(time(NULL)); // seed the random number generator with current time
    // srand(17);         // seed the random number generator with current time

    rw.Read_HF_Parameters_TXT(argv[1], prt);

    MyData paradata;
    PNbasis basis_bra(MS);
    PNbasis basis_ket(MS);
    paradata.Hinput = &Hinput;
    paradata.Bra = &basis_bra;
    paradata.Ket = &basis_ket;

    // Output shape
    normalizeBais(prt, N_p, dim_p, 1);
    normalizeBais(prt + N_p * dim_p, N_n, dim_n, 1);
    myfunc(N_p * dim_p + N_n * dim_n, prt,prt, &paradata);
    ShapeParameters(prt, &paradata);


    mkl_free(prt);
    MPI_Finalize();

}
