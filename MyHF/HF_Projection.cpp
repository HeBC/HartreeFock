#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mkl.h>
using namespace std;

#include "ReadWriteFiles.h"
#include "HFbasis.h"
#include "AngMom.h"
#include "Pfaffian_tools.h"
#include "GCM_Tools.h"
using namespace HF_Pfaffian_Tools;

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
    rw.ReadInputInfo_HF_GCM("Input_HF_GCM.txt", MS, Hinput);
    MS.InitialModelSpace_HF();
    Hinput.Prepare_MschemeH();
    // Hinput.Prepare_MschemeH_Unrestricted_ForPhaffian();

    AngMomProjection AngMomProj(MS);
    AngMomProj.InitInt_HF_Projection();

    if (myid == 0)
    {
        MS.PrintAllParameters_HF();
        Hinput.PrintHamiltonianInfo_pn();
        AngMomProj.PrintInfo();
    }

    int N_p = MS.GetProtonNum();
    int N_n = MS.GetNeutronNum();
    int dim_p = MS.Get_MScheme_dim(Proton);
    int dim_n = MS.Get_MScheme_dim(Neutron);
    //________________________________________
    srand(time(NULL)); // seed the random number generator with current time
    PNbasis basis_bra(MS, AngMomProj);
    PNbasis basis_ket(MS, AngMomProj);

    GCM_Projection myfun(MS, Hinput, AngMomProj);
    myfun.ReadBasis("Input/GCMpoints/");
    if (MS.Get_MeshType() == "LAmethod")
    {
        myfun.DoCalculation_LAmethod();
    }
    else
    {
        myfun.DoCalculation();
    }
    myfun.PrintResults();

    MPI_Finalize();
    return 0;
}
