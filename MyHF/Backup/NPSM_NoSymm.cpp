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
#include "mpi.h" /////MPI
#include <math.h>

#include "ReadWriteFiles.h"
#include "NPSMCommutator.h"
#include "CalNPSM.h"

int main(int argc, char *argv[])
{
  int myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  ///------- Initial NPSM --------///

  ReadWriteFiles rw;
  ModelSpace MS;
  Hamiltonian Hinput(MS);
  rw.ReadInputInfo_pnSystem("InputFile_pn.dat", MS, Hinput);
  MS.InitialModelSpace_pn();
  rw.Read_InteractionFile_pn(Hinput);

  //AngMomProjection AngMomProj(MS);
  //AngMomProj.InitInt_pn();

  if (myid == 0)
  {
    MS.PrintAllParameters_pn();
    Hinput.PrintHamiltonianInfo_pn();
    //AngMomProj.PrintInfo();
  }

  // Initial parameters
  NPSM_No_projection myfun(MS, Hinput);
  MnUserParameters upar;
  rw.InitVariationPara_diff_pn(MS, &upar, argc, argv);

  double tolerance = 0.1; // edm = 0.001 ∗ tolerance ∗ up
  int Max_call = MS.GetMaxNumberOfIteration();

  // MnMigrad
  MnStrategy MyStrategy(0);  // 0 low; 1 medium; 2 high
  MnUserParameterState Mypar(upar);
  //Mypar.SetEDM(tolerance);


  MnMigrad migrad(myfun, Mypar, MyStrategy);
  //MnMigrad migrad(myfun, upar);
  
  FunctionMinimum min = migrad(Max_call, tolerance);
  //FunctionMinimum min = migrad();

  if (myid == 0)
  {
    rw.OutputResults(min, MS);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
