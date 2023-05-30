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
#include "MCMC.h"

using namespace MarkovChainMC;

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

    //MS.Initial_BrokenJPairIndex(Neutron);


    srand(MS.GetRandomSeed());
    //std::cout<< rand() % 1000 << "  " <<rand()% 1000 <<std::endl;

    int nwalkers = 1;
    int ndim = 1;  // number of parameters


    std::vector< std::vector<double> > init_pos;

    for(int k = 0; k < nwalkers; k++){
        double p1 = double(rand()% 550)/10.;
        std::vector<double> pos{p1};
        init_pos.push_back(pos);
    }
    char file_name[256] = "test_state.dat"; 
    

    MarkovChainMC::MCMC my_sample(nwalkers, ndim, init_pos);
    my_sample.run_Metropolis(GuassionDis, 1000);

    /*If you want to save data */
    
    my_sample.save_chain(file_name, file_name);


    return 0;
}
