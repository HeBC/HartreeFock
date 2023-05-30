#ifndef MarkovChainMonteCarlo_h
#define MarkovChainMonteCarlo_h 1

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <random>
#include <chrono>

namespace MarkovChainMC
{
    // Walker, record the history of each random moving point
    class Walker
    {
    private:
        std::vector<std::vector<double>> history; // record the history of moving
        std::vector<double> position;             // record the parameters
        std::vector<double> P_history;            // record the energies

        int steps = 0; // steps have been moved
        int dim;       // number of parameters in this problem.
        double Previous_proposal;
        bool Initial_proposal = false;

    public:
        std::vector<double> getPosition();          // get current position
        std::vector<double> getPosition(int index); // get history position
        void setPosition(std::vector<double> new_position);
        void setPosition(std::vector<double> new_position, double value);
        void setSamePosition(std::vector<double> new_position, double value);
        void Set_proposal(double value);
        double Get_proposal();
        int getSteps();
        std::vector<std::vector<double>> getHistory() { return this->history; };
        void clearHistory();
        bool IsProposalInitial() { return this->Initial_proposal; };
        double Get_p_history(int i) { return this->P_history[i]; };

        Walker(){};
        Walker(int dim);
        Walker(int dim, std::vector<double> initial_position);
        ~Walker();
    };

    class Ensemble
    {
    private:
        std::vector<Walker> walkers;
        int nwalkers = 0; // number of walkers
        int dim;          // number of parameters in each walker

    public:
        Walker getRandomWalker();
        Walker getRandomWalkerCopy();
        Walker getWalker(int index); // return a walker and erase it from Ensemble
        Walker getWalkerCopy(int index);
        std::vector<Walker> getWalkers();
        void setWalkers(std::vector<Walker> walkers) { this->walkers = walkers; };

        int AddNewWalker(Walker walker);
        int ReplaceWalker(int index, Walker walker);
        void cleanHistory();
        std::vector<Ensemble> divideEnsemble();
        std::vector<std::vector<double>> getWalkerHistroy(int index); // input the index of walkers, return the history of this walker

        Ensemble(int numWalker, int dim, std::vector<Walker> initial_walkers);
        Ensemble(){};
        ~Ensemble(){};
    };

    // Test MCMC for serial evaluation
    class MCMC
    {
    private:
        Ensemble sample;
        int nwalkers;
        int dim;

        bool constrained = true;
        double RandomGenerator(); // return a random double type number range from (0,1)
        std::vector<double> GetNormDistribution(int dim, double mean, double sigma);

    public:
        MCMC(int numWalkers, int dim, std::vector<std::vector<double>> init_positions);
        MCMC(int numWalkers, int dim, char *file_name);
        ~MCMC(){};

        // MCMC methods
        void run_Metropolis(double (*func)(std::vector<double>), int total_draws);

        std::vector<std::vector<double>> load_data(char *file_name);
        void save_data(char *file_name);
        std::vector<std::vector<double>> get_chain();
        void save_chain(char *file_name, char *header);
        // std::vector<std::vector<double>> get_chain_walkers();
        // int save_chain_walker(char *file_name, char *header);
    };

}

double GuassionDis(std::vector<double> params);

#endif