#include "MCMC.h"

namespace MarkovChainMC
{
    //////////////////////////
    // class Walker
    //////////////////////////
    Walker::Walker(int dim)
    {
        this->dim = dim;
    };

    Walker::Walker(int dim, std::vector<double> initial_position)
    {
        this->dim = dim;
        this->setPosition(initial_position);
    };

    Walker::~Walker(){};

    std::vector<double> Walker::getPosition(int n) // get history position
    {
        return this->history[n];
    }

    std::vector<double> Walker::getPosition() // get current position
    {
        return this->position;
    }

    int Walker::getSteps()
    {
        return this->steps;
    }

    void Walker::setPosition(std::vector<double> new_position)
    {
        this->position = new_position;
        this->history.push_back(new_position);
        this->P_history.push_back(0.);
        this->steps++;
        return;
    }

    void Walker::setPosition(std::vector<double> new_position, double value)
    {
        this->Initial_proposal = true;
        this->position = new_position;
        this->history.push_back(new_position);
        this->Previous_proposal = value;
        this->P_history.push_back(value);
        this->steps++;
        return;
    }

    void Walker::setSamePosition(std::vector<double> new_position, double value)
    {
        this->Initial_proposal = true;
        this->position = new_position;
        this->history.push_back(new_position);
        this->Previous_proposal = value;
        this->steps++;
        return;
    }

    void Walker::clearHistory()
    {
        this->history.clear();
        this->history.push_back(this->position);
        this->steps = 1;
        return;
    }

    void Walker::Set_proposal(double value)
    {
        this->Initial_proposal = true;
        this->Previous_proposal = value;
        this->P_history[this->steps - 1] = value;
    };

    double Walker::Get_proposal()
    {
        if (this->Initial_proposal != true)
        {
            std::cout << " The walker doesn't initial the proposal!" << std::endl;
            exit(0);
        }
        return this->Previous_proposal;
    };

    //////////////////////////
    // class Ensemble
    //////////////////////////
    Ensemble::Ensemble(int numWalker, int dim, std::vector<Walker> initial_walkers)
    {
        this->walkers = initial_walkers;
        this->nwalkers = numWalker;
        this->dim = dim;
    };

    int Ensemble::AddNewWalker(Walker walker)
    {
        this->walkers.push_back(walker);
        this->nwalkers++;
        return this->nwalkers;
    };

    int Ensemble::ReplaceWalker(int index, Walker walker)
    {
        this->walkers[index] = walker;
        return 1;
    };

    Walker Ensemble::getWalker(int index)
    {
        Walker walker = walkers[index];
        this->nwalkers--;
        walkers.erase(walkers.begin() + index);
        return walker;
    };

    Walker Ensemble::getWalkerCopy(int index)
    {
        return this->walkers[index];
    };

    std::vector<std::vector<double>> Ensemble::getWalkerHistroy(int index)
    {
        return this->walkers[index].getHistory();
    }

    Walker Ensemble::getRandomWalker()
    {
        int k = rand() % nwalkers;
        Walker walker = walkers[k];
        walkers.erase(walkers.begin() + k);
        return walker;
    };

    Walker Ensemble::getRandomWalkerCopy()
    {
        int k = rand() % nwalkers;
        return this->walkers[k];
    };

    void Ensemble::cleanHistory()
    {
        for (int k = 0; k < nwalkers; k++)
        {
            this->walkers[k].clearHistory();
        };
        return;
    };

    std::vector<Ensemble> Ensemble::divideEnsemble()
    {
        std::size_t const half_size = this->walkers.size() / 2;
        std::vector<Walker> split_lo(this->walkers.begin(), this->walkers.begin() + half_size);
        std::vector<Walker> split_hi(this->walkers.begin() + half_size, this->walkers.end());

        Ensemble sample1((int)this->walkers.size() / 2, dim, split_lo);
        Ensemble sample2((int)this->walkers.size() / 2, dim, split_hi);

        std::vector<Ensemble> samples;
        samples.push_back(sample1);
        samples.push_back(sample2);

        return samples;
    };

    std::vector<Walker> Ensemble::getWalkers()
    {
        return this->walkers;
    };

    // class MCMC
    MCMC::MCMC(int numWalkers, int dim, std::vector<std::vector<double>> init_positions)
    {
        std::vector<Walker> walkers;
        for (int k_walk = 0; k_walk < numWalkers; k_walk++)
        {
            Walker single_walker(dim, init_positions[k_walk]);
            walkers.push_back(single_walker);
        };
        Ensemble create_sample(numWalkers, dim, walkers);
        this->sample = create_sample;
        this->nwalkers = numWalkers;
        this->dim = dim;
    };

    MCMC::MCMC(int numWalkers, int dim, char *file_name)
    {
        std::vector<std::vector<double>> init_positions = this->load_data(file_name);
        std::vector<Walker> walkers;
        for (int k_walk = 0; k_walk < numWalkers; k_walk++)
        {
            Walker single_walker(dim, init_positions[k_walk]);
            walkers.push_back(single_walker);
        };
        Ensemble create_sample(numWalkers, dim, walkers);
        this->sample = create_sample;
        this->nwalkers = numWalkers;
        this->dim = dim;
    };

    std::vector<std::vector<double>> MCMC::get_chain()
    {
        std::vector<std::vector<double>> walker_track;
        for (int k = 0; k < this->nwalkers; k++)
        {
            Walker walker;
            walker = this->sample.getWalkerCopy(k);
            int steps = walker.getSteps();
            for (int i = 0; i < steps; i++)
            {
                walker_track.push_back(walker.getHistory()[i]);
            };
        };
        return walker_track;
    };

    std::vector<std::vector<double>> MCMC::load_data(char *file_name)
    {
        std::ifstream file;
        std::vector<std::vector<double>> init_positions;
        file.open(file_name);

        if (!(file.is_open()))
        {
            std::cout << "\n\n ****You should pass an opened file!****\n\n";
            exit(0);
            return init_positions;
        };

        /*Load data*/
        do
        {
            std::string line, col;
            std::getline(file, line);

            // Create a stringstream from line
            std::stringstream ss(line);
            std::vector<double> walker_pos;
            // Extract each column name
            while (std::getline(ss, col, ','))
            {
                double value = std::stold(col);
                walker_pos.push_back(value);
            };
            init_positions.push_back(walker_pos);
        } while (!file.eof());
        return init_positions;
    }

    void MCMC::save_data(char *file_name)
    {
        std::ofstream file;
        file.open(file_name);
        if (!(file.is_open()))
        {
            std::cout << "\n\n ****You should pass an opened file!****\n\n";
            exit(0);
            return;
        };

        /*Writing samples*/
        for (int i = 0; i < this->nwalkers; i++)
        {
            Walker temp_walker;
            temp_walker = this->sample.getWalkerCopy(i);
            for (int j = 0; j < this->dim; j++)
            {
                file << temp_walker.getPosition()[j];
                if (j != this->dim - 1)
                    file << ",";
            };
            file << std::endl;
        };
        file.close();
        return;
    }

    void MCMC::save_chain(char *file_name, char *header)
    {
        std::ofstream sampling_file;
        sampling_file.open(file_name);

        std::vector<std::vector<double>> sampling_points;
        sampling_points = this->get_chain();
        sampling_file << header << std::endl;
        ;

        /*Writing samples*/
        int nsteps = sampling_points.size();
        for (int i = 0; i < nsteps; i++)
        {
            for (int j = 0; j < this->dim; j++)
            {
                sampling_file << sampling_points[i][j];
                if (j != this->dim - 1)
                    sampling_file << ",";
            };
            sampling_file << std::endl;
        };
        return;
    }

    // This special case of the algorithm, with P symmetric,
    // was first presented by Metropolis et al, 1953, and for
    // this reason it is sometimes called the “Metropolis algorithm”.

    // Initialize, X_1=x_1
    // For t=1,2,…
    //     sample y from P(y|x_t). Think of y as a “proposed” value for x_{t+1}
    //              y = x_t + N(0,1)
    //     Compute
    //                A=min(1, π(y) / π(x_t)).
    // A is often called the “acceptance probabilty”.
    // with probability A “accept” the proposed value, and set x_{t+1}=y, otherwise set x_{t+1}=x_t
    // random walk proposal P given above satisfies P(y|x)=P(x|y)
    void MCMC::run_Metropolis(double (*func)(std::vector<double>), int total_draws)
    {
        typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();
        myclock::time_point endtime;
        myclock::duration timer_now = beginning - endtime;
        unsigned seed = timer_now.count();
        std::default_random_engine generator(seed);
        std::normal_distribution<double> distribution(0, 1);

        // set normal distribution for Metropolis draws
        for (int i = 0; i < total_draws; i++) // loop draws
        {
            for (int k = 0; k < this->nwalkers; k++) // loop walkers
            {
                Walker walker;
                walker = sample.getWalker(0);
                std::vector<double> y, xt;
                double value, r, P, Current_proposal, Provious_proposal;
                xt = walker.getPosition();
                if (walker.IsProposalInitial())
                {
                    Provious_proposal = walker.Get_proposal();
                }
                else
                {
                    Provious_proposal = func(xt);
                    walker.Set_proposal(Current_proposal);
                }
                for (size_t loop_dim = 0; loop_dim < this->dim; loop_dim++)
                {
                    value = xt[loop_dim] + distribution(generator);
                    // std::cout << " Norm:  " << distribution(generator) << std::endl;
                    y.push_back(value);
                }

                // acceptance probabilty
                Current_proposal = func(y);
                P = Current_proposal / Provious_proposal;
                r = RandomGenerator();
                if (r <= std::min(1., P))
                {
                    walker.setPosition(y);
                    walker.Set_proposal(Current_proposal);
                }
                else
                {
                    // walker.setPosition(xt);
                    walker.setSamePosition(xt, Provious_proposal);
                }
                sample.AddNewWalker(walker);
            }
        }
    }

    double MCMC::RandomGenerator()
    {
        return double((rand() % 707525)) / 707525.;
    }

    std::vector<double> MCMC::GetNormDistribution(int dim, double mean, double sigma)
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(mean, sigma);
        std::vector<double> Norm_vector;
        for (size_t i = 0; i < dim; i++)
        {
            double temp = distribution(generator);
            Norm_vector.push_back(temp);
            std::cout << "Norm:   " << temp << std::endl;
        }
        return Norm_vector;
    }

} // namespace Markov chain Monte Carlo

double GuassionDis(std::vector<double> params)
{
    double value = 0;
    double mean[3] = {50, 1, 20};
    double dev[3] = {1, 1, 20};
    for (int i = 0; i < params.size(); i++)
    {
        value += exp(-(params[i] - mean[i]) * (params[i] - mean[i]) / (dev[i] * dev[i]));
    }
    // return value;
    if (params[0] >= 0.)
    {
        return exp(-params[0]);
    }
    else
    {
        return 0.;
    }
}
