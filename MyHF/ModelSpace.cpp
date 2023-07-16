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
         Codes are far away from bugs
          with the Buddha protecting
*/
#include "ModelSpace.h"
#include "AngMom.h"
#include "Hamiltonian.h"
#include "mkl.h"
#include <cmath>

MSchemeMatrix::~MSchemeMatrix()
{
    mkl_free(PairParity);
};

void MSchemeMatrix::Set_MScheme_Dim(int d) // Set M scheme dimension
{
    dim = d;
    dim2 = d * d;
}

double MSchemeMatrix::GetCGC(int t, int m, int i, int j)
{
    int SP = this->GetMschemeM_StartingPoint(t, m);
    return CGC_memory[SP + i * dim + j];
}

void ModelSpace::CheckComplexDefinition()
{
    if (sizeof(ComplexNum) != 2 * sizeof(double))
    {
        std::cout << "Please check the define of Complex number !!" << std::endl;
        ;
        exit(0);
    }
    return;
}

void ModelSpace::InitialModelSpace_Iden() // FOR NPSM
{
    InitMSMatrix(Proton);
    InitCollectivePairs(Proton);
}

void ModelSpace::InitialModelSpace_pn() // for NPSM
{
    InitMSMatrix(Proton);
    InitMSMatrix(Neutron);
    InitCollectivePairs(Proton);
    InitCollectivePairs(Neutron);
}

void ModelSpace::InitialModelSpace_HF()
{
    InitMSMatrix_HF(Proton);
    InitMSMatrix_HF(Neutron);
}

int ModelSpace::Get_MScheme_dim(int tz)
{
    if (tz == Proton) // find proton orbits
    {
        return this->Get_Proton_MScheme_dim();
    }
    else if (tz == Neutron)
    {
        return this->Get_Neutron_MScheme_dim();
    }
    else
    {
        std::cout << " Tz should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

int ModelSpace::Get_MScheme_dim2(int tz)
{
    if (tz == Proton) // find proton orbits
    {
        return this->Get_Proton_MScheme_dim2();
    }
    else if (tz == Neutron)
    {
        return this->Get_Neutron_MScheme_dim2();
    }
    else
    {
        std::cout << " Tz should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

int ModelSpace::Get_MSmatrix_2j(int isospin, int index)
{
    if (isospin == Proton) // find proton orbits
    {
        return this->MSM_p.Get_2j(index);
    }
    else if (isospin == Neutron)
    {
        return this->MSM_n.Get_2j(index);
    }
    else
    {
        std::cout << " Tz should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

int ModelSpace::Get_MSmatrix_2m(int isospin, int index)
{
    if (isospin == Proton) // find proton orbits
    {
        return this->MSM_p.Get_2m(index);
    }
    else if (isospin == Neutron)
    {
        return this->MSM_n.Get_2m(index);
    }
    else
    {
        std::cout << " Tz should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

// return 2 * Jmax
int ModelSpace::Get2Jmax()
{
    int Jmax2;
    std::vector<int> Proton_m, Neutron_m;
    int dim_proton = this->Get_Proton_MScheme_dim();
    int dim_neutron = this->Get_Neutron_MScheme_dim();
    for (size_t i = 0; i < dim_proton; i++)
    {
        Proton_m.push_back(this->MSM_p.Get_2m(i));
    }
    for (size_t i = 0; i < dim_neutron; i++)
    {
        Neutron_m.push_back(this->MSM_n.Get_2m(i));
    }

    // Sort the vector in descending order using a lambda expression
    std::sort(Proton_m.begin(), Proton_m.end(), [](int a, int b)
              {
                  return a > b; // Sort in descending order
              });

    // Sort the vector in descending order using a lambda expression
    std::sort(Neutron_m.begin(), Neutron_m.end(), [](int a, int b)
              {
                  return a > b; // Sort in descending order
              });

    Jmax2 = 0;
    for (size_t i = 0; i < this->GetProtonNum(); i++)
    {
        Jmax2 += Proton_m[i];
    }
    for (size_t i = 0; i < this->GetNeutronNum(); i++)
    {
        Jmax2 += Neutron_m[i];
    }
    return Jmax2;
}

// Look up the starting point in the M-scheme matrix
int ModelSpace::LookupStartingPoint(int isospin, int index)
{
    if (isospin == Proton) // find proton orbits
    {
        return this->MSM_p.LookupStartingPoint(index);
    }
    else if (isospin == Neutron)
    {
        return this->MSM_n.LookupStartingPoint(index);
    }
    else
    {
        std::cout << " Tz should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

MSchemeMatrix *ModelSpace::GetMSmatrixPointer(int isospin)
{
    if (isospin == Proton) // find proton orbits
    {
        return &MSM_p;
    }
    else if (isospin == Neutron)
    {
        return &MSM_n;
    }
    else
    {
        std::cout << " Tz should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

int ModelSpace::GetParticleNumber(int tz)
{
    if (tz == Proton)
    {
        return MS_N_p;
    }
    else
    {
        return MS_N_n;
    }
} // retrun particle number

int ModelSpace::GetPairNumber(int tz)
{
    if (tz == Proton) // find proton orbits
    {
        return this->GetProtonPairNum();
    }
    else if (tz == Neutron)
    {
        return this->GetNeutronPairNum();
    }
    else
    {
        std::cout << " Tz should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

double ModelSpace::Get_CGC(int isospin, int t, int m, int i, int j)
{
    if (isospin == Proton) // find proton orbits
    {
        return this->MSM_p.GetCGC(t, m, i, j);
    }
    else if (isospin == Neutron)
    {
        return this->MSM_n.GetCGC(t, m, i, j);
    }
    else
    {
        std::cout << " isospin should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

int ModelSpace::Get_OrbitIndex_Mscheme(int index, int isospin)
{
    if (isospin == Proton) // find proton orbits
    {
        return MSM_p.GetOrbitIndex(index);
    }
    else if (isospin == Neutron)
    {
        return MSM_n.GetOrbitIndex(index);
    }
    else
    {
        std::cout << " isospin should be Proton or Neutron !" << std::endl;
        exit(0);
    }
}

int ModelSpace::FindOrbit(int Tz, int n, int l, int j)
{
    if (Tz == Proton) // find proton orbits
    {
        for (size_t i = 0; i < GetProtonOrbitsNum(); i++)
        {
            if (Orbits_p[i].n == n and Orbits_p[i].l == l and Orbits_p[i].j2 == j)
            {
                return i; // return orbit index
            }
        }
    }
    else if (Tz == Neutron)
    {
        for (size_t i = 0; i < GetNeutronOrbitsNum(); i++)
        {
            if (Orbits_n[i].n == n and Orbits_n[i].l == l and Orbits_n[i].j2 == j)
            {
                return i; // return orbit index
            }
        }
    }
    printf("Don't find orbit n %d   l %d   j %d  Tz %d\n", n, l, j, Tz);
    return -1000;
}

int ModelSpace::FindOrbit(int Tz, int j) // return index
{
    if (Tz == Proton) // find proton orbits
    {
        for (size_t i = 0; i < GetProtonOrbitsNum(); i++)
        {
            if (Orbits_p[i].j2 == j)
            {
                return i; // return orbit index
            }
        }
    }
    else if (Tz == Neutron)
    {
        for (size_t i = 0; i < GetNeutronOrbitsNum(); i++)
        {
            if (Orbits_n[i].j2 == j)
            {
                return i; // return orbit index
            }
        }
    }
    printf("Don't find orbit  j %d  Tz %d\n", j, Tz);
    return -1000;
}

int ModelSpace::GetOrbitsNumber(int isospin)
{
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        return Orbits_p.size();
    }
    else if (isospin == Neutron)
    {
        return Orbits_n.size();
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
    return -1;
}

int ModelSpace::GetOrbit_2j(int index, int isospin)
{
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        return Orbits_p[index].j2;
    }
    else if (isospin == Neutron)
    {
        return Orbits_n[index].j2;
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
    return -1;
}

Orbit &ModelSpace::GetOrbit(int isospin, int i)
{
    if (isospin == Proton)
        return (Orbit &)Orbits_p[i];
    else

    {
        return (Orbit &)Orbits_n[i];
    }
}

void ModelSpace::InitMSMatrix(int isospin)
{
    int CountDim, countMSMdim;
    int i, j, jm, t, m, t_max;
    CountDim = 0;
    t_max = 0;
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        int NJ = this->GetProtonOrbitsNum();
        for (i = 0; i < NJ; i++)
        {
            MSM_p.SPj.push_back(CountDim);
            j = this->GetProtonOrbit_2j(i);
            if (t_max < j)
            {
                t_max = j;
            }
            CountDim += j + 1; // 2j+1
        }
        MSM_p.Set_MScheme_Dim(CountDim);
        j = 0;  // j
        t = 0;  // j index
        jm = 0; // count jm
        for (i = 0; i < CountDim; i++)
        {
            j = this->GetProtonOrbit_2j(t);
            MSM_p.j.push_back(j);
            MSM_p.j_index.push_back(t);
            MSM_p.m.push_back(j - 2 * jm);
            jm++;
            if (jm == (j + 1))
            {
                t++;
                jm = 0;
            }
        }
        // initial CG coefficient for M matrix
        countMSMdim = 0;
        for (t = 0; t <= t_max; t++)
        {
            for (m = -t; m <= t; m++)
            {
                MSM_p.CGC_lookup.insert(std::pair<std::array<int, 2>, int>({t, m}, countMSMdim));
                for (i = 0; i < CountDim; i++)
                {
                    for (j = 0; j < CountDim; j++)
                    {
                        MSM_p.CGC_memory.push_back(0.);
                        countMSMdim++;
                        if (MSM_p.m.at(i) + MSM_p.m.at(j) != 2 * m) // remove the restriction on i != j
                        {
                            continue;
                        }
                        MSM_p.CGC_memory[countMSMdim - 1] = AngMom::cgc(t * 1., m * 1., MSM_p.j[i] / 2., MSM_p.m[i] / 2., MSM_p.j[j] / 2., MSM_p.m[j] / 2.);
                    }
                }
            }
        }
        ///// Initial parity of operator
        MSM_p.PairParity = (ComplexNum *)mkl_malloc(CountDim * sizeof(ComplexNum), 64);
        for (i = 0; i < CountDim; i++)
        {
            t = sgn(GetProtonOrbit_l(MSM_p.j_index[i]));
            MSM_p.PairParity[i] = t;
        }
    }
    else if (isospin == Neutron)
    {
        int NJ = this->GetNeutronOrbitsNum();
        for (i = 0; i < NJ; i++)
        {
            MSM_n.SPj.push_back(CountDim);
            j = this->GetNeutronOrbit_2j(i);
            if (t_max < j)
            {
                t_max = j;
            }
            CountDim += j + 1; // 2j+1
        }
        MSM_n.Set_MScheme_Dim(CountDim);
        j = 0;  // j
        t = 0;  // j index
        jm = 0; // count jm
        for (i = 0; i < CountDim; i++)
        {
            j = this->GetNeutronOrbit_2j(t);
            MSM_n.j.push_back(j);
            MSM_n.j_index.push_back(t);
            MSM_n.m.push_back(j - 2 * jm);
            jm++;
            if (jm == (j + 1))
            {
                t++;
                jm = 0;
            }
        }
        // initial CG coefficient for M matrix
        countMSMdim = 0;
        for (t = 0; t <= t_max; t++)
        {
            for (m = -t; m <= t; m++)
            {
                MSM_n.CGC_lookup.insert(std::pair<std::array<int, 2>, int>({t, m}, countMSMdim));
                for (i = 0; i < CountDim; i++)
                {
                    for (j = 0; j < CountDim; j++)
                    {
                        MSM_n.CGC_memory.push_back(0.);
                        countMSMdim++;
                        if (MSM_n.m.at(i) + MSM_n.m.at(j) != 2 * m) // remove the restriction on i != j
                        {
                            continue;
                        }
                        MSM_n.CGC_memory[countMSMdim - 1] = AngMom::cgc(t * 1., m * 1., MSM_n.j[i] / 2., MSM_n.m[i] / 2., MSM_n.j[j] / 2., MSM_n.m[j] / 2.);
                    }
                }
            }
        }
        ///// Initial parity of pair
        MSM_n.PairParity = (ComplexNum *)mkl_malloc(CountDim * sizeof(ComplexNum), 64);
        for (i = 0; i < CountDim; i++)
        {
            t = sgn(GetProtonOrbit_l(MSM_n.j_index[i]));
            MSM_n.PairParity[i] = t;
        }
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
    return;
}

void ModelSpace::InitMSMatrix_HF(int isospin)
{
    int CountDim;
    int i, j, jm, t, m, t_max;
    CountDim = 0;
    t_max = 0;
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        int NJ = this->GetProtonOrbitsNum();
        for (i = 0; i < NJ; i++)
        {
            MSM_p.SPj.push_back(CountDim);
            j = this->GetProtonOrbit_2j(i);
            if (t_max < j)
            {
                t_max = j;
            }
            CountDim += j + 1; // 2j+1
        }
        MSM_p.Set_MScheme_Dim(CountDim);
        j = 0;  // j
        t = 0;  // j index
        jm = 0; // count jm
        for (i = 0; i < CountDim; i++)
        {
            j = this->GetProtonOrbit_2j(t);
            MSM_p.j.push_back(j);
            MSM_p.j_index.push_back(t);
            MSM_p.m.push_back(j - 2 * jm);
            jm++;
            if (jm == (j + 1))
            {
                t++;
                jm = 0;
            }
        }
        // initial CG coefficient for M matrix
        // HF don't need it
        ///// Initial parity of operator
        MSM_p.PairParity = (ComplexNum *)mkl_malloc(CountDim * sizeof(ComplexNum), 64);
        for (i = 0; i < CountDim; i++)
        {
            t = sgn(GetProtonOrbit_l(MSM_p.j_index[i]));
            MSM_p.PairParity[i] = t;
        }
    }
    else if (isospin == Neutron)
    {
        int NJ = this->GetNeutronOrbitsNum();
        for (i = 0; i < NJ; i++)
        {
            MSM_n.SPj.push_back(CountDim);
            j = this->GetNeutronOrbit_2j(i);
            if (t_max < j)
            {
                t_max = j;
            }
            CountDim += j + 1; // 2j+1
        }
        MSM_n.Set_MScheme_Dim(CountDim);
        j = 0;  // j
        t = 0;  // j index
        jm = 0; // count jm
        for (i = 0; i < CountDim; i++)
        {
            j = this->GetNeutronOrbit_2j(t);
            MSM_n.j.push_back(j);
            MSM_n.j_index.push_back(t);
            MSM_n.m.push_back(j - 2 * jm);
            jm++;
            if (jm == (j + 1))
            {
                t++;
                jm = 0;
            }
        }
        // initial CG coefficient for M matrix
        // HF don't need it
        ///// Initial parity of pair
        MSM_n.PairParity = (ComplexNum *)mkl_malloc(CountDim * sizeof(ComplexNum), 64);
        for (i = 0; i < CountDim; i++)
        {
            t = sgn(GetNeutronOrbit_l(MSM_n.j_index[i]));
            MSM_n.PairParity[i] = t;
        }
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
    return;
}

void ModelSpace::InitCollectivePairs(int isospin)
{
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        PairStructurePara_num_p = 0;
        int PairHierarchyNum = this->GetProtonCollectivePairNum();
        int NJ = this->GetProtonOrbitsNum();
        for (int t = 0; t < PairHierarchyNum; t++) // loop all pairs
        {
            vector<CollectivePairs> *ColPair = this->GetCollectivePairVectorPointer(isospin);
            int J = ColPair->at(t).GetJ();
            int PairParity = ColPair->at(t).GetParity();
            vector<int> *index_i = ColPair->at(t).GetVectorPointer_i();
            vector<int> *index_j = ColPair->at(t).GetVectorPointer_j();
            for (int i = 0; i < NJ; i++)
            {
                for (int j = i; j < NJ; j++)
                {
                    if (J % 2 == 1 && i == j)
                    {
                        continue; // blocking effect for identical nucleons
                    }
                    if ((this->GetProtonOrbit_2j(i) + this->GetProtonOrbit_2j(j)) / 2 >= J and abs((this->GetProtonOrbit_2j(i) - this->GetProtonOrbit_2j(j)) / 2) <= J)
                    {
                        if (this->GetProtonOrbits_parity(i) * this->GetProtonOrbits_parity(j) == PairParity) // here determine the parity
                        {
                            index_i->push_back(i); // record index of orbit
                            index_j->push_back(j);
                            PairStructurePara_num_p++;
                            // printf("pair index %d  J=%d  %d %d %d  %ld\n", t, J, i, j, PairParity, index_i->size());
                        }
                    }
                }
            }
        }
    }
    else if (isospin == Neutron)
    {
        PairStructurePara_num_n = 0;
        int PairHierarchyNum = this->GetNeutronCollectivePairNum();
        int NJ = this->GetNeutronOrbitsNum();
        for (int t = 0; t < PairHierarchyNum; t++) // loop all pairs
        {
            vector<CollectivePairs> *ColPair = this->GetCollectivePairVectorPointer(isospin);
            int J = ColPair->at(t).GetJ();
            int PairParity = ColPair->at(t).GetParity();
            vector<int> *index_i = ColPair->at(t).GetVectorPointer_i();
            vector<int> *index_j = ColPair->at(t).GetVectorPointer_j();
            for (int i = 0; i < NJ; i++)
            {
                for (int j = i; j < NJ; j++)
                {
                    if (J % 2 == 1 && i == j)
                    {
                        continue; // blocking effect for identical nucleons
                    }
                    if ((this->GetNeutronOrbit_2j(i) + this->GetNeutronOrbit_2j(j)) / 2 >= J and abs((this->GetNeutronOrbit_2j(i) - this->GetNeutronOrbit_2j(j)) / 2) <= J)
                    {
                        if (this->GetNeutronOrbits_parity(i) * this->GetNeutronOrbits_parity(j) == PairParity) // here determine the parity
                        {
                            index_i->push_back(i); // record index of orbit
                            index_j->push_back(j);
                            PairStructurePara_num_n++;
                            // printf("pair index %d  J=%d  %d %d %d  %ld\n", t, J, i, j, PairParity, index_i->size());
                        }
                    }
                }
            }
        }
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
    return;
}

vector<CollectivePairs> ModelSpace::GetCollectivePairVector(int isospin)
{
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        return CollectivePair_p;
    }
    else if (isospin == Neutron)
    {
        return CollectivePair_n;
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
}

vector<CollectivePairs> *ModelSpace::GetCollectivePairVectorPointer(int isospin)
{
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        return &CollectivePair_p;
    }
    else if (isospin == Neutron)
    {
        return &CollectivePair_n;
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
}

int ModelSpace::GetCollectivePairNumber(int isospin)
{
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        return CollectivePair_p.size();
    }
    else if (isospin == Neutron)
    {
        return CollectivePair_n.size();
    }
    else
    {
        printf("The particle should be proton or neutron!!  GetCollectivePairNumber()\n");
        exit(0);
    }
}

int ModelSpace::Get_NonCollecitvePairNumber(int isospin)
{
    if (isospin == Proton) // iflag=1  proton   else  nuetron
    {
        return PairStructurePara_num_p;
    }
    else if (isospin == Neutron)
    {
        return PairStructurePara_num_n;
    }
    else
    {
        printf("The particle should be proton or neutron!!\n");
        exit(0);
    }
}

int ModelSpace::GetNucleiMassA()
{
    if (MSNucleiMass != 0)
    {
        return MSNucleiMass;
    }
    else
    {
        return pcore + ncore + MS_N_p + MS_N_n;
    }
}

int ModelSpace::GetCoreMass()
{
    if (ncore != 0 or pcore != 0)
        return pcore + ncore;
    else
        return this->GetNucleiMassA() - this->GetProtonNum() - this->GetNeutronNum();
}

void ModelSpace::PrintAllParameters_Iden()
{
    std::cout << "/-----------------------------------------------------/" << std::endl;
    std::cout << " Number of Identical nucleons: " << 2 * this->GetProtonPairNum() << std::endl;
    std::cout << " Projected 2I: " << this->GetAMProjected_J() << "   2K: " << this->GetAMProjected_K() << "   2M: " << this->GetAMProjected_M() << std::endl;
    std::cout << " Read num of collective pairs:  " << this->GetCollectivePairNumber(Proton) << std::endl;
    if (this->IsJbrokenPair())
    {
        std::cout << " the num of M-scheme pair structures:   " << this->GetMSchemeNumberOfFreePara(Proton) << std::endl;
    }
    else
    {
        std::cout << " the num of pair structures:   " << this->Get_NonCollecitvePairNumber(Proton) << std::endl;
    }

    if (this->GetProjected_parity() == 0)
    {
        std::cout << " No parity projection included!" << std::endl;
    }
    else
    {
        std::cout << " Parity projection:  " << this->GetProjected_parity() << std::endl;
    }
    std::cout << " Total number of basis:  " << this->GetTotalOrders() << " number of summed basis: " << this->GetNumberOfBasisSummed() << std::endl;
}

void ModelSpace::PrintAllParameters_pn()
{
    std::cout << "/-----------------------------------------------------/" << std::endl;
    std::cout << " Number of protons:  " << 2 * this->GetProtonPairNum() << std::endl;
    std::cout << " Number of neutrons: " << 2 * this->GetNeutronPairNum() << std::endl;
    std::cout << " Projected 2I: " << this->GetAMProjected_J() << "   2K: " << this->GetAMProjected_K() << "   2M: " << this->GetAMProjected_M() << std::endl;
    std::cout << " Read num of collective pairs (Proton):  " << this->GetCollectivePairNumber(Proton) << std::endl;
    std::cout << " the num of pair structure (Proton):   " << this->Get_NonCollecitvePairNumber(Proton) << std::endl;
    std::cout << " Read num of collective pairs (Neutron):  " << this->GetCollectivePairNumber(Neutron) << std::endl;
    std::cout << " the num of pair structure (Neutron):   " << this->Get_NonCollecitvePairNumber(Neutron) << std::endl;
    if (this->GetProjected_parity() == 0)
    {
        std::cout << " No parity projection included!" << std::endl;
    }
    else
    {
        std::cout << " Parity projection:  " << this->GetProjected_parity() << std::endl;
    }
    std::cout << " Total number of basis:  " << this->GetTotalOrders() << " number of summed basis: " << this->GetNumberOfBasisSummed() << std::endl;

    if (this->GetBasisType() == 0) // 0 different pairs;    1 identical pairs
    {
        if (this->GetPairType() == 0) // 0 J conserved pairs;  1 J borken pairs
        {
            std::cout << " Basis are constructed by different J-conserved pairs! " << std::endl;
        }
        else
        {
            std::cout << " Basis are constructed by different J-broken pairs! " << std::endl;
        }
    }
    else
    {
        if (this->GetPairType() == 0) // 0 J conserved pairs;  1 J borken pairs
        {
            std::cout << " Basis are constructed by identical J-conserved pairs! " << std::endl;
        }
        else
        {
            std::cout << " Basis are constructed by identical J-broken pairs! " << std::endl;
        }
    }
}

void ModelSpace::PrintAllParameters_pn_GCM()
{
    std::cout << " Total number of basis:  " << this->GetTotalOrders() << std::endl;

    std::cout << " Number of protons:  " << 2 * this->GetProtonPairNum() << std::endl;
    std::cout << " Number of neutrons: " << 2 * this->GetNeutronPairNum() << std::endl;
    std::cout << " Projected 2I: " << this->GetAMProjected_J() << "   2K: " << this->GetAMProjected_K() << "   2M: " << this->GetAMProjected_M() << std::endl;

    if (this->GetProjected_parity() == 0)
    {
        std::cout << " No parity projection included!" << std::endl;
    }
    else
    {
        std::cout << " Parity projection:  " << this->GetProjected_parity() << std::endl;
    }
    if (this->GetBasisType() == 0) // 0 different pairs;    1 identical pairs
    {
        if (this->GetPairType() == 0) // 0 J conserved pairs;  1 J borken pairs
        {
            std::cout << " Basis are constructed by different J-conserved pairs! " << std::endl;
        }
        else
        {
            std::cout << " Basis are constructed by different J-broken pairs! " << std::endl;
        }
    }
    else
    {
        if (this->GetPairType() == 0) // 0 J conserved pairs;  1 J borken pairs
        {
            std::cout << " Basis are constructed by identical J-conserved pairs! " << std::endl;
        }
        else
        {
            std::cout << " Basis are constructed by identical J-broken pairs! " << std::endl;
        }
    }
}

void ModelSpace::PrintAllParameters_HF()
{
    std::cout << "/-----------------------------------------------------/" << std::endl;
    std::cout << "/                    Hartree-Fock                     /" << std::endl;
    std::cout << "/-----------------------------------------------------/" << std::endl;
    std::cout << "  Number of valence protons:  " << this->GetParticleNumber(Proton) << std::endl;
    std::cout << "  Number of valence neutrons: " << this->GetParticleNumber(Neutron) << std::endl;
    std::cout << "/-----------------------------------------------------/" << std::endl;
}

std::vector<double> *ModelSpace::GetCGC_prt(int isospin)
{
    if (isospin == Proton)
    {
        return MSM_p.GetCGC_prt();
    }
    else if (isospin == Neutron)
    {
        return MSM_n.GetCGC_prt();
    }
    else
    {
        std::cout << "Isospin should be Proton and Neutron!" << std::endl;
        exit(0);
    }
}

int ModelSpace::Get_CGC_StartPoint(int isospin, int t, int m)
{
    if (isospin == Proton)
    {
        return MSM_p.GetMschemeM_StartingPoint(t, m);
    }
    else if (isospin == Neutron)
    {
        return MSM_n.GetMschemeM_StartingPoint(t, m);
    }
    else
    {
        std::cout << "Isospin should be Proton and Neutron!" << std::endl;
        exit(0);
    }
}

ComplexNum *ModelSpace::GetParityProjOperator_prt(int isospin)
{
    if (isospin == Proton)
    {
        return MSM_p.GetParityProjOperator_prt();
    }
    else if (isospin == Neutron)
    {
        return MSM_n.GetParityProjOperator_prt();
    }
    else
    {
        std::cout << "Isospin should be Proton and Neutron!" << std::endl;
        exit(0);
    }
};

int ModelSpace::GetMSchemeNumberOfFreePara(int isospin)
{
    if (isospin == Proton)
    {
        return (MSM_p.Get_MScheme_Dim2() - MSM_p.Get_MScheme_Dim()) / 2;
    }
    else if (isospin == Neutron)
    {
        return (MSM_n.Get_MScheme_Dim2() - MSM_n.Get_MScheme_Dim()) / 2;
    }
    else
    {
        std::cout << "Isospin should be Proton and Neutron!" << std::endl;
        exit(0);
    }
};

void ModelSpace::SetGuassQuadMesh(int a, int b, int c)
{
    this->GQ_alpha = a;
    this->GQ_beta = b;
    this->GQ_gamma = c;
};

void ModelSpace::Initial_BrokenJPairIndex(int isospin)
{
    if (isospin == Proton)
    {
        int count = 0;
        int PairHierarchyNum = this->GetProtonCollectivePairNum();
        int NJ = this->GetProtonOrbitsNum();
        for (int t = 0; t < PairHierarchyNum; t++) // loop all pairs
        {
            vector<CollectivePairs> *ColPair = this->GetCollectivePairVectorPointer(Proton);
            int J = ColPair->at(t).GetJ();
            int PairParity = ColPair->at(t).GetParity();
            for (int i = 0; i < NJ; i++)
            {
                for (int j = i; j < NJ; j++)
                {
                    if (J % 2 == 1 && i == j)
                    {
                        continue; // blocking effect for identical nucleons
                    }
                    if ((this->GetProtonOrbit_2j(i) + this->GetProtonOrbit_2j(j)) / 2 >= J and abs((this->GetProtonOrbit_2j(i) - this->GetProtonOrbit_2j(j)) / 2) <= J)
                    {
                        if (this->GetProtonOrbits_parity(i) * this->GetProtonOrbits_parity(j) == PairParity) // here determine the parity
                        {
                            for (int M = -J; M <= J; M++)
                            {
                                for (int ma = this->GetProtonOrbit_2j(i); ma >= -this->GetProtonOrbit_2j(i); ma -= 2)
                                {
                                    for (int mb = this->GetProtonOrbit_2j(j); mb >= -this->GetProtonOrbit_2j(j); mb -= 2)
                                    {
                                        if (ma + mb != 2 * M)
                                        {
                                            continue;
                                        }
                                        int index_i = MSM_p.LookupIndexInMSmatrix(i, this->GetProtonOrbit_2j(i), ma);
                                        int index_j = MSM_p.LookupIndexInMSmatrix(j, this->GetProtonOrbit_2j(j), mb);
                                        if (index_i >= index_j)
                                        {
                                            continue;
                                        }
                                        /*if (index_i > index_j)
                                        {
                                            std::cout << "Pair index error! " << index_i << "  " << index_j << std::endl;
                                        }*/
                                        BrokenRotationalPairs temp(count, index_i, index_j, i, j, ma, mb, this->GetProtonOrbit_2j(i), this->GetProtonOrbit_2j(j), J, M, PairParity);
                                        BrokenRPairs_p.push_back(temp);
                                    }
                                }
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        // std::cout << "Total dim: " << count <<"  " << BrokenRPairs_p.size()<< std::endl;
        if (count != (this->Get_MScheme_dim2(isospin) - this->Get_MScheme_dim(isospin)) / 2)
        {
            std::cout << "Proton collective pairs don't include all kinds of pairs  " << count << "  " << (this->Get_MScheme_dim2(isospin) - this->Get_MScheme_dim(isospin)) / 2 << std::endl;
        }
    }
    else if (isospin == Neutron)
    {
        int count = 0;
        int PairHierarchyNum = this->GetNeutronCollectivePairNum();
        int NJ = this->GetNeutronOrbitsNum();
        for (int t = 0; t < PairHierarchyNum; t++) // loop all pairs
        {
            vector<CollectivePairs> *ColPair = this->GetCollectivePairVectorPointer(Neutron);
            int J = ColPair->at(t).GetJ();
            int PairParity = ColPair->at(t).GetParity();
            for (int i = 0; i < NJ; i++)
            {
                for (int j = i; j < NJ; j++)
                {
                    if (J % 2 == 1 && i == j)
                    {
                        continue; // blocking effect for identical nucleons
                    }
                    if ((this->GetNeutronOrbit_2j(i) + this->GetNeutronOrbit_2j(j)) / 2 >= J and abs((this->GetNeutronOrbit_2j(i) - this->GetNeutronOrbit_2j(j)) / 2) <= J)
                    {
                        if (this->GetNeutronOrbits_parity(i) * this->GetNeutronOrbits_parity(j) == PairParity) // here determine the parity
                        {
                            for (int M = -J; M <= J; M++)
                            {
                                for (int ma = -this->GetNeutronOrbit_2j(i); ma <= this->GetNeutronOrbit_2j(i); ma += 2)
                                {
                                    for (int mb = -this->GetNeutronOrbit_2j(j); mb <= this->GetNeutronOrbit_2j(j); mb += 2)
                                    {
                                        if (ma + mb != 2 * M)
                                        {
                                            continue;
                                        }
                                        int index_i = MSM_n.LookupIndexInMSmatrix(i, this->GetNeutronOrbit_2j(i), ma);
                                        int index_j = MSM_n.LookupIndexInMSmatrix(j, this->GetNeutronOrbit_2j(j), mb);
                                        if (index_i >= index_j)
                                        {
                                            continue;
                                        }
                                        BrokenRotationalPairs temp(count, index_i, index_j, i, j, ma, mb, this->GetNeutronOrbit_2j(i), this->GetNeutronOrbit_2j(j), J, (ma + mb) / 2, PairParity);
                                        BrokenRPairs_n.push_back(temp);
                                    }
                                }
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        // std::cout << "Total dim: " << count << "  " << BrokenRPairs_n.size() << std::endl;
        if (count != (this->Get_MScheme_dim2(isospin) - this->Get_MScheme_dim(isospin)) / 2)
        {
            std::cout << "Neutron collective pairs don't include all kinds of pairs  " << count << "  " << (this->Get_MScheme_dim2(isospin) - this->Get_MScheme_dim(isospin)) / 2 << std::endl;
        }
    }
    else
    {
        std::cout << "Isospin should be Proton and Neutron!" << std::endl;
        exit(0);
    }
};

vector<BrokenRotationalPairs> ModelSpace::GetBrokenJPairVector(int isospin)
{
    if (isospin == Proton)
    {
        return BrokenRPairs_p;
    }
    else if (isospin == Neutron)
    {
        return BrokenRPairs_n;
    }
    else
    {
        std::cout << "Isospin should be Proton and Neutron!" << std::endl;
        exit(0);
    }
};

// inspired from Ragnar's code, polished by B.C. He
void ModelSpace::GetAZfromString(std::string str, double &A, double &Z)
{
    std::vector<std::string> periodic_table = {"n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                                               "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                                               "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
                                               "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
                                               "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
                                               "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                                               "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

    int pos = 0;
    while (!isdigit(str[pos]))
        pos++;

    std::stringstream(str.substr(pos, str.size() - pos)) >> A;
    string strA = str.substr(pos, str.size() - pos);
    std::string elem;
    if (pos != std::string::npos)
    {
        // If subtractString is found, remove it from originalString
        elem = str.substr(0, pos) + str.substr(pos + std::to_string(static_cast<int>(A)).length());
        elem.erase(std::remove_if(elem.begin(), elem.end(), ::isspace), elem.end());
        std::string::size_type EnterPos = elem.find('\n');
        if (EnterPos != std::string::npos)
        {
            elem.erase(EnterPos);
        }
    }
    else
    {
        std::cout << "ModelSpace::GetAZfromString :  Trouble geting ele " << str << std::endl;
        exit(0);
    }
    auto it_elem = find(periodic_table.begin(), periodic_table.end(), elem);
    if (it_elem != periodic_table.end())
    {
        Z = it_elem - periodic_table.begin();
    }
    else
    {
        Z = -1;
        std::cout << "ModelSpace::GetAZfromString :  Trouble parsing " << str << std::endl;
        exit(0);
    }
}
