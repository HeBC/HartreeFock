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
#include "Hamiltonian.h"
#include "AngMom.h"
#include "mkl.h"
#include <cstring>
#include <algorithm>

HamiltonianElements::HamiltonianElements(int Tz2, int j1, int j2, int j3, int j4, int J, double V)
    : Tz2(Tz2), j1(j1), j2(j2), j3(j3), j4(j4), J(J), V(V)
{
}

HamiltoaninColllectiveElements::HamiltoaninColllectiveElements(int J, int Parity, int Sign, std::vector<double> y_pair)
    : J(J), Parity(Parity), Sign(Sign), y_pair(y_pair)
{
}

OneBodyOperatorChannel::OneBodyOperatorChannel(int orbit_a, int orbit_b, int t, int Tz2)
    : orbit_a(orbit_a), orbit_b(orbit_b), t(t), Tz2(Tz2)
{
}

MschemeHamiltonian& MschemeHamiltonian::operator=(const MschemeHamiltonian& other) {
    if (this != &other) {
        // Free existing resources
        mkl_free(ME_pp);
        mkl_free(ME_nn);
        mkl_free(ME_pn);

        // Copy simple and std::vector members
        // Copy ModelSpace pointer/reference and dimensions
        ms = other.ms;
        dim_p = other.dim_p;
        dim_n = other.dim_n;
        Hpp_index = other.Hpp_index;
        Hnn_index = other.Hnn_index;
        Hpn_index = other.Hpn_index;
        Hpn_OBindex = other.Hpn_OBindex;
        OB_p = other.OB_p;
        OB_n = other.OB_n;
        SPOindex_p = other.SPOindex_p;
        SPOindex_n = other.SPOindex_n;
        NumOB_p = other.NumOB_p;
        NumOB_n = other.NumOB_n;

        // Allocate and copy new resources
        int n = dim_p * dim_p * dim_p * dim_p;
        ME_pp = (double *)mkl_malloc(n * sizeof(double), 64);
        memcpy(ME_pp, other.ME_pp, sizeof(double) * n);

        n = dim_n * dim_n * dim_n * dim_n;
        ME_nn = (double *)mkl_malloc(n * sizeof(double), 64);
        memcpy(ME_nn, other.ME_nn, sizeof(double) * n);

        n = dim_p * dim_p * dim_n * dim_n;
        ME_pn = (double *)mkl_malloc(n * sizeof(double), 64);
        memcpy(ME_pn, other.ME_pn, sizeof(double) * n);
    }
    return *this;
}

void MschemeHamiltonian::Initial(ModelSpace &ms)
{
    this->ms = &ms;
    dim_p = ms.Get_MScheme_dim(Proton);
    dim_n = ms.Get_MScheme_dim(Neutron);
    double alpha = 0.0;
    int n = dim_p * dim_p * dim_p * dim_p;
    ME_pp = (double *)mkl_malloc((n) * sizeof(double), 64);
    memset(ME_pp, 0, sizeof(double) * n);

    n = dim_n * dim_n * dim_n * dim_n;
    ME_nn = (double *)mkl_malloc((n) * sizeof(double), 64);
    memset(ME_nn, 0, sizeof(double) * n);

    n = dim_p * dim_p * dim_n * dim_n;
    ME_pn = (double *)mkl_malloc((n) * sizeof(double), 64);
    memset(ME_pn, 0, sizeof(double) * n);
}

MschemeHamiltonian::~MschemeHamiltonian()
{
    mkl_free(ME_pp);
    mkl_free(ME_nn);
    mkl_free(ME_pn);
}

double MschemeHamiltonian::Vpp(int a, int b, int c, int d)
{
    return ME_pp[dim_p * dim_p * dim_p * a + dim_p * dim_p * b + dim_p * c + d];
}

double MschemeHamiltonian::Vnn(int a, int b, int c, int d)
{
    return ME_nn[dim_n * dim_n * dim_n * a + dim_n * dim_n * b + dim_n * c + d];
}

double MschemeHamiltonian::Vpn(int a, int b, int c, int d) // in format of ppnn
{
    return ME_pn[dim_p * dim_n * dim_n * a + dim_n * dim_n * b + dim_n * c + d];
}

void MschemeHamiltonian::add_Vpp(int a, int b, int c, int d, double V)
{
    ME_pp[dim_p * dim_p * dim_p * a + dim_p * dim_p * b + dim_p * c + d] += V;
}

void MschemeHamiltonian::add_Vnn(int a, int b, int c, int d, double V)
{
    ME_nn[dim_n * dim_n * dim_n * a + dim_n * dim_n * b + dim_n * c + d] += V;
}

void MschemeHamiltonian::add_Vpn(int a, int b, int c, int d, double V)
{
    ME_pn[dim_p * dim_n * dim_n * a + dim_n * dim_n * b + dim_n * c + d] += V;
}

void MschemeHamiltonian::Set_Vpp(int a, int b, int c, int d, double V)
{
    ME_pp[dim_p * dim_p * dim_p * a + dim_p * dim_p * b + dim_p * c + d] = V;
}

void MschemeHamiltonian::Set_Vnn(int a, int b, int c, int d, double V)
{
    ME_nn[dim_n * dim_n * dim_n * a + dim_n * dim_n * b + dim_n * c + d] = V;
}

void MschemeHamiltonian::Set_Vpn(int a, int b, int c, int d, double V)
{
    ME_pn[dim_p * dim_n * dim_n * a + dim_n * dim_n * b + dim_n * c + d] = V;
}

void MschemeHamiltonian::InitialMSOB() // Initial Mscheme one-body operator
{
    int dim_p = ms->Get_MScheme_dim(Proton);
    int dim_n = ms->Get_MScheme_dim(Neutron);
    NumOB_p = 0;
    for (int a = 0; a < dim_p; a++)
    {
        for (int b = 0; b < dim_p; b++)
        {
            if (a == b)
                SPOindex_p.push_back({NumOB_p, ms->Get_ProtonOrbitIndexInMscheme(a)});
            OB_p.push_back(MSOneBodyOperator(a, b, Proton));
            NumOB_p++;
        }
    }
    ////////////////
    NumOB_n = 0;
    for (int a = 0; a < dim_n; a++)
    {
        for (int b = 0; b < dim_n; b++)
        {
            if (a == b)
                SPOindex_n.push_back({NumOB_n, ms->Get_NeutronOrbitIndexInMscheme(a)});
            OB_n.push_back(MSOneBodyOperator(a, b, Neutron));
            NumOB_n++;
        }
    }
}

//---------------------------------------------------
//              class Hamiltonian
//---------------------------------------------------
Hamiltonian::Hamiltonian(ModelSpace &ms)
    : ms(&ms)
{
}

Hamiltonian::~Hamiltonian()
{
    if (SPE_malloc_p)
    {
        mkl_free(SPEmatrix_p);
    }
    if (SPE_malloc_n)
    {
        mkl_free(SPEmatrix_n);
    }
}

// copy constructor
Hamiltonian::Hamiltonian(const Hamiltonian& other)
    : ms(other.ms), 
      Vpp_filename(other.Vpp_filename), 
      Vnn_filename(other.Vnn_filename), 
      Vpn_filename(other.Vpn_filename), 
      snt_file(other.snt_file), 
      Vpp(other.Vpp), 
      Vnn(other.Vnn), 
      Vpn(other.Vpn), 
      OBEs_p(other.OBEs_p), 
      OBEs_n(other.OBEs_n), 
      VCol_pp(other.VCol_pp), 
      VCol_nn(other.VCol_nn), 
      VCol_pn(other.VCol_pn), 
      OBchannel_p(other.OBchannel_p), 
      OBchannel_n(other.OBchannel_n), 
      Vpn_PHcoupled(other.Vpn_PHcoupled),
      // Moments are not copied by design, adjust if necessary
      MSMEs(other.MSMEs), 
      // Skipping deprecated and not copied members
      H_has_been_Normlized(other.H_has_been_Normlized), 
      H_has_included_Hermitation(other.H_has_included_Hermitation), 
      H_CollectiveForm(other.H_CollectiveForm), 
      VpnISphcoupled(other.VpnISphcoupled), 
      TotalCalterm_pp(other.TotalCalterm_pp), 
      TotalCalterm_nn(other.TotalCalterm_nn), 
      TotalCalterm_pn(other.TotalCalterm_pn), 
      TotalCalOneBodyOperaotr_p(other.TotalCalOneBodyOperaotr_p), 
      TotalCalOneBodyOperaotr_n(other.TotalCalOneBodyOperaotr_n), 
      OB_max_t(other.OB_max_t), 
      t_p_max(other.t_p_max), 
      t_n_max(other.t_n_max), 
      IsMassDep(other.IsMassDep) 
{
}

// Assignment Operator (implementation)
Hamiltonian& Hamiltonian::operator=(const Hamiltonian& other) {
    if (this != &other) {
        this->ms = other.ms;
        this->Vpp_filename = other.Vpp_filename;
        this->Vnn_filename = other.Vnn_filename;
        this->Vpn_filename = other.Vpn_filename;
        this->snt_file = other.snt_file;
        this->Vpp = other.Vpp;  
        this->Vnn = other.Vnn;
        this->Vpn = other.Vpn;
        this->OBEs_p = other.OBEs_p;
        this->OBEs_n = other.OBEs_n;
        this->VCol_pp = other.VCol_pp;
        this->VCol_nn = other.VCol_nn;
        this->VCol_pn = other.VCol_pn;
        this->OBchannel_p = other.OBchannel_p;
        this->OBchannel_n = other.OBchannel_n;
        this->Vpn_PHcoupled = other.Vpn_PHcoupled;
        // currently, don't copy moments
        this->MSMEs = other.MSMEs;
        // deprecated ComplexNum *SPEmatrix_p, *SPEmatrix_n; bool SPE_malloc_p, SPE_malloc_n;
        this->H_has_been_Normlized = other.H_has_been_Normlized;
        this->H_has_included_Hermitation = other.H_has_included_Hermitation;
        this->H_CollectiveForm = other.H_CollectiveForm;
        this->VpnISphcoupled = other.VpnISphcoupled;
        this->TotalCalterm_pp = other.TotalCalterm_pp;
        this->TotalCalterm_nn = other.TotalCalterm_nn;
        this->TotalCalterm_pn = other.TotalCalterm_pn;
        this->TotalCalOneBodyOperaotr_p = other.TotalCalOneBodyOperaotr_p;
        this->TotalCalOneBodyOperaotr_n = other.TotalCalOneBodyOperaotr_n;
        this->OB_max_t = other.OB_max_t;
        this->t_p_max = other.t_p_max;
        this->t_n_max = other.t_n_max;
        this->IsMassDep = other.IsMassDep;
    }
    return *this;
}

void Hamiltonian::NormalHamiltonian() // add 1/4 and sqrt(2) for unrestricted
{
    double Normalfactor = std::sqrt(2); // add 1/4 here
    // Vpp
    int Vnum = GetVppNum();
    for (int i = 0; i < Vnum; i++)
    {
        Vpp[i].V *= 0.25;
        if (Vpp[i].j1 == Vpp[i].j2)
        {
            Vpp[i].V *= Normalfactor;
        }
        if (Vpp[i].j3 == Vpp[i].j4)
        {
            Vpp[i].V *= Normalfactor;
        }
    }
    // Vnn
    Vnum = GetVnnNum();
    for (int i = 0; i < Vnum; i++)
    {
        Vnn[i].V *= 0.25;
        if (Vnn[i].j1 == Vnn[i].j2)
        {
            Vnn[i].V *= Normalfactor;
        }
        if (Vnn[i].j3 == Vnn[i].j4)
        {
            Vnn[i].V *= Normalfactor;
        }
    }
    H_has_been_Normlized = true;
}

void Hamiltonian::AddHermitation()
{
    if (H_has_included_Hermitation)
    {
        std::cout << "The Hermitation should add only once!!" << std::endl;
    }

    // Vpp
    int Vnum = GetVppNum();
    for (size_t i = 0; i < Vnum; i++)
    {
        if (Vpp[i].j1 != Vpp[i].j3 or Vpp[i].j2 != Vpp[i].j4)
        {
            HamiltonianElements temp_ele = HamiltonianElements(Vpp[i].Tz2, Vpp[i].j3, Vpp[i].j4, Vpp[i].j1, Vpp[i].j2, Vpp[i].J, Vpp[i].V);
            Vpp.push_back(temp_ele);
        }
    }
    // Vnn
    Vnum = GetVnnNum();
    for (size_t i = 0; i < Vnum; i++)
    {
        if (Vnn[i].j1 != Vnn[i].j3 or Vnn[i].j2 != Vnn[i].j4)
        {
            HamiltonianElements temp_ele = HamiltonianElements(Vnn[i].Tz2, Vnn[i].j3, Vnn[i].j4, Vnn[i].j1, Vnn[i].j2, Vnn[i].J, Vnn[i].V);
            Vnn.push_back(temp_ele);
        }
    }
    // Vpn
    Vnum = GetVpnNum();
    for (size_t i = 0; i < Vnum; i++)
    {
        if (Vpn[i].j1 != Vpn[i].j3 or Vpn[i].j2 != Vpn[i].j4)
        {
            HamiltonianElements temp_ele = HamiltonianElements(Vpn[i].Tz2, Vpn[i].j3, Vpn[i].j4, Vpn[i].j1, Vpn[i].j2, Vpn[i].J, Vpn[i].V);
            Vpn.push_back(temp_ele);
        }
    }
    H_has_included_Hermitation = true;

    // add hermitation part for SP energy
    // Proton
    int temp_size = OBEs_p.size();
    for (size_t i = 0; i < temp_size; i++)
    {
        if (OBEs_p[i].orbit_a != OBEs_p[i].orbit_b)
        {
            OneBodyElement tempele = OneBodyElement(OBEs_p[i].orbit_b, OBEs_p[i].orbit_a, 0, Proton, OBEs_p[i].OBE);
            OBEs_p.push_back(tempele);
        }
    }
    // Neutron
    temp_size = OBEs_n.size();
    for (size_t i = 0; i < temp_size; i++)
    {
        if (OBEs_n[i].orbit_a != OBEs_n[i].orbit_b)
        {
            OneBodyElement tempele = OneBodyElement(OBEs_n[i].orbit_b, OBEs_n[i].orbit_a, 0, Neutron, OBEs_n[i].OBE);
            OBEs_n.push_back(tempele);
        }
    }
}

void Hamiltonian::RemoveWhitespaceInFilename()
{
    Vpp_filename.erase(std::remove_if(Vpp_filename.begin(), Vpp_filename.end(), ::isspace), Vpp_filename.end());
    Vpn_filename.erase(std::remove_if(Vpn_filename.begin(), Vpn_filename.end(), ::isspace), Vpn_filename.end());
    Vnn_filename.erase(std::remove_if(Vnn_filename.begin(), Vnn_filename.end(), ::isspace), Vnn_filename.end());
    snt_file.erase(std::remove_if(snt_file.begin(), snt_file.end(), ::isspace), snt_file.end());
}

void Hamiltonian::PrepareV_Identical()
{
    // for non-collective pairing interaction
    // We should include the Hermitations and normalize the
    // interaction
    if (this->H_IsCollective())
    {
        // Collective interaction is already initialized!
        std::cout << "Prepare Collective interaction!" << std::endl;
    }
    else
    {
        std::cout << "Prepare regular shell model interaction!" << std::endl;
        AddHermitation();
        NormalHamiltonian();
    }
    InitialSPEmatrix(Proton, *ms);
    // TranformHpnToPHRepresentation();

    CalNumberInitial_Iden();
    return;
}

void Hamiltonian::PrepareV_pnSystem_v1()
{
    AddHermitation();
    NormalHamiltonian();
    TranformHpnToPHRepresentation();
    InitialSPEmatrix(Proton, *ms);
    InitialSPEmatrix(Neutron, *ms);
    CalNumberInitial();
    return;
}

int Hamiltonian::FindOneBodyOperatorChannel(int a, int b, int t, int Tz)
{
    if (Tz == Proton)
    {
        for (int i = 0; i < this->OBchannel_p.size(); i++)
        {
            if (OBchannel_p[i].orbit_a == a and OBchannel_p[i].orbit_b == b and OBchannel_p[i].t == t)
            {
                return i;
            }
        }
        std::cout << "Don't find this one body operator!!  " << a << b << t << Tz << std::endl;
    }
    else if (Tz == Neutron)
    {
        for (int i = 0; i < this->OBchannel_n.size(); i++)
        {
            if (OBchannel_n[i].orbit_a == a and OBchannel_n[i].orbit_b == b and OBchannel_n[i].t == t)
            {
                return i;
            }
        }
        std::cout << "Don't find this one body operator!!  " << a << b << t << Tz << std::endl;
    }
    else
    {
        std::cout << "Tz should be Proton or Neuton!  " << Tz << std::endl;
        exit(0);
    }
    return -10000;
}

int Hamiltonian::FindVpnParticleHoleIndex(int Qp_index, int Qn_index)
{
    for (size_t i = 0; i < Vpn_PHcoupled.size(); i++)
    {
        if (Vpn_PHcoupled[i].QpOBChannelindex == Qp_index and Vpn_PHcoupled[i].QnOBChannelindex == Qn_index)
        {
            return i;
        }
    }
    std::cout << "Don't find the Vpn particle-hole channel!! " << Qp_index << "  " << Qn_index << std::endl;
    return -100000;
}

void Hamiltonian::TranformHpnToPHRepresentation() // Transform the Hpn part to Particle Hole representation
{
    if (!ISIncludedHermitationElements())
    {
        std::cout << "We should add hermitation part first!!" << std::endl;
        exit(0);
    }
    if (this->IS_Vpn_ph_coupled())
    {
        std::cout << "The Vpn has already transformed to particle-hole coupled form!" << std::endl;
        exit(0);
    }
    vector<HamiltonianElements> V_ph_form;
    // Generate One body channel
    // Proton One body
    int orb_num = ms->GetProtonOrbitsNum();
    for (size_t a = 0; a < orb_num; a++)
    {
        for (size_t b = 0; b < orb_num; b++)
        {
            int tmax = (ms->GetProtonOrbit_2j(a) + ms->GetProtonOrbit_2j(b)) / 2;
            int tmin = std::abs(ms->GetProtonOrbit_2j(a) - ms->GetProtonOrbit_2j(b)) / 2;
            for (size_t t = tmin; t <= tmax; t++)
            {
                // std::cout << OBchannel_p.size() << "  " << a << b << t << std::endl;
                OneBodyOperatorChannel tempOBC = OneBodyOperatorChannel(a, b, t, Proton);
                OBchannel_p.push_back(tempOBC);
            }
        }
    }

    // Neutron One body
    orb_num = ms->GetNeutronOrbitsNum();
    for (size_t a = 0; a < orb_num; a++)
    {
        for (size_t b = 0; b < orb_num; b++)
        {
            int tmax = (ms->GetNeutronOrbit_2j(a) + ms->GetNeutronOrbit_2j(b)) / 2;
            int tmin = std::abs(ms->GetNeutronOrbit_2j(a) - ms->GetNeutronOrbit_2j(b)) / 2;
            for (size_t t = tmin; t <= tmax; t++)
            {
                OneBodyOperatorChannel tempOBC = OneBodyOperatorChannel(a, b, t, Neutron);
                OBchannel_n.push_back(tempOBC);
            }
        }
    }

    // add Vpn particle-hole channel
    for (size_t a = 0; a < OBchannel_p.size(); a++)
    {
        for (size_t b = 0; b < OBchannel_n.size(); b++)
        {
            if (OBchannel_p[a].t == OBchannel_n[b].t)
            {
                Vpn_phCoupledElements temp = Vpn_phCoupledElements(a, b, OBchannel_p[a].t, 0.);
                Vpn_PHcoupled.push_back(temp);
            }
        }
    }

    // convert to particle hole format
    int ja, jb, jc, jd, J, t, Qp_index, Qn_index, Vpn_index;
    int a, b, c, d;
    double Vnor, Vph;
    AngMom::init_sixj();
    for (size_t i = 0; i < this->GetVpnNum(); i++) // loop all Vpn MEs
    {
        a = this->Vpn[i].j1;
        b = this->Vpn[i].j2;
        c = this->Vpn[i].j3;
        d = this->Vpn[i].j4;
        J = this->Vpn[i].J;
        Vnor = this->Vpn[i].V;
        ja = ms->GetProtonOrbit_2j(a);
        jb = ms->GetNeutronOrbit_2j(b);
        jc = ms->GetProtonOrbit_2j(c);
        jd = ms->GetNeutronOrbit_2j(d);
        for (t = std::abs((ja - jc)) / 2; t <= (ja + jc) / 2; t++)
        {
            if (t > (jb + jd) / 2 or t < std::abs(jb - jd) / 2)
            {
                continue;
            }
            Qp_index = this->FindOneBodyOperatorChannel(a, c, t, Proton);
            Qn_index = this->FindOneBodyOperatorChannel(b, d, t, Neutron);
            Vpn_index = FindVpnParticleHoleIndex(Qp_index, Qn_index);
            double tempV;
            Vph = this->GetVpn_phCoupledElements(Vpn_index);
            tempV = AngMom::U(ja, jb, jc, jd, 2 * J, 2 * t) * Vnor;
            Vph += sgn((ja + jd) / 2 + J) * std::sqrt(2 * J + 1) / std::sqrt(2 * t + 1) * tempV;
            // std::cout << Vph << std::endl;
            this->SetVpn_phCoupledElements(Vpn_index, Vph);
        }
    }
    // std::cout<< this->GetV_phCoupledChannelNum() <<std::endl;

    // remove redundant particle-hole Vpn channel
    vector<int> Removal_List;
    for (size_t i = 0; i < this->GetV_phCoupledChannelNum(); i++)
    {
        Vph = this->GetVpn_phCoupledElements(i);
        if (std::abs(Vph) < 1.e-8)
        {
            // std::cout << Vph << std::endl;
            Removal_List.push_back(i);
        }
    }
    // std::cout<< Removal_List.size() <<std::endl;
    int count_removal = 0;
    for (size_t i = 0; i < Removal_List.size(); i++)
    {
        Vpn_PHcoupled.erase(Vpn_PHcoupled.begin() + Removal_List[i] - count_removal);
        count_removal++;
    }
    VpnISphcoupled = true;
    return;
}

void Hamiltonian::PrintNonCollectiveV(int tz) // 2 Proton for Vpp, 2 Neutron for Vnn and 0 for Vpn
{

    if (tz == 2 * Proton)
    {
        int total_terms = Hamiltonian::GetVppNum();
        for (size_t i = 0; i < total_terms; i++)
        {
            std::cout << Vpp[i].j1 << " " << Vpp[i].j2 << " " << Vpp[i].j3 << " " << Vpp[i].j4 << " " << Vpp[i].Tz2 << " " << Vpp[i].J << " " << Vpp[i].V << std::endl;
        }
        std::cout << "Vpp total number: " << total_terms << std::endl;
    }
    else if (tz == 2 * Neutron)
    {
        int total_terms = Hamiltonian::GetVnnNum();
        for (size_t i = 0; i < total_terms; i++)
        {
            std::cout << Vnn[i].j1 << " " << Vnn[i].j2 << " " << Vnn[i].j3 << " " << Vnn[i].j4 << " " << Vnn[i].Tz2 << " " << Vnn[i].J << " " << Vnn[i].V << std::endl;
        }
        std::cout << "Vnn total number: " << total_terms << std::endl;
    }
    else if (tz == Proton + Neutron)
    {
        int total_terms = Hamiltonian::GetVpnNum();
        for (size_t i = 0; i < total_terms; i++)
        {
            std::cout << Vpn[i].j1 << " " << Vpn[i].j2 << " " << Vpn[i].j3 << " " << Vpn[i].j4 << " " << Vpn[i].Tz2 << " " << Vpn[i].J << " " << Vpn[i].V << std::endl;
        }
        std::cout << "Vpn total number: " << total_terms << std::endl;
    }
    return;
}

void Hamiltonian::Print_Vpn_ParticleHole()
{
    for (size_t i = 0; i < this->GetV_phCoupledChannelNum(); i++)
    {
        double Vph = this->GetVpn_phCoupledElements(i);
        std::cout << Vpn_PHcoupled[i].QpOBChannelindex << " " << Vpn_PHcoupled[i].QnOBChannelindex << " " << Vph << std::endl;
    }
    std::cout << "Vpn total number: " << this->GetV_phCoupledChannelNum() << std::endl;
    return;
}

void Hamiltonian::Print_MschemeHpp()
{
    for (int i = 0; i < MSMEs.Hpp_index.size(); i++)
    {
        int a = MSMEs.Hpp_index[i][0];
        int b = MSMEs.Hpp_index[i][1];
        int c = MSMEs.Hpp_index[i][2];
        int d = MSMEs.Hpp_index[i][3];
        std::cout << a << "  " << b << "  " << c << "  " << d << "  " << MSMEs.Vpp(a, b, c, d) << std::endl;
    }
}

void Hamiltonian::Print_MschemeHnn()
{
    for (int i = 0; i < MSMEs.Hnn_index.size(); i++)
    {
        int a = MSMEs.Hnn_index[i][0];
        int b = MSMEs.Hnn_index[i][1];
        int c = MSMEs.Hnn_index[i][2];
        int d = MSMEs.Hnn_index[i][3];
        std::cout << a << "  " << b << "  " << c << "  " << d << "  " << MSMEs.Vnn(a, b, c, d) << std::endl;
    }
}

void Hamiltonian::Print_MschemeHpn()
{
    for (int i = 0; i < MSMEs.Hpn_index.size(); i++)
    {
        int a = MSMEs.Hpn_index[i][0];
        int b = MSMEs.Hpn_index[i][1];
        int c = MSMEs.Hpn_index[i][2];
        int d = MSMEs.Hpn_index[i][3];
        std::cout << a << "  " << b << "  " << c << "  " << d << "  " << MSMEs.Vpn(a, b, c, d) << std::endl;
    }
}

void Hamiltonian::InitialSPEmatrix(int isospin, ModelSpace &ms) // Gnerate a matrix for single particle energy
{
    if (isospin == Proton)
    {
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        SPEmatrix_p = (ComplexNum *)mkl_malloc((dim2) * sizeof(ComplexNum), 64);
        memset(SPEmatrix_p, 0., dim2 * 2 * sizeof(double));
        for (int i = 0; i < dim; i++)
        {
            SPEmatrix_p[i * dim + i] = ms.GetProtonSPE(ms.Get_ProtonOrbitIndexInMscheme(i));
        }
        SPE_malloc_p = true;
    }
    else if (isospin == Neutron)
    {
        int dim2 = ms.Get_MScheme_dim2(isospin);
        int dim = ms.Get_MScheme_dim(isospin);
        SPEmatrix_n = (ComplexNum *)mkl_malloc((dim2) * sizeof(ComplexNum), 64);
        memset(SPEmatrix_n, 0., dim2 * 2 * sizeof(double));
        for (int i = 0; i < dim; i++)
        {
            SPEmatrix_n[i * dim + i] = ms.GetNeutronSPE(ms.Get_NeutronOrbitIndexInMscheme(i));
        }
        SPE_malloc_n = true;
    }
    else
    {
        std::cout << "The isospin should be Proton or Neutron! InitialSPEmatrix()" << std::endl;
        exit(0);
    }
}

ComplexNum *Hamiltonian::GetSPEpointer(int isospin)
{
    if (isospin == Proton)
    {
        return SPEmatrix_p;
    }
    else if (isospin == Neutron)
    {
        return SPEmatrix_n;
    }
    else
    {
        std::cout << "The isospin should be Proton or Neutron! InitialSPEmatrix()" << std::endl;
        exit(0);
    }
}

void Hamiltonian::CalNumberInitial_Iden() // count the number of terms need to calculate
{                                         // Only initalize TotalCalterm_pp
    if (H_CollectiveForm)                 // using collective pairng interaction
    {
        TotalCalterm_pp = 0;
        int num = GetColVppNum();
        for (size_t i = 0; i < num; i++)
        {
            TotalCalterm_pp += 2 * VCol_pp[i].J + 1;
        }
    }
    else
    {
        TotalCalterm_pp = 0;
        int num = GetVppNum();
        for (size_t i = 0; i < num; i++)
        {
            TotalCalterm_pp += 2 * Vpp[i].J + 1;
        }
    }
}

void Hamiltonian::CalNumberInitial() // count the number of terms need to calculate
{                                    // Only initalize TotalCalterm_pp
    if (H_CollectiveForm)            // using collective pairng interaction
    {
        TotalCalterm_pp = 0;
        int num = GetColVppNum();
        for (size_t i = 0; i < num; i++)
        {
            TotalCalterm_pp += 2 * VCol_pp[i].J + 1;
        }
        TotalCalterm_nn = 0;
        num = GetColVnnNum();
        for (size_t i = 0; i < num; i++)
        {
            TotalCalterm_nn += 2 * VCol_nn[i].J + 1;
        }
    }
    else
    {
        TotalCalterm_pp = 0;
        int num = GetVppNum();
        for (size_t i = 0; i < num; i++)
        {
            TotalCalterm_pp += 2 * Vpp[i].J + 1;
        }
        TotalCalterm_nn = 0;
        num = GetVppNum();
        for (size_t i = 0; i < num; i++)
        {
            TotalCalterm_nn += 2 * Vnn[i].J + 1;
        }
    }

    /// now we only use the nonCollective interaction for Vpn
    TotalCalterm_pn = 0;
    int num = GetV_phCoupledChannelNum();
    for (size_t i = 0; i < num; i++)
    {
        TotalCalterm_pn += 2 * OBchannel_p[Vpn_PHcoupled[i].QpOBChannelindex].t + 1;
    }

    //-----------------------
    TotalCalOneBodyOperaotr_p = 0;
    for (size_t i = 0; i < this->GetOneBodyOperatorNumber_p(); i++)
    {
        TotalCalOneBodyOperaotr_p += 2 * OBchannel_p[i].t + 1;
        if (OBchannel_p[i].t > t_p_max)
        {
            t_p_max = OBchannel_p[i].t;
        }
    }
    TotalCalOneBodyOperaotr_n = 0;
    for (size_t i = 0; i < this->GetOneBodyOperatorNumber_n(); i++)
    {
        TotalCalOneBodyOperaotr_n += 2 * OBchannel_n[i].t + 1;
        if (OBchannel_n[i].t > t_n_max)
        {
            t_n_max = OBchannel_n[i].t;
        }
    }
    OB_max_t = std::min(t_p_max, t_n_max);

    // Initial index for Quadrupole
    if (ms->GetIsShapeConstrained())
    {
        int tempcount;
        // Proton
        int orb_num = ms->GetProtonOrbitsNum();
        Q2MEs_p.Total_Number = 0;
        for (size_t a = 0; a < orb_num; a++)
        {
            for (size_t b = 0; b < orb_num; b++)
            {
                int tmax = (ms->GetProtonOrbit_2j(a) + ms->GetProtonOrbit_2j(b)) / 2;
                int tmin = std::abs(ms->GetProtonOrbit_2j(a) - ms->GetProtonOrbit_2j(b)) / 2;
                if (tmax < 2 or tmin > 2)
                {
                    continue;
                }
                Q2MEs_p.orbit_a.push_back(a);
                Q2MEs_p.orbit_b.push_back(b);
                Q2MEs_p.Q_MEs.push_back(this->Calculate_Q2(a, b, Proton));
                tempcount = 0;
                int channel_idx = FindOneBodyOperatorChannel(a, b, 2, Proton);
                for (size_t i = 0; i < channel_idx; i++)
                {
                    tempcount += 2 * OBchannel_p[i].t + 1;
                }
                Q2MEs_p.Q_2_list.push_back(tempcount);    // m = -2
                Q2MEs_p.Q0_list.push_back(tempcount + 2); // m = 0
                Q2MEs_p.Q2_list.push_back(tempcount + 4); // m = 2
                Q2MEs_p.Total_Number++;
            }
        }

        // Neutron
        orb_num = ms->GetNeutronOrbitsNum();
        Q2MEs_n.Total_Number = 0;
        for (size_t a = 0; a < orb_num; a++)
        {
            for (size_t b = 0; b < orb_num; b++)
            {
                int tmax = (ms->GetNeutronOrbit_2j(a) + ms->GetNeutronOrbit_2j(b)) / 2;
                int tmin = std::abs(ms->GetNeutronOrbit_2j(a) - ms->GetNeutronOrbit_2j(b)) / 2;
                if (tmax < 2 or tmin > 2)
                {
                    continue;
                }
                Q2MEs_n.orbit_a.push_back(a);
                Q2MEs_n.orbit_b.push_back(b);
                Q2MEs_n.Q_MEs.push_back(this->Calculate_Q2(a, b, Neutron));
                tempcount = 0;
                int channel_idx = FindOneBodyOperatorChannel(a, b, 2, Neutron);
                for (size_t i = 0; i < channel_idx; i++)
                {
                    tempcount += 2 * OBchannel_p[i].t + 1;
                }
                Q2MEs_n.Q_2_list.push_back(tempcount);    // m = -2
                Q2MEs_n.Q0_list.push_back(tempcount + 2); // m = 0
                Q2MEs_n.Q2_list.push_back(tempcount + 4); // m = 2
                Q2MEs_n.Total_Number++;
            }
        }
    }
}

// in unit of b^2
double Hamiltonian::Calculate_Q2(int a, int b, int tz)
{
    int phase;
    double res;
    double cc, dd;
    if (tz == Proton)
    {
        phase = (ms->GetProtonOrbit_2j(a) - 1) / 2;
        cc = 0.5 * ms->GetProtonOrbit_2j(a);
        dd = 0.5 * ms->GetProtonOrbit_2j(b);
        res = sgn(ms->GetProtonOrbit_n(a) + ms->GetProtonOrbit_n(b)) * (1 + sgn(ms->GetProtonOrbit_l(a) + ms->GetProtonOrbit_l(b))) / 2;
        res *= sgn(phase) * sqrt(ms->GetProtonOrbit_2j(a) + 1) * sqrt(ms->GetProtonOrbit_2j(b) + 1) / sqrt(4 * 3.1415926535) * AngMom::cgc(2, 0, cc, 0.5, dd, -0.5);
        if (fabs(res) < 1e-15)
            return 0;
        if (ms->GetProtonOrbit_l(a) == ms->GetProtonOrbit_l(b))
        {
            res *= ((double)(2 * ms->GetProtonOrbit_n(a) + ms->GetProtonOrbit_l(a)) + 1.5);
        }
        else if (ms->GetProtonOrbit_l(a) == ms->GetProtonOrbit_l(b) + 2)
        {
            res *= sqrt((2 * ms->GetProtonOrbit_n(b) + 2 * ms->GetProtonOrbit_l(b) + 3) * (2 * ms->GetProtonOrbit_n(b)));
        }
        else if (ms->GetProtonOrbit_l(a) == ms->GetProtonOrbit_l(b) - 2)
        {
            res *= sqrt((2 * ms->GetProtonOrbit_n(b) + 2 * ms->GetProtonOrbit_l(b) + 1) * (2 * ms->GetProtonOrbit_n(b) + 2));
        }
        else
        {
            printf("Qudruapole error!!!\n");
        }
        return res;
    }
    else // Neutron
    {
        phase = (ms->GetNeutronOrbit_2j(a) - 1) / 2;
        cc = 0.5 * ms->GetNeutronOrbit_2j(a);
        dd = 0.5 * ms->GetNeutronOrbit_2j(b);
        res = sgn(ms->GetNeutronOrbit_n(a) + ms->GetNeutronOrbit_n(b)) * (1 + sgn(ms->GetNeutronOrbit_l(a) + ms->GetNeutronOrbit_l(b))) / 2.;
        res *= sgn(phase) * sqrt(ms->GetNeutronOrbit_2j(a) + 1) * sqrt(ms->GetNeutronOrbit_2j(b) + 1) / sqrt(4 * 3.1415926535) * AngMom::cgc(2, 0, cc, 0.5, dd, -0.5);
        if (fabs(res) < 1e-15)
            return 0;
        if (ms->GetNeutronOrbit_l(a) == ms->GetNeutronOrbit_l(b))
        {
            res *= ((double)(2 * ms->GetNeutronOrbit_n(a) + ms->GetNeutronOrbit_l(a)) + 1.5);
        }
        else if (ms->GetNeutronOrbit_l(a) == ms->GetNeutronOrbit_l(b) + 2)
        {
            res *= sqrt((2 * ms->GetNeutronOrbit_n(b) + 2 * ms->GetNeutronOrbit_l(b) + 3) * (2 * ms->GetNeutronOrbit_n(b)));
        }
        else if (ms->GetNeutronOrbit_l(a) == ms->GetNeutronOrbit_l(b) - 2)
        {
            res *= sqrt((2 * ms->GetNeutronOrbit_n(b) + 2 * ms->GetNeutronOrbit_l(b) + 1) * (2 * ms->GetNeutronOrbit_n(b) + 2));
        }
        else
        {
            printf("Qudruapole error!!!\n");
        }
        return res;
    }
}

// return < r ^ lamda > in unit of b^lamda
// where b =  (b = 1.005 A^{1/6} fm)
double Hamiltonian::HarmonicRadialIntegral(int isospin, int lamda, int orbit_a, int orbit_b)
// return R_ab^lamda / b^lamda
{
    this->ms;
    int na, nb, la, lb;
    if (isospin == Proton)
    {
        int na = ms->GetProtonOrbit_n(orbit_a);
        int nb = ms->GetProtonOrbit_n(orbit_b);
        int la = ms->GetProtonOrbit_l(orbit_a);
        int lb = ms->GetProtonOrbit_l(orbit_b);
    }
    else
    {
        int na = ms->GetNeutronOrbit_n(orbit_a);
        int nb = ms->GetNeutronOrbit_n(orbit_b);
        int la = ms->GetNeutronOrbit_l(orbit_a);
        int lb = ms->GetNeutronOrbit_l(orbit_b);
    }

    double factor = sqrt(4 * gsl_sf_fact(na) * gsl_sf_fact(nb));
    factor /= sqrt(gsl_sf_gamma(na + la + 1.5) * gsl_sf_gamma(nb + lb + 1.5));

    int n = 50; // order of the quadrature rule
    gsl_integration_fixed_workspace *table = gsl_integration_fixed_alloc(gsl_integration_fixed_legendre, n, 0, 20, 0., 0.);
    // get the abscissae and weights from the table
    const double *x = gsl_integration_fixed_nodes(table);   // pointer to the abscissae
    const double *w = gsl_integration_fixed_weights(table); // pointer to the weights
    double integral = 0.0;

    for (int i = 0; i < n; i++)
    {
        double r = pow(x[i], la + lb + lamda + 2);
        double x2 = x[i] * x[i];
        double wf1 = gsl_sf_laguerre_n(na, la + 0.5, x2);
        double wf2 = gsl_sf_laguerre_n(nb, lb + 0.5, x2);
        integral += w[i] * wf1 * wf2 * r * exp(-x2);
    }
    gsl_integration_fixed_free(table);
    return integral * factor;
}

double Hamiltonian::Calculate_Q3(int a, int b, int tz)
{
    int phase;
    double res;
    double cc, dd;
    int lamda = 3;
    if (tz == Proton)
    {
        phase = (ms->GetProtonOrbit_2j(a) - 1) / 2 + lamda;
        cc = 0.5 * ms->GetProtonOrbit_2j(a);
        dd = 0.5 * ms->GetProtonOrbit_2j(b);
        res = (1 + sgn(ms->GetProtonOrbit_l(a) + ms->GetProtonOrbit_l(b) + lamda)) / 2;
        res *= sgn(phase) * sqrt(ms->GetProtonOrbit_2j(a) + 1) * sqrt(ms->GetProtonOrbit_2j(b) + 1) / sqrt(4 * 3.1415926535) * AngMom::cgc(lamda, 0, cc, 0.5, dd, -0.5);
        return res * HarmonicRadialIntegral(Proton, lamda, a, b);
    }
    else // Neutron
    {
        phase = (ms->GetNeutronOrbit_2j(a) - 1) / 2 + lamda;
        cc = 0.5 * ms->GetNeutronOrbit_2j(a);
        dd = 0.5 * ms->GetNeutronOrbit_2j(b);
        res = (1 + sgn(ms->GetNeutronOrbit_l(a) + ms->GetNeutronOrbit_l(b) + lamda)) / 2;
        res *= sgn(phase) * sqrt(ms->GetNeutronOrbit_2j(a) + 1) * sqrt(ms->GetNeutronOrbit_2j(b) + 1) / sqrt(4 * 3.1415926535) * AngMom::cgc(lamda, 0, cc, 0.5, dd, -0.5);
        return res * HarmonicRadialIntegral(Proton, lamda, a, b);
    }
}

// in unit of b^lamda
double Hamiltonian::Calculate_Qt(int lamda, int a, int b, int tz)
{
    int phase;
    double res;
    double cc, dd;
    if (tz == Proton)
    {
        phase = (ms->GetProtonOrbit_2j(a) - 1) / 2 + lamda;
        cc = 0.5 * ms->GetProtonOrbit_2j(a);
        dd = 0.5 * ms->GetProtonOrbit_2j(b);
        res = (1 + sgn(ms->GetProtonOrbit_l(a) + ms->GetProtonOrbit_l(b) + lamda)) / 2;
        res *= sgn(phase) * sqrt(ms->GetProtonOrbit_2j(a) + 1) * sqrt(ms->GetProtonOrbit_2j(b) + 1) / sqrt(4 * 3.1415926535) * AngMom::cgc(lamda, 0, cc, 0.5, dd, -0.5);
        return res * HarmonicRadialIntegral(Proton, lamda, a, b);
    }
    else // Neutron
    {
        phase = (ms->GetNeutronOrbit_2j(a) - 1) / 2 + lamda;
        cc = 0.5 * ms->GetNeutronOrbit_2j(a);
        dd = 0.5 * ms->GetNeutronOrbit_2j(b);
        res = (1 + sgn(ms->GetNeutronOrbit_l(a) + ms->GetNeutronOrbit_l(b) + lamda)) / 2;
        res *= sgn(phase) * sqrt(ms->GetNeutronOrbit_2j(a) + 1) * sqrt(ms->GetNeutronOrbit_2j(b) + 1) / sqrt(4 * 3.1415926535) * AngMom::cgc(lamda, 0, cc, 0.5, dd, -0.5);
        return res * HarmonicRadialIntegral(Proton, lamda, a, b);
    }
}

void Hamiltonian::PrintHamiltonianInfo_Iden()
{
    if (H_CollectiveForm)
    {
        std::cout << " the num of collective interaction terms: " << this->GetColVppNum() << std::endl;
    }
    else
    {
        std::cout << " the num of interaction terms: " << this->GetVppNum() << std::endl;
    }
}

void Hamiltonian::PrintHamiltonianInfo_pn()
{
    if (H_CollectiveForm)
    {
        std::cout << "  the num of collective Vpp terms: " << this->GetColVppNum() << std::endl;
        std::cout << "  the num of collective Vnn terms: " << this->GetColVnnNum() << std::endl;
        std::cout << "  the num of Vpn terms: " << this->GetV_phCoupledChannelNum() << std::endl;
    }
    else
    {
        if (std::fabs(ms->GetEnergyConstantShift()) > 0.1)
            std::cout << "  Found Zero body term: " << ms->GetEnergyConstantShift() << "  MeV"<< std::endl;
        std::cout << "  the num of Vpp terms: " << this->GetVppNum() << std::endl;
        std::cout << "  the num of Vnn terms: " << this->GetVnnNum() << std::endl;
        std::cout << "  the num of Vpn terms: " << this->GetVpnNum() << std::endl;
        std::cout << "/-----------------------------------------------------/" << std::endl;
    }
}

vector<HamiltoaninColllectiveElements> *Hamiltonian::GetColMatrixEle_prt(int isospin)
{
    if (isospin == Proton)
    {
        return &VCol_pp;
    }
    else if (isospin == Neutron)
    {
        return &VCol_nn;
    }
    else
    {
        std::cout << "isospin should be Proton or Neutron!" << std::endl;
        exit(0);
    }
}

OneBodyOperatorChannel Hamiltonian::GetOneBodyOperator(int isospin, int index)
{
    if (isospin == Proton)
    {
        return OBchannel_p[index];
    }
    else if (isospin == Neutron)
    {
        return OBchannel_n[index];
    }
    else
    {
        std::cout << "isospin should be Proton or Neutron!" << std::endl;
        exit(0);
    }
}

double Hamiltonian::USDmassScalingFactor(double mass, double powerf, double ReferenceA) //(18/A)^P
{
    double USDmass_scaling;
    USDmass_scaling = std::pow(ReferenceA / mass, powerf);
    return USDmass_scaling;
}

double Hamiltonian::GetMassDependentFactor()
{
    if (IsMassDep)
    {
        // std::cout<< ms->GetMassReferenceA()  << "  " << ms->GetNucleiMassA()<< "   " << ms->GetMassPowerFactor()<< "  " << ms->GetMassReferenceA() << "  "<<  ms->GetCoreMass()  << std::endl;
        if (ms->GetMassReferenceA() > 1)
        {
            return USDmassScalingFactor(ms->GetNucleiMassA(), ms->GetMassPowerFactor(), ms->GetMassReferenceA());
        }
        else
            return USDmassScalingFactor(ms->GetNucleiMassA(), ms->GetMassPowerFactor(), ms->GetCoreMass() + 2);
    }
    else
        return 1.;
}

int Hamiltonian::GetCoreMass()
{
    return ms->GetNucleiMassA() - ms->GetProtonNum() - ms->GetNeutronNum();
}

void Hamiltonian::Prepare_MschemeH_Unrestricted() /// work for the HF_Diag code
// Vpp and Vnn stored in a skewed way Vacbd, which may help to speed up the code
{
    int dim_p = ms->Get_MScheme_dim(Proton);
    int dim_n = ms->Get_MScheme_dim(Neutron);
    double MassDep = this->GetMassDependentFactor();
    // std::cout << MassDep <<std::endl;
    AddHermitation();
    MSMEs.Initial(*ms);
    double SqareRoot2 = sqrt(2.);
    // Vpp
    for (int i = 0; i < Vpp.size(); i++)
    {
        int a = Vpp[i].j1;
        int b = Vpp[i].j2;
        int c = Vpp[i].j3;
        int d = Vpp[i].j4;
        int J = Vpp[i].J;
        double V = Vpp[i].V;
        if (a > b or c > d)
        {
            std::cout << " Vpp format error! " << std::endl;
            exit(0);
        }
        int j1 = ms->GetProtonOrbit_2j(a);
        int j2 = ms->GetProtonOrbit_2j(b);
        int j3 = ms->GetProtonOrbit_2j(c);
        int j4 = ms->GetProtonOrbit_2j(d);
        if (a == b)
        {
            V *= SqareRoot2;
        }
        if (c == d)
        {
            V *= SqareRoot2;
        }
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vpp(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(d, j4, m4), MassDep * V * C1 * C2);
                    if (a != b and c == d)
                    {
                        MSMEs.add_Vpp(ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(d, j4, m4), -MassDep * V * C1 * C2);
                    }
                    if (a == b and c != d)
                    {
                        MSMEs.add_Vpp(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(d, j4, m4), ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(c, j3, m3), -MassDep * V * C1 * C2);
                    }
                    if (a != b and c != d)
                    {
                        MSMEs.add_Vpp(ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(d, j4, m4), -MassDep * V * C1 * C2);
                        MSMEs.add_Vpp(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(d, j4, m4), ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(c, j3, m3), -MassDep * V * C1 * C2);
                        MSMEs.add_Vpp(ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(d, j4, m4), ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(c, j3, m3), MassDep * V * C1 * C2);
                    }
                }
            }
        }
    }

    // Vnn
    for (int i = 0; i < Vnn.size(); i++)
    {
        int a = Vnn[i].j1;
        int b = Vnn[i].j2;
        int c = Vnn[i].j3;
        int d = Vnn[i].j4;
        int J = Vnn[i].J;
        double V = Vnn[i].V;
        if (a > b or c > d)
        {
            std::cout << " Vnn format error! " << std::endl;
            exit(0);
        }
        int j1 = ms->GetNeutronOrbit_2j(a);
        int j2 = ms->GetNeutronOrbit_2j(b);
        int j3 = ms->GetNeutronOrbit_2j(c);
        int j4 = ms->GetNeutronOrbit_2j(d);
        if (a == b)
        {
            V *= SqareRoot2;
        }
        if (c == d)
        {
            V *= SqareRoot2;
        }
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vnn(ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(c, j3, m3), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(d, j4, m4), MassDep * V * C1 * C2);

                    if (a != b and c == d)
                    {
                        MSMEs.add_Vnn(ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(c, j3, m3), ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(d, j4, m4), -MassDep * V * C1 * C2);
                    }
                    if (a == b and c != d)
                    {
                        MSMEs.add_Vnn(ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(d, j4, m4), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(c, j3, m3), -MassDep * V * C1 * C2);
                    }
                    if (a != b and c != d)
                    {
                        MSMEs.add_Vnn(ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(c, j3, m3), ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(d, j4, m4), -MassDep * V * C1 * C2);
                        MSMEs.add_Vnn(ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(d, j4, m4), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(c, j3, m3), -MassDep * V * C1 * C2);
                        MSMEs.add_Vnn(ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(d, j4, m4), ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(c, j3, m3), MassDep * V * C1 * C2);
                    }
                }
            }
        }
    }

    // Vpn
    MSMEs.InitialMSOB();
    for (int i = 0; i < Vpn.size(); i++)
    {
        int a = Vpn[i].j1;
        int b = Vpn[i].j2;
        int c = Vpn[i].j3;
        int d = Vpn[i].j4;
        int J = Vpn[i].J;
        double V = Vpn[i].V;
        int j1 = ms->GetProtonOrbit_2j(a);
        int j2 = ms->GetNeutronOrbit_2j(b);
        int j3 = ms->GetProtonOrbit_2j(c);
        int j4 = ms->GetNeutronOrbit_2j(d);
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vpn(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(d, j4, m4), MassDep * V * C1 * C2);
                }
            }
        }
    }

    // Initial index for Quadrupole
    if (ms->GetIsShapeConstrained())
    {
        this->Initial_Q2();
    }
}

void Hamiltonian::Prepare_MschemeH_Unrestricted_ForPhaffian() /// work for the HF_Pfaffian code
{
    int dim_p = ms->Get_MScheme_dim(Proton);
    int dim_n = ms->Get_MScheme_dim(Neutron);
    double MassDep = this->GetMassDependentFactor();
    // std::cout<< MassDep << std::endl;
    AddHermitation();
    MSMEs.Initial(*ms);
    double SqareRoot2 = sqrt(2.);
    // Vpp
    for (int i = 0; i < Vpp.size(); i++)
    {
        int a = Vpp[i].j1;
        int b = Vpp[i].j2;
        int c = Vpp[i].j3;
        int d = Vpp[i].j4;
        int J = Vpp[i].J;
        double V = 0.25 * Vpp[i].V;
        if (a > b or c > d)
        {
            std::cout << " Vpp format error! " << std::endl;
            exit(0);
        }
        int j1 = ms->GetProtonOrbit_2j(a);
        int j2 = ms->GetProtonOrbit_2j(b);
        int j3 = ms->GetProtonOrbit_2j(c);
        int j4 = ms->GetProtonOrbit_2j(d);
        if (a == b)
        {
            V *= SqareRoot2;
        }
        if (c == d)
        {
            V *= SqareRoot2;
        }
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vpp(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_p(d, j4, m4), -MassDep * V * C1 * C2);
                    if (a != b and c == d)
                    {
                        MSMEs.add_Vpp(ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_p(d, j4, m4), MassDep * V * C1 * C2);
                    }
                    if (a == b and c != d)
                    {
                        MSMEs.add_Vpp(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(d, j4, m4), ms->GetMS_index_p(c, j3, m3), MassDep * V * C1 * C2);
                    }
                    if (a != b and c != d)
                    {
                        MSMEs.add_Vpp(ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_p(d, j4, m4), MassDep * V * C1 * C2);
                        MSMEs.add_Vpp(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(d, j4, m4), ms->GetMS_index_p(c, j3, m3), MassDep * V * C1 * C2);
                        MSMEs.add_Vpp(ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(d, j4, m4), ms->GetMS_index_p(c, j3, m3), -MassDep * V * C1 * C2);
                    }
                }
            }
        }
    }
    for (int a = 0; a < dim_p; a++)
    {
        for (int b = 0; b < dim_p; b++)
        {
            if (a == b)
            {
                continue;
            }
            for (int c = 0; c < dim_p; c++)
            {
                for (int d = 0; d < dim_p; d++)
                {
                    if (c == d)
                    {
                        continue;
                    }
                    if (fabs(MSMEs.Vpp(a, b, c, d)) > 1.e-6)
                    {
                        MSMEs.Hpp_index.push_back({a, b, c, d});
                    }
                }
            }
        }
    }

    // Vnn
    for (int i = 0; i < Vnn.size(); i++)
    {
        int a = Vnn[i].j1;
        int b = Vnn[i].j2;
        int c = Vnn[i].j3;
        int d = Vnn[i].j4;
        int J = Vnn[i].J;
        double V = 0.25 * Vnn[i].V;
        if (a > b or c > d)
        {
            std::cout << " Vnn format error! " << std::endl;
            exit(0);
        }
        int j1 = ms->GetNeutronOrbit_2j(a);
        int j2 = ms->GetNeutronOrbit_2j(b);
        int j3 = ms->GetNeutronOrbit_2j(c);
        int j4 = ms->GetNeutronOrbit_2j(d);
        if (a == b)
        {
            V *= SqareRoot2;
        }
        if (c == d)
        {
            V *= SqareRoot2;
        }
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vnn(ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(c, j3, m3), ms->GetMS_index_n(d, j4, m4), -MassDep * V * C1 * C2);

                    if (a != b and c == d)
                    {
                        MSMEs.add_Vnn(ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(c, j3, m3), ms->GetMS_index_n(d, j4, m4), MassDep * V * C1 * C2);
                    }
                    if (a == b and c != d)
                    {
                        MSMEs.add_Vnn(ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(d, j4, m4), ms->GetMS_index_n(c, j3, m3), MassDep * V * C1 * C2);
                    }
                    if (a != b and c != d)
                    {
                        MSMEs.add_Vnn(ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(c, j3, m3), ms->GetMS_index_n(d, j4, m4), MassDep * V * C1 * C2);
                        MSMEs.add_Vnn(ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(d, j4, m4), ms->GetMS_index_n(c, j3, m3), MassDep * V * C1 * C2);
                        MSMEs.add_Vnn(ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(d, j4, m4), ms->GetMS_index_n(c, j3, m3), -MassDep * V * C1 * C2);
                    }
                }
            }
        }
    }
    for (int a = 0; a < dim_n; a++)
    {
        for (int b = 0; b < dim_n; b++)
        {
            if (a == b)
            {
                continue;
            }
            for (int c = 0; c < dim_n; c++)
            {
                for (int d = 0; d < dim_n; d++)
                {
                    if (c == d)
                    {
                        continue;
                    }
                    if (fabs(MSMEs.Vnn(a, b, c, d)) > 1.e-6)
                    {
                        MSMEs.Hnn_index.push_back({a, b, c, d});
                    }
                }
            }
        }
    }

    // Vpn
    MSMEs.InitialMSOB();
    for (int i = 0; i < Vpn.size(); i++)
    {
        int a = Vpn[i].j1;
        int b = Vpn[i].j2;
        int c = Vpn[i].j3;
        int d = Vpn[i].j4;
        int J = Vpn[i].J;
        double V = Vpn[i].V;
        int j1 = ms->GetProtonOrbit_2j(a);
        int j2 = ms->GetNeutronOrbit_2j(b);
        int j3 = ms->GetProtonOrbit_2j(c);
        int j4 = ms->GetNeutronOrbit_2j(d);
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vpn(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(d, j4, m4), MassDep * V * C1 * C2);
                }
            }
        }
    }
    for (int i = 0; i < MSMEs.OB_p.size(); i++)
    {
        for (int j = 0; j < MSMEs.OB_n.size(); j++)
        {
            int a = MSMEs.OB_p[i].GetIndex_a();
            int b = MSMEs.OB_p[i].GetIndex_b();
            int c = MSMEs.OB_n[j].GetIndex_a();
            int d = MSMEs.OB_n[j].GetIndex_b();
            if (fabs(MSMEs.Vpn(a, b, c, d)) > 1.e-6)
            {
                MSMEs.Hpn_OBindex.push_back({i, j});
                // std::cout << a << "  " << b << "  " << c << "  " << d << "  " << MSMEs.Vpn(a, b, c, d) << std::endl;
            }
        }
    }

    // Initial index for Quadrupole
    // if (ms->GetIsShapeConstrained())  //always initial Q2
    if (ms->GetIsShapeConstrained())
    {
        this->Initial_Q2();
    }
}

void Hamiltonian::Prepare_MschemeH() // restricted H for pfaffian code
{
    int dim_p = ms->Get_MScheme_dim(Proton);
    int dim_n = ms->Get_MScheme_dim(Neutron);
    double MassDep = this->GetMassDependentFactor();
    // std::cout<< MassDep << std::endl;
    AddHermitation();
    MSMEs.Initial(*ms);
    double SqareRoot2 = sqrt(2);
    // Vpp
    for (int i = 0; i < Vpp.size(); i++)
    {
        int a = Vpp[i].j1;
        int b = Vpp[i].j2;
        int c = Vpp[i].j3;
        int d = Vpp[i].j4;
        int J = Vpp[i].J;
        double V = Vpp[i].V;
        if (a > b or c > d)
        {
            std::cout << " Vpp format error! " << std::endl;
            exit(0);
        }
        int j1 = ms->GetProtonOrbit_2j(a);
        int j2 = ms->GetProtonOrbit_2j(b);
        int j3 = ms->GetProtonOrbit_2j(c);
        int j4 = ms->GetProtonOrbit_2j(d);
        if (a == b)
        {
            V *= SqareRoot2;
        }
        if (c == d)
        {
            V *= SqareRoot2;
        }
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vpp(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(b, j2, m2), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_p(d, j4, m4), -MassDep * V * C1 * C2);
                }
            }
        }
    }
    for (int a = 0; a < dim_p; a++)
    {
        for (int b = a + 1; b < dim_p; b++)
        {
            for (int c = 0; c < dim_p; c++)
            {
                for (int d = c + 1; d < dim_p; d++)
                {
                    if (fabs(MSMEs.Vpp(a, b, c, d)) > 1.e-6)
                    {
                        MSMEs.Hpp_index.push_back({a, b, c, d});
                    }
                }
            }
        }
    }

    // Vnn
    for (int i = 0; i < Vnn.size(); i++)
    {
        int a = Vnn[i].j1;
        int b = Vnn[i].j2;
        int c = Vnn[i].j3;
        int d = Vnn[i].j4;
        int J = Vnn[i].J;
        double V = Vnn[i].V;
        if (a > b or c > d)
        {
            std::cout << " Vnn format error! " << std::endl;
            exit(0);
        }
        int j1 = ms->GetNeutronOrbit_2j(a);
        int j2 = ms->GetNeutronOrbit_2j(b);
        int j3 = ms->GetNeutronOrbit_2j(c);
        int j4 = ms->GetNeutronOrbit_2j(d);
        if (a == b)
        {
            V *= SqareRoot2;
        }
        if (c == d)
        {
            V *= SqareRoot2;
        }
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vnn(ms->GetMS_index_n(a, j1, m1), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(c, j3, m3), ms->GetMS_index_n(d, j4, m4), -MassDep * V * C1 * C2);
                }
            }
        }
    }
    for (int a = 0; a < dim_n; a++)
    {
        for (int b = a + 1; b < dim_n; b++)
        {
            for (int c = 0; c < dim_n; c++)
            {
                for (int d = c + 1; d < dim_n; d++)
                {
                    if (fabs(MSMEs.Vnn(a, b, c, d)) > 1.e-6)
                    {
                        MSMEs.Hnn_index.push_back({a, b, c, d});
                    }
                }
            }
        }
    }

    // Vpn
    MSMEs.InitialMSOB();
    for (int i = 0; i < Vpn.size(); i++)
    {
        int a = Vpn[i].j1;
        int b = Vpn[i].j2;
        int c = Vpn[i].j3;
        int d = Vpn[i].j4;
        int J = Vpn[i].J;
        double V = Vpn[i].V;
        int j1 = ms->GetProtonOrbit_2j(a);
        int j2 = ms->GetNeutronOrbit_2j(b);
        int j3 = ms->GetProtonOrbit_2j(c);
        int j4 = ms->GetNeutronOrbit_2j(d);
        for (int M = -J; M <= J; M++)
        {
            for (int m1 = -j1; m1 <= j1; m1 += 2)
            {
                int m2 = 2 * M - m1;
                if (m2 < -j2 or m2 > j2)
                {
                    continue;
                }
                double C1 = AngMom::cgc(J, M, j1 * 0.5, m1 * 0.5, j2 * 0.5, m2 * 0.5);
                for (int m3 = -j3; m3 <= j3; m3 += 2)
                {
                    int m4 = 2 * M - m3;
                    if (m4 < -j4 or m4 > j4)
                    {
                        continue;
                    }
                    double C2 = AngMom::cgc(J, M, j3 * 0.5, m3 * 0.5, j4 * 0.5, m4 * 0.5);
                    MSMEs.add_Vpn(ms->GetMS_index_p(a, j1, m1), ms->GetMS_index_p(c, j3, m3), ms->GetMS_index_n(b, j2, m2), ms->GetMS_index_n(d, j4, m4), MassDep * V * C1 * C2);
                }
            }
        }
    }
    for (int i = 0; i < MSMEs.OB_p.size(); i++)
    {
        for (int j = 0; j < MSMEs.OB_n.size(); j++)
        {
            int a = MSMEs.OB_p[i].GetIndex_a();
            int b = MSMEs.OB_p[i].GetIndex_b();
            int c = MSMEs.OB_n[j].GetIndex_a();
            int d = MSMEs.OB_n[j].GetIndex_b();
            if (fabs(MSMEs.Vpn(a, b, c, d)) > 1.e-6)
            {
                MSMEs.Hpn_OBindex.push_back({i, j});
                // std::cout << a << "  " << b << "  " << c << "  " << d << "  " << MSMEs.Vpn(a, b, c, d) << std::endl;
            }
        }
    }

    // Initial index for Quadrupole
    // if (ms->GetIsShapeConstrained())  //always initial Q2
    if (ms->GetIsShapeConstrained())
    {
        this->Initial_Q2();
    }
}

void Hamiltonian::Initial_Q2()
{
    double QME;
    // Proton
    int orb_num = ms->GetProtonOrbitsNum();
    for (size_t a = 0; a < orb_num; a++)
    {
        for (size_t b = 0; b < orb_num; b++)
        {
            int tmax = (ms->GetProtonOrbit_2j(a) + ms->GetProtonOrbit_2j(b)) / 2;
            int tmin = std::abs(ms->GetProtonOrbit_2j(a) - ms->GetProtonOrbit_2j(b)) / 2;
            if (tmax < 2 or tmin > 2)
            {
                continue;
            }
            // Q2MEs_p.orbit_a.push_back(a);
            // Q2MEs_p.orbit_b.push_back(b);
            double QMEJ = this->Calculate_Q2(a, b, Proton);
            // std::cout << a << "   " << b << "    " << QMEJ << std::endl;
            for (size_t i = 0; i < MSMEs.OB_p.size(); i++) // loop m-scheme OB operator
            {
                int ia = MSMEs.OB_p[i].GetIndex_a(); // index of a in M scheme
                int ib = MSMEs.OB_p[i].GetIndex_b();
                int orb_ia = ms->Get_ProtonOrbitIndexInMscheme(ia); // index of j orbit a in J scheme
                int orb_ib = ms->Get_ProtonOrbitIndexInMscheme(ib);
                int aj = ms->Get_MSmatrix_2j(Proton, ia); // 2j of orbit a
                int bj = ms->Get_MSmatrix_2j(Proton, ib);
                int am = ms->Get_MSmatrix_2m(Proton, ia); // 2m of orbit a
                int bm = ms->Get_MSmatrix_2m(Proton, ib);
                if (orb_ia != a or orb_ib != b)
                {
                    continue;
                }
                if (am - bm == -4) // -2m
                {
                    double CGC = AngMom::cgc(2., -2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj - bm) / 2) * CGC;
                    Q2MEs_p.Q_2_MSMEs.push_back(QME);
                    Q2MEs_p.Q_2_list.push_back(i);
                }
                else if (am == bm)
                {
                    double CGC = AngMom::cgc(2., 0., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj - bm) / 2) * CGC;
                    Q2MEs_p.Q0_MSMEs.push_back(QME);
                    Q2MEs_p.Q0_list.push_back(i);
                }
                else if (am - bm == 4)
                {
                    double CGC = AngMom::cgc(2., 2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj - bm) / 2) * CGC;
                    Q2MEs_p.Q2_MSMEs.push_back(QME);
                    Q2MEs_p.Q2_list.push_back(i);
                }
            }
        }
    }

    // Neutron
    orb_num = ms->GetNeutronOrbitsNum();
    for (size_t a = 0; a < orb_num; a++)
    {
        for (size_t b = 0; b < orb_num; b++)
        {
            int tmax = (ms->GetNeutronOrbit_2j(a) + ms->GetNeutronOrbit_2j(b)) / 2;
            int tmin = std::abs(ms->GetNeutronOrbit_2j(a) - ms->GetNeutronOrbit_2j(b)) / 2;
            if (tmax < 2 or tmin > 2)
            {
                continue;
            }
            // Q2MEs_n.orbit_a.push_back(a);
            // Q2MEs_n.orbit_b.push_back(b);
            double QMEJ = this->Calculate_Q2(a, b, Neutron);
            for (size_t i = 0; i < MSMEs.OB_n.size(); i++) // loop m-scheme OB operator
            {
                int ia = MSMEs.OB_n[i].GetIndex_a(); // index of a
                int ib = MSMEs.OB_n[i].GetIndex_b();
                int orb_ia = ms->Get_NeutronOrbitIndexInMscheme(ia); // index of j orbit a
                int orb_ib = ms->Get_NeutronOrbitIndexInMscheme(ib);
                int aj = ms->Get_MSmatrix_2j(Neutron, ia); // 2j of orbit a
                int bj = ms->Get_MSmatrix_2j(Neutron, ib);
                int am = ms->Get_MSmatrix_2m(Neutron, ia); // 2m of orbit a
                int bm = ms->Get_MSmatrix_2m(Neutron, ib);
                if (orb_ia != a or orb_ib != b)
                {
                    continue;
                }
                if (am - bm == -4) // 2m
                {
                    double CGC = AngMom::cgc(2., -2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj - bm) / 2) * CGC;
                    Q2MEs_n.Q_2_MSMEs.push_back(QME);
                    Q2MEs_n.Q_2_list.push_back(i);
                }
                else if (am == bm)
                {
                    double CGC = AngMom::cgc(2., 0., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj - bm) / 2) * CGC;
                    Q2MEs_n.Q0_MSMEs.push_back(QME);
                    Q2MEs_n.Q0_list.push_back(i);
                }
                else if (am - bm == 4)
                {
                    double CGC = AngMom::cgc(2., 2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj - bm) / 2) * CGC;
                    Q2MEs_n.Q2_MSMEs.push_back(QME);
                    Q2MEs_n.Q2_list.push_back(i);
                }
            }
        }
    }
}

void Hamiltonian::Initial_Q4()
{
    double QME;
    int lamda = 4;
    // Proton
    int orb_num = ms->GetProtonOrbitsNum();
    for (size_t a = 0; a < orb_num; a++)
    {
        for (size_t b = 0; b < orb_num; b++)
        {
            int tmax = (ms->GetProtonOrbit_2j(a) + ms->GetProtonOrbit_2j(b)) / 2;
            int tmin = std::abs(ms->GetProtonOrbit_2j(a) - ms->GetProtonOrbit_2j(b)) / 2;
            if (tmax < lamda or tmin > lamda)
            {
                continue;
            }
            double QMEJ = this->Calculate_Qt(lamda, a, b, Proton);
            for (size_t i = 0; i < MSMEs.OB_p.size(); i++) // loop m-scheme OB operator
            {
                int ia = MSMEs.OB_p[i].GetIndex_a(); // index of a in M scheme
                int ib = MSMEs.OB_p[i].GetIndex_b();
                int orb_ia = ms->Get_ProtonOrbitIndexInMscheme(ia); // index of j orbit a in J scheme
                int orb_ib = ms->Get_ProtonOrbitIndexInMscheme(ib);
                int aj = ms->Get_MSmatrix_2j(Proton, ia); // 2j of orbit a
                int bj = ms->Get_MSmatrix_2j(Proton, ib);
                int am = ms->Get_MSmatrix_2m(Proton, ia); // 2m of orbit a
                int bm = ms->Get_MSmatrix_2m(Proton, ib);
                if (orb_ia != a or orb_ib != b)
                {
                    continue;
                }
                if (am - bm == -4) // 2m
                {
                    double CGC = AngMom::cgc(lamda * 1., -2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj + bm) / 2) * CGC;
                    Q4MEs_p.Q_2_MSMEs.push_back(QME);
                    Q4MEs_p.Q_2_list.push_back(i);
                }
                else if (am == bm)
                {
                    double CGC = AngMom::cgc(lamda * 1., 0., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj + bm) / 2) * CGC;
                    Q4MEs_p.Q0_MSMEs.push_back(QME);
                    Q4MEs_p.Q0_list.push_back(i);
                }
                else if (am - bm == 4)
                {
                    double CGC = AngMom::cgc(lamda * 1., 2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj + bm) / 2) * CGC;
                    Q4MEs_p.Q2_MSMEs.push_back(QME);
                    Q4MEs_p.Q2_list.push_back(i);
                }
            }
        }
    }

    // Neutron
    orb_num = ms->GetNeutronOrbitsNum();
    for (size_t a = 0; a < orb_num; a++)
    {
        for (size_t b = 0; b < orb_num; b++)
        {
            int tmax = (ms->GetNeutronOrbit_2j(a) + ms->GetNeutronOrbit_2j(b)) / 2;
            int tmin = std::abs(ms->GetNeutronOrbit_2j(a) - ms->GetNeutronOrbit_2j(b)) / 2;
            if (tmax < 2 or tmin > 2)
            {
                continue;
            }
            double QMEJ = this->Calculate_Qt(lamda, a, b, Neutron);
            for (size_t i = 0; i < MSMEs.OB_n.size(); i++) // loop m-scheme OB operator
            {
                int ia = MSMEs.OB_n[i].GetIndex_a(); // index of a
                int ib = MSMEs.OB_n[i].GetIndex_b();
                int orb_ia = ms->Get_NeutronOrbitIndexInMscheme(ia); // index of j orbit a
                int orb_ib = ms->Get_NeutronOrbitIndexInMscheme(ib);
                int aj = ms->Get_MSmatrix_2j(Neutron, ia); // 2j of orbit a
                int bj = ms->Get_MSmatrix_2j(Neutron, ib);
                int am = ms->Get_MSmatrix_2m(Neutron, ia); // 2m of orbit a
                int bm = ms->Get_MSmatrix_2m(Neutron, ib);
                if (orb_ia != a or orb_ib != b)
                {
                    continue;
                }
                if (am - bm == -4) // 2m
                {
                    double CGC = AngMom::cgc(lamda * 1., -2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj + bm) / 2) * CGC;
                    Q4MEs_n.Q_2_MSMEs.push_back(QME);
                    Q4MEs_n.Q_2_list.push_back(i);
                }
                else if (am == bm)
                {
                    double CGC = AngMom::cgc(lamda * 1., 0., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj + bm) / 2) * CGC;
                    Q4MEs_n.Q0_MSMEs.push_back(QME);
                    Q4MEs_n.Q0_list.push_back(i);
                }
                else if (am - bm == 4)
                {
                    double CGC = AngMom::cgc(lamda * 1., 2., aj * 0.5, am * 0.5, bj * 0.5, -bm * 0.5);
                    QME = QMEJ * sgn((bj + bm) / 2) * CGC;
                    Q4MEs_n.Q2_MSMEs.push_back(QME);
                    Q4MEs_n.Q2_list.push_back(i);
                }
            }
        }
    }
}
