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

#include "GCM_Toos.h"
using namespace NPSMCommutator;
///////////////////////////////////////////////////////
/// CountEvaluationsGCM
CountEvaluationsGCM::CountEvaluationsGCM(ModelSpace &ms)
    : ms(&ms)
{
    total_steps = ms.GetNumPara_p() + ms.GetNumPara_n();
    total_steps *= 2;
    total_steps += 1;
}

bool CountEvaluationsGCM::CountCals()
{
    // std::cout << "Number of evaluation:  " << count_cals << std::endl;
    if (count_cals % total_steps == 0)
    {
        string Filename = saved_path + "/save_points_" + std::to_string(saved_number) + ".dat";
        this->SetFilename(Filename);
        saved_number++;
        count_cals++;
        return true;
    }
    count_cals++;
    return false;
};

Cal_GCM_projection::Cal_GCM_projection(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj, MatrixIndex &MyIndex, MultiPairs &Stored_p, MultiPairs &Stored_n)
    : ms(&ms), Ham(&Ham), AngMomProj(&AngMomProj), MyIndex(&MyIndex), MP_stored_p(&Stored_p), MP_stored_n(&Stored_n)
{
}

Cal_GCM_projection::~Cal_GCM_projection()
{
}

std::vector<double> Cal_GCM_projection::Cal_All_MEs_pn()
{
    ComplexNum *OvlME, *HamME, *tempME;
    int myid, numprocs;  // MPI parameters
    int i, MEnum, MEDim; //	int i, j, k, n1, n2, j1, j2;
    std::vector<double> tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    MEnum = MyIndex->ME_total;
    MEDim = Total_order * Total_order;
    /// build array for ME
    HamME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(HamME, 0, sizeof(double) * 2 * (MEnum));
    OvlME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(OvlME, 0, sizeof(double) * 2 * (MEnum));
    tempME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(tempME, 0, sizeof(double) * 2 * (MEnum));

    /// MPI inint
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    ///------- loop ME
    for (i = myid; i < MyIndex->Ovl_total; i += numprocs)
    {
        if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
        {
            Calculate_ME_pn(i, HamME, OvlME);
        }
        else
        {
            Calculate_ME_SamePairs_pn(i, HamME, OvlME);
        }
    }
    MPI_Reduce(OvlME, tempME, MEnum * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        /// Copy Matrix
        cblas_zcopy(MEnum, tempME, 1, OvlME, 1);
    }
    MPI_Reduce(HamME, tempME, MEnum * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        // std::cout << tempME[0] << OvlME[0] << std::endl;
        tempval = DealTotalHamiltonianMatrix(OvlME, tempME);
        SaveData_pn(OvlME, tempME);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    return tempval;
}

std::vector<double> Cal_GCM_projection::Cal_Overlap_before_Porjection()
{
    ComplexNum *OvlME, *tempME;
    int myid, numprocs;  // MPI parameters
    int i, MEnum, MEDim; //	int i, j, k, n1, n2, j1, j2;
    std::vector<double> tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    MEnum = MyIndex->ME_total;
    MEDim = Total_order * Total_order;
    /// build array for ME
    OvlME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(OvlME, 0, sizeof(double) * 2 * (MEnum));
    tempME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(tempME, 0, sizeof(double) * 2 * (MEnum));

    /// MPI inint
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    ///------- loop ME
    for (i = myid; i < MEnum; i += numprocs)
    {
        MPI_Calculate_Overlap(i, OvlME);
    }
    MPI_Reduce(OvlME, tempME, MEnum * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        tempval = AnalysisOverlap(tempME);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(tempME);
    return tempval;
}

std::vector<int> Cal_GCM_projection::SelectBasis() // pick linear independent basis
{
    ComplexNum *OvlME, *tempME;
    int myid, numprocs;  // MPI parameters
    int i, MEnum, MEDim; //	int i, j, k, n1, n2, j1, j2;
    std::vector<int> SelectedBasisIndex;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    MEnum = MyIndex->ME_total;
    MEDim = Total_order * Total_order;
    /// build array for ME
    OvlME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(OvlME, 0, sizeof(double) * 2 * (MEnum));
    tempME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(tempME, 0, sizeof(double) * 2 * (MEnum));

    /// MPI inint
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    ///------- loop ME
    for (i = myid; i < MEnum; i += numprocs)
    {
        MPI_Calculate_Overlap(i, OvlME);
    }
    MPI_Reduce(OvlME, tempME, MEnum * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        PickBasis(tempME, SelectedBasisIndex);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(tempME);
    return SelectedBasisIndex;
}

void Cal_GCM_projection::MPI_Calculate_Overlap(int i, ComplexNum *MyOvlME)
{
    int i1, j1;
    ComplexNum ovlME_p, ovlME_n;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();

    i1 = MyIndex->MEindex_i[i]; // matrix index i1
    j1 = MyIndex->MEindex_j[i]; // matrix index j1

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp(MP_stored_p[i1]);
    MultiPairs MPBasis_Rp(MP_stored_p[j1]);
    MultiPairs MPBasis_Ln(MP_stored_n[i1]);
    MultiPairs MPBasis_Rn(MP_stored_n[j1]);

    if (ParityProj == 0)
    {
        // Cal ovl MEs
        ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
        ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
        MyOvlME[i] += ovlME_p * ovlME_n;
        // std::cout<< MyOvlME[i] << weightFactor<< ovlME_p << ovlME_n  <<std::endl;
    }
    else
    {
        MultiPairs MPBasis_R_Parity_p(MPBasis_Rp);
        MultiPairs MPBasis_R_Parity_n(MPBasis_Rn);
        MPBasis_R_Parity_p.ParityProjection();
        MPBasis_R_Parity_n.ParityProjection();
        double Pairty = ms->GetProjected_parity();
        ComplexNum ovlME_p2, ovlME_n2;

        // Cal ovl ME
        ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
        ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
        ovlME_p2 = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_R_Parity_p);
        ovlME_n2 = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_R_Parity_n);

        MyOvlME[i] += 0.5 * ovlME_p * ovlME_n;
        MyOvlME[i] += 0.5 * Pairty * ovlME_p2 * ovlME_n2;
    }
    //////////////////////////////////////////
}

void Cal_GCM_projection::SaveData_pn(ComplexNum *OvlME, ComplexNum *HamME)
{
    ComplexNum *Matrix1, *Matrix2;
    int Total_Order = ms->GetTotalOrders();
    int N_p = ms->GetPairNumber(Proton);
    int N_n = ms->GetPairNumber(Neutron);
    Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    ReadWriteFiles rw;

    if (Total_Order > 1)
    { /// Output Ham ME
        Build_Matrix(Total_Order, OvlME, Matrix1);
        Build_Matrix(Total_Order, HamME, Matrix2);
        rw.OutputME(Total_Order, Matrix1, Ovl_filename);
        rw.OutputME(Total_Order, Matrix2, Ham_filename);
    }
    else
    {
        rw.OutputME(Total_Order, OvlME, Ovl_filename);
        rw.OutputME(Total_Order, HamME, Ham_filename);
    }
    mkl_free(Matrix1);
    mkl_free(Matrix2);
}

void Cal_GCM_projection::Calculate_ME_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME)
{
    int i1, j1, t, k, Rindex, MEind, Hind, Hm;
    ComplexNum ovlME_p, ovlME_n, temp_ME, weightFactor;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    double mass_scaling = ms->GetMassDependentFactor();

    Rindex = i / MyIndex->ME_total;
    MEind = i % MyIndex->ME_total;

    // MEind = MyIndex->OvlInd_ME[i];
    // Rindex = MyIndex->OvlInd_R[i];

    i1 = MyIndex->MEindex_i[MEind]; // matrix index i1
    j1 = MyIndex->MEindex_j[MEind]; // matrix index j1

    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->GuassQuad_weight(alpha, beta, gamma);

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp(MP_stored_p[i1]);
    MultiPairs MPBasis_Rp(MP_stored_p[j1]);
    MultiPairs MPBasis_Ln(MP_stored_n[i1]);
    MultiPairs MPBasis_Rn(MP_stored_n[j1]);

    MPBasis_Rp.RotatedPairs(alpha, beta, gamma);
    MPBasis_Rn.RotatedPairs(alpha, beta, gamma);

    if (ParityProj == 0)
    {
        // Cal ovl MEs
        ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
        ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
        MyOvlME[MEind] += weightFactor * ovlME_p * ovlME_n;
        // std::cout<< MyOvlME[MEind] << weightFactor<< ovlME_p << ovlME_n  <<std::endl;

        // Cal S.P. MEs
        MyHamME[MEind] += weightFactor * CalOneBodyOperator(N_p, *ms, MPBasis_Lp, MPBasis_Rp, Ham->GetSPEpointer(Proton)) * ovlME_n;
        MyHamME[MEind] += weightFactor * CalOneBodyOperator(N_n, *ms, MPBasis_Ln, MPBasis_Rn, Ham->GetSPEpointer(Neutron)) * ovlME_p;

        // Cal Vpp and Vnn MEs
        if (Ham->H_IsCollective())
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_n;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME -= Prepare_Collective_pairingME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_p;
        }
        else
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_n;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME += Prepare_pairingME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_p;
        }

        // Cal Vpn MEsls
        MyHamME[MEind] += mass_scaling * weightFactor * Calculate_Vpn_MEs(N_p, N_n, MPBasis_Lp, MPBasis_Rp, MPBasis_Ln, MPBasis_Rn);
    }
    else
    {
        MultiPairs MPBasis_R_Parity_p(MPBasis_Rp);
        MultiPairs MPBasis_R_Parity_n(MPBasis_Rn);
        MPBasis_R_Parity_p.ParityProjection();
        MPBasis_R_Parity_n.ParityProjection();
        double Pairty = ms->GetProjected_parity();
        ComplexNum ovlME_p2, ovlME_n2;

        // Cal ovl ME
        ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
        ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
        ovlME_p2 = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_R_Parity_p);
        ovlME_n2 = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_R_Parity_n);

        MyOvlME[MEind] += 0.5 * weightFactor * ovlME_p * ovlME_n;
        MyOvlME[MEind] += 0.5 * Pairty * weightFactor * ovlME_p2 * ovlME_n2;

        // Cal S.P. ME
        MyHamME[MEind] += 0.5 * weightFactor * CalOneBodyOperator(N_p, *ms, MPBasis_Lp, MPBasis_Rp, Ham->GetSPEpointer(Proton)) * ovlME_n;
        MyHamME[MEind] += 0.5 * weightFactor * CalOneBodyOperator(N_n, *ms, MPBasis_Ln, MPBasis_Rn, Ham->GetSPEpointer(Neutron)) * ovlME_p;
        MyHamME[MEind] += 0.5 * Pairty * weightFactor * CalOneBodyOperator(N_p, *ms, MPBasis_Lp, MPBasis_R_Parity_p, Ham->GetSPEpointer(Proton)) * ovlME_n2;
        MyHamME[MEind] += 0.5 * Pairty * weightFactor * CalOneBodyOperator(N_n, *ms, MPBasis_Ln, MPBasis_R_Parity_n, Ham->GetSPEpointer(Neutron)) * ovlME_p2;

        // Cal Vpp ME
        if (Ham->H_IsCollective())
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_n;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_R_Parity_p, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_n2;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME -= Prepare_Collective_pairingME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_p;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME -= Prepare_Collective_pairingME(N_n, *ms, *Ham, MPBasis_R_Parity_n, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_p2;
        }
        else
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_n;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_R_Parity_p, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_n2;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME += Prepare_pairingME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_p;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME += Prepare_pairingME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_R_Parity_n, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_p2;
        }

        // Cal Vpn MEsls
        MyHamME[MEind] += mass_scaling * weightFactor * Calculate_Vpn_MEs_pairty(N_p, N_n, MPBasis_Lp, MPBasis_Rp, MPBasis_R_Parity_p, MPBasis_Ln, MPBasis_Rn, MPBasis_R_Parity_n);
    }
    //////////////////////////////////////////
}

ComplexNum Cal_GCM_projection::Calculate_Vpn_MEs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn)
{
    ComplexNum temp_ME, tempQQ, *tempQp, *tempQn;
    int k, t, Hind, Hm;
    /////////////////////////////
    tempQp = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_p() * sizeof(ComplexNum), 64);
    tempQn = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_n() * sizeof(ComplexNum), 64);
    for (k = 0; k < Ham->GetOneBodyOperatorNumber_p(); k++)
    {
        t = Ham->OBchannel_p[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQpIndex(k, t, Hm);
            tempQp[Hind] = Prepare_OB_ME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, k, Hm);
            // OneBodyOperatorChannel OP = Ham->GetOneBodyOperator(Proton, k);
            // int index_a = OP.GetIndex_a();
            // int index_b = OP.GetIndex_b();
            // printf("k %d   t  %d   m  %d  orbit: %d  %d      %lf   %lf\n", k, t, Hm, index_a, index_b, tempQp[Hind].real(), tempQp[Hind].imag());
        }
    }

    for (k = 0; k < Ham->GetOneBodyOperatorNumber_n(); k++)
    {
        t = Ham->OBchannel_n[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQnIndex(k, t, Hm);
            tempQn[Hind] = Prepare_OB_ME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, k, Hm);
            // printf("k %d   t  %d   m  %d       %lf   %lf\n", k, t, Hm, tempQn[Hind].real(),tempQn[Hind].imag());
        }
    }
    temp_ME = 0.;
    for (k = 0; k < Ham->Get_CalTerms_pn(); k++)
    {
        Hind = MyIndex->Hpn_Hindex[k];
        Hm = MyIndex->Hpn_m[k];
        Vpn_phCoupledElements Vme = Ham->GetVpn_phcoupled_ME(Hind);
        t = Vme.GeT_t();
        int index_Qp = MyIndex->GetQpIndex(Vme.GetQpindex(), t, Hm);
        int index_Qn = MyIndex->GetQnIndex(Vme.GetQnindex(), t, -Hm);
        tempQQ = sgn(Hm) * Vme.GetV();
        tempQQ *= tempQp[index_Qp] * tempQn[index_Qn];
        temp_ME += tempQQ;
    }
    mkl_free(tempQp);
    mkl_free(tempQn);
    return temp_ME;
}

ComplexNum Cal_GCM_projection::Calculate_Vpn_MEs_pairty(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n)
{
    ComplexNum temp_ME, tempQQ, *tempQp, *tempQn, *tempQp_parity, *tempQn_parity;
    int k, t, Hind, Hm;
    double Pairty = ms->GetProjected_parity();
    /////////////////////////////
    tempQp = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_p() * sizeof(ComplexNum), 64);
    tempQn = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_n() * sizeof(ComplexNum), 64);
    /// Pairty projection
    tempQp_parity = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_p() * sizeof(ComplexNum), 64);
    tempQn_parity = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_n() * sizeof(ComplexNum), 64);

    for (k = 0; k < Ham->GetOneBodyOperatorNumber_p(); k++)
    {
        t = Ham->OBchannel_p[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQpIndex(k, t, Hm);
            tempQp[Hind] = Prepare_OB_ME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, k, Hm);
            tempQp_parity[Hind] = Prepare_OB_ME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_R_Parity_p, k, Hm);
            // OneBodyOperatorChannel OP = Ham->GetOneBodyOperator(Proton, k);
            // int index_a = OP.GetIndex_a();
            // int index_b = OP.GetIndex_b();
            // printf("k %d   t  %d   m  %d  orbit: %d  %d      %lf   %lf\n", k, t, Hm, index_a, index_b, tempQp[Hind].real(), tempQp[Hind].imag());
        }
    }

    for (k = 0; k < Ham->GetOneBodyOperatorNumber_n(); k++)
    {
        t = Ham->OBchannel_n[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQnIndex(k, t, Hm);
            tempQn[Hind] = Prepare_OB_ME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, k, Hm);
            tempQn_parity[Hind] = Prepare_OB_ME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_R_Parity_n, k, Hm);
            // printf("k %d   t  %d   m  %d       %lf   %lf\n", k, t, Hm, tempQn[Hind].real(),tempQn[Hind].imag());
        }
    }

    /////
    temp_ME = 0.;
    for (k = 0; k < Ham->Get_CalTerms_pn(); k++)
    {
        Hind = MyIndex->Hpn_Hindex[k];
        Hm = MyIndex->Hpn_m[k];
        Vpn_phCoupledElements Vme = Ham->GetVpn_phcoupled_ME(Hind);
        t = Vme.GeT_t();
        int index_Qp = MyIndex->GetQpIndex(Vme.GetQpindex(), t, Hm);
        int index_Qn = MyIndex->GetQnIndex(Vme.GetQnindex(), t, -Hm);
        tempQQ = sgn(Hm) * Vme.GetV() * tempQp[index_Qp] * tempQn[index_Qn];
        tempQQ += Pairty * sgn(Hm) * Vme.GetV() * tempQp_parity[index_Qp] * tempQn_parity[index_Qn];
        temp_ME += 0.5 * tempQQ;
    }
    mkl_free(tempQp);
    mkl_free(tempQn);
    mkl_free(tempQp_parity);
    mkl_free(tempQn_parity);
    return temp_ME;
}

std::vector<double> Cal_GCM_projection::DealTotalHamiltonianMatrix(ComplexNum *OvlME, ComplexNum *HamME)
{
    ReadWriteFiles rw;
    int Total_Order = ms->GetTotalOrders();
    std::vector<double> Sum_EigenVals;
    ComplexNum *Matrix1, *Matrix2;
    Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    if (Total_Order > 1)
    { /// Eigen problem
        Build_Matrix(Total_Order, OvlME, Matrix1);
        Build_Matrix(Total_Order, HamME, Matrix2);
        Sum_EigenVals = EigenValues(Total_Order, Matrix1, Matrix2);
        // Sum_EigenVals = RealEigenValues(Total_Order, Sum_Num, Matrix1, Matrix2);
    }
    else
    {
        // std::cout << "ME " << Matrix2[0] << "    Ovl " << Matrix1[0] << '\n';
        Sum_EigenVals.push_back(HamME[0].real() / OvlME[0].real());
    }
    // std::cout << std::fixed << std::showpoint;
    // std::cout << "Total Hamiltonian = " << std::setprecision(12) << tempval << '\n';
    mkl_free(Matrix1);
    mkl_free(Matrix2);
    return Sum_EigenVals;
}

void Cal_GCM_projection::Build_Matrix(int dim, ComplexNum *ele, ComplexNum *NewEle)
{
    int i, i1, j1;
    for (i = 0; i < MyIndex->ME_total; i++)
    {
        i1 = MyIndex->MEindex_i[i];
        j1 = MyIndex->MEindex_j[i];
        NewEle[i1 * dim + j1] = ele[i];
        if (i1 != j1)
        {
            NewEle[j1 * dim + i1] = std::conj(ele[i]);
        }
    }
}

std::vector<double> Cal_GCM_projection::EigenValues(int dim, ComplexNum *Ovl, ComplexNum *Ham)
{
    int i, j;
    double *e;
    std::vector<double> returu_value;
    ComplexNum *tempHam, *tempMat;
    ComplexNum *prt_a, *prt_b, *prt_c;
    ComplexNum alpha(1, 0), beta(0, 0);
    e = (double *)mkl_malloc((dim) * sizeof(double), 64);
    tempHam = (ComplexNum *)mkl_malloc((dim * dim) * sizeof(ComplexNum), 64);
    tempMat = (ComplexNum *)mkl_malloc((dim * dim) * sizeof(ComplexNum), 64);
    /// Copy Matrix
    cblas_zcopy(dim * dim, Ovl, 1, tempMat, 1);
    /// Tridiag Ovl
    if (LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'L', dim, tempMat, dim, e) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in normal procedure!!\n");
        ReadWriteFiles rw;
        rw.OutputME(dim, tempMat, CheckMatrix);
        exit(0);
    }
    for (i = 0; i < dim; i++) // resacle
        for (j = 0; j < dim; j++)
            tempMat[i * dim + j] /= sqrt(e[j]);
    mkl_zomatcopy('R', 'C', dim, dim, alpha, tempMat, dim, Ovl, dim);

    // Normal Ham matrix
    memset(tempHam, 0, dim * dim * 2 * sizeof(double));
    prt_a = Ovl;
    prt_b = Ham;
    prt_c = tempHam;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha, prt_a, dim, prt_b, dim, &beta, prt_c, dim);
    memset(Ham, 0, dim * dim * 2 * sizeof(double));
    prt_a = tempHam;
    prt_b = tempMat;
    prt_c = Ham;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha, prt_a, dim, prt_b, dim, &beta, prt_c, dim);
    if (LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'N', 'L', dim, Ham, dim, e) != 0)
    { /// 'N' stand for only eigenvalues
        printf("Error when Computes all eigenvalues and eigenvectors in Ham procedure!!\n");
        ReadWriteFiles rw;
        rw.OutputME(dim, Ham, CheckHMatrix);
        rw.OutputME(dim, Ovl, CheckMatrix);
        exit(0);
    }

    returu_value.clear();
    for (i = 0; i < dim; i++)
    {
        returu_value.push_back(e[i]);
    }
    mkl_free(e);
    mkl_free(tempHam);
    mkl_free(tempMat);
    return returu_value;
}

void Cal_GCM_projection::Calculate_ME_SamePairs_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME)
{
    int i1, j1, t, k, Rindex, MEind, Hind, Hm;
    ComplexNum ovlME_p, ovlME_n, temp_ME, weightFactor;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    double mass_scaling = ms->GetMassDependentFactor();

    Rindex = i / MyIndex->ME_total;
    MEind = i % MyIndex->ME_total;
    // MEind = MyIndex->OvlInd_ME[i];
    // Rindex = MyIndex->OvlInd_R[i];
    i1 = MyIndex->MEindex_i[MEind]; // order
    j1 = MyIndex->MEindex_j[MEind]; // should always be TotalOrder - 1

    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->GuassQuad_weight(alpha, beta, gamma);

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp(MP_stored_p[i1]);
    MultiPairs MPBasis_Rp(MP_stored_p[j1]);
    MultiPairs MPBasis_Ln(MP_stored_n[i1]);
    MultiPairs MPBasis_Rn(MP_stored_n[j1]);
    MPBasis_Rp.RotatedPairs(alpha, beta, gamma);
    MPBasis_Rn.RotatedPairs(alpha, beta, gamma);

    if (ParityProj == 0)
    {
        // Cal ovl MEs
        ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
        ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
        MyOvlME[MEind] += weightFactor * ovlME_p * ovlME_n;
        // std::cout<< MyOvlME[MEind] << weightFactor<< ovlME_p << ovlME_n  <<std::endl;

        // Cal S.P. MEs
        MyHamME[MEind] += weightFactor * CalOneBodyOperator_SamePairs(N_p, *ms, MPBasis_Lp, MPBasis_Rp, Ham->GetSPEpointer(Proton)) * ovlME_n;
        MyHamME[MEind] += weightFactor * CalOneBodyOperator_SamePairs(N_n, *ms, MPBasis_Ln, MPBasis_Rn, Ham->GetSPEpointer(Neutron)) * ovlME_p;

        // Cal Vpp and Vnn MEs
        if (Ham->H_IsCollective())
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_n;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_p;
        }
        else
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_n;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME * ovlME_p;
        }

        // Cal Vpn MEsls
        MyHamME[MEind] += mass_scaling * weightFactor * Calculate_Vpn_MEs_SamePairs(N_p, N_n, MPBasis_Lp, MPBasis_Rp, MPBasis_Ln, MPBasis_Rn);
    }
    else
    {
        MultiPairs MPBasis_R_Parity_p(MPBasis_Rp);
        MultiPairs MPBasis_R_Parity_n(MPBasis_Rn);
        MPBasis_R_Parity_p.ParityProjection();
        MPBasis_R_Parity_n.ParityProjection();
        double Pairty = ms->GetProjected_parity();
        ComplexNum ovlME_p2, ovlME_n2;

        // Cal ovl ME
        ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
        ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
        ovlME_p2 = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_R_Parity_p);
        ovlME_n2 = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_R_Parity_n);

        MyOvlME[MEind] += 0.5 * weightFactor * ovlME_p * ovlME_n;
        MyOvlME[MEind] += 0.5 * Pairty * weightFactor * ovlME_p2 * ovlME_n2;

        // Cal S.P. ME
        MyHamME[MEind] += 0.5 * weightFactor * CalOneBodyOperator_SamePairs(N_p, *ms, MPBasis_Lp, MPBasis_Rp, Ham->GetSPEpointer(Proton)) * ovlME_n;
        MyHamME[MEind] += 0.5 * weightFactor * CalOneBodyOperator_SamePairs(N_n, *ms, MPBasis_Ln, MPBasis_Rn, Ham->GetSPEpointer(Neutron)) * ovlME_p;
        MyHamME[MEind] += 0.5 * Pairty * weightFactor * CalOneBodyOperator_SamePairs(N_p, *ms, MPBasis_Lp, MPBasis_R_Parity_p, Ham->GetSPEpointer(Proton)) * ovlME_n2;
        MyHamME[MEind] += 0.5 * Pairty * weightFactor * CalOneBodyOperator_SamePairs(N_n, *ms, MPBasis_Ln, MPBasis_R_Parity_n, Ham->GetSPEpointer(Neutron)) * ovlME_p2;

        // Cal Vpp ME
        if (Ham->H_IsCollective())
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_n;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_R_Parity_p, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_n2;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_p;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_R_Parity_n, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_p2;
        }
        else
        {
            // Vpp
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_n;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_R_Parity_p, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_n2;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME * ovlME_p;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_R_Parity_n, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME * ovlME_p2;
        }

        // Cal Vpn MEsls
        MyHamME[MEind] += mass_scaling * weightFactor * Calculate_Vpn_MEs_pairty_SamePairs(N_p, N_n, MPBasis_Lp, MPBasis_Rp, MPBasis_R_Parity_p, MPBasis_Ln, MPBasis_Rn, MPBasis_R_Parity_n);
    }
    //////////////////////////////////////////
}

ComplexNum Cal_GCM_projection::Calculate_Vpn_MEs_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn)
{
    ComplexNum temp_ME, tempQQ, *tempQp, *tempQn;
    int k, t, Hind, Hm;
    /////////////////////////////
    tempQp = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_p() * sizeof(ComplexNum), 64);
    tempQn = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_n() * sizeof(ComplexNum), 64);
    for (k = 0; k < Ham->GetOneBodyOperatorNumber_p(); k++)
    {
        t = Ham->OBchannel_p[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQpIndex(k, t, Hm);
            tempQp[Hind] = Prepare_OB_ME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, k, Hm);
            /*
            OneBodyOperatorChannel OP = Ham->GetOneBodyOperator(Proton, k);
            int index_a = OP.GetIndex_a();
            int index_b = OP.GetIndex_b();
            printf("k %d   t  %d   m  %d  orbit: %d  %d      %lf   %lf\n", k, t, Hm, index_a, index_b, tempQp[Hind].real(), tempQp[Hind].imag());
            */
        }
    }

    for (k = 0; k < Ham->GetOneBodyOperatorNumber_n(); k++)
    {
        t = Ham->OBchannel_n[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQnIndex(k, t, Hm);
            tempQn[Hind] = Prepare_OB_ME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, k, Hm);
            // printf("k %d   t  %d   m  %d       %lf   %lf\n", k, t, Hm, tempQn[Hind].real(),tempQn[Hind].imag());
        }
    }
    temp_ME = 0.;
    for (k = 0; k < Ham->Get_CalTerms_pn(); k++)
    {
        Hind = MyIndex->Hpn_Hindex[k];
        Hm = MyIndex->Hpn_m[k];
        Vpn_phCoupledElements Vme = Ham->GetVpn_phcoupled_ME(Hind);
        t = Vme.GeT_t();
        int index_Qp = MyIndex->GetQpIndex(Vme.GetQpindex(), t, Hm);
        int index_Qn = MyIndex->GetQnIndex(Vme.GetQnindex(), t, -Hm);
        tempQQ = sgn(Hm) * Vme.GetV();
        tempQQ *= tempQp[index_Qp] * tempQn[index_Qn];
        temp_ME += tempQQ;
    }
    mkl_free(tempQp);
    mkl_free(tempQn);
    return temp_ME;
}

ComplexNum Cal_GCM_projection::Calculate_Vpn_MEs_pairty_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n)
{
    ComplexNum temp_ME, tempQQ, *tempQp, *tempQn, *tempQp_parity, *tempQn_parity;
    int k, t, Hind, Hm;
    double Pairty = ms->GetProjected_parity();
    /////////////////////////////
    tempQp = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_p() * sizeof(ComplexNum), 64);
    tempQn = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_n() * sizeof(ComplexNum), 64);
    /// Pairty projection
    tempQp_parity = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_p() * sizeof(ComplexNum), 64);
    tempQn_parity = (ComplexNum *)mkl_malloc(Ham->GetCalOBOperatorNumber_n() * sizeof(ComplexNum), 64);

    for (k = 0; k < Ham->GetOneBodyOperatorNumber_p(); k++)
    {
        t = Ham->OBchannel_p[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQpIndex(k, t, Hm);
            tempQp[Hind] = Prepare_OB_ME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, k, Hm);
            tempQp_parity[Hind] = Prepare_OB_ME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_R_Parity_p, k, Hm);
            // OneBodyOperatorChannel OP = Ham->GetOneBodyOperator(Proton, k);
            // int index_a = OP.GetIndex_a();
            // int index_b = OP.GetIndex_b();
            // printf("k %d   t  %d   m  %d  orbit: %d  %d      %lf   %lf\n", k, t, Hm, index_a, index_b, tempQp[Hind].real(), tempQp[Hind].imag());
        }
    }

    for (k = 0; k < Ham->GetOneBodyOperatorNumber_n(); k++)
    {
        t = Ham->OBchannel_n[k].t; // angular momentum
        if (t > Ham->GetAllowedMaxOneBody_t())
            continue;
        for (Hm = -t; Hm <= t; Hm++)
        {
            Hind = MyIndex->GetQnIndex(k, t, Hm);
            tempQn[Hind] = Prepare_OB_ME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, k, Hm);
            tempQn_parity[Hind] = Prepare_OB_ME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_R_Parity_n, k, Hm);
            // printf("k %d   t  %d   m  %d       %lf   %lf\n", k, t, Hm, tempQn[Hind].real(),tempQn[Hind].imag());
        }
    }

    /////
    temp_ME = 0.;
    for (k = 0; k < Ham->Get_CalTerms_pn(); k++)
    {
        Hind = MyIndex->Hpn_Hindex[k];
        Hm = MyIndex->Hpn_m[k];
        Vpn_phCoupledElements Vme = Ham->GetVpn_phcoupled_ME(Hind);
        t = Vme.GeT_t();
        int index_Qp = MyIndex->GetQpIndex(Vme.GetQpindex(), t, Hm);
        int index_Qn = MyIndex->GetQnIndex(Vme.GetQnindex(), t, -Hm);
        tempQQ = sgn(Hm) * Vme.GetV() * tempQp[index_Qp] * tempQn[index_Qn];
        tempQQ += Pairty * sgn(Hm) * Vme.GetV() * tempQp_parity[index_Qp] * tempQn_parity[index_Qn];
        temp_ME += 0.5 * tempQQ;
    }
    mkl_free(tempQp);
    mkl_free(tempQn);
    mkl_free(tempQp_parity);
    mkl_free(tempQn_parity);
    return temp_ME;
}

std::vector<double> Cal_GCM_projection::AnalysisOverlap(ComplexNum *Ovl)
{
    ReadWriteFiles rw;
    int dim = ms->GetTotalOrders();
    int i, j;
    double *e;
    std::vector<double> returu_value;
    ComplexNum *Matrix1;
    ComplexNum *prt_a, *prt_b, *prt_c;
    ComplexNum alpha(1, 0), beta(0, 0);

    Matrix1 = (ComplexNum *)mkl_malloc(dim * dim * sizeof(ComplexNum), 64);
    Build_Matrix(dim, Ovl, Matrix1);
    e = (double *)mkl_malloc((dim) * sizeof(double), 64);
    /// Tridiag Ovl
    if (LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'L', dim, Matrix1, dim, e) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in Cal-Overlap procedure!!\n");
        rw.OutputME(dim, Matrix1, CheckMatrix);
        exit(0);
    }

    returu_value.clear();
    for (i = 0; i < dim; i++)
    {
        returu_value.push_back(e[i]);
    }

    mkl_free(e);
    mkl_free(Matrix1);
    return returu_value;
}

void Cal_GCM_projection::PickBasis(ComplexNum *Ovl, std::vector<int> &Index)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int dim = ms->GetTotalOrders();
    if (dim <= 1)
    {
        std::cout << "The number of basis should be at least 2!" << std::endl;
        MPI_Finalize();
        exit(0);
    }
    int loopBasis, i, j;
    int MEDim = dim * dim;
    ComplexNum *Ovl_Matrix;
    /// build array for ME
    Ovl_Matrix = (ComplexNum *)mkl_malloc((MEDim) * sizeof(ComplexNum), 64);
    memset(Ovl_Matrix, 0, sizeof(double) * 2 * (MEDim));
    Build_Matrix(dim, Ovl, Ovl_Matrix);

    Index.push_back(0); // pick the first basis
    for (loopBasis = 1; loopBasis < dim; loopBasis++)
    {
        Index.push_back(loopBasis);
        int new_dim = Index.size();
        //// construct temp matrix
        ComplexNum *Matrix1;
        ComplexNum *prt_a, *prt_b, *prt_c;
        ComplexNum alpha(1, 0), beta(0, 0);
        Matrix1 = (ComplexNum *)mkl_malloc(new_dim * new_dim * sizeof(ComplexNum), 64);
        memset(Matrix1, 0, sizeof(double) * 2 * (new_dim * new_dim));
        Build_Matrix_BasisPicking(dim, new_dim, Index, Ovl_Matrix, Matrix1);
        ///
        double *e;
        e = (double *)mkl_malloc(new_dim * sizeof(double), 64);
        /// Tridiag Ovl
        if (LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'L', new_dim, Matrix1, new_dim, e) != 0)
        { /// 'V' stand for eigenvalues and vectors
            ReadWriteFiles rw;
            printf("Error when Computes all eigenvalues and eigenvectors in Cal-Overlap procedure!!\n");
            rw.OutputME(new_dim, Matrix1, CheckMatrix);
            exit(0);
        }
        // std::cout << "Eigenvalues of Overlap:  add basis " << loopBasis << std::endl;
        for (i = 0; i < new_dim; i++)
        {
            // std::cout << i << ":    " << e[i] << std::endl;
            // if (e[i] < this->Overlap_dependence)
            if (e[i] < ms->GetOverlapMin())
            {
                break;
            }
        }
        if (i != new_dim)
        {
            Index.erase(Index.begin() + new_dim - 1);
        }
        if (myid == 0 and loopBasis == dim - 1)
        {
            std::cout << std::endl;
            for (i = 0; i < Index.size(); i++)
            {
                std::cout << " " << i + 1 << "  " << e[new_dim - i - 1] << std::endl;
            }
            std::cout << std::endl;
        }

        mkl_free(e);
        mkl_free(Matrix1);
    }
}

void Cal_GCM_projection::Build_Matrix_BasisPicking(int Original_dim, int dim, std::vector<int> &Index, ComplexNum *ele, ComplexNum *NewEle)
{
    int i, i1, j1;
    ComplexNum temp_element;
    for (i1 = 0; i1 < dim; i1++)
    {
        for (j1 = i1; j1 < dim; j1++)
        {
            temp_element = ele[Index[i1] * Original_dim + Index[j1]];
            NewEle[i1 * dim + j1] = temp_element;
            if (i1 != j1)
            {
                NewEle[j1 * dim + i1] = std::conj(temp_element);
            }
        }
    }
}

