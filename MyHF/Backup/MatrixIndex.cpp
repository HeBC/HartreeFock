#include "MatrixIndex.h"

MatrixIndex::MatrixIndex(ModelSpace &ms, AngMomProjection &AMproj, Hamiltonian &Ham)
    : ms(&ms), AMproj(&AMproj), Ham(&Ham)
{
    int i1, j1, J, m;
    int tempint;
    int order = ms.GetTotalOrders();
    /// initial Matrix elements index
    tempint = 0;
    MEindex_i = (int *)mkl_malloc((2 * order - 1) * sizeof(int *), 64);
    MEindex_j = (int *)mkl_malloc((2 * order - 1) * sizeof(int *), 64);
    for (i1 = 0; i1 < order; i1++)
    // 0 = > Total_Order - 1
    {

        j1 = order - 1;                       ///
        if (i1 < order - 1 && j1 < order - 1) /// Only the last order do variation
        {
            continue;
        }
        MEindex_i[tempint] = i1;
        MEindex_j[tempint] = j1;
        tempint++;
    }
    ME_total = tempint;
    /// initial overlap index
    int totalmeshPoints = AMproj.GetTotalMeshPoints();
    OvlInd_R = (int *)mkl_malloc((totalmeshPoints) * sizeof(int *), 64);
    OvlInd_ME = (int *)mkl_malloc((totalmeshPoints) * sizeof(int *), 64);

    /*for (i1 = 0; i1 < ME_total; i1++)
    {
        for (j1 = 0; j1 < totalmeshPoints; j1++)
        {
            OvlInd_ME[tempint] = i1;
            OvlInd_R[tempint] = j1;
            tempint++;
        }
    }*/
    Ovl_total = ME_total * totalmeshPoints;
    /// initial proton Identical Hpp index
    int CalTerms = Ham.Get_CalTerms_pp();
    Hp_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hp_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial neutron Identical Hnn index
    CalTerms = Ham.Get_CalTerms_nn();
    Hn_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hn_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial Vpn index
    CalTerms = Ham.Get_CalTerms_pn();
    Hpn_Hindex = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hpn_m = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    if (ms.GetNeutronPairNum() == 0)
    {
        InitializeHindex_Iden(Ham);
    }
    else
    {
        InitializeHindex(Ham);
    }

    return;
}

MatrixIndex::MatrixIndex(ModelSpace &ms, AngMomProjection &AMproj, Hamiltonian &Ham, bool GCMprojection) // GCM projection
    : ms(&ms), AMproj(&AMproj), Ham(&Ham)
{
    int i1, j1, J, m;
    int tempint;
    int order = ms.GetTotalOrders();
    /// initial Matrix elements index
    tempint = 0;
    MEindex_i = (int *)mkl_malloc((order * order + order) / 2 * sizeof(int *), 64);
    MEindex_j = (int *)mkl_malloc((order * order + order) / 2 * sizeof(int *), 64);
    for (i1 = 0; i1 < order; i1++)
    // 0 = > Total_Order - 1
    {
        for (j1 = i1; j1 < order; j1++)
        // 0 = > Total_Order - 1
        {
            MEindex_i[tempint] = i1;
            MEindex_j[tempint] = j1;
            tempint++;
        }
    }
    ME_total = tempint;
    /// initial overlap index
    int totalmeshPoints = AMproj.GetTotalMeshPoints();
    OvlInd_R = (int *)mkl_malloc((totalmeshPoints) * sizeof(int *), 64);
    OvlInd_ME = (int *)mkl_malloc((totalmeshPoints) * sizeof(int *), 64);
    tempint = 0;
    /*for (i1 = 0; i1 < ME_total; i1++)
    {
        for (j1 = 0; j1 < totalmeshPoints; j1++)
        {
            OvlInd_ME[tempint] = i1;
            OvlInd_R[tempint] = j1;
            tempint++;
        }
    }*/
    Ovl_total = totalmeshPoints * ME_total; // total overlap after projection

    /// initial proton Identical Hpp index
    int CalTerms = Ham.Get_CalTerms_pp();
    Hp_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hp_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial neutron Identical Hnn index
    CalTerms = Ham.Get_CalTerms_nn();
    Hn_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hn_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial Vpn index
    CalTerms = Ham.Get_CalTerms_pn();
    Hpn_Hindex = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hpn_m = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    if (ms.GetNeutronPairNum() == 0)
    {
        InitializeHindex_Iden(Ham);
    }
    else
    {
        InitializeHindex(Ham);
    }

    return;
}

MatrixIndex::MatrixIndex(ModelSpace &ms, Hamiltonian &Ham) // for NPSM without projection
    : ms(&ms), Ham(&Ham)
{
    int i1, j1, J, m;
    int tempint;
    int order = ms.GetTotalOrders();
    /// initial index for Hamiltonian
    /// initial proton Identical Hpp index
    int CalTerms = Ham.Get_CalTerms_pp();
    Hp_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hp_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial neutron Identical Hnn index
    CalTerms = Ham.Get_CalTerms_nn();
    Hn_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hn_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial Vpn index
    CalTerms = Ham.Get_CalTerms_pn();
    Hpn_Hindex = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hpn_m = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    InitializeHindex(Ham);

    /// initial Matrix elements index
    tempint = 0;
    MEindex_i = (int *)mkl_malloc((2 * order - 1) * sizeof(int *), 64);
    MEindex_j = (int *)mkl_malloc((2 * order - 1) * sizeof(int *), 64);
    for (i1 = 0; i1 < order; i1++)
    // 0 = > Total_Order - 1
    {
        j1 = order - 1;                       ///
        if (i1 < order - 1 && j1 < order - 1) /// Only the last order do variation
        {
            continue;
        }
        MEindex_i[tempint] = i1;
        MEindex_j[tempint] = j1;
        tempint++;
    }
    ME_total = tempint;

    //////////////////////////////////////////////
    tempint = ME_total;                                   // cal overlap first both proton and neutron type0
    tempint += ME_total;                                  // cal S.P  both proton and neutron          type1
    tempint += Ham.Get_CalTerms_pp() * ME_total;          // cal Vpp                                   type2
    tempint += Ham.Get_CalTerms_nn() * ME_total;          // cal Vnn                                   type3
    tempint += Ham.GetCalOBOperatorNumber_p() * ME_total; // cal Qp                                    type4
    tempint += Ham.GetCalOBOperatorNumber_n() * ME_total; // cal Qn                                    type5

    /// initial index
    OvlInd_R = (int *)mkl_malloc((tempint) * sizeof(int *), 64);  // record ME type
    OvlInd_ME = (int *)mkl_malloc((tempint) * sizeof(int *), 64); // record ME index
    tempint = 0;
    for (i1 = 0; i1 < ME_total; i1++) // overlap
    {
        OvlInd_R[tempint] = 0;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    for (i1 = 0; i1 < ME_total; i1++) // S.P.
    {
        OvlInd_R[tempint] = 1;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    for (i1 = 0; i1 < Ham.Get_CalTerms_pp() * ME_total; i1++) // Vpp
    {
        OvlInd_R[tempint] = 2;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    for (i1 = 0; i1 < Ham.Get_CalTerms_nn() * ME_total; i1++) // Vnn
    {
        OvlInd_R[tempint] = 3;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    for (i1 = 0; i1 < Ham.GetCalOBOperatorNumber_p() * ME_total; i1++) // Qp
    {
        OvlInd_R[tempint] = 4;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    for (i1 = 0; i1 < Ham.GetCalOBOperatorNumber_n() * ME_total; i1++) // Qn
    {
        OvlInd_R[tempint] = 5;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    Ovl_total = tempint;
    return;
}

MatrixIndex::MatrixIndex(ModelSpace &ms, Hamiltonian &Ham, int HalfColsed) // for half-closed NPSM without projection
    : ms(&ms), Ham(&Ham)
{
    int i1, j1, J, m;
    int tempint;
    int order = ms.GetTotalOrders();
    /// initial index for Hamiltonian
    /// initial proton Identical Hpp index
    int CalTerms = Ham.Get_CalTerms_pp();
    Hp_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hp_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial neutron Identical Hnn index
    CalTerms = Ham.Get_CalTerms_nn();
    Hn_index = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hn_index_M = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    /// initial Vpn index
    CalTerms = Ham.Get_CalTerms_pn();
    Hpn_Hindex = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);
    Hpn_m = (int *)mkl_malloc((CalTerms) * sizeof(int), 64);

    InitializeHindex_Iden(Ham);

    /// initial Matrix elements index
    tempint = 0;
    MEindex_i = (int *)mkl_malloc((2 * order - 1) * sizeof(int *), 64);
    MEindex_j = (int *)mkl_malloc((2 * order - 1) * sizeof(int *), 64);
    for (i1 = 0; i1 < order; i1++)
    // 0 = > Total_Order - 1
    {
        j1 = order - 1;                       ///
        if (i1 < order - 1 && j1 < order - 1) /// Only the last order do variation
        {
            continue;
        }
        MEindex_i[tempint] = i1;
        MEindex_j[tempint] = j1;
        tempint++;
    }
    ME_total = tempint;

    //////////////////////////////////////////////
    tempint = ME_total;                          // cal overlap first                         type0
    tempint += ME_total;                         // cal S.P  both proton                      type1
    tempint += Ham.Get_CalTerms_pp() * ME_total; // cal Vpp                                   type2

    /// initial index
    OvlInd_R = (int *)mkl_malloc((tempint) * sizeof(int *), 64);  // record ME type
    OvlInd_ME = (int *)mkl_malloc((tempint) * sizeof(int *), 64); // record ME index
    tempint = 0;
    for (i1 = 0; i1 < ME_total; i1++) // overlap
    {
        OvlInd_R[tempint] = 0;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    for (i1 = 0; i1 < ME_total; i1++) // S.P.
    {
        OvlInd_R[tempint] = 1;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    for (i1 = 0; i1 < Ham.Get_CalTerms_pp() * ME_total; i1++) // Vpp
    {
        OvlInd_R[tempint] = 2;
        OvlInd_ME[tempint] = i1;
        tempint++;
    }
    Ovl_total = tempint;
    return;
}

MatrixIndex::~MatrixIndex()
{
    mkl_free(MEindex_i);
    mkl_free(MEindex_j);
    mkl_free(OvlInd_R);
    mkl_free(OvlInd_ME);
    mkl_free(Hp_index);
    mkl_free(Hp_index_M);
    mkl_free(Hn_index);
    mkl_free(Hn_index_M);
    mkl_free(Hpn_Hindex);
    mkl_free(Hpn_m);
    mkl_free(QpListSP);
    mkl_free(QnListSP);
}

void MatrixIndex::InitializeHindex(Hamiltonian &Ham)
{
    int i, m, t, tempint;
    if (Ham.H_IsCollective())
    {
        tempint = 0;
        for (i = 0; i < Ham.GetColVppNum(); i++)
        {
            for (m = -Ham.VCol_pp[i].J; m <= Ham.VCol_pp[i].J; m++)
            {
                Hp_index[tempint] = i;
                Hp_index_M[tempint] = m;
                tempint++;
            }
        }

        tempint = 0;
        for (i = 0; i < Ham.GetColVnnNum(); i++)
        {
            for (m = -Ham.VCol_nn[i].J; m <= Ham.VCol_nn[i].J; m++)
            {
                Hn_index[tempint] = i;
                Hn_index_M[tempint] = m;
                tempint++;
            }
        }

        tempint = 0;
        for (i = 0; i < Ham.GetV_phCoupledChannelNum(); i++)
        {
            t = Ham.OBchannel_p[Ham.Vpn_PHcoupled[i].QpOBChannelindex].t;
            for (m = -t; m <= t; m++)
            {
                Hpn_Hindex[tempint] = i;
                Hpn_m[tempint] = m;
                tempint++;
            }
        }

        QpListSP = (int *)mkl_malloc((Ham.GetOneBodyOperatorNumber_p()) * sizeof(int), 64);
        QnListSP = (int *)mkl_malloc((Ham.GetOneBodyOperatorNumber_n()) * sizeof(int), 64);
        tempint = 0;
        for (i = 0; i < Ham.GetOneBodyOperatorNumber_p(); i++)
        {
            QpListSP[i] = tempint;
            tempint += 2 * Ham.OBchannel_p[i].t + 1;
        }
        tempint = 0;
        for (i = 0; i < Ham.GetOneBodyOperatorNumber_n(); i++)
        {
            QnListSP[i] = tempint;
            tempint += 2 * Ham.OBchannel_n[i].t + 1;
        }
    }
    else
    {
        tempint = 0;
        for (i = 0; i < Ham.GetVppNum(); i++)
        {
            for (m = -Ham.Vpp[i].J; m <= Ham.Vpp[i].J; m++)
            {
                Hp_index[tempint] = i;
                Hp_index_M[tempint] = m;
                tempint++;
            }
        }

        tempint = 0;
        for (i = 0; i < Ham.GetVnnNum(); i++)
        {
            for (m = -Ham.Vnn[i].J; m <= Ham.Vnn[i].J; m++)
            {
                Hn_index[tempint] = i;
                Hn_index_M[tempint] = m;
                tempint++;
            }
        }

        tempint = 0;
        for (i = 0; i < Ham.GetV_phCoupledChannelNum(); i++)
        {
            t = Ham.OBchannel_p[Ham.Vpn_PHcoupled[i].QpOBChannelindex].t;
            for (m = -t; m <= t; m++)
            {
                Hpn_Hindex[tempint] = i;
                Hpn_m[tempint] = m;
                tempint++;
            }
        }

        QpListSP = (int *)mkl_malloc((Ham.GetOneBodyOperatorNumber_p()) * sizeof(int), 64);
        QnListSP = (int *)mkl_malloc((Ham.GetOneBodyOperatorNumber_n()) * sizeof(int), 64);
        tempint = 0;
        for (i = 0; i < Ham.GetOneBodyOperatorNumber_p(); i++)
        {
            QpListSP[i] = tempint;
            tempint += 2 * Ham.OBchannel_p[i].t + 1;
        }
        tempint = 0;
        for (i = 0; i < Ham.GetOneBodyOperatorNumber_n(); i++)
        {
            QnListSP[i] = tempint;
            tempint += 2 * Ham.OBchannel_n[i].t + 1;
        }
    }
}

void MatrixIndex::InitializeHindex_Iden(Hamiltonian &Ham)
{
    int i, m, t, tempint;
    if (Ham.H_IsCollective())
    {
        tempint = 0;
        for (i = 0; i < Ham.GetColVppNum(); i++)
        {
            for (m = -Ham.VCol_pp[i].J; m <= Ham.VCol_pp[i].J; m++)
            {
                Hp_index[tempint] = i;
                Hp_index_M[tempint] = m;
                tempint++;
            }
        }
    }
    else
    {
        tempint = 0;
        for (i = 0; i < Ham.GetVppNum(); i++)
        {
            for (m = -Ham.Vpp[i].J; m <= Ham.Vpp[i].J; m++)
            {
                Hp_index[tempint] = i;
                Hp_index_M[tempint] = m;
                tempint++;
            }
        }
    }
}

void MatrixIndex::GetQpIndex(int Qtm_index, int &Qt_index, int &t, int &m)
{
    int i;
    for (i = 0; i < Ham->GetOneBodyOperatorNumber_p(); i++)
    {
        if (Qtm_index < QpListSP[i])
        {
            break;
        }
    }
    i--;
    Qt_index = i;
    t = Ham->OBchannel_p[i].t;
    m = Qtm_index - QpListSP[i] - t;
}

void MatrixIndex::GetQnIndex(int Qtm_index, int &Qt_index, int &t, int &m)
{
    int i;
    for (i = 0; i < Ham->GetOneBodyOperatorNumber_n(); i++)
    {
        if (Qtm_index < QnListSP[i])
        {
            break;
        }
    }
    i--;
    Qt_index = i;
    t = Ham->OBchannel_n[i].t;
    m = Qtm_index - QnListSP[i] - t;
}
