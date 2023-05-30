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

#include "CalNPSM.h"
using namespace NPSMCommutator;

CalNPSM::CalNPSM(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj, MatrixIndex &MyIndex, MultiPairs &Stored_p, MultiPairs &Stored_n)
    : ms(&ms), Ham(&Ham), AngMomProj(&AngMomProj), MyIndex(&MyIndex), MP_stored_p(&Stored_p), MP_stored_n(&Stored_n)
{
    MP_p = new MultiPairs(&ms, &AngMomProj, Proton);
    MP_n = new MultiPairs(&ms, &AngMomProj, Neutron);
}

CalNPSM::CalNPSM(ModelSpace &ms, Hamiltonian &Ham, MatrixIndex &MyIndex, MultiPairs &Stored_p, MultiPairs &Stored_n)
    : ms(&ms), Ham(&Ham), MyIndex(&MyIndex), MP_stored_p(&Stored_p), MP_stored_n(&Stored_n)
{
    MP_p = new MultiPairs(ms, Proton);
    MP_n = new MultiPairs(ms, Neutron);
}

CalNPSM::~CalNPSM()
{
    delete MP_p;
    delete MP_n;
}

double CalNPSM::DealTotalHamiltonianMatrix(ComplexNum *OvlME, ComplexNum *HamME)
{
    ReadWriteFiles rw;
    int Total_Order = ms->GetTotalOrders();
    int Sum_Num = ms->GetNumberOfBasisSummed();
    double Sum_EigenVals;
    ComplexNum *Matrix1, *Matrix2;
    Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    if (Total_Order > 1)
    { /// Eigen problem
        Read_Build_Matrix(Total_Order, OvlME, Matrix1, Saved_Ovl_filename);
        Read_Build_Matrix(Total_Order, HamME, Matrix2, Saved_Ham_filename);
        Sum_EigenVals = EigenValues(Total_Order, Sum_Num, Matrix1, Matrix2);
        // Sum_EigenVals = RealEigenValues(Total_Order, Sum_Num, Matrix1, Matrix2);
    }
    else
    {
        // std::cout << "ME " << Matrix2[0] << "    Ovl " << Matrix1[0] << '\n';
        Sum_EigenVals = HamME[0].real() / OvlME[0].real();
    }
    // std::cout << std::fixed << std::showpoint;
    // std::cout << "Total Hamiltonian = " << std::setprecision(12) << tempval << '\n';

    if (ms->IsPrintDiagResult())
    {
        printf("Total Hamiltonian =  %.15lf \n", Sum_EigenVals);
    }

    mkl_free(Matrix1);
    mkl_free(Matrix2);
    return Sum_EigenVals;
}

void CalNPSM::SaveData_Iden(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME)
{
    ComplexNum *Matrix1, *Matrix2;
    int Total_Order = ms->GetTotalOrders();
    int N_p = ms->GetPairNumber(Proton);
    Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    ReadWriteFiles rw;
    rw.SavePairStruc_DiffPairs_Iden(*ms, x);
    for (int i = 1; i <= N_p; i++)
    {
        rw.OutputME(MP_p->GetDim(), MP_p->GetPointer(i), this->GetMatrixFilename(Proton, i, Total_Order));
    }

    if (Total_Order > 1)
    { /// Output Ham ME
        Read_Build_Matrix(Total_Order, OvlME, Matrix1, Saved_Ovl_filename);
        Read_Build_Matrix(Total_Order, HamME, Matrix2, Saved_Ham_filename);
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

void CalNPSM::SaveData_SamePairs_Iden(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME)
{
    ComplexNum *Matrix1, *Matrix2;
    int Total_Order = ms->GetTotalOrders();
    int N_p = ms->GetPairNumber(Proton);
    Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    ReadWriteFiles rw;
    // rw.SavePairStruc_DiffPairs_Iden(*ms, x);
    rw.SavePairStruc_SamePairs_Iden(*ms, x);
    for (int i = 1; i <= N_p; i++)
    {
        rw.OutputME(MP_p->GetDim(), MP_p->GetPointer(i), this->GetMatrixFilename(Proton, i, Total_Order));
    }

    if (Total_Order > 1)
    { /// Output Ham ME
        Read_Build_Matrix(Total_Order, OvlME, Matrix1, Saved_Ovl_filename);
        Read_Build_Matrix(Total_Order, HamME, Matrix2, Saved_Ham_filename);
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

void CalNPSM::SaveData_pn(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME)
{
    ComplexNum *Matrix1, *Matrix2;
    int Total_Order = ms->GetTotalOrders();
    int N_p = ms->GetPairNumber(Proton);
    int N_n = ms->GetPairNumber(Neutron);
    Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    ReadWriteFiles rw;

    rw.SavePairStruc_DiffPairs(Proton, *ms, x);
    rw.SavePairStruc_DiffPairs(Neutron, *ms, x);
    for (int i = 1; i <= N_p; i++)
    {
        rw.OutputME(MP_p->GetDim(), MP_p->GetPointer(i), this->GetMatrixFilename(Proton, i, Total_Order));
    }
    for (int i = 1; i <= N_n; i++)
    {
        rw.OutputME(MP_n->GetDim(), MP_n->GetPointer(i), this->GetMatrixFilename(Neutron, i, Total_Order));
    }

    if (Total_Order > 1)
    { /// Output Ham ME
        Read_Build_Matrix(Total_Order, OvlME, Matrix1, Saved_Ovl_filename);
        Read_Build_Matrix(Total_Order, HamME, Matrix2, Saved_Ham_filename);
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

void CalNPSM::SaveData_SamePairs_pn(const std::vector<double> &x, ComplexNum *OvlME, ComplexNum *HamME)
{
    ComplexNum *Matrix1, *Matrix2;
    int Total_Order = ms->GetTotalOrders();
    int N_p = ms->GetPairNumber(Proton);
    int N_n = ms->GetPairNumber(Neutron);
    Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
    ReadWriteFiles rw;

    rw.SavePairStruc_SamePairs(Proton, *ms, x);
    rw.SavePairStruc_SamePairs(Neutron, *ms, x);
    for (int i = 1; i <= N_p; i++)
    {
        rw.OutputME(MP_p->GetDim(), MP_p->GetPointer(i), this->GetMatrixFilename(Proton, i, Total_Order));
    }
    for (int i = 1; i <= N_n; i++)
    {
        rw.OutputME(MP_n->GetDim(), MP_n->GetPointer(i), this->GetMatrixFilename(Neutron, i, Total_Order));
    }

    if (Total_Order > 1)
    { /// Output Ham ME
        Read_Build_Matrix(Total_Order, OvlME, Matrix1, Saved_Ovl_filename);
        Read_Build_Matrix(Total_Order, HamME, Matrix2, Saved_Ham_filename);
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

string CalNPSM::GetMatrixFilename(int isospin, int nth_pair, int Order)
{
    if (isospin == Proton)
    {
        return PairMatrixPath + to_string(nth_pair) + "_" + to_string(Order) + "_p.dat";
    }
    else if (isospin == Neutron)
    {
        return PairMatrixPath + to_string(nth_pair) + "_" + to_string(Order) + "_n.dat";
    }
    else
    {
        std::cout << "isospin should be Proton and Neutron!" << std::endl;
        return "error";
    }
}

double CalNPSM::Cal_All_MEs_Iden(const std::vector<double> &x)
{
    ComplexNum *OvlME, *HamME, *tempME;
    int myid, numprocs;  // MPI parameters
    int i, MEnum, MEDim; //	int i, j, k, n1, n2, j1, j2;
    double tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    tempval = 0;
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
        Calculate_ME_Iden(i, HamME, OvlME);
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
        tempval = DealTotalHamiltonianMatrix(OvlME, tempME);
        if (tempval < fxval)
        {
            SaveData_Iden(x, OvlME, tempME);
            fxval = tempval;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&tempval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    return tempval;
}

double CalNPSM::Cal_All_MEs_SamePairs_Iden(const std::vector<double> &x)
{
    ComplexNum *OvlME, *HamME, *tempME;
    int myid, numprocs;  // MPI parameters
    int i, MEnum, MEDim; //	int i, j, k, n1, n2, j1, j2;
    double tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    tempval = 0;
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
        Calculate_ME_Iden(i, HamME, OvlME);
        Calculate_ME_SamePairs_Iden(i, HamME, OvlME);
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
        tempval = DealTotalHamiltonianMatrix(OvlME, tempME);
        if (tempval < fxval)
        {
            SaveData_SamePairs_Iden(x, OvlME, tempME);
            fxval = tempval;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&tempval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    return tempval;
}

double CalNPSM::Cal_All_MEs_NoProjection_Iden(const std::vector<double> &x)
{
    ComplexNum *OvlME, *HamME, *tempME;
    int myid, numprocs; // MPI parameters
    int i, k, Hind, Hm, t, MEnum, MEDim;
    double tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    tempval = 0;
    MEnum = MyIndex->ME_total;
    MEDim = Total_order * Total_order;

    /// build array for ME
    OvlME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(OvlME, 0, sizeof(double) * 2 * (MEnum));
    HamME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(HamME, 0, sizeof(double) * 2 * (MEnum));

    tempME = (ComplexNum *)mkl_malloc((MEnum) * sizeof(ComplexNum), 64);
    memset(tempME, 0, sizeof(double) * 2 * (MEnum));

    /// MPI inint
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    ///------- loop ME
    for (i = myid; i < MyIndex->Ovl_total; i += numprocs)
    {
        Calculate_ME_NoProjection_Iden(i, HamME, OvlME);
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
        /// Copy Matrix
        // cblas_zcopy(MEnum, tempME, 1, HamME, 1);
    }
    ////
    if (myid == 0)
    {
        // Construct Matrix
        // vzMul(MEnum, OvlME, HamME, tempME); // contruct Identical Hamiltonian
        /*for (i = 0; i < MEnum; i++)
        {
            OvlME[i] *= OvlME[i + MEnum]; // contruct Overlap
            tempME[i] += tempME[i + MEnum];
        }*/

        // std::cout << tempME[0] << OvlME[0] << std::endl;
        tempval = DealTotalHamiltonianMatrix(OvlME, tempME);
        if (tempval < fxval)
        {
            SaveData_Iden(x, OvlME, tempME);
            fxval = tempval;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&tempval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    return tempval;
}

double CalNPSM::Cal_All_MEs_pn(const std::vector<double> &x)
{
    ComplexNum *OvlME, *HamME, *tempME;
    int myid, numprocs;  // MPI parameters
    int i, MEnum, MEDim; //	int i, j, k, n1, n2, j1, j2;
    double tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    tempval = 0;
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
        Calculate_ME_pn(i, HamME, OvlME);
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
        if (tempval < fxval)
        {
            SaveData_pn(x, OvlME, tempME);
            fxval = tempval;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&tempval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    return tempval;
}

double CalNPSM::Cal_All_MEs_pn_NoProjection(const std::vector<double> &x)
{
    ComplexNum *OvlME, *HamME, *tempME, *QpMEs, *QnMEs, *tempQpME, *tempQnME, tempQQ;
    int myid, numprocs; // MPI parameters
    int i, k, Hind, Hm, t, MEnum, MEDim;
    double tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    tempval = 0;
    MEnum = MyIndex->ME_total;
    MEDim = Total_order * Total_order;

    int CalQpTerm = Ham->GetCalOBOperatorNumber_p();
    int CalQnTerm = Ham->GetCalOBOperatorNumber_n();

    /// build array for ME
    OvlME = (ComplexNum *)mkl_malloc((MEnum * 2) * sizeof(ComplexNum), 64); //[Neutron, Proton]
    memset(OvlME, 0, sizeof(double) * 2 * 2 * (MEnum));
    HamME = (ComplexNum *)mkl_malloc((MEnum * 2) * sizeof(ComplexNum), 64); //[Proton, Neutron]
    memset(HamME, 0, sizeof(double) * 2 * 2 * (MEnum));

    QpMEs = (ComplexNum *)mkl_malloc((CalQpTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(QpMEs, 0, sizeof(double) * 2 * (CalQpTerm * MEnum));
    QnMEs = (ComplexNum *)mkl_malloc((CalQnTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(QnMEs, 0, sizeof(double) * 2 * (CalQnTerm * MEnum));

    tempQpME = (ComplexNum *)mkl_malloc((CalQpTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(tempQpME, 0, sizeof(double) * 2 * (CalQpTerm * MEnum));
    tempQnME = (ComplexNum *)mkl_malloc((CalQnTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(tempQnME, 0, sizeof(double) * 2 * (CalQnTerm * MEnum));

    tempME = (ComplexNum *)mkl_malloc((MEnum * 2) * sizeof(ComplexNum), 64);
    memset(tempME, 0, sizeof(double) * 2 * 2 * (MEnum));

    /// MPI inint
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    ///------- loop ME
    for (i = myid; i < MyIndex->Ovl_total; i += numprocs)
    {
        Calculate_ME_NoProjection_pn(i, HamME, OvlME, QpMEs, QnMEs);
    }
    MPI_Reduce(OvlME, tempME, MEnum * 2 * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        /// Copy Matrix
        cblas_zcopy(MEnum * 2, tempME, 1, OvlME, 1);
    }
    MPI_Reduce(HamME, tempME, MEnum * 2 * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        /// Copy Matrix
        cblas_zcopy(MEnum * 2, tempME, 1, HamME, 1);
    }
    //// Reduce Qp and Qn
    MPI_Reduce(QpMEs, tempQpME, 2 * CalQpTerm * MEnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(QnMEs, tempQnME, 2 * CalQnTerm * MEnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    ////
    if (myid == 0)
    {
        // Construct Matrix
        vzMul(MEnum * 2, OvlME, HamME, tempME); // contruct Identical Hamiltonian
        for (i = 0; i < MEnum; i++)
        {
            OvlME[i] *= OvlME[i + MEnum]; // contruct Overlap
            tempME[i] += tempME[i + MEnum];
            for (k = 0; k < Ham->Get_CalTerms_pn(); k++)
            {
                Hind = MyIndex->Hpn_Hindex[k];
                Hm = MyIndex->Hpn_m[k];
                Vpn_phCoupledElements Vme = Ham->GetVpn_phcoupled_ME(Hind);
                t = Vme.GeT_t();
                int index_Qp = MyIndex->GetQpIndex(Vme.GetQpindex(), t, Hm);
                int index_Qn = MyIndex->GetQnIndex(Vme.GetQnindex(), t, -Hm);
                tempQQ = sgn(Hm) * Vme.GetV();
                tempQQ *= tempQpME[CalQpTerm * i + index_Qp] * tempQnME[CalQnTerm * i + index_Qn];
                tempME[i] += tempQQ;
            }
        }

        // std::cout << tempME[0] << OvlME[0] << std::endl;
        tempval = DealTotalHamiltonianMatrix(OvlME, tempME);
        if (tempval < fxval)
        {
            SaveData_pn(x, OvlME, tempME);
            fxval = tempval;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&tempval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    mkl_free(QpMEs);
    mkl_free(QnMEs);
    mkl_free(tempQpME);
    mkl_free(tempQnME);
    return tempval;
}

double CalNPSM::Cal_All_MEs_SamePairs_pn(const std::vector<double> &x)
{
    ComplexNum *OvlME, *HamME, *tempME;
    int myid, numprocs;  // MPI parameters
    int i, MEnum, MEDim; //	int i, j, k, n1, n2, j1, j2;
    double tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    tempval = 0;
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
        Calculate_ME_SamePairs_pn(i, HamME, OvlME);
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
        if (tempval < fxval)
        {
            SaveData_SamePairs_pn(x, OvlME, tempME);
            fxval = tempval;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&tempval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    return tempval;
}

double CalNPSM::Cal_All_MEs_SamePairs_pn_NoJProjection(const std::vector<double> &x)
{
    ComplexNum *OvlME, *HamME, *tempME, *QpMEs, *QnMEs, *tempQpME, *tempQnME, tempQQ;
    int myid, numprocs; // MPI parameters
    int i, k, Hind, Hm, t, MEnum, MEDim;
    double tempval;
    int Total_order = ms->GetTotalOrders();
    /// Initial state
    tempval = 0;
    MEnum = MyIndex->ME_total;
    MEDim = Total_order * Total_order;

    int CalQpTerm = Ham->GetCalOBOperatorNumber_p();
    int CalQnTerm = Ham->GetCalOBOperatorNumber_n();

    /// build array for ME
    OvlME = (ComplexNum *)mkl_malloc((MEnum * 2) * sizeof(ComplexNum), 64); //[Neutron, Proton]
    memset(OvlME, 0, sizeof(double) * 2 * 2 * (MEnum));
    HamME = (ComplexNum *)mkl_malloc((MEnum * 2) * sizeof(ComplexNum), 64); //[Proton, Neutron]
    memset(HamME, 0, sizeof(double) * 2 * 2 * (MEnum));

    QpMEs = (ComplexNum *)mkl_malloc((CalQpTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(QpMEs, 0, sizeof(double) * 2 * (CalQpTerm * MEnum));
    QnMEs = (ComplexNum *)mkl_malloc((CalQnTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(QnMEs, 0, sizeof(double) * 2 * (CalQnTerm * MEnum));

    tempQpME = (ComplexNum *)mkl_malloc((CalQpTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(tempQpME, 0, sizeof(double) * 2 * (CalQpTerm * MEnum));
    tempQnME = (ComplexNum *)mkl_malloc((CalQnTerm * MEnum) * sizeof(ComplexNum), 64);
    memset(tempQnME, 0, sizeof(double) * 2 * (CalQnTerm * MEnum));

    tempME = (ComplexNum *)mkl_malloc((MEnum * 2) * sizeof(ComplexNum), 64);
    memset(tempME, 0, sizeof(double) * 2 * 2 * (MEnum));

    /// MPI inint
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    ///------- loop ME
    for (i = myid; i < MyIndex->Ovl_total; i += numprocs)
    {
        Calculate_ME_NoProjection_pn_SamePairs(i, HamME, OvlME, QpMEs, QnMEs);
    }
    MPI_Reduce(OvlME, tempME, MEnum * 2 * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        /// Copy Matrix
        cblas_zcopy(MEnum * 2, tempME, 1, OvlME, 1);
    }
    MPI_Reduce(HamME, tempME, MEnum * 2 * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        /// Copy Matrix
        cblas_zcopy(MEnum * 2, tempME, 1, HamME, 1);
    }
    //// Reduce Qp and Qn
    MPI_Reduce(QpMEs, tempQpME, 2 * CalQpTerm * MEnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(QnMEs, tempQnME, 2 * CalQnTerm * MEnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    ////
    if (myid == 0)
    {
        ComplexNum Qud0, Qud2, Qud_2;
        // shape constrained
        if (ms->GetIsShapeConstrained())
        {
            Qud0 = 0;
            Qud2 = 0;
            Qud_2 = 0;
            for (i = 0; i < Ham->Q2MEs_p.Total_Number; i++)
            {
                Qud0 += OvlME[0] * Ham->Q2MEs_p.Q_MEs[i] * tempQpME[Ham->Q2MEs_p.Q0_list[i]];
                Qud2 += OvlME[0] * Ham->Q2MEs_p.Q_MEs[i] * tempQpME[Ham->Q2MEs_p.Q2_list[i]];
                Qud_2 += OvlME[0] * Ham->Q2MEs_p.Q_MEs[i] * tempQpME[Ham->Q2MEs_p.Q_2_list[i]];
            }
            for (i = 0; i < Ham->Q2MEs_n.Total_Number; i++)
            {
                Qud0 += OvlME[1] * Ham->Q2MEs_n.Q_MEs[i] * tempQnME[Ham->Q2MEs_n.Q0_list[i]];
                Qud2 += OvlME[1] * Ham->Q2MEs_n.Q_MEs[i] * tempQnME[Ham->Q2MEs_n.Q2_list[i]];
                Qud_2 += OvlME[1] * Ham->Q2MEs_n.Q_MEs[i] * tempQnME[Ham->Q2MEs_n.Q_2_list[i]];
            }
        }
        // Construct Matrix
        vzMul(MEnum * 2, OvlME, HamME, tempME); // contruct Identical Hamiltonian
        for (i = 0; i < MEnum; i++)
        {
            OvlME[i] *= OvlME[i + MEnum]; // contruct Overlap
            tempME[i] += tempME[i + MEnum];
            for (k = 0; k < Ham->Get_CalTerms_pn(); k++)
            {
                Hind = MyIndex->Hpn_Hindex[k];
                Hm = MyIndex->Hpn_m[k];
                Vpn_phCoupledElements Vme = Ham->GetVpn_phcoupled_ME(Hind);
                t = Vme.GeT_t();
                int index_Qp = MyIndex->GetQpIndex(Vme.GetQpindex(), t, Hm);
                int index_Qn = MyIndex->GetQnIndex(Vme.GetQnindex(), t, -Hm);
                tempQQ = sgn(Hm) * Vme.GetV();
                tempQQ *= tempQpME[CalQpTerm * i + index_Qp] * tempQnME[CalQnTerm * i + index_Qn];
                tempME[i] += tempQQ;
            }
        }

        // std::cout << tempME[0] << OvlME[0] << std::endl;

        if (MEnum == 1)
        {
            tempval = tempME[0].real() / OvlME[0].real();
            if (ms->GetIsShapeConstrained())
            {
                std::cout << tempval << "  " << Qud0 << "  " << Qud2 << "  " << Qud_2 << std::endl;
                tempval += ms->GetShapeConstant() * pow(Qud0.real() - ms->GetShapeQ0(), 2);
                tempval += ms->GetShapeConstant() * pow(Qud2.real() - ms->GetShapeQ2(), 2);
                tempval += ms->GetShapeConstant() * pow(Qud_2.real() - ms->GetShapeQ2(), 2);
            }
        }
        else
        {
            tempval = DealTotalHamiltonianMatrix(OvlME, tempME);
        }

        if (tempval < fxval)
        {
            SaveData_SamePairs_pn(x, OvlME, tempME);
            fxval = tempval;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&tempval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /// free array
    mkl_free(OvlME);
    mkl_free(HamME);
    mkl_free(tempME);
    mkl_free(QpMEs);
    mkl_free(QnMEs);
    mkl_free(tempQpME);
    mkl_free(tempQnME);
    return tempval;
}

void CalNPSM::Calculate_ME_Iden(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME)
{
    int i1, j1, k, Rindex, MEind, Hind, Hm;
    ComplexNum ovlME_p, temp_ME, weightFactor;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N = ms->GetPairNumber(Proton); // Identical system
    double mass_scaling = ms->GetMassDependentFactor();

    // MEind = MyIndex->OvlInd_ME[i];
    // Rindex = MyIndex->OvlInd_R[i];

    Rindex = i / MyIndex->ME_total;
    MEind = i % MyIndex->ME_total;

    i1 = MyIndex->MEindex_i[MEind]; // order
    j1 = MyIndex->MEindex_j[MEind]; // should always be TotalOrder - 1

    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->GuassQuad_weight(alpha, beta, gamma);

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_L;
    MultiPairs MPBasis_R(*MP_p);
    if (i1 < j1)
    {
        MPBasis_L = MP_stored_p[i1];
    }
    else
    {
        MPBasis_L = *MP_p;
    }
    MPBasis_R.RotatedPairs(alpha, beta, gamma);

    if (ParityProj == 0)
    {
        // Cal ovl ME
        ovlME_p = Cal_Overlap(N, *ms, MPBasis_L, MPBasis_R);
        MyOvlME[MEind] += weightFactor * ovlME_p;

        // Cal S.P. ME
        MyHamME[MEind] += weightFactor * CalOneBodyOperator(N, *ms, MPBasis_L, MPBasis_R, Ham->GetSPEpointer(Proton));
        // Cal Vpp ME
        if (Ham->H_IsCollective())
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME(N, *ms, *Ham, MPBasis_L, MPBasis_R, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME;
        }
        else
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME(N, *ms, *Ham, MPBasis_L, MPBasis_R, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME;
        }
    }
    else
    {
        MultiPairs MPBasis_R_Parity(MPBasis_R);
        MPBasis_R_Parity.ParityProjection();

        // Cal ovl ME
        ovlME_p = Cal_Overlap_parity(N, *ms, MPBasis_L, MPBasis_R, MPBasis_R_Parity);
        MyOvlME[MEind] += weightFactor * ovlME_p;

        // Cal S.P. ME
        MyHamME[MEind] += weightFactor * CalOneBodyOperator_parity(N, *ms, MPBasis_L, MPBasis_R, MPBasis_R_Parity, Ham->GetSPEpointer(Proton));
        // Cal Vpp ME
        if (Ham->H_IsCollective())
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_parity(N, *ms, *Ham, MPBasis_L, MPBasis_R, MPBasis_R_Parity, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME;
        }
        else
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_parity(N, *ms, *Ham, MPBasis_L, MPBasis_R, MPBasis_R_Parity, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME;
        }
    }
    //////////////////////////////////////////
}

void CalNPSM::Calculate_ME_SamePairs_Iden(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME)
{
    int i1, j1, k, Rindex, MEind, Hind, Hm;
    ComplexNum ovlME_p, temp_ME, weightFactor;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N = ms->GetPairNumber(Proton); // Identical system
    double mass_scaling = ms->GetMassDependentFactor();

    // MEind = MyIndex->OvlInd_ME[i];
    // Rindex = MyIndex->OvlInd_R[i];

    Rindex = i / MyIndex->ME_total;
    MEind = i % MyIndex->ME_total;

    i1 = MyIndex->MEindex_i[MEind]; // order
    j1 = MyIndex->MEindex_j[MEind]; // should always be TotalOrder - 1

    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->GuassQuad_weight(alpha, beta, gamma);

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_L;
    MultiPairs MPBasis_R(*MP_p);
    if (i1 < j1)
    {
        MPBasis_L = MP_stored_p[i1];
    }
    else
    {
        MPBasis_L = *MP_p;
    }
    MPBasis_R.RotatedPairs(alpha, beta, gamma);

    if (ParityProj == 0)
    {
        // Cal ovl ME
        ovlME_p = Cal_Overlap(N, *ms, MPBasis_L, MPBasis_R);
        MyOvlME[MEind] += weightFactor * ovlME_p;

        // Cal S.P. ME
        MyHamME[MEind] += weightFactor * CalOneBodyOperator_SamePairs(N, *ms, MPBasis_L, MPBasis_R, Ham->GetSPEpointer(Proton));

        // Cal Vpp ME
        if (Ham->H_IsCollective())
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N, *ms, *Ham, MPBasis_L, MPBasis_R, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME;
        }
        else
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N, *ms, *Ham, MPBasis_L, MPBasis_R, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * weightFactor * temp_ME;
        }
    }
    else
    {
        MultiPairs MPBasis_R_Parity(MPBasis_R);
        MPBasis_R_Parity.ParityProjection();
        double Pairty = ms->GetProjected_parity();

        // Cal ovl ME
        ovlME_p = Cal_Overlap_parity(N, *ms, MPBasis_L, MPBasis_R, MPBasis_R_Parity);
        MyOvlME[MEind] += weightFactor * ovlME_p;

        // Cal S.P. ME
        MyHamME[MEind] += 0.5 * weightFactor * CalOneBodyOperator_SamePairs(N, *ms, MPBasis_L, MPBasis_R, Ham->GetSPEpointer(Proton));
        MyHamME[MEind] += 0.5 * Pairty * weightFactor * CalOneBodyOperator_SamePairs(N, *ms, MPBasis_L, MPBasis_R_Parity, Ham->GetSPEpointer(Proton));

        // Cal Vpp ME
        if (Ham->H_IsCollective())
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N, *ms, *Ham, MPBasis_L, MPBasis_R, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_SamePairs(N, *ms, *Ham, MPBasis_L, MPBasis_R_Parity, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME;
        }
        else
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N, *ms, *Ham, MPBasis_L, MPBasis_R, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * mass_scaling * weightFactor * temp_ME;

            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_SamePairs(N, *ms, *Ham, MPBasis_L, MPBasis_R_Parity, Hind, Hm);
            }
            MyHamME[MEind] += 0.5 * Pairty * mass_scaling * weightFactor * temp_ME;
        }
    }
    //////////////////////////////////////////
}

void CalNPSM::Calculate_ME_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME)
{
    int i1, j1, t, k, Rindex, MEind, Hind, Hm;
    ComplexNum ovlME_p, ovlME_n, temp_ME, weightFactor;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    double mass_scaling = ms->GetMassDependentFactor();

    // MEind = MyIndex->OvlInd_ME[i];
    // Rindex = MyIndex->OvlInd_R[i];

    Rindex = i / MyIndex->ME_total;
    MEind = i % MyIndex->ME_total;

    i1 = MyIndex->MEindex_i[MEind]; // order
    j1 = MyIndex->MEindex_j[MEind]; // should always be TotalOrder - 1

    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->GuassQuad_weight(alpha, beta, gamma);

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp;
    MultiPairs MPBasis_Rp(*MP_p);
    MultiPairs MPBasis_Ln;
    MultiPairs MPBasis_Rn(*MP_n);
    if (i1 < j1)
    {
        MPBasis_Lp = MP_stored_p[i1];
        MPBasis_Ln = MP_stored_n[i1];
    }
    else
    {
        MPBasis_Lp = *MP_p;
        MPBasis_Ln = *MP_n;
    }
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

void CalNPSM::Calculate_ME_SamePairs_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME)
{
    int i1, j1, t, k, Rindex, MEind, Hind, Hm;
    ComplexNum ovlME_p, ovlME_n, temp_ME, weightFactor;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    double mass_scaling = ms->GetMassDependentFactor();

    // MEind = MyIndex->OvlInd_ME[i];
    // Rindex = MyIndex->OvlInd_R[i];

    Rindex = i / MyIndex->ME_total;
    MEind = i % MyIndex->ME_total;

    i1 = MyIndex->MEindex_i[MEind]; // order
    j1 = MyIndex->MEindex_j[MEind]; // should always be TotalOrder - 1

    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->GuassQuad_weight(alpha, beta, gamma);

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp;
    MultiPairs MPBasis_Rp(*MP_p);
    MultiPairs MPBasis_Ln;
    MultiPairs MPBasis_Rn(*MP_n);
    if (i1 < j1)
    {
        MPBasis_Lp = MP_stored_p[i1];
        MPBasis_Ln = MP_stored_n[i1];
    }
    else
    {
        MPBasis_Lp = *MP_p;
        MPBasis_Ln = *MP_n;
    }
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

ComplexNum CalNPSM::Calculate_Vpn_MEs_pairty(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n)
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

ComplexNum CalNPSM::Calculate_Vpn_MEs_pairty_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_R_Parity_p, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn, MultiPairs &MPBasis_R_Parity_n)
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

ComplexNum CalNPSM::Calculate_Vpn_MEs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn)
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

ComplexNum CalNPSM::Calculate_Vpn_MEs_SamePairs(int N_p, int N_n, MultiPairs &MPBasis_Lp, MultiPairs &MPBasis_Rp, MultiPairs &MPBasis_Ln, MultiPairs &MPBasis_Rn)
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

void CalNPSM::Calculate_ME_NoProjection_pn(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME, ComplexNum *QpME, ComplexNum *QnME)
{
    int i1, j1, t, k, Rindex, MEind, Ht, Hind, Hm, totalME;
    ComplexNum ovlME_p, ovlME_n, temp_ME;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    double mass_scaling = ms->GetMassDependentFactor();

    Rindex = MyIndex->OvlInd_R[i]; // record ME type
    MEind = MyIndex->OvlInd_ME[i]; // record ME index
    totalME = MyIndex->ME_total;   // total number of basis
    // t the matrix index
    // k the index of Hamiltonian
    switch (Rindex)
    {
    case 0: // cal overlap
    {
        t = MEind;
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 1: // Cal S.P
    {
        t = MEind;
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 2: // cal Vpp
    {
        k = MEind % Ham->Get_CalTerms_pp();
        t = MEind / Ham->Get_CalTerms_pp();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 3: // cal Vnn
        k = MEind % Ham->Get_CalTerms_nn();
        t = MEind / Ham->Get_CalTerms_nn();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
        break;
    case 4: // Cal Qp
        k = MEind % Ham->GetCalOBOperatorNumber_p();
        t = MEind / Ham->GetCalOBOperatorNumber_p();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
        break;
    case 5: // Cal Qn
        k = MEind % Ham->GetCalOBOperatorNumber_n();
        t = MEind / Ham->GetCalOBOperatorNumber_n();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
        break;
    default:
        std::cout << "The matrix index error! Calculate_ME_NoProjection_pn() " << Rindex << std::endl;
        exit(0);
    }

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp;
    MultiPairs MPBasis_Rp(*MP_p);
    MultiPairs MPBasis_Ln;
    MultiPairs MPBasis_Rn(*MP_n);
    if (i1 < j1)
    {
        MPBasis_Lp = MP_stored_p[i1];
        MPBasis_Ln = MP_stored_n[i1];
    }
    else
    {
        MPBasis_Lp = *MP_p;
        MPBasis_Ln = *MP_n;
    }

    if (ParityProj == 0)
    {
        switch (Rindex)
        {
        case 0: // cal overlap
            MyOvlME[t] = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
            MyOvlME[totalME + t] = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
            break;
        case 1: // Cal S.P
            MyHamME[t] += CalOneBodyOperator(N_p, *ms, MPBasis_Lp, MPBasis_Rp, Ham->GetSPEpointer(Proton));
            MyHamME[totalME + t] += CalOneBodyOperator(N_n, *ms, MPBasis_Ln, MPBasis_Rn, Ham->GetSPEpointer(Neutron));
            break;
        case 2: // cal Vpp
            // Cal Vpp and Vnn MEs
            if (Ham->H_IsCollective())
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                MyHamME[t] -= mass_scaling * Prepare_Collective_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            else
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                MyHamME[t] += mass_scaling * Prepare_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            break;
        case 3:
            // Cal Vpp and Vnn MEs
            if (Ham->H_IsCollective())
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                MyHamME[totalME + t] -= mass_scaling * Prepare_Collective_pairingME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            else
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                MyHamME[totalME + t] += mass_scaling * Prepare_pairingME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            break;
        case 4: // Cal Qp
            MyIndex->GetQpIndex(k, Hind, Ht, Hm);
            // std::cout << k << " " << Hind << " " << Ht << " " << Hm << std::endl;
            QpME[MEind] = mass_scaling * Prepare_OB_ME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            break;
        case 5: // Cal Qn
            MyIndex->GetQnIndex(k, Hind, Ht, Hm);
            QnME[MEind] = Prepare_OB_ME(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            break;
        default:
            std::cout << "The matrix index error! Calculate_ME_NoProjection_pn()" << std::endl;
            exit(0);
        }
    }
    else
    {
        std::cout << "To be implemented ! " << std::endl;
        exit(0);
        MultiPairs MPBasis_R_Parity_p(MPBasis_Rp);
        MultiPairs MPBasis_R_Parity_n(MPBasis_Rn);
        MPBasis_R_Parity_p.ParityProjection();
        MPBasis_R_Parity_n.ParityProjection();

        // Cal ovl ME
        ovlME_p = Cal_Overlap_parity(N_p, *ms, MPBasis_Lp, MPBasis_Rp, MPBasis_R_Parity_p);
        ovlME_n = Cal_Overlap_parity(N_n, *ms, MPBasis_Ln, MPBasis_Rn, MPBasis_R_Parity_n);
        MyOvlME[MEind] += ovlME_p * ovlME_n;

        // Cal S.P. ME
        MyHamME[MEind] += CalOneBodyOperator_parity(N_p, *ms, MPBasis_Lp, MPBasis_Rp, MPBasis_R_Parity_p, Ham->GetSPEpointer(Proton)) * ovlME_n;
        MyHamME[MEind] += CalOneBodyOperator_parity(N_n, *ms, MPBasis_Ln, MPBasis_Rn, MPBasis_R_Parity_n, Ham->GetSPEpointer(Neutron)) * ovlME_p;

        // Cal Vpp ME
        if (Ham->H_IsCollective())
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_parity(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, MPBasis_R_Parity_p, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * temp_ME * ovlME_n;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME -= Prepare_Collective_pairingME_parity(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, MPBasis_R_Parity_n, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * temp_ME * ovlME_p;
        }
        else
        {
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_pp(); k++)
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                temp_ME += Prepare_pairingME_parity(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, MPBasis_R_Parity_p, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * temp_ME * ovlME_n;

            // Vnn
            temp_ME = 0.;
            for (k = 0; k < Ham->Get_CalTerms_nn(); k++)
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                temp_ME += Prepare_pairingME_parity(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, MPBasis_R_Parity_n, Hind, Hm);
            }
            MyHamME[MEind] += mass_scaling * temp_ME * ovlME_p;
        }

        // Cal Vpn MEsls
        MyHamME[MEind] += mass_scaling * Calculate_Vpn_MEs_pairty(N_p, N_n, MPBasis_Lp, MPBasis_Rp, MPBasis_R_Parity_p, MPBasis_Ln, MPBasis_Rn, MPBasis_R_Parity_n);
    }
    //////////////////////////////////////////
}

void CalNPSM::Calculate_ME_NoProjection_pn_SamePairs(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME, ComplexNum *QpME, ComplexNum *QnME)
{
    int i1, j1, t, k, Rindex, MEind, Ht, Hind, Hm, totalME;
    ComplexNum ovlME_p, ovlME_n, temp_ME;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    double mass_scaling = ms->GetMassDependentFactor();

    Rindex = MyIndex->OvlInd_R[i]; // record ME type
    MEind = MyIndex->OvlInd_ME[i]; // record ME index
    totalME = MyIndex->ME_total;   // total number of basis
    // t the matrix index
    // k the index of Hamiltonian
    switch (Rindex)
    {
    case 0: // cal overlap
    {
        t = MEind;
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 1: // Cal S.P
    {
        t = MEind;
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 2: // cal Vpp
    {
        k = MEind % Ham->Get_CalTerms_pp();
        t = MEind / Ham->Get_CalTerms_pp();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 3: // cal Vnn
        k = MEind % Ham->Get_CalTerms_nn();
        t = MEind / Ham->Get_CalTerms_nn();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
        break;
    case 4: // Cal Qp
        k = MEind % Ham->GetCalOBOperatorNumber_p();
        t = MEind / Ham->GetCalOBOperatorNumber_p();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
        break;
    case 5: // Cal Qn
        k = MEind % Ham->GetCalOBOperatorNumber_n();
        t = MEind / Ham->GetCalOBOperatorNumber_n();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
        break;
    default:
        std::cout << "The matrix index error! Calculate_ME_NoProjection_pn() " << Rindex << std::endl;
        exit(0);
    }

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp;
    MultiPairs MPBasis_Rp(*MP_p);
    MultiPairs MPBasis_Ln;
    MultiPairs MPBasis_Rn(*MP_n);
    if (i1 < j1)
    {
        MPBasis_Lp = MP_stored_p[i1];
        MPBasis_Ln = MP_stored_n[i1];
    }
    else
    {
        MPBasis_Lp = *MP_p;
        MPBasis_Ln = *MP_n;
    }

    if (ParityProj == 0)
    {
        switch (Rindex)
        {
        case 0: // cal overlap
            MyOvlME[t] = Cal_Overlap(N_n, *ms, MPBasis_Ln, MPBasis_Rn);
            MyOvlME[totalME + t] = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
            break;
        case 1: // Cal S.P
            MyHamME[t] += CalOneBodyOperator_SamePairs(N_p, *ms, MPBasis_Lp, MPBasis_Rp, Ham->GetSPEpointer(Proton));
            MyHamME[totalME + t] += CalOneBodyOperator_SamePairs(N_n, *ms, MPBasis_Ln, MPBasis_Rn, Ham->GetSPEpointer(Neutron));
            break;
        case 2: // cal Vpp
            // Cal Vpp and Vnn MEs
            if (Ham->H_IsCollective())
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                MyHamME[t] -= mass_scaling * Prepare_Collective_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            else
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                MyHamME[t] += mass_scaling * Prepare_pairingME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            break;
        case 3:
            // Cal Vpp and Vnn MEs
            if (Ham->H_IsCollective())
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                MyHamME[totalME + t] -= mass_scaling * Prepare_Collective_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            else
            {
                Hind = MyIndex->Hn_index[k];
                Hm = MyIndex->Hn_index_M[k];
                MyHamME[totalME + t] += mass_scaling * Prepare_pairingME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            }
            break;
        case 4: // Cal Qp
            MyIndex->GetQpIndex(k, Hind, Ht, Hm);
            // std::cout << k << " " << Hind << " " << Ht << " " << Hm << std::endl;
            QpME[MEind] = mass_scaling * Prepare_OB_ME_SamePairs(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            break;
        case 5: // Cal Qn
            MyIndex->GetQnIndex(k, Hind, Ht, Hm);
            QnME[MEind] = Prepare_OB_ME_SamePairs(N_n, *ms, *Ham, MPBasis_Ln, MPBasis_Rn, Hind, Hm);
            break;
        default:
            std::cout << "The matrix index error! Calculate_ME_NoProjection_pn_SamePairs()" << std::endl;
            exit(0);
        }
    }
    else
    {
        std::cout << "To be implemented ! " << std::endl;
        exit(0);
    }
    //////////////////////////////////////////
}

void CalNPSM::Calculate_ME_NoProjection_Iden(int i, ComplexNum *MyHamME, ComplexNum *MyOvlME)
{
    int i1, j1, t, k, Rindex, MEind, Ht, Hind, Hm, totalME;
    ComplexNum ovlME_p, ovlME_n, temp_ME;
    int alpha, beta, gamma;
    int ParityProj = ms->GetProjected_parity();
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    double mass_scaling = ms->GetMassDependentFactor();

    Rindex = MyIndex->OvlInd_R[i]; // record ME type
    MEind = MyIndex->OvlInd_ME[i]; // record ME index
    totalME = MyIndex->ME_total;   // total number of basis
    // t the matrix index
    // k the index of Hamiltonian
    switch (Rindex)
    {
    case 0: // cal overlap
    {
        t = MEind;
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 1: // Cal S.P
    {
        t = MEind;
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    case 2: // cal Vpp
    {
        k = MEind % Ham->Get_CalTerms_pp();
        t = MEind / Ham->Get_CalTerms_pp();
        i1 = MyIndex->MEindex_i[t]; // order
        j1 = MyIndex->MEindex_j[t]; // should always be TotalOrder - 1
    }
    break;
    default:
        std::cout << "The matrix index error! Calculate_ME_NoProjection_Iden() " << Rindex << std::endl;
        exit(0);
    }

    // build basis
    // Only rotate the Right-hand side multi-pair basis
    MultiPairs MPBasis_Lp;
    MultiPairs MPBasis_Rp(*MP_p);
    if (i1 < j1)
    {
        MPBasis_Lp = MP_stored_p[i1];
    }
    else
    {
        MPBasis_Lp = *MP_p;
    }

    if (ParityProj == 0)
    {
        switch (Rindex)
        {
        case 0: // cal overlap
            MyOvlME[t] = Cal_Overlap(N_p, *ms, MPBasis_Lp, MPBasis_Rp);
            break;
        case 1: // Cal S.P
            MyHamME[t] += CalOneBodyOperator(N_p, *ms, MPBasis_Lp, MPBasis_Rp, Ham->GetSPEpointer(Proton));
            break;
        case 2: // cal Vpp
            // Cal Vpp and Vnn MEs
            if (Ham->H_IsCollective())
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                MyHamME[t] -= mass_scaling * Prepare_Collective_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            else
            {
                Hind = MyIndex->Hp_index[k];
                Hm = MyIndex->Hp_index_M[k];
                MyHamME[t] += mass_scaling * Prepare_pairingME(N_p, *ms, *Ham, MPBasis_Lp, MPBasis_Rp, Hind, Hm);
            }
            break;
        default:
            std::cout << "The matrix index error! Calculate_ME_NoProjection_Iden()" << std::endl;
            exit(0);
        }
    }
    else
    {
        std::cout << "To be implemented ! " << std::endl;
        exit(0);
    }
    //////////////////////////////////////////
}

double CalNPSM::EigenValues(int dim, int sumEigenvalues, ComplexNum *Ovl, ComplexNum *Ham)
{
    int i, j;
    double returu_value, *e;
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

    returu_value = 0;
    for (i = 0; i < sumEigenvalues; i++)
    {
        returu_value += e[i];
    }
    mkl_free(e);
    mkl_free(tempHam);
    mkl_free(tempMat);
    return returu_value;
}

double CalNPSM::RealEigenValues(int dim, int sumEigenvalues, ComplexNum *Ovl, ComplexNum *Ham)
{
    int i, j;
    double returu_value, *e;
    double *tempHam, *tempHam2, *tempMat, *tempMat2, *prt_a, *prt_b, *prt_c;
    e = (double *)mkl_malloc((dim) * sizeof(double), 64);
    tempHam = (double *)mkl_malloc((dim * dim) * sizeof(double), 64);
    tempHam2 = (double *)mkl_malloc((dim * dim) * sizeof(double), 64);
    tempMat = (double *)mkl_malloc((dim * dim) * sizeof(double), 64);
    tempMat2 = (double *)mkl_malloc((dim * dim) * sizeof(double), 64);
    /// Copy Matrix
    for (i = 0; i < dim * dim; i++)
    {
        tempMat[i] = Ovl[i].real();
        tempHam[i] = Ham[i].real();
    }
    /// Diag Ovl
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim, tempMat, dim, e) !=
        0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in normal "
               "procedure!!\n");
        exit(0);
    }
    for (i = 0; i < dim; i++) // resacle
        for (j = 0; j < dim; j++)
            tempMat[i * dim + j] /= sqrt(e[j]);
    mkl_domatcopy('R', 'T', dim, dim, 1., tempMat, dim, tempMat2, dim); /// tempMat2 = tempMat

    // Normal Ham matrix
    memset(tempHam2, 0, dim * dim * sizeof(double));
    prt_a = tempMat2;
    prt_b = tempHam;
    prt_c = tempHam2;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.,
                prt_a, dim, prt_b, dim, 0, prt_c, dim);
    memset(tempHam, 0, dim * dim * sizeof(double));
    prt_a = tempHam2;
    prt_b = tempMat;
    prt_c = tempHam;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.,
                prt_a, dim, prt_b, dim, 0, prt_c, dim);
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', 'L', dim, tempHam, dim, e) !=
        0)
    { /// 'N' stand for only eigenvalues
        printf("Error when Computes all eigenvalues and eigenvectors in Ham "
               "procedure!!\n");
        exit(0);
    }

    returu_value = 0;
    for (i = 0; i < sumEigenvalues; i++)
    {
        returu_value += e[i];
    }
    mkl_free(e);
    mkl_free(tempHam);
    mkl_free(tempHam2);
    mkl_free(tempMat);
    mkl_free(tempMat2);
    return returu_value;
}

void CalNPSM::Read_Build_Matrix(int dim, ComplexNum *ele, ComplexNum *NewEle, string fileName)
{
    int i, i1, j1;
    std::vector<ComplexNum> tempMat;
    ReadWriteFiles rw;
    tempMat.resize((dim - 1) * (dim - 1));
    rw.ReadME_vector(dim - 1, tempMat, fileName);
    for (i1 = 0; i1 < dim - 1; i1++)
    {
        for (j1 = 0; j1 < dim - 1; j1++)
        {
            NewEle[i1 * dim + j1] = tempMat[i1 * (dim - 1) + j1];
        }
    }
    for (i = 0; i < MyIndex->ME_total; i++)
    {
        i1 = MyIndex->MEindex_i[i];
        j1 = MyIndex->MEindex_j[i];
        NewEle[i1 * dim + j1] = ele[i];
        NewEle[j1 * dim + i1] = std::conj(ele[i]);
    }
}

//////////////////////////////////////////////////////
/// class NPSM_VAP
NPSM_VAP::NPSM_VAP(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj)
    : ms(&ms), Ham(&Ham), AngMomProj(&AngMomProj)
{
    MyIndex = new MatrixIndex(ms, AngMomProj, Ham);

    int TotalOrders = ms.GetTotalOrders();
    MP_stored_p = new MultiPairs[TotalOrders];
    MP_stored_n = new MultiPairs[TotalOrders];
    for (size_t i = 0; i < TotalOrders - 1; i++)
    {
        MultiPairs temp_p(&ms, &AngMomProj, Proton);
        MP_stored_p[i] = temp_p;
        MultiPairs temp_n(&ms, &AngMomProj, Neutron);
        MP_stored_n[i] = temp_n;

        MP_stored_p[i].ReadPairs(i + 1);
        MP_stored_n[i].ReadPairs(i + 1);

        // MP_stored_p[i].PrintAllParameters();
        // MP_stored_n[i].PrintAllParameters();
    }
}

NPSM_VAP::~NPSM_VAP()
{
    delete MyIndex;
    delete[] MP_stored_p, MP_stored_n;
};

double NPSM_VAP::operator()(const std::vector<double> &x) const
{
    CalNPSM MyCal(*ms, *Ham, *AngMomProj, *MyIndex, *MP_stored_p, *MP_stored_n);
    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_Diff(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
        }
        return MyCal.Cal_All_MEs_pn(x);
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_SamePairs(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
        }
        return MyCal.Cal_All_MEs_SamePairs_pn(x);
    }

    // MyCal.MP_p->PrintAllParameters();
    // MyCal.MP_n->PrintAllParameters();
}
//////////////////////////////////////////////////////

/// class NPSM_V
NPSM_V::NPSM_V(ModelSpace &ms, Hamiltonian &Ham)
    : ms(&ms), Ham(&Ham)
{
    // AngMomProjection AngMomProj;
    MyIndex = new MatrixIndex(ms, Ham);

    int TotalOrders = ms.GetTotalOrders();
    MP_stored_p = new MultiPairs[TotalOrders];
    MP_stored_n = new MultiPairs[TotalOrders];
    for (size_t i = 0; i < TotalOrders - 1; i++)
    {
        MultiPairs temp_p(ms, Proton);
        MP_stored_p[i] = temp_p;
        MultiPairs temp_n(ms, Neutron);
        MP_stored_n[i] = temp_n;

        MP_stored_p[i].ReadPairs(i + 1);
        MP_stored_n[i].ReadPairs(i + 1);

        // MP_stored_p[i].PrintAllParameters();
        // MP_stored_n[i].PrintAllParameters();
    }
}

NPSM_V::~NPSM_V()
{
    delete MyIndex;
    delete[] MP_stored_p, MP_stored_n;
};

double NPSM_V::operator()(const std::vector<double> &x) const
{
    CalNPSM MyCal(*ms, *Ham, *MyIndex, *MP_stored_p, *MP_stored_n);
    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_Diff(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
        }
        return MyCal.Cal_All_MEs_pn_NoProjection(x);
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_SamePairs(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
        }
        return MyCal.Cal_All_MEs_SamePairs_pn_NoJProjection(x);
    }
}

//------------ half colsed nuclei --------------//
/// class Iden_BrokenJ_NoJproj
Iden_BrokenJ_NoJproj::Iden_BrokenJ_NoJproj(ModelSpace &ms, Hamiltonian &Ham)
    : ms(&ms), Ham(&Ham)
{
    // AngMomProjection AngMomProj;
    MyIndex = new MatrixIndex(ms, Ham, Proton);

    int TotalOrders = ms.GetTotalOrders();
    MP_stored_p = new MultiPairs[TotalOrders];
    for (size_t i = 0; i < TotalOrders - 1; i++)
    {
        MultiPairs temp_p(ms, Proton);
        MP_stored_p[i] = temp_p;

        MP_stored_p[i].ReadPairs(i + 1);

        // MP_stored_p[i].PrintAllParameters();
    }
}

Iden_BrokenJ_NoJproj::~Iden_BrokenJ_NoJproj()
{
    delete MyIndex;
    delete[] MP_stored_p;
};

double Iden_BrokenJ_NoJproj::operator()(const std::vector<double> &x) const
{
    CalNPSM MyCal(*ms, *Ham, *MyIndex, *MP_stored_p, *MP_stored_n);
    // initial multi-pair basis
    MyCal.MP_p->Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
    // MyCal.MP_p->PrintAllParameters();
    return MyCal.Cal_All_MEs_NoProjection_Iden(x);
}

/// class Iden_VAP
Iden_VAP::Iden_VAP(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj)
    : ms(&ms), Ham(&Ham), AngMomProj(&AngMomProj)
{
    MyIndex = new MatrixIndex(ms, AngMomProj, Ham);

    int TotalOrders = ms.GetTotalOrders();
    MP_stored_p = new MultiPairs[TotalOrders];
    for (size_t i = 0; i < TotalOrders - 1; i++)
    {
        MultiPairs temp_p(ms, Proton);
        MP_stored_p[i] = temp_p;
        MP_stored_p[i].ReadPairs(i + 1);
        // MP_stored_p[i].PrintAllParameters();
    }
}

Iden_VAP::~Iden_VAP()
{
    delete MyIndex;
    delete[] MP_stored_p;
};

double Iden_VAP::operator()(const std::vector<double> &x) const
{
    CalNPSM MyCal(*ms, *Ham, *AngMomProj, *MyIndex, *MP_stored_p, *MP_stored_n);
    // initial multi-pair basis
    return MyCal.Cal_All_MEs_Iden(x);

    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_Diff(Proton, x);
            // MyCal.MP_n->Build_Basis_Diff(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            // MyCal.MP_n->Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
        }
        // return MyCal.Cal_All_MEs_pn(x);
        return MyCal.Cal_All_MEs_Iden(x);
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_SamePairs(Proton, x);
            // MyCal.MP_n->Build_Basis_SamePairs(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            // MyCal.MP_n->Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
        }
        // return MyCal.Cal_All_MEs_SamePairs_pn(x);
        return MyCal.Cal_All_MEs_SamePairs_Iden(x);
    }
}

//////////////////////////////////////////////////////
//------------ GCM --------------//
/// class NPSM_GCM_variation
NPSM_GCM_variation::NPSM_GCM_variation(ModelSpace &ms, Hamiltonian &Ham)
    : ms(&ms), Ham(&Ham)
{
    // AngMomProjection AngMomProj;
    MyIndex = new MatrixIndex(ms, Ham);

    int TotalOrders = ms.GetTotalOrders();
    MP_stored_p = new MultiPairs[TotalOrders];
    MP_stored_n = new MultiPairs[TotalOrders];
    for (size_t i = 0; i < TotalOrders - 1; i++)
    {
        MultiPairs temp_p(ms, Proton);
        MP_stored_p[i] = temp_p;
        MultiPairs temp_n(ms, Neutron);
        MP_stored_n[i] = temp_n;

        MP_stored_p[i].ReadPairs(i + 1);
        MP_stored_n[i].ReadPairs(i + 1);

        // MP_stored_p[i].PrintAllParameters();
        // MP_stored_n[i].PrintAllParameters();
    }

    GCMcount = new CountEvaluationsGCM(ms);
}

NPSM_GCM_variation::~NPSM_GCM_variation()
{
    delete MyIndex;
    delete[] MP_stored_p, MP_stored_n;
};

double NPSM_GCM_variation::operator()(const std::vector<double> &x) const
{
    CalNPSM MyCal(*ms, *Ham, *MyIndex, *MP_stored_p, *MP_stored_n);
    double value = 1111111.;

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_Diff(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
        }
        value = MyCal.Cal_All_MEs_pn_NoProjection(x);
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_SamePairs(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
        }
        value = MyCal.Cal_All_MEs_SamePairs_pn_NoJProjection(x);
    }

    if (GCMcount->CountCals())
    {
        // std::cout << GCMcount->GetFilename() << "  " << value << std::endl;
        ReadWriteFiles rw;
        rw.Output_GCM_points(GCMcount->GetFilename(), *ms, x, value, GCMcount->GetSavedNum());
    }
    return value;
}

/// class GCM_Projection
GCM_Projection::GCM_Projection(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj)
    : ms(&ms), Ham(&Ham), AngMomProj(&AngMomProj)
{
}

GCM_Projection::~GCM_Projection()
{
    delete MyIndex;
    delete[] MP_stored_p, MP_stored_n;
};

void GCM_Projection::ReadBasis(string Path) // initial index and array
{
    ReadWriteFiles rw;
    std::vector<string> names;
    std::vector<double> para_x;
    names = rw.Get_all_files_names_within_folder(Path); // open all directory

    int TotalOrders = names.size() - 2;                      // count the dimension
    ms->SetTotalOrders(TotalOrders);                         // set dimension for model space
    MyIndex = new MatrixIndex(*ms, *AngMomProj, *Ham, true); // index for GCM

    MP_stored_p = new MultiPairs[TotalOrders];
    MP_stored_n = new MultiPairs[TotalOrders];

    for (size_t i = 0; i < TotalOrders; i++)
    {
        MultiPairs temp_p(ms, AngMomProj, Proton);
        MP_stored_p[i] = temp_p;
        MultiPairs temp_n(ms, AngMomProj, Neutron);
        MP_stored_n[i] = temp_n;
        rw.Read_GCM_points(*ms, Path + names[i + 2], para_x, E_calculated);

        if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
        {
            if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
            {
                MP_stored_p[i].Build_Basis_Diff(Proton, para_x);
                MP_stored_n[i].Build_Basis_Diff(Neutron, para_x);
            }
            else
            {
                MP_stored_p[i].Build_Basis_BorkenJSchemePairs_Diff(Proton, para_x);
                MP_stored_n[i].Build_Basis_BorkenJSchemePairs_Diff(Neutron, para_x);
            }
        }
        else
        {
            if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
            {
                MP_stored_p[i].Build_Basis_SamePairs(Proton, para_x);
                MP_stored_n[i].Build_Basis_SamePairs(Neutron, para_x);
            }
            else
            {
                MP_stored_p[i].Build_Basis_BorkenJSchemePairs_SamePairs(Proton, para_x);
                MP_stored_n[i].Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, para_x);
            }
        }
        record_paras.push_back(para_x);
        // MP_stored_p[i].PrintAllParameters();
        // MP_stored_n[i].PrintAllParameters();
    }
}

std::vector<double> GCM_Projection::DoCalculation()
{
    int myid, i, error = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    Cal_GCM_projection MyCal(*ms, *Ham, *AngMomProj, *MyIndex, *MP_stored_p, *MP_stored_n);

    // check linear dependence
    std::vector<double> value = MyCal.Cal_Overlap_before_Porjection();
    if (myid == 0)
    {
        std::cout << "Eigenvalues of Overlap without J projection:" << std::endl;
        for (i = 0; i < value.size(); i++)
        {
            std::cout << i << ":    " << value[i] << std::endl;
            if (value[i] < MyCal.Get_Overlap_dependence())
            {
                break;
            }
        }
        if (i != value.size())
        {
            error = 5250707;
        }
    }

    MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (error != 0)
    {
        if (myid == 0)
        {
            fprintf(stderr, "Input basis are not linear independence! \n");
        }
        MPI_Finalize();
        exit(error);
    }

    return MyCal.Cal_All_MEs_pn();
    // MyCal.MP_p->PrintAllParameters();
    // MyCal.MP_n->PrintAllParameters();
}

int GCM_Projection::SelectBasis() // select orthogonal basis
{
    int myid, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    Cal_GCM_projection MyCal(*ms, *Ham, *AngMomProj, *MyIndex, *MP_stored_p, *MP_stored_n);
    std::vector<int> para_index = MyCal.SelectBasis();
    int count = 0;
    // Output basis here
    if (myid == 0)
    {
        ReadWriteFiles rw;
        for (i = 0; i < para_index.size(); i++)
        {
            int temp_index = para_index[i];
            if (ms->GetEnergyTruncationGCM() < E_calculated[temp_index])
            {
                continue;
            }
            string path = rw.GetSelectingMCMCbasisPath() + "MCMCselected_" + std::to_string(count) + ".dat";
            rw.Output_GCM_points(path, *ms, this->record_paras[temp_index], this->E_calculated[temp_index], i);
            count++;
        }
    }
    return count;
}

double GCM_Projection::overlap_nlopt(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    ComplexNum ovlME_p, ovlME_n;
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    ComplexNum OverlapME[4];
    // build basis
    MultiPairs MPBasis_xp(*ms, Proton);
    MultiPairs MPBasis_xn(*ms, Neutron);
    MultiPairs MPBasis_yp(*ms, Proton);
    MultiPairs MPBasis_yn(*ms, Neutron);

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_Diff(Proton, x);
            MPBasis_xn.Build_Basis_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_Diff(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_Diff(Neutron, this->record_paras[0]);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_Diff(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, this->record_paras[0]);
        }
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_SamePairs(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_SamePairs(Neutron, this->record_paras[0]);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, this->record_paras[0]);
        }
    }

    // Cal ovl MEs
    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_xp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_xn);
    OverlapME[0] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_yp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_yn, MPBasis_yn);
    OverlapME[3] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_yn);
    OverlapME[1] = ovlME_p * ovlME_n;
    OverlapME[2] = OverlapME[1];

    ////
    double e[2];
    /// Tridiag Ovl
    if (LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'L', 2, OverlapME, 2, e) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in Cal-Overlap procedure!!\n");
        exit(0);
    }
    std::cout << e[0] << "   " << e[1] << "  " << e[0] / e[1] << std::endl;
    return -e[0] / e[1];
}

double GCM_Projection::overlap_cosA(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    ComplexNum ovlME_p, ovlME_n;
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    ComplexNum OverlapME[4];
    // build basis
    MultiPairs MPBasis_xp(*ms, Proton);
    MultiPairs MPBasis_xn(*ms, Neutron);
    MultiPairs MPBasis_yp(*ms, Proton);
    MultiPairs MPBasis_yn(*ms, Neutron);

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_Diff(Proton, x);
            MPBasis_xn.Build_Basis_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_Diff(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_Diff(Neutron, this->record_paras[0]);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_Diff(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, this->record_paras[0]);
        }
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_SamePairs(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_SamePairs(Neutron, this->record_paras[0]);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, this->record_paras[0]);
        }
    }

    // Cal ovl MEs
    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_xp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_xn);
    OverlapME[0] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_yp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_yn, MPBasis_yn);
    OverlapME[3] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_yn);
    OverlapME[1] = ovlME_p * ovlME_n;
    // OverlapME[2] = OverlapME[1];

    double Eretutn = OverlapME[1].real() / sqrt(OverlapME[0].real()) / sqrt(OverlapME[3].real()); // cos beta = A*B / (  sqrt(A) * sqrt(B)  )
    std::cout << OverlapME[0].real() << "   " << OverlapME[3].real() << "  " << Eretutn << std::endl;
    return std::abs(Eretutn);
}

double GCM_Projection::overlap_cosAB(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    double CosAB = 0;
    for (size_t i = 0; i < this->record_paras.size(); i++)
    {
        CosAB += std::abs(this->Cal_cosAB_between_2configruations(x, record_paras[i]));
    }
    return CosAB;
}

void GCM_Projection::OptimizeConfigruation()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10.0, 10.0);

    std::vector<double> xnew, xgrad, lb, ub;
    for (size_t i = 0; i < record_paras[0].size(); i++)
    {
        xnew.push_back(dis(gen));
        lb.push_back(-50.);
        ub.push_back(50.);
    }
    nlopt::opt opt(nlopt::LN_COBYLA, record_paras[0].size());

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    auto objective_function = [](const std::vector<double> &x,
                                 std::vector<double> &grad,
                                 void *data) -> double
    {
        auto instance = static_cast<GCM_Projection *>(data);
        return instance->overlap_cosA(x, grad, data);
    };

    opt.set_min_objective(nlopt::vfunc(objective_function), this);
    opt.set_maxeval(ms->GetMaxNumberOfIteration()); // Max function evaluations

    // opt.set_xtol_rel(ms->GetOverlapMin());
    opt.set_ftol_rel(ms->GetOverlapMin());
    opt.set_ftol_abs(ms->GetOverlapMin());
    opt.set_stopval(ms->GetOverlapMin());
    double minf;

    try
    {
        nlopt::result result = opt.optimize(xnew, minf);
        std::cout << "found minimum " << minf << "  after " << opt.get_numevals() << "  iteration" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0)
    {
        ReadWriteFiles rw;
        string path = rw.GetMCMC_output_path() + "MCMCselected_Nlopt.dat";
        rw.Output_GCM_points(path, *ms, xnew, 1000., 0);
    }
}

void GCM_Projection::GnerateNewOrthogonalConfigruation()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10.0, 10.0);

    std::vector<double> xnew, xgrad, lb, ub;
    for (size_t i = 0; i < record_paras[0].size(); i++)
    {
        xnew.push_back(dis(gen));
        lb.push_back(-50.);
        ub.push_back(50.);
    }
    nlopt::opt opt(nlopt::LN_COBYLA, record_paras[0].size());

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    auto objective_function = [](const std::vector<double> &x,
                                 std::vector<double> &grad,
                                 void *data) -> double
    {
        auto instance = static_cast<GCM_Projection *>(data);
        double cosA = instance->overlap_cosAB(x, grad, data);
        // std::cout << cosA<< std::endl;
        return cosA;
    };

    opt.set_min_objective(nlopt::vfunc(objective_function), this);
    opt.set_maxeval(ms->GetMaxNumberOfIteration()); // Max function evaluations

    // opt.set_xtol_rel(ms->GetOverlapMin());
    opt.set_ftol_rel(0.001);
    opt.set_ftol_abs(0.001);
    opt.set_stopval(ms->GetOverlapMin());
    double minf;

    try
    {
        nlopt::result result = opt.optimize(xnew, minf);
        std::cout << "found minimum " << minf << "  after " << opt.get_numevals() << "  iteration" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0)
    {
        ReadWriteFiles rw;
        string path = rw.GetMCMC_output_path() + "MCMCselected_Nlopt.dat";
        rw.Output_GCM_points(path, *ms, xnew, 1000., 0);
    }
}

void GCM_Projection::Cal_angle_between_2configruations() // return cos
{
    if (ms->GetTotalOrders() != 2)
    {
        std::cout << " This code can only works with 2 configruations! " << ms->GetTotalOrders() << std::endl;
        exit(0);
    }

    ComplexNum ovlME_p, ovlME_n;
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    ComplexNum OverlapME[4];
    // build basis
    MultiPairs MPBasis_xp(*ms, Proton);
    MultiPairs MPBasis_xn(*ms, Neutron);
    MultiPairs MPBasis_yp(*ms, Proton);
    MultiPairs MPBasis_yn(*ms, Neutron);

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_Diff(Proton, this->record_paras[1]);
            MPBasis_xn.Build_Basis_Diff(Neutron, this->record_paras[1]);
            MPBasis_yp.Build_Basis_Diff(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_Diff(Neutron, this->record_paras[0]);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_Diff(Proton, this->record_paras[1]);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, this->record_paras[1]);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_Diff(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, this->record_paras[0]);
        }
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_SamePairs(Proton, this->record_paras[1]);
            MPBasis_xn.Build_Basis_SamePairs(Neutron, this->record_paras[1]);
            MPBasis_yp.Build_Basis_SamePairs(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_SamePairs(Neutron, this->record_paras[0]);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, this->record_paras[1]);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, this->record_paras[1]);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, this->record_paras[0]);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, this->record_paras[0]);
        }
    }

    // Cal ovl MEs
    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_xp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_xn);
    OverlapME[0] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_yp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_yn, MPBasis_yn);
    OverlapME[3] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_yn);
    OverlapME[1] = ovlME_p * ovlME_n;
    // OverlapME[2] = OverlapME[1];

    double Eretutn = OverlapME[1].real() / sqrt(OverlapME[0].real()) / sqrt(OverlapME[3].real()); // cos beta = A*B / (  sqrt(A) * sqrt(B)  )
    std::cout << " " << OverlapME[0].real() << "   " << OverlapME[3].real() << "     cos (beta) = " << Eretutn << std::endl;
    return;
}

void GCM_Projection::Cal_CosAs() // return cos
{
    double CosAB = 0;
    for (size_t i = 0; i < this->record_paras.size(); i++)
    {
        for (size_t j = i; j < this->record_paras.size(); j++)
        {
            if (i == j)
            {
                continue;
            }
            CosAB = std::abs(this->Cal_cosAB_between_2configruations(record_paras[j], record_paras[i]));
            std::cout << "Angle between configruations: " << i << " and " << j << ", cos(angle) = " << fixed << std::setprecision(8) << CosAB << "  angle is " << acos(CosAB) * 180. / M_PI << " (degree)" << std::endl;
        }
    }
}

double GCM_Projection::Cal_cosAB_between_2configruations(const std::vector<double> &x, const std::vector<double> &y) // return cos
{
    ComplexNum ovlME_p, ovlME_n;
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    ComplexNum OverlapME[4];
    // build basis
    MultiPairs MPBasis_xp(*ms, Proton);
    MultiPairs MPBasis_xn(*ms, Neutron);
    MultiPairs MPBasis_yp(*ms, Proton);
    MultiPairs MPBasis_yn(*ms, Neutron);

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_Diff(Proton, x);
            MPBasis_xn.Build_Basis_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_Diff(Proton, y);
            MPBasis_yn.Build_Basis_Diff(Neutron, y);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_Diff(Proton, y);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, y);
        }
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_SamePairs(Proton, y);
            MPBasis_yn.Build_Basis_SamePairs(Neutron, y);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, y);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, y);
        }
    }

    // Cal ovl MEs
    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_xp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_xn);
    OverlapME[0] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_yp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_yn, MPBasis_yn);
    OverlapME[3] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_yn);
    OverlapME[1] = ovlME_p * ovlME_n;
    // OverlapME[2] = OverlapME[1];

    double cosAB = OverlapME[1].real() / sqrt(OverlapME[0].real()) / sqrt(OverlapME[3].real()); // cos beta = A*B / (  sqrt(A) * sqrt(B)  )
    // std::cout << " " << OverlapME[0].real() << "   " << OverlapME[3].real() << "     cos (beta) = " << cosAB << std::endl;
    return cosAB;
}

//////////////////////////////////////////////////////
//------------ MCMC FOR GCM --------------//
/// class MCMC_NPSM
MCMC_NPSM::MCMC_NPSM(ModelSpace &ms, Hamiltonian &Ham, int numWalkers, int dim, std::vector<std::vector<double>> init_positions)
    : ms(&ms), Ham(&Ham)
{
    // Initial NPSM
    MyIndex = new MatrixIndex(ms, Ham);
    int TotalOrders = ms.GetTotalOrders();
    MP_stored_p = new MultiPairs[TotalOrders];
    MP_stored_n = new MultiPairs[TotalOrders];
    for (size_t i = 0; i < TotalOrders - 1; i++)
    {
        MultiPairs temp_p(ms, Proton);
        MP_stored_p[i] = temp_p;
        MultiPairs temp_n(ms, Neutron);
        MP_stored_n[i] = temp_n;

        MP_stored_p[i].ReadPairs(i + 1);
        MP_stored_n[i].ReadPairs(i + 1);

        // MP_stored_p[i].PrintAllParameters();
        // MP_stored_n[i].PrintAllParameters();
    }

    // Initial MCMC
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
}

MCMC_NPSM::MCMC_NPSM(ModelSpace &ms, Hamiltonian &Ham, int numWalkers, int dim, char *file_name)
{
    // Initial NPSM
    MyIndex = new MatrixIndex(ms, Ham);
    int TotalOrders = ms.GetTotalOrders();
    MP_stored_p = new MultiPairs[TotalOrders];
    MP_stored_n = new MultiPairs[TotalOrders];
    for (size_t i = 0; i < TotalOrders - 1; i++)
    {
        MultiPairs temp_p(ms, Proton);
        MP_stored_p[i] = temp_p;
        MultiPairs temp_n(ms, Neutron);
        MP_stored_n[i] = temp_n;

        MP_stored_p[i].ReadPairs(i + 1);
        MP_stored_n[i].ReadPairs(i + 1);

        // MP_stored_p[i].PrintAllParameters();
        // MP_stored_n[i].PrintAllParameters();
    }

    // Initial MCMC
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
}

MCMC_NPSM::~MCMC_NPSM()
{
    delete MyIndex;
    delete[] MP_stored_p, MP_stored_n;
};

double MCMC_NPSM::CalNPSM_E(const std::vector<double> &x)
{
    CalNPSM MyCal(*ms, *Ham, *MyIndex, *MP_stored_p, *MP_stored_n);
    double value = 100000.;

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs; 2 M scheme pairs
        {
            MyCal.MP_p->Build_Basis_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_Diff(Neutron, x);
        }
        else if (ms->GetPairType() == 1)
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
        }
        else if (ms->GetPairType() == 2)
        {
            MyCal.MP_p->Build_Basis_MSchemePairs_Diff(Proton, x);
            MyCal.MP_n->Build_Basis_MSchemePairs_Diff(Neutron, x);
        }
        else
        {
            std::cout<< "Pair type error! " << std::endl;
        }
        value = MyCal.Cal_All_MEs_pn_NoProjection(x);
    }
    else  // Identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MyCal.MP_p->Build_Basis_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_SamePairs(Neutron, x);
        }
        else
        {
            MyCal.MP_p->Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MyCal.MP_n->Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
        }
        value = MyCal.Cal_All_MEs_SamePairs_pn_NoJProjection(x);
    }
    return value;
}

double MCMC_NPSM::CalNPSM_MCMC_overlap_ratio(const std::vector<double> &x, const std::vector<double> &y)
{
    ComplexNum ovlME_p, ovlME_n;
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    ComplexNum OverlapME[4];
    // build basis
    MultiPairs MPBasis_xp(*ms, Proton);
    MultiPairs MPBasis_xn(*ms, Neutron);
    MultiPairs MPBasis_yp(*ms, Proton);
    MultiPairs MPBasis_yn(*ms, Neutron);

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_Diff(Proton, x);
            MPBasis_xn.Build_Basis_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_Diff(Proton, y);
            MPBasis_yn.Build_Basis_Diff(Neutron, y);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_Diff(Proton, y);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, y);
        }
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_SamePairs(Proton, y);
            MPBasis_yn.Build_Basis_SamePairs(Neutron, y);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, y);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, y);
        }
    }

    // Cal ovl MEs
    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_xp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_xn);
    OverlapME[0] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_yp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_yn, MPBasis_yn);
    OverlapME[3] = ovlME_p * ovlME_n;

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_yn);
    OverlapME[1] = ovlME_p * ovlME_n;
    OverlapME[2] = OverlapME[1];

    ////
    double e[2];
    /// Tridiag Ovl
    if (LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'L', 2, OverlapME, 2, e) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in Cal-Overlap procedure!!\n");
        exit(0);
    }
    // std::cout << e[0] << "   " << e[1] << "  " << e[0] / e[1] << std::endl << std::endl;
    return e[0] / e[1];
}

double MCMC_NPSM::CalNPSM_MCMC_overlap(const std::vector<double> &x, const std::vector<double> &y)
{
    ComplexNum ovlME_p, ovlME_n;
    int N_p = ms->GetProtonPairNum();
    int N_n = ms->GetNeutronPairNum();
    ComplexNum OverlapME;
    // build basis
    MultiPairs MPBasis_xp(*ms, Proton);
    MultiPairs MPBasis_xn(*ms, Neutron);
    MultiPairs MPBasis_yp(*ms, Proton);
    MultiPairs MPBasis_yn(*ms, Neutron);

    // initial multi-pair basis
    if (ms->GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_Diff(Proton, x);
            MPBasis_xn.Build_Basis_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_Diff(Proton, y);
            MPBasis_yn.Build_Basis_Diff(Neutron, y);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_Diff(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_Diff(Proton, y);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_Diff(Neutron, y);
        }
    }
    else
    {
        if (ms->GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
        {
            MPBasis_xp.Build_Basis_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_SamePairs(Proton, y);
            MPBasis_yn.Build_Basis_SamePairs(Neutron, y);
        }
        else
        {
            MPBasis_xp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, x);
            MPBasis_xn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, x);
            MPBasis_yp.Build_Basis_BorkenJSchemePairs_SamePairs(Proton, y);
            MPBasis_yn.Build_Basis_BorkenJSchemePairs_SamePairs(Neutron, y);
        }
    }

    // Cal ovl MEs

    ovlME_p = Cal_Overlap(N_p, *ms, MPBasis_xp, MPBasis_yp);
    ovlME_n = Cal_Overlap(N_n, *ms, MPBasis_xn, MPBasis_yn);
    OverlapME = ovlME_p * ovlME_n;

    return OverlapME.real();
}

void MCMC_NPSM::run_Metropolis(int total_draws)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    unsigned seed = ms->GetRandomSeed();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    double Temperature = ms->GetGCMTemperature();
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
                Provious_proposal = CalNPSM_E(xt);
                walker.Set_proposal(Current_proposal);
            }
            for (size_t loop_dim = 0; loop_dim < this->dim; loop_dim++)
            {
                value = xt[loop_dim] + 1.0 * distribution(generator);
                // std::cout << " Random number:  " << distribution(generator) << std::endl;
                y.push_back(value);
            }

            // acceptance probabilty
            Current_proposal = CalNPSM_E(y);
            // P = exp(-std::abs((Provious_proposal - Current_proposal)) / Temperature);
            P = exp((Provious_proposal - Current_proposal) / Temperature);
            r = RandomGenerator();
            if (myid == 0)
            {
                std::cout << i << " draw in walker " << k << "  The acceptance probailty  " << P << "   " << Provious_proposal << "  " << Current_proposal << std::endl;
            }
            if (r <= P)
            {
                // walker.setPosition(y);
                // walker.Set_proposal(Current_proposal);
                walker.setPosition(y, Current_proposal);
            }
            else
            {
                // walker.setPosition(xt);
                // walker.setSamePosition(xt, Provious_proposal);
                //
                walker.Set_proposal(Provious_proposal);
            }
            sample.AddNewWalker(walker);
        }
    }
}

void MCMC_NPSM::run_Metropolis_HeatUp(int total_draws)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    unsigned seed = ms->GetRandomSeed();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    double Temperature = ms->GetGCMTemperature();
    // set normal distribution for Metropolis draws
    for (int i = 0; i < total_draws; i++) // loop draws
    {
        for (int k = 0; k < this->nwalkers; k++) // loop walkers
        {
            Walker walker;
            walker = sample.getWalker(0);
            std::vector<double> y, xt;
            double value, r, P, Current_proposal, Provious_proposal, P_ovl;
            xt = walker.getPosition();
            if (walker.IsProposalInitial())
            {
                Provious_proposal = walker.Get_proposal();
            }
            else
            {
                Provious_proposal = CalNPSM_E(xt);
                walker.Set_proposal(Current_proposal);
            }

            for (size_t loop_dim = 0; loop_dim < this->dim; loop_dim++)
            {
                value = xt[loop_dim] + distribution(generator);
                // std::cout << " Norm:  " << distribution(generator) << std::endl;
                y.push_back(value);
            }

            // acceptance probabilty
            Current_proposal = CalNPSM_E(y);
            // P = exp(-std::abs((Provious_proposal - Current_proposal)) / Temperature);
            P = exp((Provious_proposal - Current_proposal) / Temperature);
            P_ovl = CalNPSM_MCMC_overlap_ratio(xt, y);
            r = RandomGenerator();
            if (myid == 0)
            {
                std::cout << i << " draw in walker " << k << "  The acceptance probailty  " << P << "       " << P_ovl << "   " << Provious_proposal << "  " << Current_proposal << std::endl;
            }
            if (Current_proposal < -70. and r <= P_ovl * 10.)
            {
                // walker.setPosition(y);
                // walker.Set_proposal(Current_proposal);
                walker.setPosition(y, Current_proposal);
            }
            else
            {
                // walker.setPosition(xt);
                // walker.setSamePosition(xt, Provious_proposal);
                //
                walker.Set_proposal(Provious_proposal);
            }
            sample.AddNewWalker(walker);
        }
    }
}

void MCMC_NPSM::run_ConstrainedMetropolis(int total_draws)
{
    int myid;
    ReadWriteFiles rw;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0)
    {
        std::cout << " Reading constrained points for walkers!" << std::endl;
        std::cout << " Reading from " << rw.GetGCMInput() << std::endl;
    }
    std::vector<string> names;
    names = rw.Get_all_files_names_within_folder(rw.GetMCMC_output_path()); // open all directory  /Output/MCMC
    int totalInputs = names.size() - 2;                                     // count the dimension
    std::vector<double> pos;
    for (int k = 0; k < totalInputs; k++) // initial ensamble
    {
        rw.Read_GCM_points(*ms, rw.GetMCMC_output_path() + names[k + 2], pos, E_calculated);
        Previous_paras.push_back(pos);
    }
    for (size_t i = 0; i < totalInputs; i++)
    {
        double tempOvl = CalNPSM_MCMC_overlap(Previous_paras[i], Previous_paras[i]);
        Overlap_calculated.push_back(sqrt(tempOvl));
    }

    ////////////////////////////////////////////////////////////////////////
    unsigned seed = ms->GetRandomSeed();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    double Temperature = ms->GetGCMTemperature();
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
                Provious_proposal = CalNPSM_E(xt);
                walker.Set_proposal(Current_proposal);
            }
            for (size_t loop_dim = 0; loop_dim < this->dim; loop_dim++)
            {
                value = xt[loop_dim] + 1.0 * distribution(generator);
                // std::cout << " Random number:  " << distribution(generator) << std::endl;
                y.push_back(value);
            }

            // acceptance probabilty
            Current_proposal = CalNPSM_E(y);
            double P_ovl = 0;
            double Current_ovlap = sqrt(CalNPSM_MCMC_overlap(y, y));
            for (size_t j = 0; j < totalInputs; j++)
            {
                P_ovl += abs(CalNPSM_MCMC_overlap(Previous_paras[j], y) / Overlap_calculated[j] / Current_ovlap);
                // std::cout<< CalNPSM_MCMC_overlap(Previous_paras[j], y) << "  " << Overlap_calculated[j] << "  " << Current_ovlap << std::endl ;
            }
            // P = exp(-std::abs((Provious_proposal - Current_proposal)) / Temperature);
            P = exp((Provious_proposal - Current_proposal) / Temperature - 1000. * P_ovl);
            r = RandomGenerator();
            if (myid == 0)
            {
                std::cout << i << " draw in walker " << k << "  The acceptance probailty  " << P << "  " << 1000. * P_ovl << "   " << Provious_proposal << "  " << Current_proposal << std::endl;
            }
            if (r <= P)
            {
                // walker.setPosition(y);
                // walker.Set_proposal(Current_proposal);
                walker.setPosition(y, Current_proposal);
            }
            else
            {
                // walker.setPosition(xt);
                // walker.setSamePosition(xt, Provious_proposal);
                //
                walker.Set_proposal(Provious_proposal);
            }
            sample.AddNewWalker(walker);
        }
    }
}

double MCMC_NPSM::RandomGenerator()
{
    return double((rand() % 707525)) / 707525.;
}

std::vector<std::vector<double>> MCMC_NPSM::load_data(char *file_name)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    std::ifstream file;
    std::vector<std::vector<double>> init_positions;
    file.open(file_name);

    if (!(file.is_open()))
    {
        std::cout << "\n\n ****You should pass an opened file!****\n\n";
        std::cout << "myid:  " << myid << std::endl;
        exit(0);
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

double MCMC_NPSM::Objective_function_MCMC(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    double sum_ovl = 0;
    double Current_ovlap = sqrt(CalNPSM_MCMC_overlap(x, x));
    for (size_t i = 0; i < this->Previous_paras.size(); i++)
    {
        sum_ovl += abs(CalNPSM_MCMC_overlap(Previous_paras[i], x) / Overlap_calculated[i] / Current_ovlap);
    }
    return CalNPSM_E(x) + 1000. * sum_ovl;
}

void MCMC_NPSM::run_SearchNlopt()
{
    int myid;
    ReadWriteFiles rw;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0)
    {
        std::cout << " Reading constrained points for walkers!" << std::endl;
        std::cout << " Reading from " << rw.GetGCMInput() << std::endl;
    }
    std::vector<string> names;
    names = rw.Get_all_files_names_within_folder(rw.GetMCMC_output_path()); // open all directory  /Output/MCMC
    int totalInputs = names.size() - 2;                                     // count the dimension
    std::vector<double> pos;
    for (int k = 0; k < totalInputs; k++) // initial ensamble
    {
        rw.Read_GCM_points(*ms, rw.GetMCMC_output_path() + names[k + 2], pos, E_calculated);
        Previous_paras.push_back(pos);
    }
    for (size_t i = 0; i < totalInputs; i++)
    {
        double tempOvl = CalNPSM_MCMC_overlap(Previous_paras[i], Previous_paras[i]);
        Overlap_calculated.push_back(sqrt(tempOvl));
    }
    ////////////////////////////////////////////////////////////////////////

    std::vector<double> record_paras = sample.getWalkerHistroy(0)[0];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10.0, 10.0);

    std::vector<double> xnew, xgrad, lb, ub;
    for (size_t i = 0; i < record_paras.size(); i++)
    {
        xnew.push_back(dis(gen));
        lb.push_back(-50.);
        ub.push_back(50.);
    }
    nlopt::opt opt(nlopt::LN_COBYLA, record_paras.size());

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    auto objective_function = [](const std::vector<double> &x,
                                 std::vector<double> &grad,
                                 void *data) -> double
    {
        auto instance = static_cast<MCMC_NPSM *>(data);
        double cosA = instance->Objective_function_MCMC(x, grad, data);
        // std::cout << cosA << std::endl;
        return cosA;
    };

    opt.set_min_objective(nlopt::vfunc(objective_function), this);
    opt.set_maxeval(ms->GetMaxNumberOfIteration()); // Max function evaluations

    // opt.set_xtol_rel(ms->GetOverlapMin());
    opt.set_ftol_rel(0.001);
    opt.set_ftol_abs(0.001);
    // opt.set_stopval(ms->GetOverlapMin());
    double minf;

    try
    {
        nlopt::result result = opt.optimize(xnew, minf);
        std::cout << "found minimum " << minf << "  after " << opt.get_numevals() << "  iteration" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    if (myid == 0)
    {
        ReadWriteFiles rw;
        string path = rw.GetMCMC_output_path() + "MCMC_newNlopt.dat";
        rw.Output_GCM_points(path, *ms, xnew, 1000., 0);
    }
}

double MCMC_NPSM::Objective_E_function(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    if (!grad.empty())
    {
        // Compute the gradient using numerical differentiation
        // nlopt::num_gradient(Objective_E_function, x, grad, 1e-4);
    }
    return CalNPSM_E(x);
}

void MCMC_NPSM::run_NloptE_min()
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    std::vector<double> record_paras = sample.getWalkerHistroy(0)[0];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10.0, 10.0);

    std::vector<double> xnew, xgrad, lb, ub;
    for (size_t i = 0; i < record_paras.size(); i++)
    {
        xnew.push_back(dis(gen));
        lb.push_back(-50.);
        ub.push_back(50.);
    }
    nlopt::opt opt(nlopt::LN_COBYLA, record_paras.size());

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    auto objective_function = [](const std::vector<double> &x,
                                 std::vector<double> &grad,
                                 void *data) -> double
    {
        auto instance = static_cast<MCMC_NPSM *>(data);
        double cosA = instance->Objective_E_function(x, grad, data);
        // std::cout << cosA << std::endl;
        return cosA;
    };

    opt.set_min_objective(nlopt::vfunc(objective_function), this);
    opt.set_maxeval(ms->GetMaxNumberOfIteration()); // Max function evaluations

    // opt.set_xtol_rel(ms->GetOverlapMin());
    opt.set_ftol_rel(0.01);
    opt.set_ftol_abs(0.01);
    // opt.set_stopval(ms->GetOverlapMin());
    double minf;

    try
    {
        nlopt::result result = opt.optimize(xnew, minf);
        std::cout << "found minimum " << minf << "  after " << opt.get_numevals() << "  iteration" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    if (myid == 0)
    {
        ReadWriteFiles rw;
        string path = rw.GetMCMC_output_path() + "MCMC_newNlopt.dat";
        rw.Output_GCM_points(path, *ms, xnew, 1000., 0);
    }
}
