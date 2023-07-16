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

///////////////////////////////////////////////////////////////////////////////////
//   Deformed Hatree Fock code for the valence space calculation
//   Ragnar's IMSRG code inspired this code.
//   Copyright (C) 2023  Bingcheng He
///////////////////////////////////////////////////////////////////////////////////

#include "HartreeFock.h"

HartreeFock::HartreeFock(Hamiltonian &H)
    : Ham(&H), modelspace(H.GetModelSpace()), tolerance(1e-8)
{
    this->N_p = modelspace->GetProtonNum();
    this->N_n = modelspace->GetNeutronNum();
    this->dim_p = modelspace->Get_MScheme_dim(Proton);
    this->dim_n = modelspace->Get_MScheme_dim(Neutron);
    this->U_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    memset(U_p, 0, (dim_p) * (dim_p) * sizeof(double));
    for (size_t i = 0; i < dim_p; i++)
    {
        U_p[i * (dim_p) + i] = 1.;
    }
    this->U_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    memset(U_n, 0, (dim_n) * (dim_n) * sizeof(double));
    for (size_t i = 0; i < dim_n; i++)
    {
        U_n[i * (dim_n) + i] = 1.;
    }
    this->FockTerm_p = (double *)mkl_malloc((dim_p) * (dim_p) * sizeof(double), 64);
    memset(FockTerm_p, 0, (dim_p) * (dim_p) * sizeof(double));
    this->FockTerm_n = (double *)mkl_malloc((dim_n) * (dim_n) * sizeof(double), 64);
    memset(FockTerm_n, 0, (dim_n) * (dim_n) * sizeof(double));
    this->Vij_p = (double *)mkl_malloc((dim_p) * (dim_p) * sizeof(double), 64);
    this->Vij_n = (double *)mkl_malloc((dim_n) * (dim_n) * sizeof(double), 64);
    this->rho_p = (double *)mkl_malloc((dim_p) * (dim_p) * sizeof(double), 64);
    memset(rho_p, 0, (dim_p) * (dim_p) * sizeof(double));
    this->rho_n = (double *)mkl_malloc((dim_n) * (dim_n) * sizeof(double), 64);
    memset(rho_n, 0, (dim_n) * (dim_n) * sizeof(double));

    // One body part
    T_term_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    T_term_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    memset(T_term_p, 0, (dim_p) * (dim_p) * sizeof(double));
    memset(T_term_n, 0, (dim_n) * (dim_n) * sizeof(double));

    for (size_t i = 0; i < Ham->OBEs_p.size(); i++)
    {
        int index_a, index_b;
        index_a = Ham->OBEs_p[i].GetIndex_a();
        index_b = Ham->OBEs_p[i].GetIndex_b();
        double temp_E = Ham->OBEs_p[i].GetE();
        int temp_j = modelspace->Orbits_p[index_a].j2;
        int M_i = modelspace->LookupStartingPoint(Proton, index_a);
        int M_j = modelspace->LookupStartingPoint(Proton, index_b);
        for (size_t j = 0; j < temp_j + 1; j++)
        {
            T_term_p[(M_i + j) * dim_p + (M_j + j)] = temp_E;
        }
    }
    for (size_t i = 0; i < Ham->OBEs_n.size(); i++)
    {
        int index_a, index_b;
        index_a = Ham->OBEs_n[i].GetIndex_a();
        index_b = Ham->OBEs_n[i].GetIndex_b();
        double temp_E = Ham->OBEs_n[i].GetE();
        int temp_j = modelspace->Orbits_n[index_a].j2;
        int M_i = modelspace->LookupStartingPoint(Neutron, index_a);
        int M_j = modelspace->LookupStartingPoint(Neutron, index_b);
        for (size_t j = 0; j < temp_j + 1; j++)
            T_term_n[(M_i + j) * dim_n + (M_j + j)] = temp_E;
    }

    // SPE in ascending order
    std::vector<Triple> SPEpairs_p(dim_p);
    for (int i = 0; i < dim_p; ++i)
    {
        // std::cout << i << "  " << T_term[i] << "  " << modelspace->Get_MSmatrix_2j(Proton, i) << "  " << modelspace->Get_MSmatrix_2m(Proton, i) << std::endl;
        SPEpairs_p[i] = Triple(T_term_p[i * dim_p + i], i, modelspace->Get_MSmatrix_2m(Proton, i));
    }
    std::sort(SPEpairs_p.begin(), SPEpairs_p.end(), compareTriples);
    /*
    for (size_t i = 0; i < dim_p; i++)
    {
        std::cout << SPEpairs_p[i].first << "  " << SPEpairs_p[i].second << "  " << modelspace->Get_MSmatrix_2j(Proton, SPEpairs_p[i].second) << "  " << modelspace->Get_MSmatrix_2m(Proton, SPEpairs_p[i].second) << std::endl;
    }*/
    this->holeorbs_p = (int *)mkl_malloc((N_p) * sizeof(int), 64);
    for (size_t i = 0; i < N_p; i++)
    {
        holeorbs_p[i] = SPEpairs_p[i].second;
        // std::cout << SPEpairs_p[i].first << "  " << SPEpairs_p[i].second << std::endl;
    }

    //-----------
    // Sort the vector of pairs based on the first element of each pair, which is the value of the
    // element in the original array.
    std::vector<Triple> SPEpairs_n(dim_n);
    for (int i = 0; i < dim_n; ++i)
    {
        // std::cout << i << "  " << T_term[i] << "  " << modelspace->Get_MSmatrix_2j(Proton, i) << "  " << modelspace->Get_MSmatrix_2m(Proton, i) << std::endl;
        SPEpairs_n[i] = Triple(T_term_n[i * dim_n + i], i, modelspace->Get_MSmatrix_2m(Neutron, i));
    }
    std::sort(SPEpairs_n.begin(), SPEpairs_n.end(), compareTriples);

    this->holeorbs_n = (int *)mkl_malloc((N_n) * sizeof(int), 64);
    for (size_t i = 0; i < N_n; i++)
    {
        holeorbs_n[i] = SPEpairs_n[i].second;
        // std::cout << SPEpairs_n[i].first << "  " << SPEpairs_n[i].second << std::endl;
    }

    this->energies = (double *)mkl_malloc((dim_p + dim_n) * sizeof(double), 64);
    this->prev_energies = (double *)mkl_malloc((dim_p + dim_n) * sizeof(double), 64);
    memset(energies, 0, (dim_p + dim_n) * sizeof(double));
    memset(prev_energies, 0, (dim_p + dim_n) * sizeof(double));
}

HartreeFock::~HartreeFock()
{
    mkl_free(FockTerm_p);
    mkl_free(FockTerm_n);
    mkl_free(Vij_p);
    mkl_free(Vij_n);
    mkl_free(U_p);
    mkl_free(U_n);
    mkl_free(holeorbs_p);
    mkl_free(holeorbs_n);
    mkl_free(rho_p);
    mkl_free(rho_n);
    mkl_free(energies);
    mkl_free(prev_energies);
    if (T_term_p != nullptr)
    {
        mkl_free(T_term_p);
    }
    if (T_term_n != nullptr)
    {
        mkl_free(T_term_n);
    }
}

//*********************************************************************
// The hybrid minimization method
void HartreeFock::Solve_hybrid()
{

    double density_mixing_factor = 0.2;
    double field_mixing_factor = 0.0;
    UpdateDensityMatrix();
    UpdateF();
    double *rho_last_p, *rho_last_n;
    rho_last_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    rho_last_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    // std::cout << "HF start  iterations. " << std::endl;
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        Diagonalize(); // Diagonalize the Fock matrix
        //-------------------------------------------
        if (iterations == 500)
        {
            density_mixing_factor = 0.7;
            field_mixing_factor = 0.5;
            std::cout << "Still not converged after 500 iterations. Setting density_mixing_factor => " << density_mixing_factor
                      << " field_mixing_factor => " << field_mixing_factor << std::endl;
        }
        if (iterations > 600 and iterations % 20 == 2) // if we've made it to 600, really put on the brakes with a big mixing factor, with random noise
        {
            field_mixing_factor = 1 - 0.005 * (std::rand() % 100);
            density_mixing_factor = 1 - 0.005 * (std::rand() % 100);
        }
        cblas_daxpby(dim_p * dim_p, density_mixing_factor, rho_p, 1, 0.0, rho_last_p, 1);
        cblas_daxpby(dim_n * dim_n, density_mixing_factor, rho_n, 1, 0.0, rho_last_n, 1);

        UpdateF();
        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);
        std::vector<double> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<double> Z_n(dim_n * dim_n, 0);

        // Cal_Gradient(Z_p.data(), Z_n.data());
        // Cal_Gradient_preconditioned(Z_p.data(), Z_n.data());
        Cal_Gradient_preconditioned_SRG(Z_p.data(), Z_n.data());
        UpdateU_Thouless_pade(Z_p.data(), Z_n.data());

        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n

        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - density_mixing_factor), rho_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - density_mixing_factor), rho_n, 1);

        cblas_daxpby(dim_p * dim_p, field_mixing_factor, FockTerm_p, 1, 0.0, rho_last_p, 1);
        cblas_daxpby(dim_n * dim_n, field_mixing_factor, FockTerm_n, 1, 0.0, rho_last_n, 1);
        UpdateF(); // Update the Fock matrix
        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - field_mixing_factor), FockTerm_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - field_mixing_factor), FockTerm_n, 1);
        // CalcEHF();
        // save hole parameters
        // string filename = "Output/HF_para_" + std::to_string(iterations) + ".dat";
        // this->SaveHoleParameters(filename);

        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        if (CheckConvergence())
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();
    mkl_free(rho_last_p);
    mkl_free(rho_last_n);

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
}

void HartreeFock::Solve_hybrid_Constraint()
{
    /// inital Q operator
    std::vector<double> Q2_p(dim_p * dim_p, 0);
    std::vector<double> Q0_p(dim_p * dim_p, 0);
    std::vector<double> Q2_n(dim_n * dim_n, 0);
    std::vector<double> Q0_n(dim_n * dim_n, 0);

    double Q0_expect = modelspace->GetShapeQ0();
    double Q2_expect = modelspace->GetShapeQ2();
    double deltaQ0, deltaQ2, deltaQ_2;
    double tempQp, tempQn;
    int number_of_Q = 2;

    memset(Q0_p.data(), 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q0_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
    }
    memset(Q2_p.data(), 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
    }
    // memset(Q_2_p.data(), 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
    }

    memset(Q0_n.data(), 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q0_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
    }
    memset(Q2_n.data(), 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
    }
    // memset(Q_2_n.data(), 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
    }

    ////////////////////////////////////////////////////////////
    double density_mixing_factor = 0.2;
    double field_mixing_factor = 0.;
    double *rho_last_p, *rho_last_n;
    rho_last_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    rho_last_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    gradient_eta = 0.01;
    iterations = 0; // count number of iterations

    double E_previous = 1.e10;
    double constrainedQ, ConstarinedLargeNum = 1000.;

    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
        UpdateF();
        CalOnebodyOperator(Q0_p.data(), Q0_n.data(), tempQp, tempQn);
        deltaQ0 = tempQp + tempQn - Q0_expect;
        // std::cout << deltaQ0 << "  " << Q0_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        CalOnebodyOperator(Q2_p.data(), Q2_n.data(), tempQp, tempQn);
        deltaQ2 = (tempQp + tempQn) - Q2_expect;
        // std::cout << deltaQ2 << "  " << Q2_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;

        constrainedQ = ConstarinedLargeNum * deltaQ0 * deltaQ0;
        constrainedQ += ConstarinedLargeNum * deltaQ2 * deltaQ2;
        CalcEHF(constrainedQ);

        //-------------------------------------------
        if (iterations == 500)
        {
            density_mixing_factor = 0.5;
            gradient_eta *= 0.1;
            // field_mixing_factor = 0.5;
            std::cout << "Still not converged after 500 iterations. Setting density_mixing_factor => " << density_mixing_factor
                      << " field_mixing_factor => " << field_mixing_factor << std::endl;
        }
        if (iterations > 600 and iterations % 20 == 2) // if we've made it to 600, really put on the brakes with a big mixing factor, with random noise
        {
            // field_mixing_factor = 1 - 0.005 * (std::rand() % 100);
            density_mixing_factor = 1 - 0.005 * (std::rand() % 100);
        }
        cblas_daxpby(dim_p * dim_p, density_mixing_factor, rho_p, 1, 0.0, rho_last_p, 1);
        cblas_daxpby(dim_n * dim_n, density_mixing_factor, rho_n, 1, 0.0, rho_last_n, 1);
        std::vector<double> Qorb0_p(dim_p * dim_p, 0);
        std::vector<double> Qorb0_n(dim_n * dim_n, 0);
        std::vector<double> Qorb2_p(dim_p * dim_p, 0);
        std::vector<double> Qorb2_n(dim_n * dim_n, 0);
        std::vector<double> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<double> Z_n(dim_n * dim_n, 0);

        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n.data(), 1);
        Operator_ph(Z_p.data(), Z_n.data());

        cblas_dcopy(dim_p * dim_p, Q0_p.data(), 1, Qorb0_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q0_n.data(), 1, Qorb0_n.data(), 1);
        cblas_dcopy(dim_p * dim_p, Q2_p.data(), 1, Qorb2_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q2_n.data(), 1, Qorb2_n.data(), 1);

        TransferOperatorToHFbasis(Qorb0_p.data(), Qorb0_n.data());
        TransferOperatorToHFbasis(Qorb2_p.data(), Qorb2_n.data());
        Operator_ph(Qorb0_p.data(), Qorb0_n.data());
        Operator_ph(Qorb2_p.data(), Qorb2_n.data());

        //---------------------------------------------
        std::vector<double> Fph_p(dim_p * dim_p, 0);
        std::vector<double> Fph_n(dim_n * dim_n, 0);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Fph_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Fph_n.data(), 1);
        Operator_ph(Fph_p.data(), Fph_n.data());

        double dotProduct_p, QdotHProduct_p, dotProduct_n, QdotHProduct_n;
        QdotHProduct_p = -cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb0_p.data(), 1);
        dotProduct_p = -cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb0_p.data(), 1);
        QdotHProduct_n = -cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb0_n.data(), 1);
        dotProduct_n = -cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb0_n.data(), 1);

        double Q2dotProduct_p, Q2dotHProduct_p, Q2dotProduct_n, Q2dotHProduct_n;
        Q2dotHProduct_p = -cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb2_p.data(), 1);
        Q2dotProduct_p = -cblas_ddot(dim_p * dim_p, Qorb2_p.data(), 1, Qorb2_p.data(), 1);
        Q2dotHProduct_n = -cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb2_n.data(), 1);
        Q2dotProduct_n = -cblas_ddot(dim_n * dim_n, Qorb2_n.data(), 1, Qorb2_n.data(), 1);

        double Q1Q2dotProduct_p, Q1Q2dotProduct_n;
        Q1Q2dotProduct_p = -cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb2_p.data(), 1);
        Q1Q2dotProduct_n = -cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb2_n.data(), 1);

        //----------------------------------------
        std::vector<double> A_p(number_of_Q * number_of_Q, 0);
        std::vector<double> b_p(number_of_Q, 0);
        std::vector<int> ipiv_p(number_of_Q, 0);

        A_p[0] = dotProduct_p;
        A_p[1] = Q1Q2dotProduct_p;
        A_p[2] = Q1Q2dotProduct_p;
        A_p[3] = Q2dotProduct_p;

        b_p[0] = QdotHProduct_p + deltaQ0 / (gradient_eta);
        b_p[1] = Q2dotHProduct_p + deltaQ2 / (gradient_eta);
        b_p[0] = QdotHProduct_p;
        b_p[1] = Q2dotHProduct_p;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_p.data(), number_of_Q, ipiv_p.data(), b_p.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        std::vector<double> A_n(number_of_Q * number_of_Q, 0);
        std::vector<double> b_n(number_of_Q, 0);
        std::vector<int> ipiv_n(number_of_Q, 0);

        A_n[0] = dotProduct_n;
        A_n[1] = Q1Q2dotProduct_n;
        A_n[2] = Q1Q2dotProduct_n;
        A_n[3] = Q2dotProduct_n;

        b_n[0] = QdotHProduct_n + deltaQ0 / (gradient_eta);
        b_n[1] = Q2dotHProduct_n + deltaQ2 / (gradient_eta);
        b_n[0] = QdotHProduct_n;
        b_n[1] = Q2dotHProduct_n;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_n.data(), number_of_Q, ipiv_n.data(), b_n.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        cblas_daxpy(dim_p * dim_p, -b_p[0], Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_p * dim_p, -b_p[1], Qorb2_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -b_n[0], Qorb0_n.data(), 1, Z_n.data(), 1);
        cblas_daxpy(dim_n * dim_n, -b_n[1], Qorb2_n.data(), 1, Z_n.data(), 1);
        Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());
        // Cal_Gradient_given_gradient(Z_p.data(), Z_n.data());
        //--------------------------------
        //  cblas_daxpy(dim_p * dim_p, -1. * QdotHProduct_p / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        //  cblas_daxpy(dim_n * dim_n, -1. * QdotHProduct_n / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);
        //  Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());
        //  cblas_daxpy(dim_p * dim_p, -1. * deltaQ0 / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        //  cblas_daxpy(dim_n * dim_n, -1. * deltaQ0 / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);

        A_p[0] = dotProduct_p;
        A_p[1] = Q1Q2dotProduct_p;
        A_p[2] = Q1Q2dotProduct_p;
        A_p[3] = Q2dotProduct_p;

        b_p[0] = deltaQ0;
        b_p[1] = deltaQ2;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_p.data(), number_of_Q, ipiv_p.data(), b_p.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        A_n[0] = dotProduct_n;
        A_n[1] = Q1Q2dotProduct_n;
        A_n[2] = Q1Q2dotProduct_n;
        A_n[3] = Q2dotProduct_n;

        b_n[0] = deltaQ0;
        b_n[1] = deltaQ2;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_n.data(), number_of_Q, ipiv_n.data(), b_n.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        cblas_daxpy(dim_p * dim_p, 1. * -b_p[0], Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_p * dim_p, 1. * -b_p[1], Qorb2_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, 1. * -b_n[0], Qorb0_n.data(), 1, Z_n.data(), 1);
        cblas_daxpy(dim_n * dim_n, 1. * -b_n[1], Qorb2_n.data(), 1, Z_n.data(), 1);

        // UpdateU_Thouless_pade(Z_p.data(), Z_n.data());
        UpdateU_Thouless_1st(Z_p.data(), Z_n.data());
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n

        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - density_mixing_factor), rho_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - density_mixing_factor), rho_n, 1);
        // cblas_daxpby(dim_p * dim_p, field_mixing_factor, FockTerm_p, 1, 0.0, rho_last_p, 1);
        // cblas_daxpby(dim_n * dim_n, field_mixing_factor, FockTerm_n, 1, 0.0, rho_last_n, 1);
        UpdateF(); // Update the Fock matrix
        // cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - field_mixing_factor), FockTerm_p, 1);
        // cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - field_mixing_factor), FockTerm_n, 1);

        Diagonalize();

        if (fabs(E_previous - EHF) < this->tolerance)
            break;
        E_previous = this->EHF;
        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        // if (CheckConvergence())
        //    break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();
    mkl_free(rho_last_p);
    mkl_free(rho_last_n);
    CalOnebodyOperator(Q0_p.data(), Q0_n.data(), tempQp, tempQn);
    deltaQ0 = tempQn + tempQp;
    CalOnebodyOperator(Q2_p.data(), Q2_n.data(), tempQp, tempQn);
    deltaQ2 = tempQn + tempQp;
    std::cout << "  Q0  " << Q0_expect << "   " << deltaQ0 << "    Q2 " << Q2_expect << "   " << deltaQ2 << std::endl;

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Restricted Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
}

/*
void HartreeFock::Solve_hybrid_Constraint()
{
    /// inital Q operator
    double *Q2_p, *Q0_p, *Q_2_p, *Q2_n, *Q0_n, *Q_2_n;
    Q2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q0_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q_2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q0_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q_2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);

    double Q0_expect = modelspace->GetShapeQ0();
    double Q2_expect = modelspace->GetShapeQ2();
    double deltaQ0, deltaQ2, deltaQ_2;
    double tempQp, tempQn;

    memset(Q0_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q0_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
    }
    memset(Q2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
    }
    memset(Q_2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q_2_p[ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
    }

    memset(Q0_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q0_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
    }
    memset(Q2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
    }
    memset(Q_2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q_2_n[ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
    }

    ////////////////////////////////////////////////////////////
    double density_mixing_factor = 0.2;
    double field_mixing_factor = 0.0;
    double *rho_last_p, *rho_last_n;
    rho_last_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    rho_last_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    gradient_eta = 0.00001;
    double E_previous = 1.e10;
    double ConstarinedLargeNum = 1.e5;
    double constrainedQ = 0;
    iterations = 0; // count number of iterations
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
        UpdateF();
        CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
        deltaQ0 = Q0_expect - tempQp - tempQn;
        // constrainedQ = ConstarinedLargeNum * deltaQ0 * deltaQ0;
        // std::cout << deltaQ0 << "  " << Q0_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
        // deltaQ2 = Q2_expect - tempQp - tempQn;
        //    std::cout << deltaQ2 << "  " << Q2_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        // CalcEHF(constrainedQ);
        // CalcEHF();

        if (iterations == 500)
        {
            density_mixing_factor = 0.7;
            field_mixing_factor = 0.5;
            std::cout << "Still not converged after 500 iterations. Setting density_mixing_factor => " << density_mixing_factor
                      << " field_mixing_factor => " << field_mixing_factor << std::endl;
        }
        if (iterations > 600 and iterations % 20 == 2) // if we've made it to 600, really put on the brakes with a big mixing factor, with random noise
        {
            field_mixing_factor = 1 - 0.005 * (std::rand() % 100);
            density_mixing_factor = 1 - 0.005 * (std::rand() % 100);
        }

        // E_previous = this->EHF;
        std::vector<double> Qorb0_p(dim_p * dim_p, 0);
        std::vector<double> Qorb0_n(dim_n * dim_n, 0);
        std::vector<double> Qorb2_p(dim_p * dim_p, 0);
        std::vector<double> Qorb2_n(dim_n * dim_n, 0);
        std::vector<double> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<double> Z_n(dim_n * dim_n, 0);

        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);

        // Cal_Gradient(Z_p.data(), Z_n.data());
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n.data(), 1);
        Operator_ph(Z_p.data(), Z_n.data());

        cblas_dcopy(dim_p * dim_p, Q0_p, 1, Qorb0_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q0_n, 1, Qorb0_n.data(), 1);
        // cblas_dcopy(dim_p * dim_p, Q2_p, 1, Qorb2_p.data(), 1);
        // cblas_dcopy(dim_n * dim_n, Q2_n, 1, Qorb2_n.data(), 1);

        TransferOperatorToHFbasis(Qorb0_p.data(), Qorb0_n.data());
        Operator_ph(Qorb0_p.data(), Qorb0_n.data());

        //---------------------------------------------
        std::vector<double> Fph_p(dim_p * dim_p, 0);
        std::vector<double> Fph_n(dim_n * dim_n, 0);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Fph_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Fph_n.data(), 1);
        Operator_ph(Fph_p.data(), Fph_n.data());

        double dotProduct_p, QdotHProduct_p, dotProduct_n, QdotHProduct_n, factor_p, factor_n;

        QdotHProduct_p = cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb0_p.data(), 1);
        dotProduct_p = cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb0_p.data(), 1);
        // factor_p = deltaQ0 / dotProduct;
        // factor_p = -1. * gradient_eta * QdotHProduct / (dotProduct)-deltaQ0 / (dotProduct);

        QdotHProduct_n = cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb0_n.data(), 1);
        dotProduct_n = cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb0_n.data(), 1);
        // factor_n = deltaQ0 / (dotProduct);
        // factor_n = -1. * gradient_eta * QdotHProduct / (dotProduct)-deltaQ0 / (dotProduct);

        //--------------------------------
        cblas_daxpy(dim_p * dim_p, -1. * QdotHProduct_p / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -1. * QdotHProduct_n / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);
        Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());

        cblas_daxpy(dim_p * dim_p, -deltaQ0 / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -deltaQ0 / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);

        // UpdateU_Thouless_pade(Z_p.data(), Z_n.data());
        UpdateU_Thouless_1st(Z_p.data(), Z_n.data());
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n

        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - density_mixing_factor), rho_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - density_mixing_factor), rho_n, 1);

        cblas_daxpby(dim_p * dim_p, field_mixing_factor, FockTerm_p, 1, 0.0, rho_last_p, 1);
        cblas_daxpby(dim_n * dim_n, field_mixing_factor, FockTerm_n, 1, 0.0, rho_last_n, 1);
        UpdateF(); // Update the Fock matrix
        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - field_mixing_factor), FockTerm_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - field_mixing_factor), FockTerm_n, 1);

        // TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);
        // cblas_daxpy(dim_p * dim_p, -1. * QdotHProduct_p / (dotProduct_p), Qorb0_p.data(), 1, FockTerm_p, 1);
        // cblas_daxpy(dim_n * dim_n, -1. * QdotHProduct_n / (dotProduct_n), Qorb0_n.data(), 1, FockTerm_n, 1);

        Diagonalize();
        // if (fabs(E_previous - EHF) < this->tolerance)
        //     break;
        //---------------------------------------------------------------

        // CalcEHF();
        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        if (CheckConvergence())
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();
    mkl_free(rho_last_p);
    mkl_free(rho_last_n);
    CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
    deltaQ0 = tempQn + tempQp;
    CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
    deltaQ2 = tempQn + tempQp;
    std::cout << "  Q0  " << Q0_expect << "   " << deltaQ0 << "    Q2 " << Q2_expect << "   " << deltaQ2 << std::endl;

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Restricted Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
    mkl_free(Q2_p);
    mkl_free(Q0_p);
    mkl_free(Q_2_p);
    mkl_free(Q2_n);
    mkl_free(Q0_n);
    mkl_free(Q_2_n);
}

void HartreeFock::Solve_hybrid_Constraint()
{
    /// inital Q operator
    std::vector<double> Q2_p(dim_p * dim_p, 0);
    std::vector<double> Q0_p(dim_p * dim_p, 0);
    std::vector<double> Q2_n(dim_n * dim_n, 0);
    std::vector<double> Q0_n(dim_n * dim_n, 0);

    double Q0_expect = modelspace->GetShapeQ0();
    double Q2_expect = modelspace->GetShapeQ2();
    double deltaQ0, deltaQ2, deltaQ_2;
    double tempQp, tempQn;

    memset(Q0_p.data(), 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q0_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
    }
    memset(Q2_p.data(), 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
    }
    // memset(Q_2_p.data(), 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
    }

    memset(Q0_n.data(), 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q0_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
    }
    memset(Q2_n.data(), 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
    }
    // memset(Q_2_n.data(), 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
    }

    ////////////////////////////////////////////////////////////
    double density_mixing_factor = 0.2;
    double field_mixing_factor = 0.;
    double *rho_last_p, *rho_last_n;
    rho_last_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    rho_last_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    gradient_eta = 0.000001;
    double E_previous = 1.e10;
    double ConstarinedLargeNum = 1.e5;
    double constrainedQ = 0;
    iterations = 0; // count number of iterations
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
        UpdateF();
        CalOnebodyOperator(Q0_p.data(), Q0_n.data(), tempQp, tempQn);
        deltaQ0 = Q0_expect - tempQp - tempQn;
        // constrainedQ = ConstarinedLargeNum * deltaQ0 * deltaQ0;
        std::cout << deltaQ0 << "  " << Q0_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        CalOnebodyOperator(Q2_p.data(), Q2_n.data(), tempQp, tempQn);
        deltaQ2 = Q2_expect - tempQp - tempQn;
        std::cout << deltaQ2 << "  " << Q2_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        // CalcEHF(constrainedQ);
        // CalcEHF();
        //-------------------------------------------
        if (iterations == 500)
        {
            density_mixing_factor = 0.7;
            gradient_eta *= 0.1;
            // field_mixing_factor = 0.5;
            std::cout << "Still not converged after 500 iterations. Setting density_mixing_factor => " << density_mixing_factor
                      << " field_mixing_factor => " << field_mixing_factor << std::endl;
        }
        if (iterations > 600 and iterations % 20 == 2) // if we've made it to 600, really put on the brakes with a big mixing factor, with random noise
        {
            // field_mixing_factor = 1 - 0.005 * (std::rand() % 100);
            density_mixing_factor = 1 - 0.005 * (std::rand() % 100);
        }

        // E_previous = this->EHF;
        std::vector<double> Qorb0_p(dim_p * dim_p, 0);
        std::vector<double> Qorb0_n(dim_n * dim_n, 0);
        std::vector<double> Qorb2_p(dim_p * dim_p, 0);
        std::vector<double> Qorb2_n(dim_n * dim_n, 0);
        std::vector<double> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<double> Z_n(dim_n * dim_n, 0);

        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n.data(), 1);
        Operator_ph(Z_p.data(), Z_n.data());

        cblas_dcopy(dim_p * dim_p, Q0_p.data(), 1, Qorb0_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q0_n.data(), 1, Qorb0_n.data(), 1);
        cblas_dcopy(dim_p * dim_p, Q2_p.data(), 1, Qorb2_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q2_n.data(), 1, Qorb2_n.data(), 1);

        TransferOperatorToHFbasis(Qorb0_p.data(), Qorb0_n.data());
        TransferOperatorToHFbasis(Qorb2_p.data(), Qorb2_n.data());
        Operator_ph(Qorb0_p.data(), Qorb0_n.data());
        Operator_ph(Qorb2_p.data(), Qorb2_n.data());

        //---------------------------------------------
        std::vector<double> Fph_p(dim_p * dim_p, 0);
        std::vector<double> Fph_n(dim_n * dim_n, 0);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Fph_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Fph_n.data(), 1);
        Operator_ph(Fph_p.data(), Fph_n.data());

        double dotProduct_p, QdotHProduct_p, dotProduct_n, QdotHProduct_n;
        QdotHProduct_p = cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb0_p.data(), 1);
        dotProduct_p = cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb0_p.data(), 1);
        QdotHProduct_n = cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb0_n.data(), 1);
        dotProduct_n = cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb0_n.data(), 1);

        double Q2dotProduct_p, Q2dotHProduct_p, Q2dotProduct_n, Q2dotHProduct_n;
        Q2dotHProduct_p = cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb2_p.data(), 1);
        Q2dotProduct_p = cblas_ddot(dim_p * dim_p, Qorb2_p.data(), 1, Qorb2_p.data(), 1);
        Q2dotHProduct_n = cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb2_n.data(), 1);
        Q2dotProduct_n = cblas_ddot(dim_n * dim_n, Qorb2_n.data(), 1, Qorb2_n.data(), 1);

        double Q1Q2dotProduct_p, Q1Q2dotProduct_n;
        Q1Q2dotProduct_p = cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb2_p.data(), 1);
        Q1Q2dotProduct_n = cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb2_n.data(), 1);

        double factor_p, factor_n;
        // Q0
        if (fabs(Q1Q2dotProduct_p) > 1.e-3)
        {
            factor_p = (QdotHProduct_p) / Q1Q2dotProduct_p - (Q2dotHProduct_p) / Q2dotProduct_p;
        }
        else
            factor_p = -(Q2dotHProduct_p) / Q2dotProduct_p;
        factor_p /= (dotProduct_p - Q1Q2dotProduct_p);
        if (fabs(Q1Q2dotProduct_n) > 1.e-3)
        {

            factor_n = (QdotHProduct_n) / Q1Q2dotProduct_n - (Q2dotHProduct_n) / Q2dotProduct_n;
        }
        else
            factor_n = -(Q2dotHProduct_n - deltaQ2 / gradient_eta) / Q2dotProduct_n;
        factor_n /= (dotProduct_n - Q1Q2dotProduct_n);
        cblas_daxpy(dim_p * dim_p, -factor_p, Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -factor_n, Qorb0_n.data(), 1, Z_n.data(), 1);

        // Q2
        if (fabs(Q1Q2dotProduct_p) > 1.e-3)
        {
            factor_p = (QdotHProduct_p) / dotProduct_p - (Q2dotHProduct_p) / Q1Q2dotProduct_p;
        }
        else
            factor_p = (QdotHProduct_p) / dotProduct_p;
        factor_p /= (Q1Q2dotProduct_p - Q2dotProduct_p);

        if (fabs(Q1Q2dotProduct_n) > 1.e-3)
        {
            factor_n = (QdotHProduct_n) / dotProduct_n - (Q2dotHProduct_n) / Q1Q2dotProduct_n;
        }
        else
            factor_n = (QdotHProduct_n) / dotProduct_n;
        factor_n /= (Q1Q2dotProduct_p - Q2dotProduct_n);
        // std::cout << factor_p << "  " << factor_n << "   " << Q2dotProduct_p << "  " << Q2dotHProduct_p << "  " << Q1Q2dotProduct_p << std::endl;
        // std::cout << QdotHProduct_p << "  " << deltaQ0 << "  " << gradient_eta << "  " << dotProduct_p << "  " << Q2dotHProduct_p << "  " << deltaQ2 << "  " << gradient_eta << "  " << Q1Q2dotProduct_p << std::endl;
        cblas_daxpy(dim_p * dim_p, -factor_p, Qorb2_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -factor_n, Qorb2_n.data(), 1, Z_n.data(), 1);
        Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());

        //--------------------------------
        // cblas_daxpy(dim_p * dim_p, -1. * QdotHProduct_p / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        // cblas_daxpy(dim_n * dim_n, -1. * QdotHProduct_n / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);
        // Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());
        // cblas_daxpy(dim_p * dim_p, -1. * deltaQ0 / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        // cblas_daxpy(dim_n * dim_n, -1. * deltaQ0 / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);

        factor_p = -1. * deltaQ0 / (dotProduct_p);
        if (fabs(Q1Q2dotProduct_p) > 1.e-3)
            factor_p += -1. * deltaQ2 / (Q1Q2dotProduct_p);
        cblas_daxpy(dim_p * dim_p, factor_p, Qorb0_p.data(), 1, Z_p.data(), 1);

        factor_p = -1. * deltaQ2 / (Q2dotProduct_p);
        if (fabs(Q1Q2dotProduct_p) > 1.e-3)
            factor_p += -1. * deltaQ0 / (Q1Q2dotProduct_p);
        cblas_daxpy(dim_p * dim_p, factor_p, Qorb0_p.data(), 1, Z_p.data(), 1);


        factor_n = -1. * deltaQ0 / (dotProduct_n);
        if (fabs(Q1Q2dotProduct_n) > 1.e-3)
            factor_n += -1. * deltaQ2 / (Q1Q2dotProduct_n);
        cblas_daxpy(dim_n * dim_n, factor_n, Qorb0_n.data(), 1, Z_n.data(), 1);

        factor_n = -1. * deltaQ2 / (Q2dotProduct_n);
        if (fabs(Q1Q2dotProduct_n) > 1.e-3)
            factor_n += -1. * deltaQ0 / (Q1Q2dotProduct_n);
        cblas_daxpy(dim_n * dim_n, factor_n, Qorb0_n.data(), 1, Z_n.data(), 1);

        // UpdateU_Thouless_pade(Z_p.data(), Z_n.data());
        UpdateU_Thouless_1st(Z_p.data(), Z_n.data());
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n

        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - density_mixing_factor), rho_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - density_mixing_factor), rho_n, 1);

        cblas_daxpby(dim_p * dim_p, field_mixing_factor, FockTerm_p, 1, 0.0, rho_last_p, 1);
        cblas_daxpby(dim_n * dim_n, field_mixing_factor, FockTerm_n, 1, 0.0, rho_last_n, 1);
        UpdateF(); // Update the Fock matrix
        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - field_mixing_factor), FockTerm_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - field_mixing_factor), FockTerm_n, 1);

        Diagonalize();
        // if (fabs(E_previous - EHF) < this->tolerance)
        //     break;
        //---------------------------------------------------------------

        // CalcEHF();
        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        if (CheckConvergence())
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();
    mkl_free(rho_last_p);
    mkl_free(rho_last_n);
    CalOnebodyOperator(Q0_p.data(), Q0_n.data(), tempQp, tempQn);
    deltaQ0 = tempQn + tempQp;
    CalOnebodyOperator(Q2_p.data(), Q2_n.data(), tempQp, tempQn);
    deltaQ2 = tempQn + tempQp;
    std::cout << "  Q0  " << Q0_expect << "   " << deltaQ0 << "    Q2 " << Q2_expect << "   " << deltaQ2 << std::endl;

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Restricted Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
}


*/
//*********************************************************************
/// Diagonalize and update the Fock matrix until convergence.
void HartreeFock::Solve_diag()
{
    iterations = 0; // count number of iterations
    double density_mixing_factor = 0.2;
    double field_mixing_factor = 0.0;
    UpdateDensityMatrix();
    UpdateF();
    double *rho_last_p, *rho_last_n;
    rho_last_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    rho_last_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    // std::cout << "HF start  iterations. " << std::endl;
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        Diagonalize(); // Diagonalize the Fock matrix
        //-------------------------------------------
        if (iterations == 500)
        {
            density_mixing_factor = 0.7;
            field_mixing_factor = 0.5;
            std::cout << "Still not converged after 500 iterations. Setting density_mixing_factor => " << density_mixing_factor
                      << " field_mixing_factor => " << field_mixing_factor << std::endl;
        }
        if (iterations > 600 and iterations % 20 == 2) // if we've made it to 600, really put on the brakes with a big mixing factor, with random noise
        {
            field_mixing_factor = 1 - 0.005 * (std::rand() % 100);
            density_mixing_factor = 1 - 0.005 * (std::rand() % 100);
        }

        // if (DIIS_error_mats_p.size() > 0)
        //     std::cout << DIIS_error_mats_p.size() << "    " << frobenius_norm(DIIS_error_mats_n.back()) << "   " << frobenius_norm(DIIS_error_mats_p.back()) << std::endl;
        //-------------------------------------------
        if (iterations > 100 and (DIIS_error_mats_p.size() < 1 or frobenius_norm(DIIS_error_mats_n.back()) > 0.01 or frobenius_norm(DIIS_error_mats_p.back()) > 0.01))
        // if (iterations > 100 and iterations % 20 == 1)
        {
            std::cout << "Still not converged after " << iterations << " iterations. Switching to DIIS." << std::endl;
            UpdateDensityMatrix_DIIS();
            if (frobenius_norm(DIIS_error_mats_p.back()) < 0.01 and frobenius_norm(DIIS_error_mats_n.back()) < 0.01)
            {
                std::cout << "DIIS error matrix below 0.01, switching back to simpler SCF algorithm." << std::endl;
            }
        }
        else
        {
            cblas_daxpby(dim_p * dim_p, density_mixing_factor, rho_p, 1, 0.0, rho_last_p, 1);
            cblas_daxpby(dim_n * dim_n, density_mixing_factor, rho_n, 1, 0.0, rho_last_n, 1);
            UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
            cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - density_mixing_factor), rho_p, 1);
            cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - density_mixing_factor), rho_n, 1);
        }

        cblas_daxpby(dim_p * dim_p, field_mixing_factor, FockTerm_p, 1, 0.0, rho_last_p, 1);
        cblas_daxpby(dim_n * dim_n, field_mixing_factor, FockTerm_n, 1, 0.0, rho_last_n, 1);
        UpdateF(); // Update the Fock matrix
        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - field_mixing_factor), FockTerm_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - field_mixing_factor), FockTerm_n, 1);

        CalcEHF();
        // save hole parameters
        // string filename = "Output/HF_para_" + std::to_string(iterations) + ".dat";
        // this->SaveHoleParameters(filename);

        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        if (CheckConvergence())
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();
    mkl_free(rho_last_p);
    mkl_free(rho_last_n);

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
}

void HartreeFock::Solve_noCore()
{
    iterations = 0; // count number of iterations
    double density_mixing_factor = 0.2;
    double field_mixing_factor = 0.0;
    UpdateDensityMatrix();
    UpdateF_noCore();
    double *rho_last_p, *rho_last_n;
    rho_last_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    rho_last_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    // std::cout << "HF start  iterations. " << std::endl;
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        Diagonalize(); // Diagonalize the Fock matrix

        //-------------------------------------------
        if (iterations == 500)
        {
            density_mixing_factor = 0.7;
            field_mixing_factor = 0.5;
            std::cout << "Still not converged after 500 iterations. Setting density_mixing_factor => " << density_mixing_factor
                      << " field_mixing_factor => " << field_mixing_factor << std::endl;
        }
        if (iterations > 600 and iterations % 20 == 2) // if we've made it to 600, really put on the brakes with a big mixing factor, with random noise
        {
            field_mixing_factor = 1 - 0.005 * (std::rand() % 100);
            density_mixing_factor = 1 - 0.005 * (std::rand() % 100);
        }

        // if (DIIS_error_mats_p.size() > 0)
        //     std::cout << DIIS_error_mats_p.size() << "    " << frobenius_norm(DIIS_error_mats_n.back()) << "   " << frobenius_norm(DIIS_error_mats_p.back()) << std::endl;
        //-------------------------------------------
        if (iterations > 100 and (DIIS_error_mats_p.size() < 1 or frobenius_norm(DIIS_error_mats_n.back()) > 0.01 or frobenius_norm(DIIS_error_mats_p.back()) > 0.01))
        // if (iterations > 100 and iterations % 20 == 1)
        {
            std::cout << "Still not converged after " << iterations << " iterations. Switching to DIIS." << std::endl;
            UpdateDensityMatrix_DIIS();
            if (frobenius_norm(DIIS_error_mats_p.back()) < 0.01 and frobenius_norm(DIIS_error_mats_n.back()) < 0.01)
            {
                std::cout << "DIIS error matrix below 0.01, switching back to simpler SCF algorithm." << std::endl;
            }
        }
        else
        {
            cblas_daxpby(dim_p * dim_p, density_mixing_factor, rho_p, 1, 0.0, rho_last_p, 1);
            cblas_daxpby(dim_n * dim_n, density_mixing_factor, rho_n, 1, 0.0, rho_last_n, 1);
            UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
            cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - density_mixing_factor), rho_p, 1);
            cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - density_mixing_factor), rho_n, 1);
        }

        cblas_daxpby(dim_p * dim_p, field_mixing_factor, FockTerm_p, 1, 0.0, rho_last_p, 1);
        cblas_daxpby(dim_n * dim_n, field_mixing_factor, FockTerm_n, 1, 0.0, rho_last_n, 1);
        UpdateF_noCore(); // Update the Fock matrix
        cblas_daxpby(dim_p * dim_p, 1., rho_last_p, 1, (1.0 - field_mixing_factor), FockTerm_p, 1);
        cblas_daxpby(dim_n * dim_n, 1., rho_last_n, 1, (1.0 - field_mixing_factor), FockTerm_n, 1);

        // CalcEHF();
        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        if (CheckConvergence())
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF_noCore();      // Update the Fock matrix
    CalcEHF_noCore();
    mkl_free(rho_last_p);
    mkl_free(rho_last_n);

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
}

//*********************************************************************
// Gradient method
// First order gradient are considered
void HartreeFock::Solve_gradient()
{
    double E_previous = 1.e10;
    iterations = 0; // count number of iterations
    UpdateDensityMatrix();
    UpdateF();
    CalcEHF();

    // std::cout << "HF start  iterations. " << std::endl;
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        E_previous = this->EHF;
        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);
        std::vector<double> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<double> Z_n(dim_n * dim_n, 0);

        // Cal_Gradient(Z_p.data(), Z_n.data());
        // Cal_Gradient_preconditioned(Z_p.data(), Z_n.data());
        Cal_Gradient_preconditioned_SRG(Z_p.data(), Z_n.data());

        // UpdateU_Thouless_pade(Z_p.data(), Z_n.data());
        UpdateU_Thouless_1st(Z_p.data(), Z_n.data());

        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
        UpdateF();             // Update the Fock matrix
        // Diagonalize();
        CalcEHF();
        // std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        //  if (CheckConvergence())
        if (fabs(E_previous - EHF) < this->tolerance)
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
    return;
}

void HartreeFock::Solve_gradient_Constraint()
{
    /// inital Q operator
    double *Q2_p, *Q0_p, *Q_2_p, *Q2_n, *Q0_n, *Q_2_n;
    Q2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q0_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q_2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q0_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q_2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);

    double Q0_expect = modelspace->GetShapeQ0();
    double Q2_expect = modelspace->GetShapeQ2();
    double deltaQ0, deltaQ2, deltaQ_2;
    double tempQp, tempQn, constrainedQ;

    memset(Q0_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q0_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
    }
    memset(Q2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
    }
    memset(Q_2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
    }

    memset(Q0_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q0_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
    }
    memset(Q2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
    }
    memset(Q_2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
    }

    ////////////////////////////////////////////////////////////
    double E_previous = 1.e10;
    int number_of_Q = 2;
    iterations = 0;        // count number of iterations
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    double ConstarinedLargeNum = 1000.;
    UpdateF();
    CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
    deltaQ0 = Q0_expect - tempQp - tempQn;
    constrainedQ = ConstarinedLargeNum * deltaQ0 * deltaQ0;
    // std::cout << deltaQ0 << "  " << Q0_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
    CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
    deltaQ2 = Q2_expect - tempQp - tempQn;
    constrainedQ += ConstarinedLargeNum * deltaQ2 * deltaQ2;
    // std::cout << deltaQ2 << "  " << Q2_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
    CalcEHF(constrainedQ);
    // CalcEHF();
    tolerance = 1.e-4;
    gradient_eta = 0.05;
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        E_previous = this->EHF;

        std::vector<double> Qorb0_p(dim_p * dim_p, 0);
        std::vector<double> Qorb0_n(dim_n * dim_n, 0);
        std::vector<double> Qorb2_p(dim_p * dim_p, 0);
        std::vector<double> Qorb2_n(dim_n * dim_n, 0);
        std::vector<double> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<double> Z_n(dim_n * dim_n, 0);

        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n.data(), 1);
        Operator_ph(Z_p.data(), Z_n.data());

        cblas_dcopy(dim_p * dim_p, Q0_p, 1, Qorb0_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q0_n, 1, Qorb0_n.data(), 1);
        cblas_dcopy(dim_p * dim_p, Q2_p, 1, Qorb2_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q2_n, 1, Qorb2_n.data(), 1);

        TransferOperatorToHFbasis(Qorb0_p.data(), Qorb0_n.data());
        TransferOperatorToHFbasis(Qorb2_p.data(), Qorb2_n.data());
        Operator_ph(Qorb0_p.data(), Qorb0_n.data());
        Operator_ph(Qorb2_p.data(), Qorb2_n.data());

        //---------------------------------------------
        std::vector<double> Fph_p(dim_p * dim_p, 0);
        std::vector<double> Fph_n(dim_n * dim_n, 0);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Fph_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Fph_n.data(), 1);
        Operator_ph(Fph_p.data(), Fph_n.data());

        double dotProduct_p, QdotHProduct_p, dotProduct_n, QdotHProduct_n;
        QdotHProduct_p = -cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb0_p.data(), 1);
        dotProduct_p = -cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb0_p.data(), 1);
        QdotHProduct_n = -cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb0_n.data(), 1);
        dotProduct_n = -cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb0_n.data(), 1);

        double Q2dotProduct_p, Q2dotHProduct_p, Q2dotProduct_n, Q2dotHProduct_n;
        Q2dotHProduct_p = -cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb2_p.data(), 1);
        Q2dotProduct_p = -cblas_ddot(dim_p * dim_p, Qorb2_p.data(), 1, Qorb2_p.data(), 1);
        Q2dotHProduct_n = -cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb2_n.data(), 1);
        Q2dotProduct_n = -cblas_ddot(dim_n * dim_n, Qorb2_n.data(), 1, Qorb2_n.data(), 1);

        double Q1Q2dotProduct_p, Q1Q2dotProduct_n;
        Q1Q2dotProduct_p = -cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb2_p.data(), 1);
        Q1Q2dotProduct_n = -cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb2_n.data(), 1);

        //----------------------------------------
        std::vector<double> A_p(number_of_Q * number_of_Q, 0);
        std::vector<double> b_p(number_of_Q, 0);
        std::vector<int> ipiv_p(number_of_Q, 0);

        A_p[0] = dotProduct_p;
        A_p[1] = Q1Q2dotProduct_p;
        A_p[2] = Q1Q2dotProduct_p;
        A_p[3] = Q2dotProduct_p;

        b_p[0] = QdotHProduct_p + deltaQ0 / (gradient_eta);
        b_p[1] = Q2dotHProduct_p + deltaQ2 / (gradient_eta);
        b_p[0] = QdotHProduct_p;
        b_p[1] = Q2dotHProduct_p;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_p.data(), number_of_Q, ipiv_p.data(), b_p.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        std::vector<double> A_n(number_of_Q * number_of_Q, 0);
        std::vector<double> b_n(number_of_Q, 0);
        std::vector<int> ipiv_n(number_of_Q, 0);

        A_n[0] = dotProduct_n;
        A_n[1] = Q1Q2dotProduct_n;
        A_n[2] = Q1Q2dotProduct_n;
        A_n[3] = Q2dotProduct_n;

        b_n[0] = QdotHProduct_n + deltaQ0 / (gradient_eta);
        b_n[1] = Q2dotHProduct_n + deltaQ2 / (gradient_eta);
        b_n[0] = QdotHProduct_n;
        b_n[1] = Q2dotHProduct_n;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_n.data(), number_of_Q, ipiv_n.data(), b_n.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        cblas_daxpy(dim_p * dim_p, -b_p[0], Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_p * dim_p, -b_p[1], Qorb2_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -b_n[0], Qorb0_n.data(), 1, Z_n.data(), 1);
        cblas_daxpy(dim_n * dim_n, -b_n[1], Qorb2_n.data(), 1, Z_n.data(), 1);
        Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());
        // Cal_Gradient_given_gradient(Z_p.data(), Z_n.data());
        //--------------------------------
        //  cblas_daxpy(dim_p * dim_p, -1. * QdotHProduct_p / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        //  cblas_daxpy(dim_n * dim_n, -1. * QdotHProduct_n / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);
        //  Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());
        //  cblas_daxpy(dim_p * dim_p, -1. * deltaQ0 / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        //  cblas_daxpy(dim_n * dim_n, -1. * deltaQ0 / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);

        A_p[0] = dotProduct_p;
        A_p[1] = Q1Q2dotProduct_p;
        A_p[2] = Q1Q2dotProduct_p;
        A_p[3] = Q2dotProduct_p;

        b_p[0] = deltaQ0;
        b_p[1] = deltaQ2;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_p.data(), number_of_Q, ipiv_p.data(), b_p.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        A_n[0] = dotProduct_n;
        A_n[1] = Q1Q2dotProduct_n;
        A_n[2] = Q1Q2dotProduct_n;
        A_n[3] = Q2dotProduct_n;

        b_n[0] = deltaQ0;
        b_n[1] = deltaQ2;

        if (LAPACKE_dgesv(LAPACK_ROW_MAJOR, number_of_Q, 1, A_n.data(), number_of_Q, ipiv_n.data(), b_n.data(), 1) != 0)
        {
            std::cout << "  Linear equation error!" << std::endl;
            exit(0);
        }

        cblas_daxpy(dim_p * dim_p, 1. * -b_p[0], Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_p * dim_p, 1. * -b_p[1], Qorb2_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, 1. * -b_n[0], Qorb0_n.data(), 1, Z_n.data(), 1);
        cblas_daxpy(dim_n * dim_n, 1. * -b_n[1], Qorb2_n.data(), 1, Z_n.data(), 1);

        UpdateU_Thouless_pade(Z_p.data(), Z_n.data());
        // UpdateU_Thouless_1st(Z_p.data(), Z_n.data());
        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n

        UpdateF(); // Update the Fock matrix

        CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
        deltaQ0 = tempQp + tempQn - Q0_expect;
        // std::cout << deltaQ0 << "  " << Q0_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
        deltaQ2 = tempQp + tempQn - Q2_expect;
        // std::cout << deltaQ2 << "  " << Q2_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;

        constrainedQ = ConstarinedLargeNum * deltaQ0 * deltaQ0;
        constrainedQ += ConstarinedLargeNum * deltaQ2 * deltaQ2;
        CalcEHF(constrainedQ);
        // CalcEHF();
        //   std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        //   if (CheckConvergence())

        if (fabs(E_previous - EHF) < this->tolerance)
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();

    CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
    deltaQ0 = tempQn + tempQp;
    CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
    deltaQ2 = tempQn + tempQp;
    std::cout << "  Q0  " << Q0_expect << "   " << deltaQ0 << "    Q2 " << Q2_expect << "   " << deltaQ2 << std::endl;

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Restricted Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
    mkl_free(Q2_p);
    mkl_free(Q0_p);
    mkl_free(Q_2_p);
    mkl_free(Q2_n);
    mkl_free(Q0_n);
    mkl_free(Q_2_n);
}

/*
void HartreeFock::Solve_gradient_Constraint()
{
    /// inital Q operator
    double *Q2_p, *Q0_p, *Q_2_p, *Q2_n, *Q0_n, *Q_2_n;
    Q2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q0_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q_2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q0_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q_2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);

    double Q0_expect = modelspace->GetShapeQ0();
    double Q2_expect = modelspace->GetShapeQ2();
    double deltaQ0, deltaQ2, deltaQ_2;
    double tempQp, tempQn, constrainedQ;

    memset(Q0_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q0_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
    }
    memset(Q2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
    }
    memset(Q_2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
    }

    memset(Q0_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q0_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
    }
    memset(Q2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
    }
    memset(Q_2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
    }

    ////////////////////////////////////////////////////////////
    double E_previous = 1.e10;
    double last_shift = 1000.;
    iterations = 0;        // count number of iterations
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();
    CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
    deltaQ0 = Q0_expect - tempQp - tempQn;
    // std::cout << deltaQ0 << "  " << Q0_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
    CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
    deltaQ2 = Q2_expect - tempQp - tempQn;
    // constrainedQ += ConstarinedLargeNum * deltaQ2 * deltaQ2;
    // std::cout << deltaQ2 << "  " << Q2_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
    constrainedQ = 0;
    // CalcEHF(constrainedQ);
    CalcEHF();
    tolerance = 1.e-5;
    gradient_eta = 0.0001;
    for (iterations = 0; iterations < maxiter; ++iterations)
    {
        E_previous = this->EHF;
        std::vector<double> Qorb0_p(dim_p * dim_p, 0);
        std::vector<double> Qorb0_n(dim_n * dim_n, 0);
        std::vector<double> Qorb2_p(dim_p * dim_p, 0);
        std::vector<double> Qorb2_n(dim_n * dim_n, 0);
        std::vector<double> Z_p(dim_p * dim_p, 0); // gradient matrix
        std::vector<double> Z_n(dim_n * dim_n, 0);

        TransferOperatorToHFbasis(FockTerm_p, FockTerm_n);

        // Cal_Gradient(Z_p.data(), Z_n.data());
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n.data(), 1);
        Operator_ph(Z_p.data(), Z_n.data());

        cblas_dcopy(dim_p * dim_p, Q0_p, 1, Qorb0_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, Q0_n, 1, Qorb0_n.data(), 1);
        // cblas_dcopy(dim_p * dim_p, Q2_p, 1, Qorb2_p.data(), 1);
        // cblas_dcopy(dim_n * dim_n, Q2_n, 1, Qorb2_n.data(), 1);

        TransferOperatorToHFbasis(Qorb0_p.data(), Qorb0_n.data());
        Operator_ph(Qorb0_p.data(), Qorb0_n.data());

        //---------------------------------------------
        std::vector<double> Fph_p(dim_p * dim_p, 0);
        std::vector<double> Fph_n(dim_n * dim_n, 0);
        cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Fph_p.data(), 1);
        cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Fph_n.data(), 1);
        Operator_ph(Fph_p.data(), Fph_n.data());

        double dotProduct_p, QdotHProduct_p, dotProduct_n, QdotHProduct_n, factor_p, factor_n;

        QdotHProduct_p = cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Qorb0_p.data(), 1);
        dotProduct_p = cblas_ddot(dim_p * dim_p, Qorb0_p.data(), 1, Qorb0_p.data(), 1);
        // factor_p = deltaQ0 / dotProduct;
        // factor_p = -1. * gradient_eta * QdotHProduct / (dotProduct)-deltaQ0 / (dotProduct);

        QdotHProduct_n = cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Qorb0_n.data(), 1);
        dotProduct_n = cblas_ddot(dim_n * dim_n, Qorb0_n.data(), 1, Qorb0_n.data(), 1);
        // factor_n = deltaQ0 / (dotProduct);
        // factor_n = -1. * gradient_eta * QdotHProduct / (dotProduct)-deltaQ0 / (dotProduct);

        //--------------------------------
        cblas_daxpy(dim_p * dim_p, -1. * QdotHProduct_p / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -1. * QdotHProduct_n / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);
        Cal_Gradient_preconditioned_given_gradient(Z_p.data(), Z_n.data());

        cblas_daxpy(dim_p * dim_p, -deltaQ0 / (dotProduct_p), Qorb0_p.data(), 1, Z_p.data(), 1);
        cblas_daxpy(dim_n * dim_n, -deltaQ0 / (dotProduct_n), Qorb0_n.data(), 1, Z_n.data(), 1);

        UpdateU_Thouless_pade(Z_p.data(), Z_n.data());
        // UpdateU_Thouless_1st(Z_p.data(), Z_n.data());

        UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
        UpdateF();

        CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
        deltaQ0 = Q0_expect - tempQp - tempQn;
        // std::cout << deltaQ0 << "  " << Q0_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
        // deltaQ2 = Q2_expect - tempQp - tempQn;
        //    std::cout << deltaQ2 << "  " << Q2_expect << "   " << tempQn + tempQp << "   " << tempQp << "   " << tempQn << std::endl;
        // CalcEHF(constrainedQ);
        CalcEHF();

        // CalcEHF();
        //   std::cout << "   " << std::setw(5) << iterations << "   " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << EHF << std::endl;
        //   if (CheckConvergence())

        if (fabs(E_previous - EHF) < this->tolerance)
            break;
    }
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();

    CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
    deltaQ0 = tempQn + tempQp;
    CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
    deltaQ2 = tempQn + tempQp;
    std::cout << "  Q0  " << Q0_expect << "   " << deltaQ0 << "    Q2 " << Q2_expect << "   " << deltaQ2 << std::endl;

    std::cout << std::setw(15) << std::setprecision(10);
    if (iterations < maxiter)
    {
        std::cout << "  HF converged after " << iterations << " iterations. " << std::endl;
    }
    else
    {
        std::cout << "\033[31m!!!! Warning: Restricted Hartree-Fock calculation didn't converge after " << iterations << " iterations.\033[0m" << std::endl;
        std::cout << std::endl;
    }
    PrintEHF();
    mkl_free(Q2_p);
    mkl_free(Q0_p);
    mkl_free(Q_2_p);
    mkl_free(Q2_n);
    mkl_free(Q0_n);
    mkl_free(Q_2_n);
}
*/

//*********************************************************************
// Random transformation matrix U
void HartreeFock::RandomTransformationU(int RandomSeed)
{
    // std::cout << RandomSeed << std::endl;
    srand(RandomSeed); // seed the random number generator with current time
    for (size_t i = 0; i < dim_p * dim_p; i++)
    {
        U_p[i] = (rand() % 1000) / 1000.;
    }
    for (size_t i = 0; i < dim_n * dim_n; i++)
    {
        U_n[i] = (rand() % 1000) / 1000.;
    }
    std::vector<double> Energy_vec_p(dim_p, 0);
    std::vector<double> Energy_vec_n(dim_n, 0);
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim_p, U_p, dim_p, Energy_vec_p.data()) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in proton random U!\n");
        exit(0);
    }
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim_n, U_n, dim_n, Energy_vec_n.data()) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in Neutron random U!\n");
        exit(0);
    }
    UpdateDensityMatrix();
    UpdateF();
    Diagonalize();
    return;
}

/// one-body density matrix
/// \f$ <i|\rho|j> = \sum\limits_{\beta} C_{ i beta} C_{ beta j } \f$
/// where \f$n_{\beta} \f$ ensures that beta runs over HF orbits in
/// the core (i.e. below the fermi surface)
//*********************************************************************
void HartreeFock::UpdateDensityMatrix()
{
    double *tmp_p = (double *)mkl_malloc((N_p * dim_p) * sizeof(double), 64);
    double *tmp_p_copy = (double *)mkl_malloc((N_p * dim_p) * sizeof(double), 64);
    double *tmp_n = (double *)mkl_malloc((N_n * dim_n) * sizeof(double), 64);
    double *tmp_n_copy = (double *)mkl_malloc((N_n * dim_n) * sizeof(double), 64);
    for (size_t i = 0; i < N_p; i++)
    {
        cblas_dcopy(dim_p, U_p + holeorbs_p[i], dim_p, tmp_p + i, N_p);
    }
    for (size_t i = 0; i < N_n; i++)
    {
        cblas_dcopy(dim_n, U_n + holeorbs_n[i], dim_n, tmp_n + i, N_n);
    }

    cblas_dcopy(dim_p * N_p, tmp_p, 1, tmp_p_copy, 1);
    cblas_dcopy(dim_n * N_n, tmp_n, 1, tmp_n_copy, 1);
    if (N_p > 0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim_p, dim_p, N_p, 1., tmp_p, N_p, tmp_p_copy, N_p, 0, rho_p, dim_p);
    if (N_n > 0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim_n, dim_n, N_n, 1., tmp_n, N_n, tmp_n_copy, N_n, 0, rho_n, dim_n);

    mkl_free(tmp_p);
    mkl_free(tmp_n);
    mkl_free(tmp_p_copy);
    mkl_free(tmp_n_copy);
}

// indicate orbits
void HartreeFock::UpdateDensityMatrix(const std::vector<int> proton_vec, const std::vector<int> neutron_vec)
{
    if (proton_vec.size() != N_p or neutron_vec.size() != N_n)
    {
        std::cout << "  Number of particle error!  UpdateDensityMatrix() " << std::endl;
        exit(0);
    }
    double *tmp_p = (double *)mkl_malloc((N_p * dim_p) * sizeof(double), 64);
    double *tmp_p_copy = (double *)mkl_malloc((N_p * dim_p) * sizeof(double), 64);
    double *tmp_n = (double *)mkl_malloc((N_n * dim_n) * sizeof(double), 64);
    double *tmp_n_copy = (double *)mkl_malloc((N_n * dim_n) * sizeof(double), 64);
    for (size_t i = 0; i < N_p; i++)
    {
        cblas_dcopy(dim_p, U_p + proton_vec[i], dim_p, tmp_p + i, N_p);
    }
    for (size_t i = 0; i < N_n; i++)
    {
        cblas_dcopy(dim_n, U_n + neutron_vec[i], dim_n, tmp_n + i, N_n);
    }

    cblas_dcopy(dim_p * N_p, tmp_p, 1, tmp_p_copy, 1);
    cblas_dcopy(dim_n * N_n, tmp_n, 1, tmp_n_copy, 1);
    if (N_p > 0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim_p, dim_p, N_p, 1., tmp_p, N_p, tmp_p_copy, N_p, 0, rho_p, dim_p);
    if (N_n > 0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim_n, dim_n, N_n, 1., tmp_n, N_n, tmp_n_copy, N_n, 0, rho_n, dim_n);

    mkl_free(tmp_p);
    mkl_free(tmp_n);
    mkl_free(tmp_p_copy);
    mkl_free(tmp_n_copy);
}

//*********************************************************************
// The hybrid minimization method
// update the unitary transformation matrix U′ = U X Uη
// where Uη diagonalizes the reduced off-diagonal Fock term
// Fη_|ij = δ_ij Fii + η (1 − δij) Fij
// See more detail in G.F. Bertsch, J.M. Mehlhaff, Computer Physics Communications 207 (2016) 518–523
void HartreeFock::UpdateU_hybrid()
{
    // move energies
    cblas_dcopy(dim_p + dim_n, energies, 1, prev_energies, 1);
    std::vector<double> Heta_p(dim_p * dim_p, 0);
    std::vector<double> Heta_n(dim_n * dim_n, 0);
    std::vector<double> tempU_p(dim_p * dim_p, 0);
    std::vector<double> tempU_n(dim_n * dim_n, 0);
    // F_p = Fii + η (1 − δij) Fij
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Heta_p.data(), 1);
    cblas_dscal(dim_p * dim_p, eta, Heta_p.data(), 1);
    for (size_t i = 0; i < dim_p; i++)
    {
        Heta_p[i * dim_p + i] = FockTerm_p[i * dim_p + i];
    }
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Heta_n.data(), 1);
    cblas_dscal(dim_n * dim_n, eta, Heta_n.data(), 1);
    for (size_t i = 0; i < dim_n; i++)
    {
        Heta_n[i * dim_n + i] = FockTerm_n[i * dim_n + i];
    }

    /// Diag Proton Fock term
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim_p, Heta_p.data(), dim_p, energies) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in proton Fock term!\n");
        PrintFockMatrix();
        exit(0);
    }

    /// Diag Neutron Fock term
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim_n, Heta_n.data(), dim_n, energies + dim_p) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in neutron Fock term!\n");
        PrintFockMatrix();
        exit(0);
    }

    // update C
    cblas_dcopy(dim_p * dim_p, U_p, 1, tempU_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, U_n, 1, tempU_n.data(), 1);
    if (N_p > 0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., Heta_p.data(), dim_p, tempU_p.data(), dim_p, 0, U_p, dim_p);
    if (N_n > 0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., Heta_n.data(), dim_n, tempU_n.data(), dim_n, 0, U_n, dim_n);

    // Update hole orbits
    for (size_t i = 0; i < N_p; i++)
    {
        holeorbs_p[i] = i;
    }
    for (size_t i = 0; i < N_n; i++)
    {
        holeorbs_n[i] = i;
    }
    return;
}

//*********************************************************************
// update U casued by constraining ⟨Q⟩
// The code uses a simple Padé approximant to preserve the orthogonal character
// e^Z \aprox (1 + Z/2)(1 − Z/2)^-1    /// here -1 stand for the inverse of the matrix
// Padé approximant https://en.wikipedia.org/wiki/Pad%C3%A9_approximant
// Zij = (q - <Q>)/ ( Tr[ Q^ph x Q^ph ^T ] ) Q^ph
// where Q^ph = Q^orb_ij  (fi − fj)    /// Q^orb is the operator in HF basis fi
// is the occupation
void HartreeFock::UpdateU_Qconstraint(double deltaQ, double *O_p, double *O_n)
{
    double factor_p, factor_n;
    std::vector<double> Oorb_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_n(dim_n * dim_n, 0);
    std::vector<double> Oorb_ph_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_ph_n(dim_n * dim_n, 0);
    std::vector<double> Zp_p(dim_p * dim_p, 0);
    std::vector<double> Zn_p(dim_p * dim_p, 0);
    std::vector<double> Zp_n(dim_n * dim_n, 0);
    std::vector<double> Zn_n(dim_n * dim_n, 0);

    cblas_dcopy(dim_p * dim_p, O_p, 1, Oorb_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, O_n, 1, Oorb_n.data(), 1);
    // TransferOperatorToHFbasis(Oorb_p.data(), Oorb_n.data());
    cblas_dcopy(dim_p * dim_p, Oorb_p.data(), 1, Oorb_ph_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, Oorb_n.data(), 1, Oorb_ph_n.data(), 1);
    Operator_ph(Oorb_ph_p.data(), Oorb_ph_n.data());

    double dotProduct;
    dotProduct = cblas_ddot(dim_p * dim_p, Oorb_ph_p.data(), 1, Oorb_ph_p.data(), 1);
    factor_p = deltaQ / dotProduct;
    /// update Horb, the Fock term should be transfor to HF basis first
    /// double HdotQ = cblas_ddot(dim_p * dim_p, Oorb_ph_p.data(), 1, FockTerm_p, 1);
    /// cblas_daxpy(dim_p * dim_p, -HdotQ / dotProduct, Oorb_p.data(), 1, FockTerm_p, 1);
    /// End
    dotProduct = cblas_ddot(dim_n * dim_n, Oorb_ph_n.data(), 1, Oorb_ph_n.data(), 1);
    factor_n = deltaQ / dotProduct;
    /// update Horb, the Fock term should be transfor to HF basis first
    /// HdotQ = cblas_ddot(dim_n * dim_n, Oorb_ph_n.data(), 1, FockTerm_n, 1);
    /// cblas_daxpy(dim_n * dim_n, -HdotQ / dotProduct, Oorb_n.data(), 1, FockTerm_n, 1);
    /// End

    // std::cout<< factor_p << "  " << factor_n << std::endl;

    cblas_dscal(dim_p * dim_p, factor_p, Oorb_ph_p.data(), 1);
    cblas_dscal(dim_n * dim_n, factor_n, Oorb_ph_n.data(), 1);

    cblas_dcopy(dim_p * dim_p, Oorb_ph_p.data(), 1, Zp_p.data(), 1);
    cblas_dscal(dim_p * dim_p, 0.5, Zp_p.data(), 1);
    cblas_dcopy(dim_p * dim_p, Oorb_ph_p.data(), 1, Zn_p.data(), 1);
    cblas_dscal(dim_p * dim_p, -0.5, Zn_p.data(), 1);
    for (size_t i = 0; i < dim_p; i++)
    {
        Zp_p[i * dim_p + i] += 1.;
        Zn_p[i * dim_p + i] += 1.;
    }

    cblas_dcopy(dim_n * dim_n, Oorb_ph_n.data(), 1, Zp_n.data(), 1);
    cblas_dscal(dim_n * dim_n, 0.5, Zp_n.data(), 1);
    cblas_dcopy(dim_n * dim_n, Oorb_ph_n.data(), 1, Zn_n.data(), 1);
    cblas_dscal(dim_n * dim_n, -0.5, Zn_n.data(), 1);
    for (size_t i = 0; i < dim_n; i++)
    {
        Zp_n[i * dim_n + i] += 1.;
        Zn_n[i * dim_n + i] += 1.;
    }

    /*******************************************************************/
    std::vector<int> ipiv_p(dim_p); // Allocate memory for the ipiv array
    std::vector<int> ipiv_n(dim_n); // Allocate memory for the ipiv array
    // Perform LU factorization
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim_p, dim_p, Zn_p.data(), dim_p, ipiv_p.data());
    if (info != 0)
    {
        std::cerr << "LU factorization failed with error code: " << info << std::endl;
        return;
    }
    // Compute the inverse
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, dim_p, Zn_p.data(), dim_p, ipiv_p.data());
    if (info != 0)
    {
        std::cerr << "Matrix inverse calculation failed with error code: " << info << std::endl;
        return;
    }
    // Perform LU factorization
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim_n, dim_n, Zn_n.data(), dim_n, ipiv_n.data());
    if (info != 0)
    {
        std::cerr << "LU factorization failed with error code: " << info << std::endl;
        return;
    }
    // Compute the inverse
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, dim_n, Zn_n.data(), dim_n, ipiv_n.data());
    if (info != 0)
    {
        std::cerr << "Matrix inverse calculation failed with error code: " << info << std::endl;
        return;
    }

    ///  update U

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1.0, Zp_p.data(), dim_p, Zn_p.data(), dim_p, 0.0, Oorb_p.data(), dim_p);
    cblas_dcopy(dim_p * dim_p, U_p, 1, Zp_p.data(), 1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., Oorb_p.data(), dim_p, Zp_p.data(), dim_p, 0, U_p, dim_p);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1.0, Zp_n.data(), dim_n, Zn_n.data(), dim_n, 0.0, Oorb_n.data(), dim_n);
    cblas_dcopy(dim_n * dim_n, U_n, 1, Zp_n.data(), 1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., Oorb_n.data(), dim_n, Zp_n.data(), dim_n, 0, U_n, dim_n);

    return;
}

void HartreeFock::UpdateU_Qconstraint_generatZ(double deltaQ, double *O_p, double *O_n)
{
    double factor_p, factor_n;
    std::vector<double> Oorb_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_n(dim_n * dim_n, 0);
    std::vector<double> Oorb_ph_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_ph_n(dim_n * dim_n, 0);
    std::vector<double> Zp_p(dim_p * dim_p, 0);
    std::vector<double> Zn_p(dim_p * dim_p, 0);
    std::vector<double> Zp_n(dim_n * dim_n, 0);
    std::vector<double> Zn_n(dim_n * dim_n, 0);

    cblas_dcopy(dim_p * dim_p, O_p, 1, Oorb_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, O_n, 1, Oorb_n.data(), 1);
    // TransferOperatorToHFbasis(Oorb_p.data(), Oorb_n.data());
    cblas_dcopy(dim_p * dim_p, Oorb_p.data(), 1, Oorb_ph_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, Oorb_n.data(), 1, Oorb_ph_n.data(), 1);
    Operator_ph(Oorb_ph_p.data(), Oorb_ph_n.data());

    std::vector<double> Fph_p(dim_n * dim_n, 0);
    std::vector<double> Fph_n(dim_n * dim_n, 0);
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Fph_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Fph_n.data(), 1);
    Operator_ph(Fph_p.data(), Fph_n.data());

    double dotProduct;
    double QdotHProduct;

    QdotHProduct = cblas_ddot(dim_p * dim_p, Fph_p.data(), 1, Oorb_ph_p.data(), 1);
    dotProduct = cblas_ddot(dim_p * dim_p, Oorb_ph_p.data(), 1, Oorb_ph_p.data(), 1);
    // dotProduct = cblas_ddot(dim_p * dim_p, Oorb_p.data(), 1, Oorb_p.data(), 1);
    factor_p = deltaQ / dotProduct;
    factor_p = -0.5 * QdotHProduct / (dotProduct)-deltaQ / (dotProduct);

    QdotHProduct = cblas_ddot(dim_n * dim_n, Fph_n.data(), 1, Oorb_ph_n.data(), 1);
    dotProduct = cblas_ddot(dim_n * dim_n, Oorb_ph_n.data(), 1, Oorb_ph_n.data(), 1);
    // dotProduct = cblas_ddot(dim_n * dim_n, Oorb_n.data(), 1, Oorb_n.data(), 1);
    factor_n = deltaQ / (dotProduct);
    factor_n = -0.5 * QdotHProduct / (dotProduct)-deltaQ / (dotProduct);

    cblas_dscal(dim_p * dim_p, factor_p, Oorb_ph_p.data(), 1);
    cblas_dscal(dim_n * dim_n, factor_n, Oorb_ph_n.data(), 1);

    cblas_dcopy(dim_p * dim_p, Oorb_ph_p.data(), 1, O_p, 1);
    cblas_dcopy(dim_n * dim_n, Oorb_ph_n.data(), 1, O_n, 1);
    return;
}

void HartreeFock::UpdateU_Qconstraint_generatZ_v2(double deltaQ, double *O_p, double *O_n)
{
    double factor_p, factor_n;
    std::vector<double> Oorb_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_n(dim_n * dim_n, 0);
    std::vector<double> Oorb_ph_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_ph_n(dim_n * dim_n, 0);
    std::vector<double> Zp_p(dim_p * dim_p, 0);
    std::vector<double> Zn_p(dim_p * dim_p, 0);
    std::vector<double> Zp_n(dim_n * dim_n, 0);
    std::vector<double> Zn_n(dim_n * dim_n, 0);

    cblas_dcopy(dim_p * dim_p, O_p, 1, Oorb_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, O_n, 1, Oorb_n.data(), 1);
    // TransferOperatorToHFbasis(Oorb_p.data(), Oorb_n.data());
    cblas_dcopy(dim_p * dim_p, Oorb_p.data(), 1, Oorb_ph_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, Oorb_n.data(), 1, Oorb_ph_n.data(), 1);
    Operator_ph(Oorb_ph_p.data(), Oorb_ph_n.data());

    factor_p = deltaQ;
    factor_n = deltaQ;

    cblas_dscal(dim_p * dim_p, factor_p, Oorb_ph_p.data(), 1);
    cblas_dscal(dim_n * dim_n, factor_n, Oorb_ph_n.data(), 1);

    cblas_dcopy(dim_p * dim_p, Oorb_ph_p.data(), 1, O_p, 1);
    cblas_dcopy(dim_n * dim_n, Oorb_ph_n.data(), 1, O_n, 1);
    return;
}

// frobenius norm
double HartreeFock::frobenius_norm(const std::vector<double> &A)
{
    return std::sqrt(std::accumulate(A.begin(), A.end(), 0.0, [](double acc, double x)
                                     { return acc + x * x; }));
}

// DIIS: Direct Inversion in the Iterative Subspace
// an approach for accelerating the convergence of the HF iterations
// See P Pulay J. Comp. Chem. 3(4) 556 (1982), and  Garza & Scuseria J. Chem. Phys. 137 054110 (2012)
void HartreeFock::UpdateDensityMatrix_DIIS()
{
    size_t N_MATS_STORED = 5; // How many past density matrices and error matrices to store

    // If we're at the solution, the Fock matrix F and rho will commute
    std::vector<double> error_mat_p(dim_p * dim_p);
    std::vector<double> error_mat_n(dim_n * dim_n);

    // Proton
    if (N_p > 0)
    {
        // compute matrix product F * rho
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1.0, FockTerm_p, dim_p, rho_p, dim_p, 0.0, error_mat_p.data(), dim_p);
        // compute matrix difference F * rho - rho * F
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, -1.0, rho_p, dim_p, FockTerm_p, dim_p, 1.0, error_mat_p.data(), dim_p);
    }

    // Neutron
    if (N_n > 0)
    {
        // compute matrix product F * rho
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1.0, FockTerm_n, dim_n, rho_n, dim_n, 0.0, error_mat_n.data(), dim_n);
        // compute matrix difference F * rho - rho * F
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, -1.0, rho_n, dim_n, FockTerm_n, dim_n, 1.0, error_mat_n.data(), dim_n);
    }
    // Compute the new density matrix rho
    UpdateDensityMatrix();

    // save this error matrix in a list
    size_t nsave_p = DIIS_error_mats_p.size();
    if (nsave_p > N_MATS_STORED)
        DIIS_error_mats_p.pop_front();        // out with the old
    DIIS_error_mats_p.push_back(error_mat_p); // in with the new

    // save the new rho in the list of rhos
    if (DIIS_density_mats_p.size() > N_MATS_STORED)
        DIIS_density_mats_p.pop_front(); // out with the old
    std::vector<double> veU_p(rho_p, rho_p + dim_p * dim_p);
    DIIS_density_mats_p.push_back(veU_p); // in with the new

    // save this error matrix in a list
    size_t nsave_n = DIIS_error_mats_n.size();
    if (nsave_n > N_MATS_STORED)
        DIIS_error_mats_n.pop_front(); // out with the old
    // Insert the elements from the array into the vector
    DIIS_error_mats_n.push_back(error_mat_n); // in with the new

    // save the new rho in the list of rhos
    if (DIIS_density_mats_n.size() > N_MATS_STORED)
        DIIS_density_mats_n.pop_front(); // out with the old
    std::vector<double> veU_n(rho_n, rho_n + dim_n * dim_n);
    DIIS_density_mats_n.push_back(veU_n); // in with the new

    if (std::max(nsave_p, nsave_n) < N_MATS_STORED)
    {
        return; // check that have enough previous rhos and errors stored to do this
    }

    // Now construct the B matrix which is inverted to find the combination of
    // previous density matrices which will minimize the error
    // Proton
    int nsave = nsave_p;
    int n = nsave + 1;
    // Define additional arrays for LAPACK's output
    int ipiv[n];
    int info;
    // Define the arrays to hold the matrix and right-hand side vector
    std::vector<double> b(n, 0);
    std::vector<double> Bij(n * n);
    for (size_t i = 0; i < nsave; i++)
    {
        Bij[i * (n) + nsave] = 1;
        Bij[nsave * (n) + i] = 1;
    }
    Bij[nsave * (n) + nsave] = 0;
    double sum;
    std::vector<double> Cmatrix_p(dim_p * dim_p);
    if (N_p > 0)
    {
        for (size_t i = 0; i < nsave; i++)
        {
            for (size_t j = 0; j < nsave; j++)
            {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim_p, dim_p, dim_p, 1.0, DIIS_error_mats_p[i].data(), dim_p, DIIS_error_mats_p[j].data(), dim_p, 0.0, Cmatrix_p.data(), dim_p);
                Bij[i * (n) + j] = frobenius_norm(Cmatrix_p);
            }
        }

        b[nsave] = 1;

        // solve the system of linear equations
        info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, Bij.data(), n, ipiv, b.data(), 1);
        for (size_t i = 0; i < nsave + 1; i++)
        {
            b[i] = std::abs(b[i]);
        }

        // Calculate the sum of the first nsave elements
        sum = std::accumulate(b.begin(), b.begin() + nsave, 0.0);

        // Divide each element by the sum
        if (fabs(sum) > 1.e-8)
            for (size_t i = 0; i < nsave; i++)
            {
                // std::cout << i << "  " << b[i] << "  " << sum << std::endl;
                b[i] /= sum;
            }
        memset(rho_p, 0, sizeof(rho_p));
        for (size_t i = 0; i < nsave; i++)
        {
            cblas_daxpy(dim_p * dim_p, b[i], DIIS_density_mats_p[i].data(), 1, rho_p, 1);
        }
    }

    // Neutron
    for (size_t i = 0; i < nsave; i++)
    {
        Bij[i * (nsave + 1) + nsave] = 1;
        Bij[nsave * (nsave + 1) + i] = 1;
    }
    Bij[nsave * (nsave + 1) + nsave] = 0;

    std::vector<double> Cmatrix_n(dim_n * dim_n);
    if (N_n > 0)
    {
        for (size_t i = 0; i < nsave; i++)
        {
            for (size_t j = 0; j < nsave; j++)
            {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim_n, dim_n, dim_n, 1.0, DIIS_error_mats_n[i].data(), dim_n, DIIS_error_mats_n[j].data(), dim_n, 0.0, Cmatrix_n.data(), dim_n);
                Bij[i * (nsave + 1) + j] = frobenius_norm(Cmatrix_n);
            }
        }
        // Define the arrays to hold the matrix and right-hand side vector
        memset(b.data(), 0, sizeof(b.data()));
        b[nsave] = 1;
        // solve the system of linear equations
        info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, Bij.data(), n, ipiv, b.data(), 1);

        for (size_t i = 0; i < nsave + 1; i++)
        {
            b[i] = std::abs(b[i]);
        }

        // Calculate the sum of the first nsave elements
        sum = std::accumulate(b.begin(), b.begin() + nsave, 0.0);

        // Divide each element by the sum
        if (fabs(sum) > 1.e-8)
            for (size_t i = 0; i < nsave; i++)
            {
                b[i] /= sum;
            }
        memset(rho_n, 0, sizeof(rho_n));
        for (size_t i = 0; i < nsave; i++)
            cblas_daxpy(dim_n * dim_n, b[i], DIIS_density_mats_n[i].data(), 1, rho_n, 1);
    }
}

// EDIIS: Energy Direct Inversion in the Iterative Subspace
// an approach for accelerating the convergence of the HF iterations
// See P Pulay J. Comp. Chem. 3(4) 556 (1982), and  Garza & Scuseria J. Chem. Phys. 137 054110 (2012)
// Will developed

//*********************************************************************
///  [See Suhonen eq 4.85] page 83
///   H_{ij} = t_{ij}  + \sum_{b} \sum_{j_1 j_2}  \rho_{ab} \bar{V}^{(2)}_{iajb}
///   where (e_b < E_f)
//*********************************************************************
void HartreeFock::UpdateF()
{
    memset(FockTerm_p, 0, dim_p * dim_p * sizeof(double));
    memset(FockTerm_n, 0, dim_n * dim_n * sizeof(double));

    double *Vpp, *Vpn, *Vnn;
    Vpp = Ham->MSMEs.GetVppPrt();
    Vnn = Ham->MSMEs.GetVnnPrt();
    Vpn = Ham->MSMEs.GetVpnPrt();

    // Proton subspace
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = i; j < dim_p; j++)
        {
            // add Vpp term
            FockTerm_p[i * dim_p + j] += cblas_ddot(dim_p * dim_p, rho_p, 1, Vpp + (dim_p * dim_p * dim_p * i + dim_p * dim_p * j), 1);

            // add Vpn term
            FockTerm_p[i * dim_p + j] += cblas_ddot(dim_n * dim_n, rho_n, 1, Vpn + (dim_p * dim_n * dim_n * i + dim_n * dim_n * j), 1);
            if (i != j)
                FockTerm_p[j * dim_p + i] = FockTerm_p[i * dim_p + j];
        }
    }

    // Neutron subspace
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = i; j < dim_n; j++)
        {
            // add Vnn term
            FockTerm_n[i * dim_n + j] += cblas_ddot(dim_n * dim_n, rho_n, 1, Vnn + dim_n * dim_n * dim_n * i + dim_n * dim_n * j, 1);

            // add Vpn term
            FockTerm_n[i * dim_n + j] += cblas_ddot(dim_p * dim_p, rho_p, 1, Vpn + dim_n * i + j, dim_n * dim_n);

            if (i != j)
                FockTerm_n[j * dim_n + i] = FockTerm_n[i * dim_n + j];
        }
    }
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Vij_p, 1);
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Vij_n, 1);

    // add SP term
    // add SP term
    if (dim_p != 0)
        cblas_daxpy(dim_p * dim_p, 1., T_term_p, 1, FockTerm_p, 1);

    if (dim_n != 0)
        cblas_daxpy(dim_n * dim_n, 1., T_term_n, 1, FockTerm_n, 1);
}

/// For no core calculation, diag method
void HartreeFock::UpdateF_noCore()
{
    memset(FockTerm_p, 0, dim_p * dim_p * sizeof(double));
    memset(FockTerm_n, 0, dim_n * dim_n * sizeof(double));

    double *Vpp, *Vpn, *Vnn;
    Vpp = Ham->MSMEs.GetVppPrt();
    Vnn = Ham->MSMEs.GetVnnPrt();
    Vpn = Ham->MSMEs.GetVpnPrt();

    // Proton subspace
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = i; j < dim_p; j++)
        {
            // add Vpp term
            FockTerm_p[i * dim_p + j] += cblas_ddot(dim_p * dim_p, rho_p, 1, Vpp + (dim_p * dim_p * dim_p * i + dim_p * dim_p * j), 1);

            // add Vpn term
            FockTerm_p[i * dim_p + j] += cblas_ddot(dim_n * dim_n, rho_n, 1, Vpn + (dim_p * dim_n * dim_n * i + dim_n * dim_n * j), 1);
            if (i != j)
                FockTerm_p[j * dim_p + i] = FockTerm_p[i * dim_p + j];
        }
    }

    // Neutron subspace
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = i; j < dim_n; j++)
        {
            // add Vnn term
            FockTerm_n[i * dim_n + j] += cblas_ddot(dim_n * dim_n, rho_n, 1, Vnn + dim_n * dim_n * dim_n * i + dim_n * dim_n * j, 1);

            // add Vpn term
            FockTerm_n[i * dim_n + j] += cblas_ddot(dim_p * dim_p, rho_p, 1, Vpn + dim_n * i + j, dim_n * dim_n);

            if (i != j)
                FockTerm_n[j * dim_n + i] = FockTerm_n[i * dim_n + j];
        }
    }
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Vij_p, 1);
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Vij_n, 1);

    // add SP term
    if (dim_p != 0)
        cblas_daxpy(dim_p * dim_p, 1., T_term_p, 1, FockTerm_p, 1);

    if (dim_n != 0)
        cblas_daxpy(dim_n * dim_n, 1., T_term_n, 1, FockTerm_n, 1);
}

// the F matrix should transfer to HF orbits first
// add constribution to Horb
// Horb' = Horb -  Tr[ Horb x Q^ph ^T ] / ( Tr[ Q^ph x Q^ph ^T ] ) Q^ph
void HartreeFock::UpdateF_FromQ(double *O_p, double *O_n)
{
    std::vector<double> Oorb_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_n(dim_n * dim_n, 0);
    std::vector<double> Oorb_ph_p(dim_p * dim_p, 0);
    std::vector<double> Oorb_ph_n(dim_n * dim_n, 0);

    cblas_dcopy(dim_p * dim_p, O_p, 1, Oorb_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, O_n, 1, Oorb_n.data(), 1);
    TransferOperatorToHFbasis(Oorb_p.data(), Oorb_n.data());
    cblas_dcopy(dim_p * dim_p, Oorb_p.data(), 1, Oorb_ph_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, Oorb_n.data(), 1, Oorb_ph_n.data(), 1);
    Operator_ph(Oorb_ph_p.data(), Oorb_ph_n.data());

    std::vector<double> Hph_p(dim_p * dim_p, 0);
    std::vector<double> Hph_n(dim_n * dim_n, 0);
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Hph_p.data(), 1);
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Hph_n.data(), 1);
    Operator_ph(Hph_p.data(), Hph_n.data());
    std::vector<double> temp_p(dim_p * dim_p, 0);
    std::vector<double> temp_n(dim_n * dim_n, 0);

    /// update Horb, the Fock term should be transfor to HF basis first
    double dotProduct = cblas_ddot(dim_p * dim_p, Oorb_ph_p.data(), 1, Oorb_ph_p.data(), 1);
    double HdotQ;
    // HdotQ = cblas_ddot(dim_p * dim_p, FockTerm_p, 1, Oorb_ph_p.data(), 1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., FockTerm_p, dim_p, Oorb_ph_p.data(), dim_p, 0, temp_p.data(), dim_p);
    HdotQ = 0;
    for (size_t i = 0; i < dim_p; i++)
    {
        HdotQ += temp_p[i * dim_p + i];
    }
    cblas_daxpy(dim_p * dim_p, -HdotQ / dotProduct, Oorb_p.data(), 1, Vij_p, 1);
    // std::cout << "  " << dotProduct << "  " << HdotQ << std::endl;
    /// End

    /// update Horb, the Fock term should be transfor to HF basis first
    dotProduct = cblas_ddot(dim_n * dim_n, Oorb_ph_n.data(), 1, Oorb_ph_n.data(), 1);
    // HdotQ = cblas_ddot(dim_n * dim_n, FockTerm_n, 1, Oorb_ph_n.data(), 1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., FockTerm_n, dim_n, Oorb_ph_n.data(), dim_n, 0, temp_n.data(), dim_n);
    HdotQ = 0;
    for (size_t i = 0; i < dim_n; i++)
    {
        HdotQ += temp_n[i * dim_n + i];
    }
    cblas_daxpy(dim_n * dim_n, -HdotQ / dotProduct, Oorb_n.data(), 1, Vij_n, 1);
    // std::cout << "  " << dotProduct << "  " << HdotQ <<std::endl;
    /// End
}

//*********************************************************************
/// [See Suhonen eq. 4.85]
/// Diagonalize the fock matrix \f$ <i|F|j> \f$ and put the
/// eigenvectors in \f$C(i,\alpha) = <i|\alpha> \f$
/// and eigenvalues in the vector energies.
/// Save the last vector of energies to check for convergence.
//*********************************************************************
void HartreeFock::Diagonalize()
{
    // move energies
    cblas_dcopy(dim_p + dim_n, energies, 1, prev_energies, 1);
    ////
    double *EigenVector_p, *EigenVector_n;
    EigenVector_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    EigenVector_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    /// Diag Proton Fock term
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, EigenVector_p, 1);
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim_p, EigenVector_p, dim_p, energies) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in proton Fock term!\n");
        PrintFockMatrix();
        exit(0);
    }
    /// Diag Neutron Fock term
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, EigenVector_n, 1);
    if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim_n, EigenVector_n, dim_n, energies + dim_p) != 0)
    { /// 'V' stand for eigenvalues and vectors
        printf("Error when Computes all eigenvalues and eigenvectors in neutron Fock term!\n");
        PrintFockMatrix();
        exit(0);
    }

    // record C parameters
    cblas_dcopy(dim_p * dim_p, EigenVector_p, 1, U_p, 1);
    cblas_dcopy(dim_n * dim_n, EigenVector_n, 1, U_n, 1);

    // Update hole orbits
    for (size_t i = 0; i < N_p; i++)
    {
        holeorbs_p[i] = i;
    }
    for (size_t i = 0; i < N_n; i++)
    {
        holeorbs_n[i] = i;
    }

    /// free array
    mkl_free(EigenVector_p);
    mkl_free(EigenVector_n);
    return;
}

//*********************************************************************
// zκλ = ηz (∂E)/(∂zκλ)
// Given the gradient, the simplest algorithm to update U is the steepest descent method
// (∂E)/(∂zκλ) = Hκλ_orb (fκ − fλ).
// where Hκλ_orb is single-particle Hamiltonian in the orbital basis,
// Horb = U Hsp UT, where Hsp is the normal Fock term,
// this should be done before call this function
void HartreeFock::Cal_Gradient(double *Z_p, double *Z_n)
{
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p, 1);
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n, 1);
    // this should be done before call this function
    // TransferOperatorToHFbasis(Oorb_p.data(), Oorb_n.data());
    Operator_ph(Z_p, Z_n);

    cblas_dscal(dim_p * dim_p, gradient_eta, Z_p, 1);
    cblas_dscal(dim_n * dim_n, gradient_eta, Z_n, 1);
    return;
}

void HartreeFock::Cal_Gradient_given_gradient(double *Z_p, double *Z_n)
{
    cblas_dscal(dim_p * dim_p, gradient_eta, Z_p, 1);
    cblas_dscal(dim_n * dim_n, gradient_eta, Z_n, 1);
    return;
}

//*********************************************************************
// zκλ = ηz  1/| H^orb_kk - H^orb_ll |  (∂E)/(∂zκλ)
// Given the gradient, the simplest algorithm to update U is the steepest descent method
// (∂E)/(∂zκλ) = Hκλ_orb (fκ − fλ).
// where Hκλ_orb is single-particle Hamiltonian in the orbital basis,
// Horb = U Hsp UT, where Hsp is the normal Fock term,
// this should be done before call this function
// found more from http://dx.doi.org/10.1016/j.cpc.2016.06.023
// Computer Physics Communications 207 (2016) 518–523
void HartreeFock::Cal_Gradient_preconditioned(double *Z_p, double *Z_n)
{
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p, 1);
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n, 1);
    // this should be done before call this function
    // TransferOperatorToHFbasis(Oorb_p.data(), Oorb_n.data());
    Operator_ph(Z_p, Z_n);
    double denominator;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = i + 1; j < dim_p; j++)
        {
            denominator = std::abs(FockTerm_p[i * dim_p + i] - FockTerm_p[j * dim_p + j]);
            if (fabs(denominator) > 1.e-5)
            {
                Z_p[i * dim_p + j] *= 1. / denominator;
                Z_p[j * dim_p + i] *= 1. / denominator;
            }
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = i + 1; j < dim_n; j++)
        {
            denominator = std::abs(FockTerm_n[i * dim_n + i] - FockTerm_n[j * dim_n + j]);
            if (fabs(denominator) > 1.e-5)
            {
                Z_n[i * dim_n + j] *= 1. / denominator;
                Z_n[j * dim_n + i] *= 1. / denominator;
            }
        }
    }
    cblas_dscal(dim_p * dim_p, gradient_eta, Z_p, 1);
    cblas_dscal(dim_n * dim_n, gradient_eta, Z_n, 1);
    return;
}

void HartreeFock::Cal_Gradient_preconditioned_given_gradient(double *Z_p, double *Z_n)
{
    double denominator;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = i + 1; j < dim_p; j++)
        {
            denominator = std::abs(FockTerm_p[i * dim_p + i] - FockTerm_p[j * dim_p + j]);
            if (fabs(denominator) > 1.e-5)
            {
                Z_p[i * dim_p + j] *= 1. / denominator;
                Z_p[j * dim_p + i] *= 1. / denominator;
            }
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = i + 1; j < dim_n; j++)
        {
            denominator = std::abs(FockTerm_n[i * dim_n + i] - FockTerm_n[j * dim_n + j]);
            if (fabs(denominator) > 1.e-5)
            {
                Z_n[i * dim_n + j] *= 1. / denominator;
                Z_n[j * dim_n + i] *= 1. / denominator;
            }
        }
    }
    cblas_dscal(dim_p * dim_p, gradient_eta, Z_p, 1);
    cblas_dscal(dim_n * dim_n, gradient_eta, Z_n, 1);
    return;
}

// zκλ =  H^orb_kj  /| H^orb_kk - H^orb_ll |  (∂E)/(∂zκλ)
// Given the gradient, the simplest algorithm to update U is the steepest descent method
// (∂E)/(∂zκλ) = Hκλ_orb (fκ − fλ).
// where Hκλ_orb is single-particle Hamiltonian in the orbital basis,
// Horb = U Hsp UT, where Hsp is the normal Fock term,
// this should be done before call this function
// this idea comes from Ragnar. Talk more detail about it.
void HartreeFock::Cal_Gradient_preconditioned_SRG(double *Z_p, double *Z_n)
{
    cblas_dcopy(dim_p * dim_p, FockTerm_p, 1, Z_p, 1);
    cblas_dcopy(dim_n * dim_n, FockTerm_n, 1, Z_n, 1);
    // this should be done before call this function
    // TransferOperatorToHFbasis(Oorb_p.data(), Oorb_n.data());
    Operator_ph(Z_p, Z_n);
    double denominator;
    if (N_p > 0)
        for (size_t i = 0; i < dim_p; i++)
        {
            for (size_t j = i; j < dim_p; j++)
            {
                denominator = (FockTerm_p[i * dim_p + i] - FockTerm_p[j * dim_p + j]);
                if (fabs(denominator) > 1.e-5)
                {
                    Z_p[i * dim_p + j] *= FockTerm_p[i * dim_p + j] / denominator;
                    // Z_p[j * dim_p + i] *= FockTerm_p[j * dim_p + i] / denominator;
                }
            }
        }

    if (N_n > 0)
        for (size_t i = 0; i < dim_n; i++)
        {
            for (size_t j = i; j < dim_n; j++)
            {
                denominator = (FockTerm_n[i * dim_n + i] - FockTerm_n[j * dim_n + j]);
                if (fabs(denominator) > 1.e-5)
                {
                    Z_n[i * dim_p + j] *= FockTerm_n[i * dim_n + j] / denominator;
                    // Z_n[j * dim_p + i] *= FockTerm_n[j * dim_n + i] / denominator;
                }
            }
        }
    cblas_dscal(dim_p * dim_p, gradient_eta, Z_p, 1);
    cblas_dscal(dim_n * dim_n, gradient_eta, Z_n, 1);
    return;
}

//*********************************************************************
// The update from U to U' can be expressed as a Thouless transformation of U
// U' = e^Z U
// e^Z \aprox (1 + Z/2)(1 − Z/2)^-1
// where -1 stand for the inverse of the matrix
// Read more about Padé approximant https://en.wikipedia.org/wiki/Pad%C3%A9_approximant
void HartreeFock::UpdateU_Thouless_pade(double *Z_p, double *Z_n)
{
    std::vector<double> Zp_p(dim_p * dim_p, 0);
    std::vector<double> Zn_p(dim_p * dim_p, 0);
    std::vector<double> Zp_n(dim_n * dim_n, 0);
    std::vector<double> Zn_n(dim_n * dim_n, 0);

    cblas_dcopy(dim_p * dim_p, Z_p, 1, Zp_p.data(), 1);
    cblas_dscal(dim_p * dim_p, 0.5, Zp_p.data(), 1);
    cblas_dcopy(dim_p * dim_p, Z_p, 1, Zn_p.data(), 1);
    cblas_dscal(dim_p * dim_p, -0.5, Zn_p.data(), 1);
    for (size_t i = 0; i < dim_p; i++)
    {
        Zp_p[i * dim_p + i] += 1.;
        Zn_p[i * dim_p + i] += 1.;
    }

    cblas_dcopy(dim_n * dim_n, Z_n, 1, Zp_n.data(), 1);
    cblas_dscal(dim_n * dim_n, 0.5, Zp_n.data(), 1);
    cblas_dcopy(dim_n * dim_n, Z_n, 1, Zn_n.data(), 1);
    cblas_dscal(dim_n * dim_n, -0.5, Zn_n.data(), 1);
    for (size_t i = 0; i < dim_n; i++)
    {
        Zp_n[i * dim_n + i] += 1.;
        Zn_n[i * dim_n + i] += 1.;
    }

    /*******************************************************************/
    std::vector<int> ipiv_p(dim_p); // Allocate memory for the ipiv array
    std::vector<int> ipiv_n(dim_n); // Allocate memory for the ipiv array
    // Perform LU factorization
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim_p, dim_p, Zn_p.data(), dim_p, ipiv_p.data());
    if (info != 0)
    {
        std::cerr << "Proton LU factorization failed with error code: " << info << std::endl;
        return;
    }
    // Compute the inverse
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, dim_p, Zn_p.data(), dim_p, ipiv_p.data());
    if (info != 0)
    {
        std::cerr << "Matrix inverse calculation failed with error code: " << info << std::endl;
        return;
    }
    // Perform LU factorization
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim_n, dim_n, Zn_n.data(), dim_n, ipiv_n.data());
    if (info != 0)
    {
        std::cerr << "Neutron LU factorization failed with error code: " << info << std::endl;
        return;
    }
    // Compute the inverse
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, dim_n, Zn_n.data(), dim_n, ipiv_n.data());
    if (info != 0)
    {
        std::cerr << "Matrix inverse calculation failed with error code: " << info << std::endl;
        return;
    }

    std::vector<double> TempU_p(dim_p * dim_p, 0);
    std::vector<double> TempU_n(dim_n * dim_n, 0);
    ///  update U
    if (N_p > 0)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., Zp_p.data(), dim_p, Zn_p.data(), dim_p, 0.0, TempU_p.data(), dim_p);
        cblas_dcopy(dim_p * dim_p, U_p, 1, Zp_p.data(), 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., TempU_p.data(), dim_p, Zp_p.data(), dim_p, 0, U_p, dim_p);
    }

    if (N_n > 0)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., Zp_n.data(), dim_n, Zn_n.data(), dim_n, 0.0, TempU_n.data(), dim_n);
        cblas_dcopy(dim_n * dim_n, U_n, 1, Zp_n.data(), 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., TempU_n.data(), dim_n, Zp_n.data(), dim_n, 0, U_n, dim_n);
    }
    return;
}

void HartreeFock::gram_schmidt(double *vectors, int num_vectors, int vector_size)
{
    // Orthogonalize the first vector
    double norm = cblas_dnrm2(vector_size, vectors, 1);
    cblas_dscal(vector_size, 1.0 / norm, vectors, 1);

    for (int i = 1; i < num_vectors; ++i)
    {
        // Compute the orthogonal component of the current vector
        for (int j = 0; j < i; ++j)
        {
            double projection = cblas_ddot(vector_size, vectors + j * vector_size, 1, vectors + i * vector_size, 1);
            cblas_daxpy(vector_size, -projection, vectors + j * vector_size, 1, vectors + i * vector_size, 1);
        }

        // Normalize the orthogonalized vector
        norm = cblas_dnrm2(vector_size, vectors + i * vector_size, 1);
        cblas_dscal(vector_size, 1.0 / norm, vectors + i * vector_size, 1);
    }
}

// only keep the thouless up to the first order
// e^Z U \aprox  \sum_ik (U_ki + \sum_a Z_ai U_ki)
// i are hole states, a are particle states
// k are H.O. states
void HartreeFock::UpdateU_Thouless_1st(double *Z_p, double *Z_n) // Thouless up to first order
{
    std::vector<int> Holes_p, Holes_n, Particles_p, Particles_n;
    Holes_p = this->GetHoleList(Proton);
    Holes_n = this->GetHoleList(Neutron);
    Particles_p = this->GetParticleList(Proton);
    Particles_n = this->GetParticleList(Neutron);
    for (size_t i = 0; i < Holes_p.size(); i++)
    {
        for (size_t k = 0; k < dim_p; k++)
        {
            for (size_t a = 0; a < Particles_p.size(); a++)
            {
                U_p[k * dim_p + Holes_p[i]] += Z_p[Particles_p[a] * dim_p + Holes_p[i]] * U_p[k * dim_p + Particles_p[a]];
                U_p[k * dim_p + Particles_p[a]] += Z_p[Holes_p[i] * dim_p + Particles_p[a]] * U_p[k * dim_p + Holes_p[i]];
            }
            // cblas_daxpy(dim_p, Z_p[a * dim_p + i], U_p + Particles_p[a], dim_p, U_p + Holes_p[i], dim_p);
        }
    }

    for (size_t i = 0; i < Holes_n.size(); i++)
    {
        for (size_t k = 0; k < dim_n; k++)
        {
            for (size_t a = 0; a < Particles_n.size(); a++)
            {
                U_n[k * dim_n + Holes_n[i]] += Z_n[Particles_n[a] * dim_n + Holes_n[i]] * U_n[k * dim_n + Particles_n[a]];
                U_n[k * dim_n + Particles_n[a]] += Z_n[Holes_n[i] * dim_n + Particles_n[a]] * U_n[k * dim_n + Holes_n[i]];
            }
            // cblas_daxpy(dim_n, Z_n[a * dim_n + i], U_n + Particles_n[a], dim_n, U_n + Holes_n[i], dim_n);
        }
    }
    gram_schmidt(U_p, dim_p, dim_p);
    gram_schmidt(U_n, dim_n, dim_n);
    return;
}

//********************************************************
/// Check for convergence using difference in s.p. energies
/// between iterations.
/// Converged when
/// \f[ \delta_{e} \equiv \sqrt{ \sum_{i}(e_{i}^{(n)}-e_{i}^{(n-1)})^2} < \textrm{tolerance} \f]
/// where \f$ e_{i}^{(n)} \f$ is the \f$ i \f$th eigenvalue of the Fock matrix after \f$ n \f$ iterations.
///
//********************************************************
bool HartreeFock::CheckConvergence()
{
    double ediff = 0;
    std::vector<double> Ediffarray(dim_p + dim_n, 0);
    vdSub(dim_p + dim_n, energies, prev_energies, Ediffarray.data());
    for (size_t i = 0; i < dim_p + dim_n; i++)
    {
        ediff += Ediffarray[i] * Ediffarray[i];
    }
    // std::cout << ediff << std::endl;
    // PrintAllHFEnergies();
    return (sqrt(ediff) < tolerance);
}

///********************************************************************
/// Calculate the HF energy.
/// E_{HF} &=& \sum_{\alpha} t_{\alpha\alpha}
///          + \frac{1}{2}\sum_{\alpha\beta} V_{\alpha\beta\alpha\beta}
/// have already been calculated by UpdateF().
///********************************************************************
void HartreeFock::CalcEHF()
{
    EHF = 0;
    e1hf = 0;
    e2hf = 0;

    // Proton part
    e1hf += cblas_ddot(dim_p * dim_p, rho_p, 1, T_term_p, 1);
    e2hf += cblas_ddot(dim_p * dim_p, rho_p, 1, Vij_p, 1);

    // Neutron part
    e1hf += cblas_ddot(dim_n * dim_n, rho_n, 1, T_term_n, 1);
    e2hf += cblas_ddot(dim_n * dim_n, rho_n, 1, Vij_n, 1);

    // Total HF energy
    EHF = e1hf + 0.5 * e2hf;
    // std::cout << "      " << e1hf << "  " << 0.5 * e2hf << std::endl;
}

void HartreeFock::CalcEHF(double constrainedQ)
{
    EHF = 0;
    e1hf = 0;
    e2hf = 0;

    // Proton part
    e1hf += cblas_ddot(dim_p * dim_p, rho_p, 1, T_term_p, 1);
    e2hf += cblas_ddot(dim_p * dim_p, rho_p, 1, Vij_p, 1);

    // Neutron part
    e1hf += cblas_ddot(dim_n * dim_n, rho_n, 1, T_term_n, 1);
    e2hf += cblas_ddot(dim_n * dim_n, rho_n, 1, Vij_n, 1);

    // Total HF energy
    EHF = e1hf + 0.5 * e2hf + constrainedQ;
    // std::cout << "      " << e1hf << "  " << 0.5 * e2hf << std::endl;
}

void HartreeFock::CalcEHF_noCore()
{
    EHF = 0;
    e1hf = 0;
    e2hf = 0;

    // Proton part
    e1hf += cblas_ddot(dim_p * dim_p, rho_p, 1, T_term_p, 1);
    e2hf += cblas_ddot(dim_p * dim_p, rho_p, 1, Vij_p, 1);

    // Neutron part
    e1hf += cblas_ddot(dim_n * dim_n, rho_n, 1, T_term_n, 1);
    e2hf += cblas_ddot(dim_n * dim_n, rho_n, 1, Vij_n, 1);

    // Total HF energy
    EHF = e1hf + 0.5 * e2hf;
    // std::cout << "      " << e1hf << "  " << 0.5 * e2hf << std::endl;
}

double HartreeFock::CalcEHF(const std::vector<int> proton_vec, const std::vector<int> neutron_vec)
{
    this->UpdateDensityMatrix(proton_vec, neutron_vec);
    this->UpdateF();
    EHF = 0;
    e1hf = 0;
    e2hf = 0;

    // Proton part
    e1hf += cblas_ddot(dim_p * dim_p, rho_p, 1, T_term_p, 1);
    e2hf += cblas_ddot(dim_p * dim_p, rho_p, 1, Vij_p, 1);

    // Neutron part
    e1hf += cblas_ddot(dim_n * dim_n, rho_n, 1, T_term_n, 1);
    e2hf += cblas_ddot(dim_n * dim_n, rho_n, 1, Vij_n, 1);

    // Total HF energy
    EHF = e1hf + 0.5 * e2hf;
    // std::cout << "      " << e1hf << "  " << 0.5 * e2hf << std::endl;
    return EHF;
}

double HartreeFock::CalcEHF_HForbits(const std::vector<int> proton_vec, const std::vector<int> neutron_vec)
{
    this->UpdateDensityMatrix(proton_vec, neutron_vec);
    this->UpdateF();
    this->TransferOperatorToHFbasis(Vij_p, Vij_n);
    e1hf = 0;
    e1hf += cblas_ddot(dim_p * dim_p, rho_p, 1, T_term_p, 1);
    e1hf += cblas_ddot(dim_n * dim_n, rho_n, 1, T_term_n, 1);
    e2hf = 0;
    // Check_matrix(dim_p, FockTerm_p);
    for (size_t i = 0; i < proton_vec.size(); i++)
    {
        // for (size_t j = 0; j < proton_vec.size(); j++)
        {
            e2hf += Vij_p[proton_vec[i] * dim_p + proton_vec[i]];
        }
    }

    // Check_matrix(dim_n, FockTerm_n);
    for (size_t i = 0; i < neutron_vec.size(); i++)
    {
        // for (size_t j = 0; j < neutron_vec.size(); j++)
        {
            e2hf += Vij_n[neutron_vec[i] * dim_n + neutron_vec[i]];
        }
    }
    EHF = e1hf + 0.5 * e2hf;
    return EHF;
}

/// operator in the HF orbital basis
/// O^{orb} = UT * O * U
/// IN my code U are U_p and U_n
void HartreeFock::TransferOperatorToHFbasis(double *Op_p, double *Op_n)
{
    double *O_temp_p, *O_temp_n;
    O_temp_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    O_temp_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);

    if (N_p > 0)
    {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., U_p, dim_p, Op_p, dim_p, 0, O_temp_p, dim_p);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_p, dim_p, dim_p, 1., O_temp_p, dim_p, U_p, dim_p, 0, Op_p, dim_p);
    }
    if (N_n > 0)
    {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., U_n, dim_n, Op_n, dim_n, 0, O_temp_n, dim_n);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim_n, dim_n, dim_n, 1., O_temp_n, dim_n, U_n, dim_n, 0, Op_n, dim_n);
    }
    mkl_free(O_temp_p);
    mkl_free(O_temp_n);
    return;
}

/// evaluate one body term
/// Qp = \sum_{ij} Op_{ij} * rho_{ij}
void HartreeFock::CalOnebodyOperator(double *Op_p, double *Op_n, double &Qp, double &Qn)
{
    Qp = cblas_ddot(dim_p * dim_p, rho_p, 1, Op_p, 1);
    Qn = cblas_ddot(dim_n * dim_n, rho_n, 1, Op_n, 1);
    return;
}

/// evaluate one body term
//  multiply (fa-fb) to operator Oorb_ab
void HartreeFock::Operator_ph(double *Op_p, double *Op_n)
{
    std::vector<int> Occ_p(dim_p, 0);
    std::vector<int> Occ_n(dim_n, 0);
    for (size_t i = 0; i < N_p; i++)
    {
        Occ_p[holeorbs_p[i]] = 1;
    }
    for (size_t i = 0; i < N_n; i++)
    {
        Occ_n[holeorbs_n[i]] = 1;
    }
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            Op_p[i * dim_p + j] *= (Occ_p[i] - Occ_p[j]);
        }
    }
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            Op_n[i * dim_n + j] *= (Occ_n[i] - Occ_n[j]);
        }
    }
    return;
}

// Reset the transformation matrix
void HartreeFock::Reset_U()
{
    // Reset U
    memset(U_p, 0, (dim_p) * (dim_p) * sizeof(double));
    for (size_t i = 0; i < dim_p; i++)
    {
        U_p[i * (dim_p) + i] = 1.;
    }
    memset(U_n, 0, (dim_n) * (dim_n) * sizeof(double));
    for (size_t i = 0; i < dim_n; i++)
    {
        U_n[i * (dim_n) + i] = 1.;
    }

    // Reset hole orbits
    std::vector<Triple> SPEpairs_p(dim_p);
    for (int i = 0; i < dim_p; ++i)
    {
        // std::cout << i << "  " << T_term[i] << "  " << modelspace->Get_MSmatrix_2j(Proton, i) << "  " << modelspace->Get_MSmatrix_2m(Proton, i) << std::endl;
        SPEpairs_p[i] = Triple(T_term_p[i * dim_p + i], i, modelspace->Get_MSmatrix_2m(Proton, i));
    }
    std::sort(SPEpairs_p.begin(), SPEpairs_p.end(), compareTriples);
    for (size_t i = 0; i < N_p; i++)
    {
        holeorbs_p[i] = SPEpairs_p[i].second;
        // std::cout << SPEpairs_p[i].first << "  " << SPEpairs_p[i].second << std::endl;
    }

    //-----------
    // Sort the vector of pairs based on the first element of each pair, which is the value of the
    // element in the original array.
    std::vector<Triple> SPEpairs_n(dim_n);
    for (int i = 0; i < dim_n; ++i)
    {
        // std::cout << i << "  " << T_term[i] << "  " << modelspace->Get_MSmatrix_2j(Proton, i) << "  " << modelspace->Get_MSmatrix_2m(Proton, i) << std::endl;
        SPEpairs_n[i] = Triple(T_term_n[i * dim_n + i], i, modelspace->Get_MSmatrix_2m(Neutron, i));
    }
    std::sort(SPEpairs_n.begin(), SPEpairs_n.end(), compareTriples);
    for (size_t i = 0; i < N_n; i++)
    {
        holeorbs_n[i] = SPEpairs_n[i].second;
        // std::cout << SPEpairs_n[i].first << "  " << SPEpairs_n[i].second << std::endl;
    }
}

/// for debug **********************************************************
/// check { a^+_i, a_j } = delta_ij
void HartreeFock::Check_orthogonal_U_p(int i, int j)
{
    std::cout << "  The overlap of two vectors is " << cblas_ddot(dim_p, U_p + i, dim_p, U_p + j, dim_p) << std::endl;
}

void HartreeFock::Check_orthogonal_U_n(int i, int j)
{
    std::cout << "  The overlap of two vectors is " << cblas_ddot(dim_n, U_n + i, dim_n, U_n + j, dim_n) << std::endl;
}

void HartreeFock::Check_matrix(int dim, double *Matrix)
{
    std::cout << "   Chcecking matrix:  dim: " << dim << std::endl;
    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << Matrix[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void HartreeFock::PrintParameters_Hole()
{
    std::cout << "   Proton holes:" << std::endl;
    std::cout << "   HO basis  HF basis" << std::endl;
    for (size_t j = 0; j < N_p; j++)
    {
        for (size_t i = 0; i < dim_p; i++)
        {
            std::cout << "   " << std::setw(5) << i << "   " << std::setw(5) << j << "       " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << U_p[i * dim_p + j] << std::endl;
        }
    }
    std::cout << std::endl;
    std::cout << "   Neutron holes:" << std::endl;
    for (size_t j = 0; j < N_n; j++)
    {
        for (size_t i = 0; i < dim_n; i++)
        {
            std::cout << "   " << std::setw(5) << i << "   " << std::setw(5) << j << "       " << std::setw(10) << std::setfill(' ') << std::fixed << std::setprecision(4) << U_n[i * dim_n + j] << std::endl;
        }
    }
    return;
}

void HartreeFock::PrintHoleOrbitsIndex()
{
    std::cout << "  Proton Hole orbits index:" << std::endl;
    for (size_t i = 0; i < N_p; i++)
    {
        std::cout << "  " << holeorbs_p[i];
    }
    std::cout << std::endl;

    std::cout << "  Neutron Hole orbits index:" << std::endl;
    for (size_t i = 0; i < N_n; i++)
    {
        std::cout << "  " << holeorbs_n[i];
    }
    std::cout << std::endl;
    return;
}

void HartreeFock::PrintAllParameters()
{
    std::cout << "   Proton holes:" << std::endl;
    std::cout << "   [HO basis,  HF basis]" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << U_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   Neutron holes:" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << U_n[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
}

void HartreeFock::PrintAllHFEnergies()
{
    Diagonalize();
    std::vector<int> tempOcU_p(dim_p, 0);
    std::vector<int> tempOcU_n(dim_n, 0);
    for (size_t i = 0; i < N_p; i++)
    {
        tempOcU_p[holeorbs_p[i]] = 1;
    }
    for (size_t i = 0; i < N_n; i++)
    {
        tempOcU_n[holeorbs_n[i]] = 1;
    }
    std::cout << std::setw(6) << "\n   index:"
              << "    " << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(7) << std::setw(11) << "SPE"
              << "     " << std::setw(8) << std::setfill(' ') << "Occ." << std::endl;
    std::cout << "   Proton orbits:" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        std::cout << std::setw(6) << i << "    " << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(7) << std::setw(14) << prev_energies[i] << "     " << std::setw(8) << std::setfill(' ') << tempOcU_p[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "   Neutron Orbits:" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        std::cout << std::setw(6) << i << "    " << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(7) << std::setw(14) << prev_energies[i + dim_p] << "     " << std::setw(8) << std::setfill(' ') << tempOcU_n[i] << std::endl;
    }
    std::cout << std::endl;
    return;
}

void HartreeFock::PrintDensity()
{
    std::cout << "   Proton Density:" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << rho_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "   Neutron Density:" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << rho_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    return;
}

void HartreeFock::CheckDensity()
{
    double CalNp, CalNn;
    CalNp = 0;
    CalNn = 0;
    std::cout << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        CalNp += rho_p[i * dim_p + i];
    }
    std::cout << "  Proton  Number :  " << CalNp << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        CalNn += rho_n[i * dim_n + i];
    }

    std::cout << "  Neutron Number :  " << CalNn << std::endl;
    return;
}

void HartreeFock::PrintFockMatrix()
{
    std::cout << "   Proton Fock matrix:" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << FockTerm_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "   Neutron Fock matrix:" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << FockTerm_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    return;
}

void HartreeFock::PrintVtb()
{
    std::cout << "   Proton Vij:" << std::endl;
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << Vij_p[i * dim_p + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "   Neutron Vij:" << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << Vij_n[i * dim_n + j];
        }
        std::cout << std::endl;
    }
    return;
}

void HartreeFock::PrintEHF()
{
    std::cout << std::fixed << std::setprecision(7);
    std::cout << "  One body term = " << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(12) << e1hf << std::endl;
    std::cout << "  Two body term = " << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(12) << 0.5 * e2hf << std::endl;
    std::cout << "  E_HF          = " << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(12) << EHF << std::endl;
}

/// Print out the single particle orbits with their energies.
void HartreeFock::PrintOccupationHO()
{
    std::vector<double> tempOcU_p(dim_p, 0);
    std::vector<double> tempOcU_n(dim_n, 0);
    for (size_t i = 0; i < N_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            tempOcU_p[j] += rho_p[j * dim_p + holeorbs_p[i]];
        }
    }
    for (size_t i = 0; i < N_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            tempOcU_n[j] += rho_n[j * dim_n + holeorbs_n[i]];
        }
    }
    std::cout << "  Proton:" << std::endl;
    std::cout << std::fixed << std::setw(3) << "\n i"
              << ": " << std::setw(3) << "n"
              << " " << std::setw(3) << "l"
              << " "
              << std::setw(3) << "2j"
              << " " << std::setw(3) << "2m"
              << " " << std::setw(3) << "2tz"
              << " " << std::setw(12) << "occ." << std::endl;

    for (size_t i = 0; i < dim_p; i++)
    {
        Orbit &oi = modelspace->GetOrbit(Proton, modelspace->Get_OrbitIndex_Mscheme(i, Proton));
        std::cout << std::fixed << std::setw(3) << i << " " << std::setw(3) << oi.n << " " << std::setw(3) << oi.l << " " << std::setw(3) << oi.j2 << " " << std::setw(3) << modelspace->Get_MSmatrix_2m(Proton, i) << " " << std::setw(3) << oi.tz2 << "   " << std::setw(14) << tempOcU_p[i] << std::endl;
    }
    std::cout << "  Neutron:" << std::endl;
    std::cout << std::fixed << std::setw(3) << "\n i"
              << ": " << std::setw(3) << "n"
              << " " << std::setw(3) << "l"
              << " "
              << std::setw(3) << "2j"
              << " " << std::setw(3) << "2m"
              << " " << std::setw(3) << "2tz"
              << " " << std::setw(12) << "occ." << std::endl;
    for (size_t i = 0; i < dim_n; i++)
    {
        Orbit &oi = modelspace->GetOrbit(Neutron, modelspace->Get_OrbitIndex_Mscheme(i, Neutron));
        std::cout << std::fixed << std::setw(3) << i << " " << std::setw(3) << oi.n << " " << std::setw(3) << oi.l << " " << std::setw(3) << oi.j2 << " " << std::setw(3) << modelspace->Get_MSmatrix_2m(Neutron, i) << " " << std::setw(3) << oi.tz2 << "   " << std::setw(14) << tempOcU_n[i] << std::endl;
    }
}

void HartreeFock::PrintQudrapole()
{
    /// inital Q operator
    double *Q2_p, *Q0_p, *Q_2_p, *Q2_n, *Q0_n, *Q_2_n;
    Q2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q0_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q_2_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Q2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q0_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);
    Q_2_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);

    double Q0_expect = modelspace->GetShapeQ0();
    double Q2_expect = modelspace->GetShapeQ2();
    double deltaQ0, deltaQ2, deltaQ_2;
    double tempQp, tempQn, constrainedQ;

    memset(Q0_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q0_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q0_MSMEs[i];
    }
    memset(Q2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] = Ham->Q2MEs_p.Q2_MSMEs[i];
    }
    memset(Q_2_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_p.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_p.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_p[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_p[index].GetIndex_b();
        Q2_p[ia * dim_p + ib] += Ham->Q2MEs_p.Q_2_MSMEs[i];
    }

    memset(Q0_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q0_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q0_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q0_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q0_MSMEs[i];
    }
    memset(Q2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q2_list[i];          // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] = Ham->Q2MEs_n.Q2_MSMEs[i];
    }
    memset(Q_2_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < Ham->Q2MEs_n.Q_2_list.size(); i++)
    {
        int index = Ham->Q2MEs_n.Q_2_list[i];         // index of M scheme One body operator
        int ia = Ham->MSMEs.OB_n[index].GetIndex_a(); // index of a in M scheme
        int ib = Ham->MSMEs.OB_n[index].GetIndex_b();
        Q2_n[ia * dim_n + ib] += Ham->Q2MEs_n.Q_2_MSMEs[i];
    }
    //////////////////////////////////////////////////////////
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    CalcEHF();

    CalOnebodyOperator(Q0_p, Q0_n, tempQp, tempQn);
    deltaQ0 = tempQn + tempQp;
    CalOnebodyOperator(Q2_p, Q2_n, tempQp, tempQn);
    deltaQ2 = tempQn + tempQp;
    std::cout << "  Q0  " << Q0_expect << "   " << deltaQ0 << "    Q2 " << Q2_expect << "   " << deltaQ2 << std::endl;

    mkl_free(Q2_p);
    mkl_free(Q0_p);
    mkl_free(Q_2_p);
    mkl_free(Q2_n);
    mkl_free(Q0_n);
    mkl_free(Q_2_n);
}

void HartreeFock::Print_Jz()
{
    /// inital Q operator
    double *Jz_p, *Jz_n;
    Jz_p = (double *)mkl_malloc((dim_p * dim_p) * sizeof(double), 64);
    Jz_n = (double *)mkl_malloc((dim_n * dim_n) * sizeof(double), 64);

    memset(Jz_p, 0, (dim_p * dim_p) * sizeof(double));
    for (size_t i = 0; i < dim_p; i++)
    {
        for (size_t j = 0; j < dim_p; j++)
        {
            if (modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(i)].l == modelspace->Orbits_p[modelspace->Get_ProtonOrbitIndexInMscheme(j)].l)
            {
                if (modelspace->Get_MSmatrix_2j(Proton, i) == modelspace->Get_MSmatrix_2j(Proton, j))
                {
                    if (modelspace->Get_MSmatrix_2m(Proton, i) == modelspace->Get_MSmatrix_2m(Proton, j))
                    {
                        Jz_p[i * dim_p + j] = modelspace->Get_MSmatrix_2m(Proton, j) * 0.5;
                    }
                }
            }
        }
    }

    memset(Jz_n, 0, (dim_n * dim_n) * sizeof(double));
    for (size_t i = 0; i < dim_n; i++)
    {
        for (size_t j = 0; j < dim_n; j++)
        {
            if (modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(i)].l == modelspace->Orbits_n[modelspace->Get_NeutronOrbitIndexInMscheme(j)].l)
            {
                if (modelspace->Get_MSmatrix_2j(Neutron, i) == modelspace->Get_MSmatrix_2j(Neutron, j))
                {
                    if (modelspace->Get_MSmatrix_2m(Neutron, i) == modelspace->Get_MSmatrix_2m(Neutron, j))
                    {
                        Jz_n[i * dim_p + j] = modelspace->Get_MSmatrix_2m(Neutron, j) * 0.5;
                    }
                }
            }
        }
    }
    double Total_Jz_p, Total_Jz_n;
    //////////////////////////////////////////////////////////
    UpdateDensityMatrix(); // Update density matrix rho_p and rho_n
    UpdateF();             // Update the Fock matrix
    // CalcEHF();

    CalOnebodyOperator(Jz_p, Jz_n, Total_Jz_p, Total_Jz_n);
    std::cout << "  Expectation value of Jz are  " << Total_Jz_p + Total_Jz_n << "      Jz_p " << Total_Jz_p << "    Jz_n " << Total_Jz_n << std::endl;

    mkl_free(Jz_p);
    mkl_free(Jz_n);
}

void HartreeFock::SaveHoleParameters(string filename)
{
    ReadWriteFiles rw;
    double *prt = (double *)mkl_malloc((N_p * dim_p + N_n * dim_n) * sizeof(double), 64);
    for (size_t i = 0; i < N_p; i++)
        cblas_dcopy(dim_p, U_p + holeorbs_p[i], dim_p, prt + i * dim_p, 1);
    for (size_t i = 0; i < N_n; i++)
        cblas_dcopy(dim_n, U_n + holeorbs_n[i], dim_n, prt + i * dim_n + N_p * dim_p, 1);
    rw.Save_HF_Parameters_TXT(N_p, dim_p, N_n, dim_n, prt, EHF, filename);
    mkl_free(prt);
    return;
}

bool HartreeFock::compareTriples(const Triple &t1, const Triple &t2)
{

    if (fabs(t1.first - t2.first) < 1.e-5)
    {
        if (abs(t1.third) < abs(t2.third))
        {
            return true;
        }
        else
            return false;
    }
    else
    {
        if (t1.first < t2.first)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

// output particle-hole states
// Recursive helper function to generate combinations
void HartreeFock::generateCombinationsRecursive(const std::vector<int> &numbers, std::vector<int> &combination,
                                                int startIndex, int n, std::vector<std::vector<int>> &combinations)
{
    // Base case: combination size is n
    if (combination.size() == n)
    {
        combinations.push_back(combination);
        return;
    }

    // Generate combinations recursively
    for (int i = startIndex; i < numbers.size(); ++i)
    {
        // Include the current number in the combination
        combination.push_back(numbers[i]);

        // Generate combinations with the remaining numbers
        generateCombinationsRecursive(numbers, combination, i + 1, n, combinations);

        // Exclude the current number from the combination
        combination.pop_back();
    }
}

// Function to generate combinations
// Function to generate combinations, pick n numbers from the array numbers
std::vector<std::vector<int>> HartreeFock::generateCombinations(const std::vector<int> &numbers, int n)
{
    std::vector<std::vector<int>> combinations;

    // Generate combinations recursively
    std::vector<int> combination;
    generateCombinationsRecursive(numbers, combination, 0, n, combinations);
    return combinations;
}

std::vector<int> HartreeFock::GetHoleList(int Isospin)
{
    if (Isospin == Proton)
    {
        std::vector<int> Holes_p(N_p, 0);
        for (size_t i = 0; i < N_p; i++)
        {
            Holes_p[i] = holeorbs_p[i];
        }
        return Holes_p;
    }
    else
    {
        std::vector<int> Holes_n(N_n, 0);
        for (size_t i = 0; i < N_n; i++)
        {
            Holes_n[i] = holeorbs_n[i];
        }
        return Holes_n;
    }
}

std::vector<int> HartreeFock::GetParticleList(int Isospin)
{
    if (Isospin == Proton)
    {
        std::vector<int> Particles_p(dim_p);
        std::iota(Particles_p.begin(), Particles_p.end(), 0);
        for (size_t i = 0; i < N_p; i++)
        {
            Particles_p.erase(std::remove(Particles_p.begin(), Particles_p.end(), holeorbs_p[i]), Particles_p.end());
        }
        return Particles_p;
    }
    else
    {
        std::vector<int> Particles_n(dim_n);
        std::iota(Particles_n.begin(), Particles_n.end(), 0);
        for (size_t i = 0; i < N_n; i++)
        {
            Particles_n.erase(std::remove(Particles_n.begin(), Particles_n.end(), holeorbs_n[i]), Particles_n.end());
        }
        return Particles_n;
    }
}

std::vector<int> HartreeFock::ConstructParticleHoleState(int isospin, const std::vector<int> &hole_vec, const std::vector<int> &part_vec)
{
    if (hole_vec.size() != part_vec.size())
    {
        std::cout << "  The particle-hole STATE error! " << std::endl;
    }
    std::vector<int> vectorState;
    if (isospin == Proton)
    {
        vectorState.resize(N_p, 0);
        memcpy(vectorState.data(), holeorbs_p, N_p * sizeof(int));

        // remove hole states
        for (size_t i = 0; i < hole_vec.size(); i++)
        {
            vectorState.erase(std::remove(vectorState.begin(), vectorState.end(), hole_vec[i]), vectorState.end());
        }

        // add particle states
        for (size_t i = 0; i < part_vec.size(); i++)
        {
            vectorState.push_back(part_vec[i]);
        }
    }
    else
    {
        vectorState.resize(N_n, 0);
        memcpy(vectorState.data(), holeorbs_n, N_n * sizeof(int));

        // remove hole states
        for (size_t i = 0; i < hole_vec.size(); i++)
        {
            vectorState.erase(std::remove(vectorState.begin(), vectorState.end(), hole_vec[i]), vectorState.end());
        }

        // add particle states
        for (size_t i = 0; i < part_vec.size(); i++)
        {
            vectorState.push_back(part_vec[i]);
        }
    }
    return vectorState;
}

// save Num particle Num hole states
void HartreeFock::SaveParticleHoleStates(int Num)
{
    string file_path = "Output/";
    if ((Num > N_p or Num > dim_p - N_p) and (N_p > 0))
    {
        std::cout << "  The number of particle-hole excitation is invalid !" << std::endl;
    }

    if ((Num > N_n or Num > dim_n - N_n) and (N_n > 0))
    {
        std::cout << "  The number of particle-hole excitation is invalid !" << std::endl;
    }

    std::vector<int> Hole_p, Hole_n, Particle_p, Particle_n;
    Hole_p = GetHoleList(Proton);
    Hole_n = GetHoleList(Neutron);
    Particle_p = GetParticleList(Proton);
    Particle_n = GetParticleList(Neutron);

    std::vector<std::vector<int>> combinations_hole_p, combinations_hole_n, combinations_part_p, combinations_part_n;
    if (N_p > 0)
    {
        combinations_hole_p = this->generateCombinations(Hole_p, Num);
        combinations_part_p = this->generateCombinations(Particle_p, Num);
    }
    if (N_n > 0)
    {
        combinations_hole_n = this->generateCombinations(Hole_n, Num);
        combinations_part_n = this->generateCombinations(Particle_n, Num);
    }

    // loop all possible combinations
    if (N_p > 0 and N_n > 0)
    {
        int count = 0;
        for (size_t h_p = 0; h_p < combinations_hole_p.size(); h_p++)
        {
            for (size_t h_n = 0; h_n < combinations_hole_n.size(); h_n++)
            {
                for (size_t p_p = 0; p_p < combinations_part_p.size(); p_p++)
                {
                    for (size_t p_n = 0; p_n < combinations_part_n.size(); p_n++)
                    {
                        std::vector<int> proton_vector, neutron_vector;
                        proton_vector = ConstructParticleHoleState(Proton, combinations_hole_p[h_p], combinations_part_p[p_p]);
                        neutron_vector = ConstructParticleHoleState(Neutron, combinations_hole_n[h_n], combinations_part_n[p_n]);
                        double NewE = this->CalcEHF(proton_vector, neutron_vector);
                        string filename = file_path + "HF_ph_" + std::to_string(count) + ".dat";
                        ReadWriteFiles rw;
                        std::vector<double> prt(N_p * dim_p + N_n * dim_n, 0);
                        for (size_t i = 0; i < N_p; i++)
                            cblas_dcopy(dim_p, U_p + proton_vector[i], dim_p, prt.data() + i * dim_p, 1);
                        for (size_t i = 0; i < N_n; i++)
                            cblas_dcopy(dim_n, U_n + neutron_vector[i], dim_n, prt.data() + i * dim_n + N_p * dim_p, 1);
                        rw.Save_HF_Parameters_TXT(N_p, dim_p, N_n, dim_n, prt.data(), NewE, filename);
                        count++;
                    }
                }
            }
        }
    }
    else if (N_p > 0 and N_n == 0)
    {
        /* code */
    }
    else if (N_p == 0 and N_n > 0)
    {
        /* code */
    }
}

///
int main(int argc, char *argv[])
{
    ReadWriteFiles rw;
    ModelSpace MS;
    Hamiltonian Hinput(MS);
    // Read OSLO format interaction
    // rw.Read_OSLO_HF_input("InputFile_OSLO.dat", MS, Hinput);

    // read Kshell format interaction
    rw.Read_KShell_HF_input("Input_HF.txt", MS, Hinput);

    // print information
    // std::cout << std::endl << "  -------------------------------- "  << std::endl;
    // std::cout << "  Dealing nuclei : " << MS.Get_RefString() << std::endl;
    MS.PrintAllParameters_HF();
    Hinput.PrintHamiltonianInfo_pn();

    //----------------------------------------------
    HartreeFock hf(Hinput);
    std::cout << " --------------------------------   Diag" << std::endl;
    hf.Solve_diag();

    std::cout << " --------------------------------   Diag_hybrid with random vector" << std::endl;
    hf.RandomTransformationU(15);
    hf.Solve_hybrid();

    std::cout << " --------------------------------   Gradient" << std::endl;
    hf.Reset_U();
    // hf.RandomTransformationU(15);
    hf.Solve_gradient();
    hf.Print_Jz();
    // hf.PrintQudrapole();
    hf.SaveHoleParameters("Output/HF_para.dat");
    // hf.SaveParticleHoleStates(1);
    // hf.PrintQudrapole();

    std::cout << " --------------------------------   Gradient with constrained" << std::endl;
    hf.Reset_U();
    hf.Solve_gradient_Constraint();
    hf.Print_Jz();

    hf.SaveHoleParameters("Output/HF_para.dat");

    return 0;
}
