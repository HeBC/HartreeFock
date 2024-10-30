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

#include "GCM_Tools.h"
///////////////////////////////////////////////////////
/// class MatrixIndex
MatrixIndex::MatrixIndex(ModelSpace &ms, AngMomProjection &AMJ) // GCM projection
    : ms(&ms), AMproj(&AMJ)
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
  int totalmeshPoints = AMproj->GetTotalMeshPoints();
  Ovl_total = totalmeshPoints * ME_total; // total overlap after projection
  return;
}

MatrixIndex::~MatrixIndex()
{
  mkl_free(MEindex_i);
  mkl_free(MEindex_j);
}
//////////////////////////////////////////////

/// class GCM_Projection
GCM_Projection::GCM_Projection(ModelSpace &ms, Hamiltonian &Ham, AngMomProjection &AngMomProj)
    : ms(&ms), Ham(&Ham), AngMomProj(&AngMomProj)
{
}

GCM_Projection::~GCM_Projection()
{
  if (MyIndex != nullptr)
  {
    delete MyIndex;
  }
  if (basis_stored != nullptr)
  {
    delete[] basis_stored;
  }
};

void GCM_Projection::ReadBasis(string Path) // initial index and array
{
  ReadWriteFiles rw;
  std::vector<string> names;
  std::vector<double> para_x;
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  names = rw.Get_all_files_names_within_folder(Path); // open all directory

  int TotalOrders = names.size() - 2; // count the dimension, name include . and ..
  if (TotalOrders < 1)
  {
    std::cout << "  No input files find in Path " << Path << TotalOrders << std::endl;
    exit(0);
  }
  if (myid == 0)
    std::cout << "  Read " << TotalOrders << " configurations! from " << Path << std::endl;
  ms->SetTotalOrders(TotalOrders);             // set dimension for model space
  MyIndex = new MatrixIndex(*ms, *AngMomProj); // index for GCM
  basis_stored = new PNbasis[TotalOrders];
  for (size_t i = 0; i < TotalOrders; i++)
  {
    PNbasis temp_basis(*ms, *AngMomProj);
    basis_stored[i] = temp_basis;
    rw.Read_GCM_HF_points(Path + names[i + 2], para_x, E_calculated);
    if (para_x.size() != ms->GetProtonNum() * ms->Get_Proton_MScheme_dim() + ms->GetNeutronNum() * ms->Get_Neutron_MScheme_dim())
    {
      std::cout << "  Wrong configration !" << std::endl;
      exit(0);
    }

    basis_stored[i].FullBasis(para_x);
    record_paras.push_back(para_x);
  }
}

void GCM_Projection::Build_Matrix(int dim, ComplexNum *ele, ComplexNum *NewEle)
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

void GCM_Projection::Build_Matrix(int dim, double *ele, double *NewEle)
{
  int i, i1, j1;
  for (i = 0; i < MyIndex->ME_total; i++)
  {
    i1 = MyIndex->MEindex_i[i];
    j1 = MyIndex->MEindex_j[i];
    NewEle[i1 * dim + j1] = ele[i];
    if (i1 != j1)
    {
      NewEle[j1 * dim + i1] = ele[i];
    }
  }
}

std::vector<double> GCM_Projection::EigenValues(int dim, ComplexNum *Ovl, ComplexNum *Ham)
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
    rw.OutputME(dim, tempMat, CheckOvlMatrix);
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
    rw.OutputME(dim, Ovl, CheckOvlMatrix);
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

std::vector<double> GCM_Projection::DealTotalHamiltonianMatrix(ComplexNum *OvlME, ComplexNum *HamME)
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

void GCM_Projection::solveLinearSystem(int n, double *A, double *b, double *x) // A x = b
{
  int *ipiv = (int *)mkl_malloc(n * sizeof(int), 64);
  int info;
  info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, A, n, ipiv, b, 1);
  if (info == 0)
  {
    // Solution successful, retrieve the result
    memcpy(x, b, n * sizeof(double));
  }
  else
  {
    // Solution failed
    // Handle the error
  }
  mkl_free(ipiv);
}

void GCM_Projection::solveLinearSystem(int n, ComplexNum *A, ComplexNum *b, ComplexNum *x) // A x = b
{
  MKL_INT *ipiv = (MKL_INT *)mkl_malloc(n * sizeof(MKL_INT), 64);
  MKL_INT info;

  info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, 1, A, n, ipiv, b, 1);

  if (info == 0)
  {
    // Solution successful, retrieve the result
    memcpy(x, b, n * 2 * sizeof(double));
  }
  else
  {
    // Solution failed
    // Handle the error
  }

  mkl_free(ipiv);
}

void GCM_Projection::SolveLinearEquationMatrix(int NumJ, ComplexNum *OvlME, ComplexNum *HamME)
{
  // Solve Norm part
  ComplexNum *MatrixA, *MatrixXN, *MatrixXH, *Matrixb;
  int M2 = ms->GetAMProjected_M();
  int K2 = ms->GetAMProjected_K();
  MatrixA = (ComplexNum *)mkl_malloc(NumJ * NumJ * sizeof(ComplexNum), 64); //[ d_ij ]  i indicate beta and j indicate J
  MatrixXN = (ComplexNum *)mkl_malloc(NumJ * sizeof(ComplexNum), 64);
  MatrixXH = (ComplexNum *)mkl_malloc(NumJ * sizeof(ComplexNum), 64);
  Matrixb = (ComplexNum *)mkl_malloc(NumJ * sizeof(ComplexNum), 64);
  for (size_t i = 0; i < NumJ; i++)
  {
    for (size_t j = 0; j < NumJ; j++)
    {
      MatrixA[i * NumJ + j] = AngMom::Wigner_d(GCM_results_J2[j], M2, K2, this->AngMomProj->GetBeta_x(i));
    }
    Matrixb[i] = OvlME[i];
  }
  solveLinearSystem(NumJ, MatrixA, Matrixb, MatrixXN);
  for (size_t i = 0; i < NumJ; i++)
  {
    for (size_t j = 0; j < NumJ; j++)
    {
      MatrixA[i * NumJ + j] = AngMom::Wigner_d(GCM_results_J2[j], M2, K2, this->AngMomProj->GetBeta_x(i));
    }
    Matrixb[i] = HamME[i];
  }
  solveLinearSystem(NumJ, MatrixA, Matrixb, MatrixXH);

  for (size_t j = 0; j < NumJ; j++)
  {
    // std::cout << MatrixXH[j] << MatrixXN[j] << std::endl;
    if (std::abs(MatrixXN[j].real()) < 1.e-8)
    {
      GCM_results_E.push_back(0.);
    }
    else
      GCM_results_E.push_back((MatrixXH[j] / MatrixXN[j]).real());
  }

  mkl_free(MatrixA);
  mkl_free(MatrixXH);
  mkl_free(MatrixXN);
  mkl_free(Matrixb);

  // sort Energy
  // Create a vector of pairs, each containing an element from E and the corresponding element from J
  std::vector<std::pair<double, int>> sortedData;
  for (size_t i = 0; i < GCM_results_E.size(); ++i)
  {
    sortedData.push_back(std::make_pair(GCM_results_E[i], GCM_results_J2[i]));
  }

  // Sort the vector of pairs based on the values in E
  std::sort(sortedData.begin(), sortedData.end());

  // Print the sorted E and corresponding J from the sorted vector of pairs
  for (size_t i = 0; i < sortedData.size(); ++i)
  {
    GCM_results_E[i] = sortedData[i].first;
    GCM_results_J2[i] = sortedData[i].second;
  }

  return;
}

void GCM_Projection::CheckConfigruationValid(ComplexNum *OvlME)
{
  int Total_order = ms->GetTotalOrders();
  bool Valid = true;
  for (size_t i = 0; i < Total_order; i++)
  {
    if (std::abs(OvlME[i * Total_order + i]) < 1.e-10)
    {
      std::cout << "\033[31m   There is a invalid configruation! The \033[0m" << i + 1 << "th configuration (1-" << Total_order << ")" << std::endl;
      for (size_t j = 0; j < Total_order; j++)
        std::cout << "   Overlap of the " << i + 1 << "th configuration  " << OvlME[j * Total_order + j] << std::endl;
      exit(0);
    }
  }
}

void GCM_Projection::SaveData(ComplexNum *OvlME, ComplexNum *HamME)
{
  ComplexNum *Matrix1, *Matrix2;
  int Total_Order = ms->GetTotalOrders();
  Matrix1 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
  Matrix2 = (ComplexNum *)mkl_malloc(Total_Order * Total_Order * sizeof(ComplexNum), 64);
  ReadWriteFiles rw;

  if (Total_Order > 1)
  { /// Output Ham ME
    Build_Matrix(Total_Order, OvlME, Matrix1);
    Build_Matrix(Total_Order, HamME, Matrix2);
    rw.OutputME(Total_Order, Matrix1, Output_Ovl_filename);
    rw.OutputME(Total_Order, Matrix2, Output_Ham_filename);
  }
  else
  {
    rw.OutputME(Total_Order, OvlME, Output_Ovl_filename);
    rw.OutputME(Total_Order, HamME, Output_Ham_filename);
  }
  mkl_free(Matrix1);
  mkl_free(Matrix2);
}

void GCM_Projection::DoCalculation() // Do projection for GCM
{
  ComplexNum *MEmatrix, *tempME;
  int myid, numprocs;           // MPI parameters
  int invalid_configration = 0; // check configrations
  int MEind, Rindex;            // matrix index, rotation index
  ComplexNum weightFactor;
  int alpha, beta, gamma;
  int i, MEnum, MEDim, i1, j1;

  for (i = 0; i < ms->GetTotalOrders(); i++)
  {
    GCM_results_J2.push_back(ms->GetAMProjected_J());
  }

  /// MPI inint
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  // check linear dependence
  std::vector<double> value = this->Cal_Overlap_before_Porjection();
  if (myid == 0)
  {
    int index;
    std::cout << "  Eigenvalues of Overlaps without J projection:" << std::endl;
    for (index = 0; index < value.size(); index++)
    {
      std::cout << "      " << index + 1 << "th  configuration :  " << value[index] << std::endl;
      if (value[index] < this->Get_Overlap_dependence())
      {
        invalid_configration = 1;
      }
    }
  }
  MPI_Bcast(&invalid_configration, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (invalid_configration)
  {
    if (myid == 0)
    {
      std::cout << "Input basis are not linear independence! " << myid << std::endl;
    }
    MPI_Finalize();
    exit(0);
  }
  //---------------------------------------------------------------------
  // begin MPI parallel

  int Total_order = ms->GetTotalOrders();
  int ParityProj = ms->GetProjected_parity();
  /// Initial state
  MEnum = MyIndex->ME_total;
  MEDim = Total_order * Total_order;
  /// build array for ME
  MEmatrix = (ComplexNum *)mkl_malloc((MEnum)*2 * sizeof(ComplexNum), 64); // [Ham, Ovl]
  memset(MEmatrix, 0, sizeof(double) * 4 * (MEnum));

  tempME = (ComplexNum *)mkl_malloc((MEnum)*2 * sizeof(ComplexNum), 64);
  memset(tempME, 0, sizeof(double) * 4 * (MEnum));

  ///------- loop ME
  for (i = myid; i < MyIndex->Ovl_total; i += numprocs)
  {

    Rindex = i / MEnum;
    MEind = i % MEnum;
    i1 = MyIndex->MEindex_i[MEind]; // order
    j1 = MyIndex->MEindex_j[MEind]; // should always be TotalOrder - 1
    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->GuassQuad_weight(alpha, beta, gamma);

    PNbasis HFBasis_Bra(basis_stored[i1]);
    PNbasis HFBasis_Ket(basis_stored[j1]);
    HFBasis_Ket.RotatedOperator(alpha, beta, gamma);
    // HFBasis_Ket.GetProtonPrt()->PrintAllParameters_Complex();

    if (ParityProj == 0)
    {
      ComplexNum temp_HamME, temp_OvlME;
      CalHFKernels_Complex(*Ham, HFBasis_Bra, HFBasis_Ket, temp_HamME, temp_OvlME);
      MEmatrix[MEind] += weightFactor * temp_HamME;         // Ham
      MEmatrix[MEind + MEnum] += weightFactor * temp_OvlME; // Ovl

      // output
      // std::cout << alpha << "  " << this->AngMomProj->GetAlpha_x(alpha) << "  " << beta << "  " << this->AngMomProj->GetBeta_x(beta) << "  " << gamma << "  " << this->AngMomProj->GetGamma_x(gamma) << "   " << temp_HamME.real() << "  " << temp_HamME.imag() << "   |   " << temp_OvlME.real() << "  " << temp_OvlME.imag() << "   " << weightFactor.real()  << "  "<< weightFactor.imag()  << std::endl;

      //////////////////// Testing derivative
      /*
      double h_diff = 0.0001;
      ComplexNum derivative_a, derivative_b, derivative_c;
      /// alpha
      PNbasis HFBasis_Ket_new(basis_stored[j1]);
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI + h_diff, this->AngMomProj->GetBeta_x(beta) * PI, this->AngMomProj->GetGamma_x(gamma) * 2 * PI);
      derivative_a = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);
      derivative_a -= temp_OvlME;
      derivative_a /= h_diff;

      /// beta
      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI, this->AngMomProj->GetBeta_x(beta) * PI + h_diff, this->AngMomProj->GetGamma_x(gamma) * 2 * PI);
      derivative_b = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);
      derivative_b -= temp_OvlME;
      derivative_b /= h_diff;

      /// gamma
      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI, this->AngMomProj->GetBeta_x(beta) * PI, this->AngMomProj->GetGamma_x(gamma) * 2 * PI + h_diff);
      derivative_c = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);
      derivative_c -= temp_OvlME;
      derivative_c /= h_diff;


      std::cout << alpha << "  " << this->AngMomProj->GetAlpha_x(alpha) << "  " << beta << "  " << this->AngMomProj->GetBeta_x(beta) << "  " << gamma << "  " << this->AngMomProj->GetGamma_x(gamma) << "   " << temp_HamME.real() << "  " << temp_HamME.imag() << "   |   " << temp_OvlME.real() << "  " << temp_OvlME.imag() << "   " << weightFactor.real() << "  " << weightFactor.imag() << "   |   " << derivative_a.real() << "  " << derivative_a.imag() << "  " << derivative_b.real() << "  " << derivative_b.imag() << "  " << derivative_c.real() << "  " << derivative_c.imag() << std::endl;
      */
      // derivative
      /*
      double h_diff = 0.0001;
      ComplexNum derivative_a, derivative_b, derivative_c;
      ComplexNum derivative_abh, derivative_bch, derivative_ach;
      ComplexNum derivative_a_h, derivative_b_h, derivative_c_h;

      ComplexNum derivative_aa, derivative_bb, derivative_cc;
      ComplexNum derivative_ab, derivative_bc, derivative_ac;

      /// alpha
      PNbasis HFBasis_Ket_new(basis_stored[j1]);
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI + h_diff, this->AngMomProj->GetBeta_x(beta) * PI, this->AngMomProj->GetGamma_x(gamma) * 2 * PI);
      derivative_a = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI - h_diff, this->AngMomProj->GetBeta_x(beta) * PI, this->AngMomProj->GetGamma_x(gamma) * 2 * PI);
      derivative_a_h = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      /// beta
      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI, this->AngMomProj->GetBeta_x(beta) * PI + h_diff, this->AngMomProj->GetGamma_x(gamma) * 2 * PI);
      derivative_b = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI, this->AngMomProj->GetBeta_x(beta) * PI - h_diff, this->AngMomProj->GetGamma_x(gamma) * 2 * PI);
      derivative_b_h = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      /// gamma
      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI, this->AngMomProj->GetBeta_x(beta) * PI, this->AngMomProj->GetGamma_x(gamma) * 2 * PI + h_diff);
      derivative_c = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI, this->AngMomProj->GetBeta_x(beta) * PI, this->AngMomProj->GetGamma_x(gamma) * 2 * PI - h_diff);
      derivative_c_h = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      ///
      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI + h_diff, this->AngMomProj->GetBeta_x(beta) * PI + h_diff, this->AngMomProj->GetGamma_x(gamma) * 2 * PI);
      derivative_abh = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI, this->AngMomProj->GetBeta_x(beta) * PI + h_diff, this->AngMomProj->GetGamma_x(gamma) * 2 * PI + h_diff);
      derivative_bch = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      HFBasis_Ket_new = basis_stored[j1];
      HFBasis_Ket_new.RotatedOperator(this->AngMomProj->GetAlpha_x(alpha) * 2 * PI + h_diff, this->AngMomProj->GetBeta_x(beta) * PI, this->AngMomProj->GetGamma_x(gamma) * 2 * PI + h_diff);
      derivative_ach = CalHFOverlap_Complex(HFBasis_Bra, HFBasis_Ket_new);

      //============================================
      derivative_aa = (-derivative_a_h + 2. * temp_OvlME - derivative_a) / h_diff / h_diff;
      derivative_bb = (-derivative_b_h + 2. * temp_OvlME - derivative_b) / h_diff / h_diff;
      derivative_cc = (-derivative_c_h + 2. * temp_OvlME - derivative_c) / h_diff / h_diff;
      derivative_ab = ( derivative_abh - derivative_a - derivative_b + temp_OvlME ) / h_diff / h_diff;
      derivative_bc = ( derivative_bch - derivative_b - derivative_c + temp_OvlME ) / h_diff / h_diff;
      derivative_ac = ( derivative_ach - derivative_a - derivative_c + temp_OvlME ) / h_diff / h_diff;

      //--------------------------------------------
      derivative_a -= temp_OvlME;
      derivative_a /= h_diff;

      derivative_b -= temp_OvlME;
      derivative_b /= h_diff;

      derivative_c -= temp_OvlME;
      derivative_c /= h_diff;

      std::cout << alpha << "  " << this->AngMomProj->GetAlpha_x(alpha) << "  " << beta << "  " << this->AngMomProj->GetBeta_x(beta) << "  " << gamma << "  " << this->AngMomProj->GetGamma_x(gamma) << "   " << temp_HamME.real() << "  " << temp_HamME.imag() << "   |   " << temp_OvlME.real() << "  " << temp_OvlME.imag() << "   " << weightFactor.real() << "  " << weightFactor.imag() << "   |   " << derivative_a.real() << "  " << derivative_a.imag() << "  " << derivative_b.real() << "  " << derivative_b.imag() << "  " << derivative_c.real() << "  " << derivative_c.imag() << "  |  " << derivative_aa.real() << "  " << derivative_bb.real() << "  " << derivative_cc.real() << "  " << derivative_ab.real() << "  " << derivative_bc.real() << "  " << derivative_ac.real() << std::endl;
      */
      //////////////////// End Test
    }
    else
    {
      // \hat P = 1/2 ( 1 +- \hat Pi )
      ComplexNum temp_HamME, temp_OvlME;
      CalHFKernels_Complex(*Ham, HFBasis_Bra, HFBasis_Ket, temp_HamME, temp_OvlME);
      MEmatrix[MEind] += 0.5 * weightFactor * temp_HamME;         // Ham
      MEmatrix[MEind + MEnum] += 0.5 * weightFactor * temp_OvlME; // Ovl

      HFBasis_Ket.ParityProjection();
      CalHFKernels_Complex(*Ham, HFBasis_Bra, HFBasis_Ket, temp_HamME, temp_OvlME);
      MEmatrix[MEind] += ParityProj * 0.5 * weightFactor * temp_HamME;         // Ham
      MEmatrix[MEind + MEnum] += ParityProj * 0.5 * weightFactor * temp_OvlME; // Ovl
    }
  }

  //----------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(MEmatrix, tempME, MEnum * 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myid == 0)
  {
    // for (size_t i = 0; i < MEnum; i++)
    //    std::cout << "     " << tempME[i] << "   " << MEmatrix[i] << "  " << MyIndex->MEindex_i[i] << " " << MyIndex->MEindex_j[i] << "  " << myid << std::endl;
    memset(MEmatrix, 0, sizeof(double) * 4 * (MEnum));
    cblas_zcopy(MEnum, tempME + MEnum, 1, MEmatrix, 1);
    SaveData(MEmatrix, tempME);
    CheckConfigruationValid(MEmatrix);
    GCM_results_E = DealTotalHamiltonianMatrix(MEmatrix, tempME); // input ovl and Ham
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /// free array
  mkl_free(MEmatrix);
  mkl_free(tempME);
  return;
}

void GCM_Projection::DoCalculation_LAmethod() // Do projection for GCM with linear algebra method
{
  ComplexNum *MEmatrix, *tempME;
  int myid, numprocs;           // MPI parameters
  int invalid_configration = 0; // check configrations
  int MEind, Rindex;            // matrix index, rotation index
  ComplexNum weightFactor;
  int alpha, beta, gamma;
  int i1, j1;

  /// MPI inint
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  // check linear dependence
  std::vector<double> value = this->Cal_Overlap_before_Porjection();
  if (myid == 0)
  {
    int index;
    std::cout << "  Eigenvalues of Overlaps without J projection:" << std::endl;
    for (index = 0; index < value.size(); index++)
    {
      std::cout << "      " << index + 1 << "th  configuration :  " << value[index] << std::endl;
      if (value[index] < this->Get_Overlap_dependence())
      {
        invalid_configration = 1;
      }
    }
  }
  MPI_Bcast(&invalid_configration, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (invalid_configration)
  {
    if (myid == 0)
    {
      std::cout << "Input basis are not linear independence! " << myid << std::endl;
    }
    MPI_Finalize();
    exit(0);
  }
  //---------------------------------------------------------------------
  // begin MPI parallel
  int i, MEnum, MEDim;
  int Total_order = ms->GetTotalOrders();
  if (Total_order > 1)
  {
    std::cout << "  To be developed in the upcoming future! " << std::endl;
    exit(0);
  }
  int ParityProj = ms->GetProjected_parity();
  /// Initial state
  MEnum = MyIndex->ME_total;
  MEDim = Total_order * Total_order;
  int NumJ;
  // int minJ = std::max(abs(ms->GetAMProjected_M()), abs(ms->GetAMProjected_K()));
  if (ms->GetAMProjected_J() % 2) // odd
  {
    NumJ = (ms->GetAMProjected_J() + 1) / 2;
    for (i = 0; i < NumJ; i++)
    {
      GCM_results_J2.push_back(i * 2 + 1);
    }
  }
  else // even
  {
    NumJ = (ms->GetAMProjected_J() / 2 + 1);
    for (i = 0; i < NumJ; i++)
    {
      GCM_results_J2.push_back(i * 2);
    }
  }
  /// build array for ME
  MEmatrix = (ComplexNum *)mkl_malloc((MEnum * NumJ) * 2 * sizeof(ComplexNum), 64); // [Ham, Ovl]
  memset(MEmatrix, 0, sizeof(double) * 4 * (MEnum * NumJ));
  tempME = (ComplexNum *)mkl_malloc((MEnum * NumJ) * 2 * sizeof(ComplexNum), 64);
  memset(tempME, 0, sizeof(double) * 4 * (MEnum * NumJ));
  ///------- loop ME
  for (i = myid; i < MyIndex->Ovl_total; i += numprocs)
  {

    Rindex = i / MEnum;
    MEind = i % MEnum;
    i1 = MyIndex->MEindex_i[MEind]; // index of bra
    j1 = MyIndex->MEindex_j[MEind]; // index of ket
    /// Rotation index
    alpha = this->AngMomProj->GetIndex_Alpha(Rindex);
    beta = this->AngMomProj->GetIndex_Beta(Rindex);
    gamma = this->AngMomProj->GetIndex_Gamma(Rindex);
    weightFactor = AngMomProj->LinearAlgebra_weight(alpha, gamma);

    PNbasis HFBasis_Bra(basis_stored[i1]);
    PNbasis HFBasis_Ket(basis_stored[j1]);
    HFBasis_Ket.RotatedOperator(alpha, beta, gamma);
    // HFBasis_Ket.GetProtonPrt()->PrintAllParameters_Complex();

    if (ParityProj == 0)
    {
      ComplexNum temp_HamME, temp_OvlME;
      CalHFKernels_Complex(*Ham, HFBasis_Bra, HFBasis_Ket, temp_HamME, temp_OvlME);
      MEmatrix[MEind * MEnum + beta] += weightFactor * temp_HamME;                // Ham
      MEmatrix[MEnum * NumJ + MEind * MEnum + beta] += weightFactor * temp_OvlME; // Ovl
      // std::cout << alpha << "  " << this->AngMomProj->GetAlpha_x(alpha) << "  " << beta << "  " << this->AngMomProj->GetBeta_x(beta) << "  " << gamma << "  " << this->AngMomProj->GetGamma_x(gamma) << "   " << temp_HamME.real() << "  " << temp_HamME.imag() << "   |   " << temp_OvlME.real() << "  " << temp_OvlME.imag() << "   " << weightFactor.real()  << "  "<< weightFactor.imag()  << std::endl;
    }
    else
    {
      // \hat P = 1/2 ( 1 +- \hat Pi )
      ComplexNum temp_HamME, temp_OvlME;
      CalHFKernels_Complex(*Ham, HFBasis_Bra, HFBasis_Ket, temp_HamME, temp_OvlME);
      MEmatrix[MEind * MEnum + beta] += 0.5 * weightFactor * temp_HamME;                // Ham
      MEmatrix[MEnum * NumJ + MEind * MEnum + beta] += 0.5 * weightFactor * temp_OvlME; // Ovl

      HFBasis_Ket.ParityProjection();
      CalHFKernels_Complex(*Ham, HFBasis_Bra, HFBasis_Ket, temp_HamME, temp_OvlME);
      MEmatrix[MEind * MEnum + beta] += ParityProj * 0.5 * weightFactor * temp_HamME;                // Ham
      MEmatrix[MEnum * NumJ + MEind * MEnum + beta] += ParityProj * 0.5 * weightFactor * temp_OvlME; // Ovl
    }
  }

  //----------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  // for (size_t i = 0; i < MEnum; i++)
  //   std::cout << "     " << tempME[i] << "   " << MEmatrix[i] << "  " << MyIndex->MEindex_i[i] << " " << MyIndex->MEindex_j[i] << "  " << myid << std::endl;
  MPI_Reduce(MEmatrix, tempME, MEnum * NumJ * 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myid == 0)
  {
    memset(MEmatrix, 0, sizeof(double) * 4 * (MEnum * NumJ));
    cblas_zcopy(MEnum * NumJ, tempME + MEnum * NumJ, 1, MEmatrix, 1);
    SolveLinearEquationMatrix(NumJ, MEmatrix, tempME);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /// free array
  mkl_free(MEmatrix);
  mkl_free(tempME);
  return;
}

std::vector<double> GCM_Projection::Cal_Overlap_before_Porjection()
{
  double *OvlME, *tempME;
  int myid, numprocs; // MPI parameters
  int i, MEnum, MEDim, MEind;
  std::vector<double> tempval;
  int Total_order = ms->GetTotalOrders();
  /// Initial state
  MEnum = MyIndex->ME_total;
  MEDim = Total_order * Total_order;
  /// build array for ME
  OvlME = (double *)mkl_malloc((MEnum) * sizeof(double), 64);
  memset(OvlME, 0, sizeof(double) * (MEnum));
  tempME = (double *)mkl_malloc((MEnum) * sizeof(double), 64);
  memset(tempME, 0, sizeof(double) * (MEnum));

  /// MPI inint
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  ///------- loop ME
  for (i = myid; i < MEnum; i += numprocs)
  {
    int i1, j1;
    MEind = i % MyIndex->ME_total;
    i1 = MyIndex->MEindex_i[MEind]; // order
    j1 = MyIndex->MEindex_j[MEind]; // should always be TotalOrder - 1
    PNbasis MPBasis_Bra(basis_stored[i1]);
    PNbasis MPBasis_Ket(basis_stored[j1]);
    OvlME[i] = CalHFOverlap_Complex(MPBasis_Bra, MPBasis_Ket).real();
  }
  MPI_Reduce(OvlME, tempME, MEnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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

std::vector<double> GCM_Projection::AnalysisOverlap(double *Ovl)
{
  ReadWriteFiles rw;
  int dim = ms->GetTotalOrders();
  int i, j;
  double *e;
  std::vector<double> returu_value;
  double *Matrix1;
  Matrix1 = (double *)mkl_malloc(dim * dim * sizeof(double), 64);
  Build_Matrix(dim, Ovl, Matrix1);
  e = (double *)mkl_malloc((dim) * sizeof(double), 64);
  /// Tridiag Ovl
  if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', dim, Matrix1, dim, e) != 0)
  { /// 'V' stand for eigenvalues and vectors
    printf("Error when Computes all eigenvalues and eigenvectors in Cal-Overlap procedure!!\n");
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

void GCM_Projection::PrintResults()
{
  int myid; // MPI parameters
  /// MPI inint
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid == 0)
  {
    std::cout << "/-----------------------------------------------------/" << std::endl;
    std::cout << "  GCM results output:" << std::endl;
    if (ms->GetAMProjected_J() % 2)
    {
      for (size_t i = 0; i < GCM_results_J2.size(); i++)
      {
        std::cout << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(8) << GCM_results_E[i] << "   J = " << GCM_results_J2[i] << "/2" << std::endl;
      }
    }
    else
    {
      for (size_t i = 0; i < GCM_results_J2.size(); i++)
      {
        std::cout << std::setw(18) << std::setfill(' ') << std::fixed << std::setprecision(8) << GCM_results_E[i] << "   J = " << GCM_results_J2[i] / 2 << std::endl;
      }
    }
    std::cout << "/-----------------------------------------------------/" << std::endl;
  }
  return;
}

void GCM_Projection::PrintInfo()
{
  int myid; // MPI parameters
  /// MPI inint
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid == 0)
  {
    ms->PrintAllParameters_HF();
    Ham->PrintHamiltonianInfo_pn();
    AngMomProj->PrintInfo();
  }
  return;
}

void GCM_Projection::Do_Projection()
{
    if (ms->Get_MeshType() == "LAmethod")
    {
      DoCalculation_LAmethod();
    }
    else
    {
      DoCalculation();
    }
}



// GCM-MPI function for pybind
// Initialize MPI
void mpi_initialize() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        MPI_Init(nullptr, nullptr);
    }
}

// Finalize MPI
void mpi_finalize() {
    int is_finalized;
    MPI_Finalized(&is_finalized);
    if (!is_finalized) {
        MPI_Finalize();
    }
}


