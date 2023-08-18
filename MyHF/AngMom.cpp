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
#include "ModelSpace.h"
#include "mkl.h"
#include "AngMom.h"
// GSL
#include <gsl/gsl_integration.h>

double step(int i)
{
  int j;
  double res;

  if (i < 0)
    return 0;
  if (i <= 1)
    return 1;
  res = 1;
  for (j = 2; j < i + 1; j++)
    res *= j;
  return res;
}

double gma[150], sqg[150];

double AngMom::threej(double j1, double j2, double j3,
                      double m1, double m2, double m3)
{
  int k;
  double fact;
  double sum;
  double denom;
  int stop;
  int phase;

  fact = (step(j1 + j2 - j3) * step(j1 - j2 + j3) * step(-j1 + j2 + j3) * step(j1 + m1) *
          step(j1 - m1) * step(j2 + m2) * step(j2 - m2) * step(j3 + m3) *
          step(j3 - m3)) /
         step(j1 + j2 + j3 + 1);
  phase = j1 - j2 - m3;
  fact = sgn(phase) * sqrt(fact);
  stop = 0;
  sum = 0;
  for (k = 0; k <= j1 - m1; k++)
  {
    denom = (step(k) * step(j1 + j2 - j3 - k) * step(j1 - m1 - k) * step(j2 + m2 - k) *
             step(j3 - j2 + m1 + k) * step(j3 - j1 - m2 + k));
    if (fabs(denom) > 1e-15)
    {
      sum += sgn(k) / denom;
      stop = 1;
    }
    else if (stop)
      break;
  }
  sum *= fact;
  return sum;
}

double AngMom::cgc(double j3, double m3, double j1, double m1, double j2, double m2)
{
  double res;
  int phase;

  phase = j2 - j1 - m3;
  res = threej(j1, j2, j3, m1, m2, -m3);
  res *= sgn(phase) * sqrt(2 * j3 + 1);
  return res;
}

void AngMom::init_sixj()
{
  int i;
  gma[0] = sqg[0] = 0;
  for (i = 2; i < 150; i += 2)
  {
    gma[i] = gma[i - 2] + log((double)i / 2);
    sqg[i] = gma[i] / 2;
  }
  return;
}

double AngMom::sixJ(int a, int b, int e, int d, int c, int f) /* in units of 2*j */
{
  int i;
  double res;

  res = U(a, b, c, d, e, f);
  if (res == 0)
    return 0;
  i = a + b + c + d;
  i /= 2;
  return sgn(i) * res / (sqrt(e + 1) * sqrt(f + 1));
}

double AngMom::U(int a, int b, int c, int d, int e, int f)
{
  double fact;
  int k;
  double sum;
  double denom, term;
  double del;
  int stop;
  int tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  int kmin, kmax;

  /*  i = rint(a+b+d+c); */

  tmp1 = (a + b - e);
  tmp2 = (d + c - e);
  tmp3 = (a + c - f);
  tmp4 = (d + b - f);
  if (tmp1 < 0 || b + e < a || a + e < b || tmp2 < 0 || c + e < d || d + e < c || tmp4 < 0 || b + f < d || d + f < b || tmp3 < 0 || c + f < a || a + f < c)
    return 0;
  if (a == 0 || b == 0 || c == 0 || d == 0)
    return 1;
  kmax = AngMomMin(tmp1, AngMomMin(tmp2, AngMomMin(tmp3, tmp4)));
  sum = 0;
  stop = 0;
  tmp5 = (e + f - a - d);
  tmp6 = (e + f - b - c);
  kmin = AngMomMax(AngMomMax(-tmp5, -tmp6), 0);
  tmp7 = (a + b + c + d) + 2;
  if (kmin > kmax)
    return 0;
  denom = (gma[tmp7 - kmin] - gma[tmp1 - kmin] - gma[tmp2 - kmin] - gma[tmp3 - kmin] -
           gma[tmp4 - kmin] - gma[tmp5 + kmin] - gma[tmp6 + kmin] - gma[kmin]);
  term = exp(denom) * sgn(kmin / 2);
  sum = term;
  for (k = kmin + 2; k <= kmax; k += 2)
  {
    term *= -(double)((tmp1 - k + 2) * (tmp2 - k + 2) * (tmp3 - k + 2) *
                      (tmp4 - k + 2)) /
            (double)(k * (tmp7 - k + 2) * (tmp5 + k) * (tmp6 + k));
    sum += term;
  }
  if (sum == 0)
    return 0;
  del = logdel(a, b, e) + logdel(d, c, e) + logdel(d, b, f) + logdel(a, c, f);
  fact = sqrt(e + 1) * sqrt(f + 1) * exp(del);
  sum *= fact;
  return sum;
}

double AngMom::Factorial(int N) /* Factorial function; assumes N >= 0 */
{
  double product;
  int i;
  product = 1;
  if (N < 0)
  {
    printf("double Factorial function error  %d\n\n", N);
    exit(0);
  }
  else if (N <= 1)
    return product;
  else
  {
    for (i = 2; i <= N; i++)
    {
      product *= i;
    }
    return product;
  }
}

double AngMom::Wigner_d(int j, int m1, int m2, double beta) // Wigner small d function
{                                                           /// j, m1 and m2 are twiced, beta in unit of Pi
  double value, temp;
  int upperl_x, lowerl_x;
  int x;
  value = 0;
  lowerl_x = std::max((m2 - m1) / 2, 0);
  upperl_x = std::min((j + m2) / 2, (j - m1) / 2);
  for (x = lowerl_x; x <= upperl_x; x++)
  {
    temp = sgn((m1 - m2) / 2 + x) * pow(cos(PI * beta / 2.), j - 2 * x + (m2 - m1) / 2) * pow(sin(PI * beta / 2.), 2 * x + (m1 - m2) / 2);
    value += temp / (Factorial((j + m2) / 2 - x) * Factorial(x) * Factorial((m1 - m2) / 2 + x) * Factorial((j - m1) / 2 - x));
  }
  value *= sqrt(Factorial((j + m1) / 2) * Factorial((j - m1) / 2) * Factorial((j + m2) / 2) * Factorial((j - m2) / 2));
  return value;
}

ComplexNum AngMom::Wigner_D(int j, int m1, int m2, double alpha, double beta, double gamma) // Wigner small d function
{                                                                                           /// j, m1 and m2 are twiced, beta in unit of Pi
  double value, small_d;
  ComplexNum factor, expPart;
  expPart = -1.i * ((m1 * alpha + m2 * gamma) * PI);
  factor = std::exp(expPart);
  small_d = Wigner_d(j, m1, m2, beta);
  return value * factor;
}

///////////////////////////////////////////
// Angular momentum projection

QuadratureClass::~QuadratureClass()
{
  if (mesh_x != nullptr)
  {
    mkl_free(mesh_x);
    mkl_free(mesh_w);
  }
}

void QuadratureClass::MallocMemory()
{
  if (this->GetTotalNumber() != 0)
  {
    this->mesh_x = (double *)mkl_malloc((this->GetTotalNumber()) * sizeof(double), 64);
    this->mesh_w = (double *)mkl_malloc((this->GetTotalNumber()) * sizeof(double), 64);
  }
}

AngMomProjection::~AngMomProjection()
{
  if (initial_proton)
  {
    mkl_free(RotatePairs_p);
  }
  if (initial_neutron)
  {
    mkl_free(RotatePairs_n);
  }
  if (initial_proton or initial_neutron)
  {
    mkl_free(WDTab);
  }
}

void AngMomProjection::ReadMesh()
{
  //////////////////// File names

  FILE *fp;
  int i, j, k, temp;
  int j1, j2, m1, m2;
  int index1, index2;
  /// Alpha
  /// Abscissa
  fp = fopen(FileName_ax.c_str(), "r");
  if (fscanf(fp, "%d", GQAlpha.GetNumberPointer()) != 1)
  {
    printf("number of GQAlpha mesh point reading error 1!\n");
    exit(0);
  }
  GQAlpha.MallocMemory();
  for (i = 0; i < GQAlpha.GetTotalNumber(); i++)
  {
    if (fscanf(fp, "%lf", &(GQAlpha.GetXpointer()[i])) != 1)
    {
      printf("Abscissa of GQAlpha mesh point reading error!\n");
      exit(0);
    }
    // printf("%d   %lf\n", i, GQAlpha.GetXpointer()[i]);
  }
  fclose(fp);

  /// weight
  fp = fopen(FileName_aw.c_str(), "r");
  if (fscanf(fp, "%d", &i) != 1 || i != GQAlpha.GetTotalNumber())
  {
    printf("number of GQAlpha mesh point reading error 2!\n");
    exit(0);
  }
  for (i = 0; i < GQAlpha.GetTotalNumber(); i++)
  {
    if (fscanf(fp, "%lf", &(GQAlpha.GetWeightPointer()[i])) != 1)
    {
      printf("Weight of GQAlpha mesh point reading error!  %d\n", i);
      exit(0);
    }
    // printf("%d   %lf\n", i, GQAlpha.GetWeightPointer()[i]);
  }
  fclose(fp);

  ///--------------------------------------------
  /// Beta
  /// Abscissa
  fp = fopen(FileName_bx.c_str(), "r");
  if (fscanf(fp, "%d", GQBeta.GetNumberPointer()) != 1)
  {
    printf("number of GQBeta mesh point reading error 1!\n");
    exit(0);
  }
  GQBeta.MallocMemory();
  for (i = 0; i < GQBeta.GetTotalNumber(); i++)
  {
    if (fscanf(fp, "%lf", &(GQBeta.GetXpointer()[i])) != 1)
    {
      printf("Abscissa of GQBeta mesh point reading error!\n");
      exit(0);
    }
    // printf("%d   %lf\n", i, GQBeta.GetXpointer()[i]);
  }
  fclose(fp);

  /// weight
  fp = fopen(FileName_bw.c_str(), "r");
  if (fscanf(fp, "%d", &i) != 1 || i != GQBeta.GetTotalNumber())
  {
    printf("number of GQBeta mesh point reading error 2!\n");
    exit(0);
  }
  for (i = 0; i < GQBeta.GetTotalNumber(); i++)
  {
    if (fscanf(fp, "%lf", &(GQBeta.GetWeightPointer()[i])) != 1)
    {
      printf("Weight of GQBeta mesh point reading error!  %d\n", i);
      exit(0);
    }
    // printf("%d   %lf\n", i, GQBeta.GetWeightPointer()[i]);
  }
  fclose(fp);

  ///--------------------------------------------------
  /// Gamma
  /// Abscissa
  fp = fopen(FileName_cx.c_str(), "r");
  if (fscanf(fp, "%d", GQGamma.GetNumberPointer()) != 1)
  {
    printf("number of GQGamma mesh point reading error 1!\n");
    exit(0);
  }
  GQGamma.MallocMemory();
  for (i = 0; i < GQGamma.GetTotalNumber(); i++)
  {
    if (fscanf(fp, "%lf", &(GQGamma.GetXpointer()[i])) != 1)
    {
      printf("Abscissa of GQGamma mesh point reading error!\n");
      exit(0);
    }
    // printf("%d   %lf\n", i, GQGamma.GetXpointer()[i]);
  }
  fclose(fp);

  /// weight
  fp = fopen(FileName_cw.c_str(), "r");
  if (fscanf(fp, "%d", &i) != 1 || i != GQGamma.GetTotalNumber())
  {
    printf("number of GQGamma mesh point reading error 2!\n");
    exit(0);
  }
  for (i = 0; i < GQGamma.GetTotalNumber(); i++)
  {
    if (fscanf(fp, "%lf", &(GQGamma.GetWeightPointer()[i])) != 1)
    {
      printf("Weight of GQGamma mesh point reading error!  %d\n", i);
      exit(0);
    }
    // printf("%d   %lf\n", i, GQGamma.GetWeightPointer()[i]);
  }
  fclose(fp);

  ///
  GQdim = GQAlpha.GetTotalNumber() * GQBeta.GetTotalNumber() * GQGamma.GetTotalNumber();
  for (i = 0; i < GQAlpha.GetTotalNumber(); i++)
  {
    for (j = 0; j < GQBeta.GetTotalNumber(); j++)
    {
      for (k = 0; k < GQGamma.GetTotalNumber(); k++)
      {
        GasussQuadMap.push_back(i); // 3 * temp + 0/
        GasussQuadMap.push_back(j); // 3 * temp + 1/
        GasussQuadMap.push_back(k); // 3 * temp + 2/
      }
    }
  }
}

void AngMomProjection::Generate_GQ_Mesh(QuadratureClass &QCprt, std::string type) /// generate Gauss quadrature
{
  int n = QCprt.GetTotalNumber();
  gsl_integration_fixed_workspace *table;
  if (type == "legendre")
  {
    // This specifies Legendre quadrature integration. The parameters alpha and beta are ignored for this type.
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_legendre, n, 0, 1., 0., 0.);
  }
  else if (type == "chebyshev")
  {
    // This specifies Chebyshev type 1 quadrature integration. The parameters alpha and beta are ignored for this type.
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_chebyshev, n, 0, 1., 0., 0.);
  }
  else if (type == "gegenbauer")
  {
    // This specifies Gegenbauer quadrature integration. The parameter beta is ignored for this type.
    // alpha is fixed as 1.
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_gegenbauer, n, 0, 1., 1., 0.);
  }
  else if (type == "jacobi")
  {
    // This specifies Jacobi quadrature integration.
    // alpha is fixed as 1., beta is fixed as 1.
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_jacobi, n, 0, 1., 1., 1.);
  }
  else if (type == "laguerre")
  {
    // This specifies Jacobi quadrature integration.
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_laguerre, n, 0, 1., 1., 0.);
  }
  else if (type == "hermite")
  {
    // This specifies Hermite quadrature integration. The parameter beta is ignored for this type.
    // alpha is fixed as 1.,
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_hermite, n, 0, 1., 1., 0.);
  }
  else if (type == "exponential")
  {
    // This specifies exponential quadrature integration. The parameter beta is ignored for this type.
    // alpha is fixed as 1.,
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_exponential, n, 0, 1., 1., 0.);
  }
  else if (type == "rational")
  {
    // This specifies rational quadrature integration.
    // alpha is fixed as 1.,  beta is fixed as 1.
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_rational, n, 0, 1., 1., 0.);
  }
  else if (type == "chebyshev2")
  {
    // This specifies Chebyshev type 2 quadrature integration. The parameters alpha and beta are ignored for this type.
    table = gsl_integration_fixed_alloc(gsl_integration_fixed_chebyshev2, n, 0, 1., 0., 0.);
  }
  else
  {
    std::cout << " Unknow quadrature  type! " << type << std::endl;
    exit(0);
  }

  // get the abscissae and weights from the table
  const double *x = gsl_integration_fixed_nodes(table);   // pointer to the abscissae
  const double *w = gsl_integration_fixed_weights(table); // pointer to the weights
  double *Xprt, *Wprt;
  QCprt.MallocMemory();
  Xprt = QCprt.GetXpointer();
  Wprt = QCprt.GetWeightPointer();
  for (int i = 0; i < n; i++)
  {
    Xprt[i] = x[i];
    Wprt[i] = w[i];
    // std::cout << i << "  " << Xprt[i] << "   " << Wprt[i] << std::endl;
  }
  gsl_integration_fixed_free(table);
}

void AngMomProjection::Generate_LA_Mesh(QuadratureClass &QCprt, std::string type) /// generate linear algebra
// find more detail in    J. Phys. G: Nucl. Part. Phys. 46 (2019) 015101
{
  double *Xprt, *Wprt;
  QCprt.MallocMemory();
  Xprt = QCprt.GetXpointer();
  Wprt = QCprt.GetWeightPointer();
  int Jmax2 = ms->GetAMProjected_J();
  /// for angle alpha and beta
  if (type == "alpha" or type == "gamma")
  {
    int mu;
    int Nmesh = Jmax2 + 1;
    if ((Jmax2 + 1) % 4 == 1) // J_max = 0, 2, 4, 6 ...
    {
      mu = Jmax2 / 2;
    }
    else if ((Jmax2 + 1) % 4 == 2) // J_max = 1/2, 5/2, 9/2 ...
    {
      mu = (Jmax2 - 1) / 2;
    }
    else if ((Jmax2 + 1) % 4 == 3) // J_max = 1, 3, 5, 7 ...
    {
      mu = (Jmax2 - 2) / 2;
    }
    else if ((Jmax2 + 1) % 4 == 0) // J_max 3/2, 7/2, 11/2, ...
    {
      mu = (Jmax2 + 1) / 2;
    }
    else
    {
      std::cout << "  generate linear algebra mesh error! " << std::endl;
    }
    /// generate mesh
    /////////////////////////////////////////////////////////////
    for (int k = 0; k < mu; k++)
    {
      Xprt[k] = 0.5 * (k + 1.) / (mu + 1.);
      Wprt[k] = 1. / Nmesh;
      // std::cout << k << "  " << Xprt[k] << "   " << Wprt[k] << std::endl;
    }
    for (int k = mu; k < 2 * mu; k++)
    {
      Xprt[k] = 0.5 + 0.5 * (k - mu + 1.) / (mu + 1.);
      Wprt[k] = 1. / Nmesh;
      // std::cout << k << "  " << Xprt[k] << "   " << Wprt[k] << std::endl;
    }
    ///////////////////////////////////////////////////////
    if ((Jmax2 + 1) % 4 == 1) // J_max = 0, 2, 4, 6 ...
    {
      mu = Jmax2 / 2;
      Xprt[2 * mu] = 1.;
      Wprt[2 * mu] = 1. / Nmesh;
    }
    else if ((Jmax2 + 1) % 4 == 2) // J_max = 1/2, 5/2, 9/2 ...
    {
      mu = (Jmax2 - 1) / 2;
      Xprt[2 * mu] = 0;
      Wprt[2 * mu] = 1. / Nmesh;
      Xprt[2 * mu + 1] = 1.;
      Wprt[2 * mu + 1] = 1. / Nmesh;
    }
    else if ((Jmax2 + 1) % 4 == 3) // J_max = 1, 3, 5, 7 ...
    {
      mu = (Jmax2 - 2) / 2;
      Xprt[2 * mu] = 0.;
      Wprt[2 * mu] = 1. / Nmesh;
      Xprt[2 * mu + 1] = 0.5;
      Wprt[2 * mu + 1] = 1. / Nmesh;
      Xprt[2 * mu + 2] = 1.;
      Wprt[2 * mu + 2] = 1. / Nmesh;
    }
    else if ((Jmax2 + 1) % 4 == 0) // J_max 3/2, 7/2, 11/2, ...
    {
      mu = (Jmax2 + 1) / 2;
    }
    else
    {
      std::cout << "  generate linear algebra mesh error! " << std::endl;
    }
  }
  else if (type == "beta")
  {
    int N;
    if (Jmax2 % 4 == 0)
    {
      N = Jmax2 / 2 + 1;
    }
    else
    {
      N = (Jmax2 + 1) / 2;
    }
    for (int j = 1; j <= N; j++)
    {
      Xprt[j - 1] = (j - 0.5) / N;
      Wprt[j - 1] = 1;
      // std::cout << j-1 << "  " << Xprt[j-1] << "   " << Wprt[j-1] << std::endl;
    }
  }
  else
  {
    std::cout << " the angle shoubld be alpha beta or gamma!  Generate_LA_Mesh()" << std::endl;
    exit(0);
  }
  return;
}

void AngMomProjection::InitializeMatrix(int isospin)
{
  int dim = ms->Get_MScheme_dim(isospin);
  int count = 0;
  int j1, j2, m1, m2, index1, index2, i;
  if (isospin == Proton) // find proton orbits
  {
    RotatePairs_p = (double *)mkl_malloc(GQBeta.GetTotalNumber() * dim * dim * sizeof(double), 64);
    for (i = 0; i < GQBeta.GetTotalNumber(); i++)
    {
      /// proton
      count = i * dim * dim;
      RP_StartPoint_p.push_back(count);
      for (index1 = 0; index1 < dim; index1++)
      {
        for (index2 = 0; index2 < dim; index2++)
        {
          RotatePairs_p[count + index1 * dim + index2] = 0.;
          j1 = ms->Get_MSmatrix_2j(isospin, index1);
          j2 = ms->Get_MSmatrix_2j(isospin, index2);
          if (j1 == j2)
          {
            m1 = ms->Get_MSmatrix_2m(isospin, index1);
            m2 = ms->Get_MSmatrix_2m(isospin, index2);
            RotatePairs_p[count + index1 * dim + index2] = AngMom::Wigner_d(j1, m1, m2, GQBeta.GetX(i));
          }
        }
      }
    }
    initial_proton = true;
  }
  else if (isospin == Neutron)
  {
    RotatePairs_n = (double *)mkl_malloc(GQBeta.GetTotalNumber() * dim * dim * sizeof(double), 64);
    /// Full beta matrix
    for (i = 0; i < GQBeta.GetTotalNumber(); i++)
    {
      /// neutron
      count = i * dim * dim;
      RP_StartPoint_n.push_back(count);
      for (index1 = 0; index1 < dim; index1++)
      {
        for (index2 = 0; index2 < dim; index2++)
        {
          RotatePairs_n[count + index1 * dim + index2] = 0.;
          j1 = ms->Get_MSmatrix_2j(isospin, index1);
          j2 = ms->Get_MSmatrix_2j(isospin, index2);
          if (j1 == j2)
          {
            m1 = ms->Get_MSmatrix_2m(isospin, index1);
            m2 = ms->Get_MSmatrix_2m(isospin, index2);
            RotatePairs_n[count + index1 * dim + index2] = AngMom::Wigner_d(j1, m1, m2, GQBeta.GetX(i));
          }
        }
      }
    }
    initial_neutron = true;
  }
  else
  {
    std::cout << " Tz should be Proton or Neutron !" << std::endl;
    exit(0);
  }
}

void AngMomProjection::InitializeBetaFuncs()
{
  WDTab = (double *)mkl_malloc(GQBeta.GetTotalNumber() * sizeof(double), 64);
  int J = ms->GetAMProjected_J();
  int K = ms->GetAMProjected_K();
  int M = ms->GetAMProjected_M();
  /// Full beta matrix
  for (int i = 0; i < GQBeta.GetTotalNumber(); i++)
  {
    // Wigner D function table
    WDTab[i] = (J + 1) * 0.5 * PI * sin(GQBeta.GetX(i) * PI) * AngMom::Wigner_d(J, M, K, GQBeta.GetX(i));
    // printf("%d   %.15lf    %.15lf \n", i, GQBeta.GetX(i),  AngMom::Wigner_d(J, K, M, GQBeta.GetX(i)));
  }
}

void AngMomProjection::UpdateFilenames()
{
  string foldername = std::to_string(ms->GetGQ_alpha()) + "_points";
  FileName_ax = "mesh/" + foldername + "/GaussQua_x.txt";
  FileName_aw = "mesh/" + foldername + "/GaussQua_w.txt";

  foldername = std::to_string(ms->GetGQ_beta()) + "_points";
  FileName_bx = "mesh/" + foldername + "/GaussQua_x.txt";
  FileName_bw = "mesh/" + foldername + "/GaussQua_w.txt";

  foldername = std::to_string(ms->GetGQ_gamma()) + "_points";
  FileName_cx = "mesh/" + foldername + "/GaussQua_x.txt";
  FileName_cw = "mesh/" + foldername + "/GaussQua_w.txt";
}

void AngMomProjection::InitInt_pn()
{
  UpdateFilenames();
  ReadMesh(); // Read mesh points
  InitializeMatrix(Proton);
  InitializeMatrix(Neutron);
  InitializeBetaFuncs();
  return;
}

void AngMomProjection::InitInt_Iden()
{
  UpdateFilenames();
  ReadMesh(); // Read mesh points
  InitializeMatrix(Proton);
  InitializeBetaFuncs();
  return;
}

void AngMomProjection::InitInt_HF_Projection()
{
  std::string MeshType = ms->Get_MeshType();
  if (MeshType.substr(0, 3) == "Qud")
  {
    this->GQAlpha.SetNumber(ms->GetGQ_alpha());
    this->Generate_GQ_Mesh(GQAlpha, MeshType.substr(4));

    this->GQBeta.SetNumber(ms->GetGQ_beta());
    this->Generate_GQ_Mesh(GQBeta, MeshType.substr(4));

    this->GQGamma.SetNumber(ms->GetGQ_gamma());
    this->Generate_GQ_Mesh(GQGamma, MeshType.substr(4));
    ///
    this->GQdim = GQAlpha.GetTotalNumber() * GQBeta.GetTotalNumber() * GQGamma.GetTotalNumber();
    for (int i = 0; i < GQAlpha.GetTotalNumber(); i++)
    {
      for (int j = 0; j < GQBeta.GetTotalNumber(); j++)
      {
        for (int k = 0; k < GQGamma.GetTotalNumber(); k++)
        {
          GasussQuadMap.push_back(i); // 3 * temp + 0/
          GasussQuadMap.push_back(j); // 3 * temp + 1/
          GasussQuadMap.push_back(k); // 3 * temp + 2/
        }
      }
    }
  }
  else if (MeshType == "ReadFiles")
  {
    this->FileName_ax = "mesh/alpha/GaussQua_x.txt";
    this->FileName_aw = "mesh/alpha/GaussQua_w.txt";
    this->FileName_bx = "mesh/beta/GaussQua_x.txt";
    this->FileName_bw = "mesh/beta/GaussQua_w.txt";
    this->FileName_cx = "mesh/gamma/GaussQua_x.txt";
    this->FileName_cw = "mesh/gamma/GaussQua_w.txt";
    ReadMesh(); // Read mesh points
  }
  else if (MeshType == "LAmethod")
  {
    // int Jmax2 = ms->GetAMProjected_J();
    int Jmax2 = ms->Get2Jmax();
    ms->SetAMProjected_J(Jmax2);
    int Nmesh = Jmax2 + 1;
    this->GQAlpha.SetNumber(Nmesh);
    this->Generate_LA_Mesh(GQAlpha, "alpha");

    if (Jmax2 % 2 == 0) // even
    {
      this->GQBeta.SetNumber(Jmax2 / 2 + 1);
      this->Generate_LA_Mesh(GQBeta, "beta");
      ms->SetGuassQuadMesh(Nmesh, Jmax2 / 2 + 1, Nmesh);
      ms->SetAMProjected_K(0);
      ms->SetAMProjected_M(0);
    }
    else
    {
      this->GQBeta.SetNumber((Jmax2 + 1) / 2);
      this->Generate_LA_Mesh(GQBeta, "beta");
      ms->SetGuassQuadMesh(Nmesh, (Jmax2 + 1) / 2, Nmesh);
      ms->SetAMProjected_K(1);
      ms->SetAMProjected_M(1);
    }

    this->GQGamma.SetNumber(Nmesh);
    this->Generate_LA_Mesh(GQGamma, "gamma");
    ///
    this->GQdim = GQAlpha.GetTotalNumber() * GQBeta.GetTotalNumber() * GQGamma.GetTotalNumber();
    for (int i = 0; i < GQAlpha.GetTotalNumber(); i++)
    {
      for (int j = 0; j < GQBeta.GetTotalNumber(); j++)
      {
        for (int k = 0; k < GQGamma.GetTotalNumber(); k++)
        {
          GasussQuadMap.push_back(i); // 3 * temp + 0/
          GasussQuadMap.push_back(j); // 3 * temp + 1/
          GasussQuadMap.push_back(k); // 3 * temp + 2/
        }
      }
    }
  }
  else
  {
    std::cout << " Unknow  mesh grid type! " << MeshType << std::endl;
    exit(0);
  }
  InitializeMatrix(Proton);
  InitializeMatrix(Neutron);
  InitializeBetaFuncs();
  return;
}

double AngMomProjection::GetWigner_d_beta(int isospin, int beta, int i, int j)
{
  if (isospin == Proton)
  {
    int SP = RP_StartPoint_p[beta];
    int dim = ms->Get_MScheme_dim(isospin);
    return RotatePairs_p[SP + i * dim + j];
  }
  else if (isospin == Neutron)
  {
    int SP = RP_StartPoint_n[beta];
    int dim = ms->Get_MScheme_dim(isospin);
    return RotatePairs_n[SP + i * dim + j];
  }
  else
  {
    std::cout << " Tz should be Proton or Neutron !" << std::endl;
    exit(0);
  }
}

double *AngMomProjection::GetWigner_d_prt(int isospin, int beta)
{
  if (isospin == Proton)
  {
    int SP = RP_StartPoint_p[beta];
    return RotatePairs_p + SP;
  }
  else if (isospin == Neutron)
  {
    int SP = RP_StartPoint_n[beta];
    return RotatePairs_n + SP;
  }
  else
  {
    std::cout << " Tz should be Proton or Neutron !" << std::endl;
    exit(0);
  }
}

void AngMomProjection::PrintMatrix_p()
{
  int beta_dim = this->GQBeta.GetTotalNumber();
  int dim2 = ms->Get_MScheme_dim2(Proton);
  for (size_t i = 0; i < beta_dim; i++)
  {
    int SP = this->RP_StartPoint_p[i];
    for (size_t j = 0; j < dim2; j++)
    {
      std::cout << "x_i = " << i << "  " << j << "  " << RotatePairs_n[SP + j] << std::endl;
    }
  }
  return;
}

int AngMomProjection::GetMeshDimensionIn(int type) // type = 0 alpha, 1 beta, 2 gamma
{
  if (type == 0)
  {
    return this->GQAlpha.GetTotalNumber();
  }
  else if (type == 1)
  {
    return this->GQBeta.GetTotalNumber();
  }
  else if (type == 2)
  {
    return this->GQGamma.GetTotalNumber();
  }
  return -1000;
}

ComplexNum AngMomProjection::GuassQuad_weight(int alpha, int beta, int gamma)
{ // WHERE alpha gamma are in unit of 2 * PI, beta is in unit of PI
  ComplexNum factor, expPart;
  int J = ms->GetAMProjected_J();
  int M = ms->GetAMProjected_M();
  int K = ms->GetAMProjected_K();
  expPart = 1.i * ((M * GQAlpha.GetX(alpha) + K * GQGamma.GetX(gamma)) * PI);
  factor = std::exp(expPart);
  // std::cout<< WDTab[beta] * GQAlpha.GetWeight(alpha) * GQBeta.GetWeight(beta) * GQGamma.GetWeight(gamma) * factor  <<"  "<<WDTab[beta] << GQAlpha.GetWeight(alpha) << GQBeta.GetWeight(beta) << GQGamma.GetWeight(gamma) << factor << std::endl;
  return WDTab[beta] * GQAlpha.GetWeight(alpha) * GQBeta.GetWeight(beta) * GQGamma.GetWeight(gamma) * factor;
}

ComplexNum AngMomProjection::LinearAlgebra_weight(int alpha, int gamma)
{ // WHERE alpha gamma are in unit of 2 * PI, beta is in unit of PI
  ComplexNum factor, expPart;
  int J = ms->GetAMProjected_J();
  int K = ms->GetAMProjected_K();
  int M = ms->GetAMProjected_M();
  expPart = 1.i * ((M * GQAlpha.GetX(alpha) + K * GQGamma.GetX(gamma)) * PI);
  factor = std::exp(expPart);
  return GQAlpha.GetWeight(alpha) * GQGamma.GetWeight(gamma) * factor;
}

void AngMomProjection::PrintInfo()
{
  std::cout << "  mesh points type:  " << ms->Get_MeshType() << std::endl;
  std::cout << "  Projected 2I: " << ms->GetAMProjected_J() << "   2M: " << ms->GetAMProjected_M() << "   2K: " << ms->GetAMProjected_K() << std::endl;
  std::cout << "  number of mesh points: alpha: " << GQAlpha.GetTotalNumber() << "  beta: " << GQBeta.GetTotalNumber() << "  gamma: " << GQGamma.GetTotalNumber() << std::endl;
  if (ms->GetProjected_parity() == 0)
    std::cout << "  No parity projection!  " << std::endl;
  else if (ms->GetProjected_parity() == 1)
    std::cout << "  Parity projection:  +" << std::endl;
  else if (ms->GetProjected_parity() == -1)
    std::cout << "  Parity projection:  -" << std::endl;
  std::cout << "/-----------------------------------------------------/" << std::endl;
}
