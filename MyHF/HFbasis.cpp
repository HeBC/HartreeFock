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

#include "HFbasis.h"

HFbasis::HFbasis(ModelSpace &ms, int isospin)
    : ms(&ms), isospin(isospin)
{
    N = ms.GetParticleNumber(isospin);
    dim = ms.Get_MScheme_dim(isospin);
}

HFbasis::HFbasis(ModelSpace &ms, AngMomProjection &AMJ, int isospin)
    : ms(&ms), AMJ(&AMJ), isospin(isospin)
{
    N = ms.GetParticleNumber(isospin);
    dim = ms.Get_MScheme_dim(isospin);
    MallocMemoryComplex();
}

void HFbasis::MallocMemoryComplex()  // only work for the HF with AMJ
{
    if (Malloced_Memory == false)
    {
        // std::cout << " alloc particle number: " << N << std::endl;
        this->Basis_complex = (ComplexNum *)mkl_malloc((this->N * this->dim) * sizeof(ComplexNum), 64);
        Malloced_Memory = true;
        Is_basis_complex = true;
    }
}

HFbasis::~HFbasis()
{
    FreeMemory();
}

void HFbasis::FreeMemory()
{
    if (Malloced_Memory == true)
    {
        if (Is_basis_complex)
        {
            mkl_free(Basis_complex);
        }
        else
        {
            // mkl_free(Basis_double);
        }
    }
}

void HFbasis::SetArrayPrt(const double *Array_double)
{
    Basis_double = Array_double;
    Malloced_Memory = true;
}

void HFbasis::SetArrayPrt(const double *Array_double, int inner_dim)
{
    Basis_double = Array_double;
    Malloced_Memory = true;
    this->vector_dim = inner_dim;
}

void HFbasis::SetArrayPrt(ComplexNum *Array_ComplexNum)
{
    Basis_complex = Array_ComplexNum;
    Malloced_Memory = true;
    Is_basis_complex = true;
}

void HFbasis::ZeroOperatorStructure(ComplexNum *ystruc)
{
    memset(ystruc, 0, sizeof(double) * 2 * this->GetDim());
}

void HFbasis::MatrixCope(ComplexNum *destination, const ComplexNum *source, int number)
{                                                   // source = destination
    cblas_zcopy(number, source, 1, destination, 1); // b -> a
    return;
}

void HFbasis::MatrixCope(ComplexNum *destination, const double *source, int number)
{ // source = destination
    memset(destination, 0, (number) * 2 * sizeof(double));
    cblas_dcopy(number, source, 1, (double *)destination, 2); // b -> a
    return;
}

void HFbasis::RotatedOperator(int alpha, int beta, int gamma)
{
    int i, m1, m, k1, k2;
    ComplexNum factor1, factor2, expPart;
    ComplexNum *ystruct, *sourceStruc;
    double *rotateMatrix;
    rotateMatrix = AMJ->GetWignerDFunc_prt(isospin, beta);
    ystruct = (ComplexNum *)mkl_malloc(this->GetDim() * sizeof(ComplexNum), 64);
    for (i = 0; i < N; i++)
    {
        sourceStruc = this->GetArrayPointerComplex(i);
        this->ZeroOperatorStructure(ystruct); /// Zero
        for (m1 = 0; m1 < dim; m1++)
        {
            for (m = 0; m < dim; m++)
            {
                expPart = -1.i * (AMJ->GetAlpha_x(alpha) * (ms->Get_MSmatrix_2m(isospin, m1)) * PI);
                factor1 = std::exp(expPart); // Pi and m1 is twiced
                expPart = -1.i * (AMJ->GetGamma_x(gamma) * (ms->Get_MSmatrix_2m(isospin, m)) * PI);
                factor2 = std::exp(expPart); // Pi and  m is twiced
                ystruct[m1] += factor1 * factor2 * rotateMatrix[m1 * dim + m] * (sourceStruc)[m];
            }
        }
        this->MatrixCope(sourceStruc, ystruct, this->GetDim());
    }
    mkl_free(ystruct);
}

void HFbasis::PrintAllParameters_Double()
{
    std::cout << "Total number of particle: " << this->N << std::endl;
    std::cout << "Isospin: " << this->isospin << "    dim: " << this->dim << std::endl;
    for (size_t i = 0; i < this->N; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << " N-i = " << i << " - " << j << "    " << Basis_double[i * dim + j] << std::endl;
        }
    }
    return;
}

void HFbasis::PrintAllParameters_Complex()
{
    std::cout << "Total number of particle: " << this->N << std::endl;
    std::cout << "Isospin: " << this->isospin << "    dim: " << this->dim << std::endl;
    for (size_t i = 0; i < this->N; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << " N-i = " << i << " - " << j << "    " << Basis_complex[i * dim + j] << std::endl;
        }
    }
    return;
}

//---------------------------------------------------------------------
// class PNbasis
//---------------------------------------------------------------------
PNbasis::PNbasis(ModelSpace &ms)
    : ms(&ms)
{
    basis_p = new HFbasis(ms, Proton);
    basis_n = new HFbasis(ms, Neutron);
}

PNbasis::PNbasis(ModelSpace &ms, AngMomProjection &AMJ)
    : ms(&ms), AMJ(&AMJ)
{
    basis_p = new HFbasis(ms, AMJ, Proton);
    basis_n = new HFbasis(ms, AMJ, Neutron);
}

PNbasis::PNbasis(PNbasis &anotherBasis) /// Only work with AMJ
{
    ms = anotherBasis.ms;
    AMJ = anotherBasis.AMJ;
    Is_basis_complex = anotherBasis.Is_basis_complex;
    InnerVectorParamters = anotherBasis.InnerVectorParamters;
    this->basis_p = new HFbasis(*ms, *AMJ, Proton);
    this->basis_n = new HFbasis(*ms, *AMJ, Neutron);
    this->basis_p->MatrixCope(this->basis_p->GetArrayPointerComplex(), anotherBasis.basis_p->GetArrayPointerComplex(), basis_p->GetTotoalDim());
    this->basis_n->MatrixCope(this->basis_n->GetArrayPointerComplex(), anotherBasis.basis_n->GetArrayPointerComplex(), basis_n->GetTotoalDim());
}

// Copy basis
PNbasis &PNbasis::operator=(PNbasis &rhs)
{
    this->ms = rhs.ms;
    this->AMJ = rhs.AMJ;
    this->Is_basis_complex = rhs.Is_basis_complex;
    this->InnerVectorParamters = rhs.InnerVectorParamters;
    this->basis_p = new HFbasis(*ms, *AMJ, Proton);
    this->basis_n = new HFbasis(*ms, *AMJ, Neutron);

    if (rhs.basis_p != nullptr)
    {
        this->basis_p->MatrixCope(this->basis_p->GetArrayPointerComplex(), rhs.basis_p->GetArrayPointerComplex(), this->basis_p->GetTotoalDim());
    }
    if (rhs.basis_n != nullptr)
    {
        this->basis_n->MatrixCope(this->basis_n->GetArrayPointerComplex(), rhs.basis_n->GetArrayPointerComplex(), this->basis_n->GetTotoalDim());
    }
    return *this;
}

PNbasis::~PNbasis()
{
    if (basis_p != nullptr)
        delete basis_p;
    if (basis_n != nullptr)
        delete basis_n;
}

void PNbasis::SetArray(const double *Array_p, const double *Array_n)
{
    basis_p->SetArrayPrt(Array_p);
    basis_n->SetArrayPrt(Array_n);
    InnerVectorParamters = false;
}

void PNbasis::SetArray(const double *Array_p, int inner_dim_p, const double *Array_n, int inner_dim_n)
{
    basis_p->SetArrayPrt(Array_p, inner_dim_p);
    basis_n->SetArrayPrt(Array_n, inner_dim_n);
    InnerVectorParamters = true;
}

void PNbasis::SetArray(ComplexNum *Array_p, ComplexNum *Array_n)
{
    basis_p->SetArrayPrt(Array_p);
    basis_n->SetArrayPrt(Array_n);
    this->Is_basis_complex = true;
    InnerVectorParamters = false;
}

void PNbasis::SetBaiss(HFbasis &inputbasis_p, HFbasis &inputbasis_n)
{
    free(basis_p);
    free(basis_n);
    this->basis_p = &inputbasis_p;
    this->basis_n = &inputbasis_n;
}

void PNbasis::RotatedOperator(int alpha, int beta, int gamma)
{
    this->basis_p->RotatedOperator(alpha, beta, gamma);
    this->basis_n->RotatedOperator(alpha, beta, gamma);
    return;
}

void PNbasis::FullBasis(const std::vector<double> para_vector) // Only work for Complex members
{
    if (this->GetTotalDim() != para_vector.size())
    {
        std::cout << " Wrong dim in FullBasis() " << this->GetTotalDim() << "  " << para_vector.size() << std::endl;
        exit(0);
    }
    this->basis_p->MatrixCope(this->basis_p->GetArrayPointerComplex(), para_vector.data(), this->basis_p->GetTotoalDim());
    this->basis_n->MatrixCope(this->basis_n->GetArrayPointerComplex(), para_vector.data() + this->basis_p->GetTotoalDim(), this->basis_n->GetTotoalDim());

    //this->basis_p->PrintAllParameters_Complex();
    //this->basis_n->PrintAllParameters_Complex();
    return;
}
