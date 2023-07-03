#ifndef ReadWriteFiles_h
#define ReadWriteFiles_h 1

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/types.h>

#include "Hamiltonian.h"
#include "ModelSpace.h"
#include "mpi.h"
#include "mkl.h"
// #include "Minuit/MnUserParameters.h"
// #include "Minuit/FunctionMinimum.h"

class ReadWriteFiles
{
#define OneLine 200
public:
    /// wrapped interface for Hartree-Fock 
    void Read_KShell_HF_input(string filename, ModelSpace &ms, Hamiltonian &inputH);// read input parameters and Hamiltonian
    void Read_OSLO_HF_input(string filename, ModelSpace &ms, Hamiltonian &inputH);// read input parameters and Hamiltonian

    /// read Kshell format interaction
    void ReadInput_HF(string filename, ModelSpace &ms, Hamiltonian &inputH);
    void ReadInputInfo_HF_GCM(string filename, ModelSpace &ms, Hamiltonian &inputH);
    void ReadTokyo(std::string filename, ModelSpace &ms, Hamiltonian &inputH);

    /// Read input files for OSLO format interaction
    void ReadInputInfo_Identical(string filename, ModelSpace &ms, Hamiltonian &inputH);
    void ReadInputInfo_pnSystem(string filename, ModelSpace &ms, Hamiltonian &inputH);
    void ReadInputInfo_pnSystem_GCM(string filename, ModelSpace &ms, Hamiltonian &inputH);
    void Read_InteractionFile_Identical(Hamiltonian &ReadH);
    void Read_InteractionFile_pn(Hamiltonian &ReadH);
    void Read_InteractionFile_Mscheme(Hamiltonian &ReadH);
    void Read_InteractionFile_Mscheme_Unrestricted(Hamiltonian &ReadH); // for naive HF
    void Read_InteractionFile_Mscheme_Unrestricted_ForPhaffian(Hamiltonian &ReadH);

    /// Tools for reading
    vector<string> Get_all_files_names_within_folder(string folder);
    void Read_GCM_points(ModelSpace &ms, string Filename, std::vector<double> &para_x, std::vector<double> &E);
    void Read_GCM_HF_points(string Filename, std::vector<double> &para_x, std::vector<double> &E);
    void ReadME_vector(int dim, std::vector<ComplexNum> &ele, string filename);
    void MPI_ReadMatrix(int dim, ComplexNum *ele, const std::string &filename);

    /// Output files
    void OutputME(int dim, ComplexNum *ele, string filename);
    void SavePairStruc_DiffPairs_Iden(ModelSpace &ms, const std::vector<double> &x);
    void SavePairStruc_SamePairs_Iden(ModelSpace &ms, const std::vector<double> &x);
    void SavePairStruc_DiffPairs(int isospin, ModelSpace &ms, const std::vector<double> &x);
    void SavePairStruc_SamePairs(int isospin, ModelSpace &ms, const std::vector<double> &x);
    // int OutputResults(FunctionMinimum min, ModelSpace &ms);
    // int OutputResults_Iden(FunctionMinimum min, ModelSpace &ms);
    void Output_GCM_points(string filename, ModelSpace &ms, const std::vector<double> &x, double value, int saved_number);
    void SavePairStruc_MCMC(string filename, int isospin, int num_p, int num_n, const std::vector<double> &x);
    void Save_MCMC_GCMinput(string filename, int num_p, int num_n, int step, const std::vector<double> &x, double Einput);
    void Save_HF_Parameters(int N_p, int dim_p, int N_n, int dim_n, double *prt, string filename);
    void Save_HF_Parameters_TXT(int N_p, int dim_p, int N_n, int dim_n, double *prt, string filename);
    void Read_HF_Parameters_TXT(string Filename, double *para_x);
    void Save_HF_For_NPSM_TXT(int N_p, int dim_p, int N_n, int dim_n, double *prt, double E, string filename);
    void Save_HF_Parameters_TXT(int N_p, int dim_p, int N_n, int dim_n, double *prt, double EHF, string filename);

    /// Initial parameters
    // void InitVariationPara_diff_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);
    // void InitVariationPara_diff_Mscheme_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);
    // void InitVariationPara_SamePairs_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);
    // void InitVariationPara_SamePairs_Mscheme_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);

    // void InitVariationPara_diff_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);
    // void InitVariationPara_diff_Mscheme_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);
    // void InitVariationPara_SamePairs_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);
    // void InitVariationPara_SamePairs_Mscheme_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[]);

    // Get path
    string GetMCMC_output_path() { return MCMC_output_path; };
    string GetSelectingMCMCbasisPath() { return SelectingMCMCbasisPath; };
    string GetGCMInput() { return InputPointsPath; };

private:
    // for input file
    int ReadOrbits(ifstream &input_file, vector<Orbit> *Input_Orbits);
    void ReadCollectivePiars(ifstream &input_file, vector<CollectivePairs> *Input_Pairs);
    void ReadVariationParameters(ifstream &input_file, ModelSpace &ms);
    void Read_GCM_Parameters(ifstream &input_file, ModelSpace &ms);
    void ReadInteractionFileName(ifstream &input_file, Hamiltonian &inputH);
    void Read_Iden_InteractionFileName(ifstream &input_file, Hamiltonian &inputH);
    void Read_Collective_Ham(int tz2, Hamiltonian &ReadH); // tz2 = 2 read Vnn;  tz2 = 0 read Vpn;  tz2 = -2 read Vpp
    void Read_OSLO_Format_Identical(Hamiltonian &ReadH);
    void Read_OSLO_Format_pnSystem(Hamiltonian &ReadH);
    void Read_OSLO_Ham(int tz2, Hamiltonian &ReadH); // tz2 = 2 read Vnn;  tz2 = 0 read Vpn;  tz2 = -2 read Vpp
    void Read_Collective_Format_Identical(Hamiltonian &ReadH);
    void Read_Collective_Format_pnSystem(Hamiltonian &ReadH);
    void skip_comments(std::ifstream &in);
    double skip_comments_Zerobody(std::ifstream &in);
    bool isInteger(const std::string &input);
    std::string extractFirstWord(const std::string &input);

    string Save_Parameters_p = "Output/para_p.dat";
    string Save_Parameters_n = "Output/para_n.dat";
    string Final_Report_path = "Output/Final_report.dat";

    // GCM
    string InputPointsPath = "./Input/GCMpoints/";
    string MCMC_output_path = "./Output/MCMC/";
    string SelectingMCMCbasisPath = "./Output/selectedMCMC/";
};

#endif
