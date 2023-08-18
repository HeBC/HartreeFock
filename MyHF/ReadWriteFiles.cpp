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
#include "ReadWriteFiles.h"

// wrapped interface

// Kshell format interaction and HF only
// read input parameters and Hamiltonian
void ReadWriteFiles::Read_KShell_HF_input(string filename, ModelSpace &ms, Hamiltonian &inputH)
{
  this->ReadInput_HF(filename, ms, inputH);
  ms.InitialModelSpace_HF();
  inputH.Prepare_MschemeH_Unrestricted();
  return;
}

void ReadWriteFiles::Read_OSLO_HF_input(string filename, ModelSpace &ms, Hamiltonian &inputH)
{
  this->ReadInputInfo_pnSystem_GCM("InputFile_OSLO.dat", ms, inputH);
  ms.InitialModelSpace_pn();
  if (std::fabs(ms.GetMassPowerFactor() - 1) > 1.e-7)
  {
    inputH.SetMassDep(true);
  }
  this->Read_InteractionFile_Mscheme_Unrestricted(inputH);
  return;
}

// Kshell format interaction and HFB only
// read input parameters and Hamiltonian
void ReadWriteFiles::Read_KShell_HFB_input(string filename, ModelSpace &ms, Hamiltonian &inputH)
{
  this->ReadInput_HFB(filename, ms, inputH);
  ms.InitialModelSpace_HF(); // also work for HFB
  inputH.Prepare_MschemeH_Unrestricted();
  return;
}

// Tools
void ReadWriteFiles::ReadInputInfo_Identical(string filename, ModelSpace &ms, Hamiltonian &inputH)
{

  ifstream input_file;
  int N;
  input_file.open(filename, std::ios_base::in);
  if (!input_file.is_open())
  {
    cerr << "Could not open the identical input file - '" << filename << "'" << endl;
    exit(0);
  }
  N = ReadOrbits(input_file, &ms.Orbits_p);
  ms.SetProtonPairNum(N);
  ms.SetNeutronPairNum(0);
  // N = ReadOrbits(input_file, &ms.Orbits_n);
  // ms.SetNeutronPairNum(N);
  ReadCollectivePiars(input_file, ms.GetCollectivePairVectorPointer(Proton));
  // ReadCollectivePiars(input_file, &ms.CollectivePair_n);
  ReadVariationParameters(input_file, ms);
  // Read interaction file name
  // ReadInteractionFileName(input_file, inputH);
  Read_Iden_InteractionFileName(input_file, inputH);
  input_file.close();
}

void ReadWriteFiles::ReadInputInfo_pnSystem(string filename, ModelSpace &ms, Hamiltonian &inputH)
{

  ifstream input_file;
  int N;
  input_file.open(filename, std::ios_base::in);
  if (!input_file.is_open())
  {
    cerr << "Could not open the pn-case input file - '"
         << filename << "'" << endl;
    exit(0);
  }
  N = ReadOrbits(input_file, &ms.Orbits_p);
  ms.SetProtonPairNum(N);
  ReadCollectivePiars(input_file, ms.GetCollectivePairVectorPointer(Proton));
  N = ReadOrbits(input_file, &ms.Orbits_n);
  ms.SetNeutronPairNum(N);
  ReadCollectivePiars(input_file, ms.GetCollectivePairVectorPointer(Neutron));
  ReadVariationParameters(input_file, ms);
  // Read interaction file name
  ReadInteractionFileName(input_file, inputH);
  input_file.close();
}

void ReadWriteFiles::ReadInputInfo_pnSystem_GCM(string filename, ModelSpace &ms, Hamiltonian &inputH)
{

  ifstream input_file;
  int N;
  input_file.open(filename, std::ios_base::in);
  if (!input_file.is_open())
  {
    cerr << "Could not open the pn-case input file - '"
         << filename << "'" << endl;
    exit(0);
  }
  N = ReadOrbits(input_file, &ms.Orbits_p);
  ms.SetProtonPairNum(N);
  ReadCollectivePiars(input_file, ms.GetCollectivePairVectorPointer(Proton));
  N = ReadOrbits(input_file, &ms.Orbits_n);
  ms.SetNeutronPairNum(N);
  ReadCollectivePiars(input_file, ms.GetCollectivePairVectorPointer(Neutron));
  Read_GCM_Parameters(input_file, ms);
  // Read interaction file name
  ReadInteractionFileName(input_file, inputH);

  // Set SPE for OSLO interaction
  // add hermitation part for SP energy
  // Proton
  int temp_size = ms.Orbits_p.size();
  for (size_t i = 0; i < temp_size; i++)
  {
    OneBodyElement tempele = OneBodyElement(i, i, 0, Proton, ms.Orbits_p[i].SPE);
    inputH.OBEs_p.push_back(tempele);
  }
  // Neutron
  temp_size = ms.Orbits_n.size();
  for (size_t i = 0; i < temp_size; i++)
  {
    OneBodyElement tempele = OneBodyElement(i, i, 0, Neutron, ms.Orbits_n[i].SPE);
    inputH.OBEs_n.push_back(tempele);
  }

  input_file.close();
}

bool ReadWriteFiles::isInteger(const std::string &input)
{
  std::regex regexPattern("\\s*(\\d+)\\s*,\\s*(\\d+)\\s*");
  std::regex regexPattern2("\\s*(\\d+)\\s*(\\d+)\\s*");
  if (std::regex_match(input, regexPattern))
  {
    return true;
  }
  else if (std::regex_match(input, regexPattern2))
  {
    return true;
  }
  else
  {
    return false;
  }
}

void ReadWriteFiles::ReadInput_HF(string filename, ModelSpace &ms, Hamiltonian &inputH)
{
  ifstream input_file;
  int N;
  input_file.open(filename, std::ios_base::in);
  if (!input_file.is_open())
  {
    cerr << "Could not open the pn-case input file - '"
         << filename << "'" << endl;
    exit(0);
  }

  // --------------------------------  # read element
  string comment_string;
  getline(input_file, comment_string); // read comment
  getline(input_file, comment_string); // read comment
  bool ReadEle = false;
  double A = 0, Z = 0;
  double N_p = 0, N_n = 0;
  if (!isInteger(comment_string))
  {
    ms.GetAZfromString(comment_string, A, Z);
    ms.Set_RefString(comment_string);
    // std::cout << "The input is a string: " << comment_string << std::endl;
    //  std::cout << A << "   " << Z << std::endl;
    ReadEle = true;
  }
  else
  {
    std::regex regexPattern("\\s*(\\d+)\\s*,\\s*(\\d+)\\s*");
    std::regex regexPattern2("\\s*(\\d+)\\s*(\\d+)\\s*");
    std::smatch matches;
    if (std::regex_match(comment_string, matches, regexPattern))
    {
      // matches[1] contains the first number, matches[2] contains the second number
      N_p = std::stoi(matches[1]);
      N_n = std::stoi(matches[2]);
    }
    else if (std::regex_match(comment_string, matches, regexPattern2))
    {
      // matches[1] contains the first number, matches[2] contains the second number
      N_p = std::stoi(matches[1]);
      N_n = std::stoi(matches[2]);
    }
    else
    {
      std::cout << " Trouble parsing A and Z " << std::endl;
      exit(0);
    }
  }
  getline(input_file, comment_string);
  // Read interaction file name
  getline(input_file, inputH.snt_file);
  inputH.RemoveWhitespaceInFilename();

  //--------------------------------- Read Kshell interaction
  this->ReadTokyo(inputH.GetKshellSntFile(), ms, inputH);
  if (ReadEle)
  {
    ms.SetProtonNum(Z - ms.GetCoreProtonNum());
    ms.SetNeutronNum(A - Z - ms.GetCoreNeutronNum());
  }
  else
  {
    ms.SetProtonNum(N_p);
    ms.SetNeutronNum(N_n);
  }
  //--------------------------------- End Read Kshell interaction
  // -------------------------------- Read constrains
  getline(input_file, comment_string);
  double doubleData, doubleData2;

  getline(input_file, comment_string);
  std::istringstream iss(comment_string); // Create an input string stream from the line
  std::string strValue;
  int intValue;
  // Read the string and integer from the line
  iss >> strValue >> doubleData >> doubleData2;

  if (strValue == "No" or strValue == "no")
  {
    ms.SetShapeConstrained(false);
  }
  else if (strValue == "Yes" or strValue == "yes")
  {
    ms.SetShapeConstrained(true);
  }
  else
  {
    std::cout << "   ShapeConstrained parameter should be no or yes!  " << strValue << std::endl;
    exit(0);
  }
  ms.SetShapeQ(doubleData, doubleData2);

  //---------------------- read Jz
  getline(input_file, comment_string);
  iss.clear(); // Reset the state of the stream if needed
  iss.str(comment_string);
  // Read the string and integer from the line
  iss >> strValue >> doubleData;
  if (strValue == "No" or strValue == "no")
  {
    ms.Set_Jz_constraint(false);
  }
  else if (strValue == "Yes" or strValue == "yes")
  {
    ms.Set_Jz_constraint(true);
  }
  else
  {
    std::cout << "   ShapeConstrained parameter should be no or yes!  " << strValue << std::endl;
    exit(0);
  }
  ms.SetTargetJz(doubleData);

  //---------------------- read Jx
  getline(input_file, comment_string);
  iss.clear(); // Reset the state of the stream if needed
  iss.str(comment_string);
  // Read the string and integer from the line
  iss >> strValue >> doubleData;
  if (strValue == "No" or strValue == "no")
  {
    ms.Set_Jx_constraint(false);
  }
  else if (strValue == "Yes" or strValue == "yes")
  {
    ms.Set_Jx_constraint(true);
  }
  else
  {
    std::cout << "   ShapeConstrained parameter should be no or yes!  " << strValue << std::endl;
    exit(0);
  }
  ms.SetTargetJx(doubleData);

  input_file.close();
}

void ReadWriteFiles::ReadInput_HFB(string filename, ModelSpace &ms, Hamiltonian &inputH)
{
  ifstream input_file;
  int N;
  input_file.open(filename, std::ios_base::in);
  if (!input_file.is_open())
  {
    cerr << "Could not open the pn-case input file - '"
         << filename << "'" << endl;
    exit(0);
  }

  // --------------------------------  # read element
  string comment_string;
  getline(input_file, comment_string); // read comment
  getline(input_file, comment_string); // read comment
  bool ReadEle = false;
  double A = 0, Z = 0;
  double N_p = 0, N_n = 0;
  if (!isInteger(comment_string))
  {
    ms.GetAZfromString(comment_string, A, Z);
    ms.Set_RefString(comment_string);
    // std::cout << "The input is a string: " << comment_string << std::endl;
    //  std::cout << A << "   " << Z << std::endl;
    ReadEle = true;
  }
  else
  {
    std::regex regexPattern("\\s*(\\d+)\\s*,\\s*(\\d+)\\s*");
    std::regex regexPattern2("\\s*(\\d+)\\s*(\\d+)\\s*");
    std::smatch matches;
    if (std::regex_match(comment_string, matches, regexPattern))
    {
      // matches[1] contains the first number, matches[2] contains the second number
      N_p = std::stoi(matches[1]);
      N_n = std::stoi(matches[2]);
    }
    else if (std::regex_match(comment_string, matches, regexPattern2))
    {
      // matches[1] contains the first number, matches[2] contains the second number
      N_p = std::stoi(matches[1]);
      N_n = std::stoi(matches[2]);
    }
    else
    {
      std::cout << " Trouble parsing A and Z " << std::endl;
      exit(0);
    }
  }
  getline(input_file, comment_string);
  // Read interaction file name
  getline(input_file, inputH.snt_file);
  inputH.RemoveWhitespaceInFilename();

  //--------------------------------- Read Kshell interaction
  this->ReadTokyo(inputH.GetKshellSntFile(), ms, inputH);
  if (ReadEle)
  {
    ms.SetProtonNum(Z - ms.GetCoreProtonNum());
    ms.SetNeutronNum(A - Z - ms.GetCoreNeutronNum());
  }
  else
  {
    ms.SetProtonNum(N_p);
    ms.SetNeutronNum(N_n);
  }
  //--------------------------------- End Read Kshell interaction
  // -------------------------------- Read constrains
  getline(input_file, comment_string);
  double doubleData, doubleData2;
  //------------------------------------------- Particle number Constraint
  getline(input_file, comment_string);
  std::istringstream iss(comment_string); // Create an input string stream from the line
  std::string strValue;
  iss >> strValue;
  if (strValue == "No" or strValue == "no")
  {
    ms.Set_ParticleNumberConstrained(false);
  }
  else if (strValue == "Yes" or strValue == "yes")
  {
    ms.Set_ParticleNumberConstrained(true);
  }
  else
  {
    std::cout << "   Particle number constaint option should be no or yes!  " << strValue << std::endl;
    exit(0);
  }

  //------------------------------------------- shape constaint
  getline(input_file, comment_string);
  iss.clear(); // Reset the state of the stream if needed
  iss.str(comment_string);
  int intValue;
  // Read the string and integer from the line
  iss >> strValue >> doubleData >> doubleData2;

  if (strValue == "No" or strValue == "no")
  {
    ms.SetShapeConstrained(false);
  }
  else if (strValue == "Yes" or strValue == "yes")
  {
    ms.SetShapeConstrained(true);
  }
  else
  {
    std::cout << "   ShapeConstrained parameter should be no or yes!  " << strValue << std::endl;
    exit(0);
  }
  ms.SetShapeQ(doubleData, doubleData2);
  getline(input_file, comment_string);

  input_file.close();
}

std::string ReadWriteFiles::extractFirstWord(const std::string &input)
{
  std::size_t spacePos = input.find_first_of(" ");
  if (spacePos != std::string::npos)
  {
    return input.substr(0, spacePos);
  }
  return input;
}

void ReadWriteFiles::ReadInputInfo_HF_GCM(string filename, ModelSpace &ms, Hamiltonian &inputH)
{
  std::ifstream input_file;
  int intData, intData2, intData3;
  double doubleData, doubleData2;

  input_file.open(filename, std::ios_base::in);
  if (!input_file.is_open())
  {
    cerr << "Could not open the GCM input file - '"
         << filename << "'" << endl;
    exit(0);
  }

  // --------------------------------  # read element
  string comment_string;
  getline(input_file, comment_string); // read comment
  getline(input_file, comment_string); // read comment
  bool ReadEle = false;
  double A = 0, Z = 0;
  double N_p = 0, N_n = 0;
  if (!isInteger(comment_string))
  {
    ms.GetAZfromString(comment_string, A, Z);
    // std::cout << "The input is a string: " << comment_string << std::endl;
    //  std::cout << A << "   " << Z << std::endl;
    ReadEle = true;
  }
  else
  {
    std::regex regexPattern("\\s*(\\d+)\\s*,\\s*(\\d+)\\s*");
    std::regex regexPattern2("\\s*(\\d+)\\s*(\\d+)\\s*");
    std::smatch matches;
    if (std::regex_match(comment_string, matches, regexPattern))
    {
      // matches[1] contains the first number, matches[2] contains the second number
      N_p = std::stoi(matches[1]);
      N_n = std::stoi(matches[2]);
    }
    else if (std::regex_match(comment_string, matches, regexPattern2))
    {
      // matches[1] contains the first number, matches[2] contains the second number
      N_p = std::stoi(matches[1]);
      N_n = std::stoi(matches[2]);
    }
    else
    {
      std::cout << " Trouble parsing A and Z " << std::endl;
      exit(0);
    }
  }

  // --------------------------------
  getline(input_file, comment_string);
  // Read interaction file name
  getline(input_file, inputH.snt_file);
  inputH.RemoveWhitespaceInFilename();

  //--------------------------------- Read Kshell interaction
  this->ReadTokyo(inputH.GetKshellSntFile(), ms, inputH);
  if (ReadEle)
  {
    ms.SetProtonNum(Z - ms.GetCoreProtonNum());
    ms.SetNeutronNum(A - Z - ms.GetCoreNeutronNum());
  }
  else
  {
    ms.SetProtonNum(N_p);
    ms.SetNeutronNum(N_n);
  }
  //--------------------------------- End Read Kshell interaction
  ///////////////////////////////////
  getline(input_file, comment_string);
  skip_comments(input_file);

  getline(input_file, comment_string);
  ms.Set_MeshType(extractFirstWord(comment_string)); /// read mesh grid type

  input_file >> intData >> intData2 >> intData3; /// read JMK
  ms.SetAMProjected_J(intData);
  ms.SetAMProjected_M(intData2);
  ms.SetAMProjected_K(intData3);

  getline(input_file, comment_string);
  getline(input_file, comment_string);

  if (comment_string.substr(0, 10).find("No") < 10)
  {
    ms.SetProjected_parity(false);
  }
  else if (comment_string.substr(0, 10).find("+") < 10)
  {
    ms.SetProjected_parity(1);
  }
  else if (comment_string.substr(0, 10).find("-") < 10)
  {
    ms.SetProjected_parity(-1);
  }
  else
  {
    std::cout << " The option for parity projection shoubld be No , + or -!" << std::endl;
    exit(0);
  }

  input_file >> intData >> intData2 >> intData3; // Read Guass quadrature meshh
  ms.SetGuassQuadMesh(intData, intData2, intData3);

  getline(input_file, comment_string);

  input_file.close();
}

int ReadWriteFiles::ReadOrbits(ifstream &input_file, vector<Orbit> *Input_Orbits)
{
  int read_pairN;
  int number_orbits;
  int read_n, read_l, read_j, read_parity;
  double read_SPE;
  string comment_string;
  getline(input_file, comment_string);
  input_file >> read_pairN; // Read number of pairs
  getline(input_file, comment_string);
  getline(input_file, comment_string);
  input_file >> number_orbits; // Read number of orbits
  // cout<< number_orbits <<endl;
  getline(input_file, comment_string);
  getline(input_file, comment_string);
  for (size_t i = 0; i < number_orbits; i++)
  {
    input_file >> read_SPE;
    input_file >> read_n;
    input_file >> read_l;
    input_file >> read_j;
    read_parity = sgn(read_l);
    Orbit temp_orbit = Orbit(read_n, read_l, read_j, Proton, read_parity, read_SPE);
    Input_Orbits->push_back(temp_orbit);
  }
  getline(input_file, comment_string);
  return read_pairN;
}

void ReadWriteFiles::ReadCollectivePiars(ifstream &input_file, vector<CollectivePairs> *Input_Pairs)
{
  int J, parity, pair_Number;
  string comment_string;
  getline(input_file, comment_string);
  getline(input_file, comment_string);
  input_file >> pair_Number; // Read number of pairs
  getline(input_file, comment_string);
  getline(input_file, comment_string);
  for (size_t i = 0; i < pair_Number; i++)
  {
    input_file >> J;
    input_file >> parity;
    CollectivePairs temp_Pair = CollectivePairs(J, parity);
    Input_Pairs->push_back(temp_Pair);
  }
  getline(input_file, comment_string);
}

void ReadWriteFiles::ReadVariationParameters(ifstream &input_file, ModelSpace &ms)
{
  string comment_string;
  int intData, intData2, intDat3;
  double doubleData;

  getline(input_file, comment_string);
  input_file >> intData; // Read Total number of states
  ms.SetTotalOrders(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read type of basis
  ms.SetBasisType(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read type of pairs
  ms.SetPairType(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Sum_Num
  ms.SetNumberOfBasisSummed(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read NucleiMass
  ms.SetNucleiMassA(intData);

  getline(input_file, comment_string);
  input_file >> doubleData; // Read mass_scaling
  ms.SetMassPowerFactor(doubleData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_J
  ms.SetAMProjected_J(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_K
  ms.SetAMProjected_K(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_M
  ms.SetAMProjected_M(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_Parity
  ms.SetProjected_parity(intData);

  getline(input_file, comment_string);
  input_file >> intData >> intData2 >> intDat3; // Read Guass quadrature meshh
  ms.SetGuassQuadMesh(intData, intData2, intDat3);

  getline(input_file, comment_string);
  input_file >> doubleData; // Read EnergyShift
  ms.SetEnergyConstantShift(doubleData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Max number of iteration
  ms.SetMaxNumberOfIteration(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Random seed
  ms.SetRandomSeed(intData);

  getline(input_file, comment_string);
}

void ReadWriteFiles::Read_GCM_Parameters(ifstream &input_file, ModelSpace &ms)
{
  string comment_string;
  int intData, intData2, intDat3;
  double doubleData, doubleData2;

  // getline(input_file, comment_string);
  // input_file >> intData; // Read Total number of states
  ms.SetTotalOrders(1);

  getline(input_file, comment_string);
  input_file >> intData; // Read type of basis
  ms.SetBasisType(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read type of pairs
  ms.SetPairType(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read NucleiMass
  ms.SetNucleiMassA(intData);

  getline(input_file, comment_string);
  input_file >> doubleData; // Read mass_scaling
  ms.SetMassPowerFactor(doubleData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_J
  ms.SetAMProjected_J(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_K
  ms.SetAMProjected_K(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_M
  ms.SetAMProjected_M(intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Total_Parity
  ms.SetProjected_parity(intData);

  getline(input_file, comment_string);
  input_file >> intData >> intData2 >> intDat3; // Read Guass quadrature meshh
  ms.SetGuassQuadMesh(intData, intData2, intDat3);

  getline(input_file, comment_string);
  input_file >> doubleData; // Read EnergyShift
  ms.SetEnergyConstantShift(doubleData);

  getline(input_file, comment_string);
  input_file >> intData >> doubleData >> doubleData2; // Shape constrains Q0 Q2
  ms.SetShapeConstrained((bool)intData);
  ms.SetShapeQ(doubleData, doubleData2);

  getline(input_file, comment_string); // Dividing line //////////////////////////////////////

  getline(input_file, comment_string);
  input_file >> intData; // Read Max number of iteration
  ms.SetMaxNumberOfIteration(intData);

  getline(input_file, comment_string);
  input_file >> intData; // IS Reading starting points for RWMH
  ms.SetReadingSP_RWMH((bool)intData);

  getline(input_file, comment_string);
  input_file >> intData; // Read Random seed
  ms.SetRandomSeed(intData);

  getline(input_file, comment_string);
  input_file >> intData; // The number of walkers
  ms.SetWalkerAmount(intData);

  getline(input_file, comment_string);
  input_file >> doubleData; // Read Temperature
  ms.SetGCMTemperature(doubleData);

  getline(input_file, comment_string);
  input_file >> intData; // Turn off the Diagonalization output, 0 Turn off; 1 turn on
  ms.SetPrintDiagResult((bool)intData);

  getline(input_file, comment_string);
  input_file >> intData; // save MCMC walking history, 0 no, 1 yes
  ms.SetSavingWalkingHistory((bool)intData);

  /////////////////////
  getline(input_file, comment_string);
  input_file >> intData; // Do selection for basis obtained from MCMC
  ms.SetSelectBasis(intData);

  getline(input_file, comment_string);
  input_file >> doubleData; // Read the min allowed overlap EigenValues
  ms.SetOverlapMin(doubleData);

  // getline(input_file, comment_string);
  // input_file >> intData; // Do projection in GCM procedure
  // ms.SetDoGCMprojection((bool)intData);

  getline(input_file, comment_string);
  input_file >> doubleData; // Energy truncation to pick up configurations
  ms.SetEnergyTruncationGCM(doubleData);

  getline(input_file, comment_string);
}

void ReadWriteFiles::ReadInteractionFileName(ifstream &input_file, Hamiltonian &inputH)
{
  // VppFileName, VnnFileName, VpnFileName;
  string comment_string;
  int intData;
  getline(input_file, comment_string);
  input_file >> intData;          // Read Total number of states
  inputH.Set_H_ColOrNot(intData); // 0 shell model; 1 collective Hamiltonian
  getline(input_file, comment_string);

  getline(input_file, inputH.Vpp_filename);
  getline(input_file, inputH.Vnn_filename);
  getline(input_file, inputH.Vpn_filename);
  inputH.RemoveWhitespaceInFilename();
}

void ReadWriteFiles::Read_Iden_InteractionFileName(ifstream &input_file, Hamiltonian &inputH)
{
  // VppFileName, VnnFileName, VpnFileName;
  string comment_string;
  int intData;
  getline(input_file, comment_string);
  input_file >> intData;          // Read Total number of states
  inputH.Set_H_ColOrNot(intData); // 0 shell model; 1 collective Hamiltonian
  getline(input_file, comment_string);

  getline(input_file, inputH.Vpp_filename);
  inputH.RemoveWhitespaceInFilename();
}

void ReadWriteFiles::Read_InteractionFile_Identical(Hamiltonian &ReadH)
{
  if (ReadH.H_IsCollective())
  {
    Read_Collective_Ham(2 * Proton, ReadH); // Read Vpp
    // Read_Collective_Ham( 2 * Neutron, ReadH); // Read Vnn
    // Read_Collective_Ham( Neutron + Proton, ReadH); // Read Vpn
    // ReadH.Set_H_ColOrNot(true); // Set using the normal shell model interaction
    ReadH.PrepareV_Identical();
  }
  else
  {
    Read_OSLO_Ham(2 * Proton, ReadH); // Read Vpp
    // Read_OSLO_Ham( 2 * Neutron, ReadH); // Read Vnn
    // Read_OSLO_Ham( Neutron + Proton, ReadH); // Read Vpn
    // ReadH.Set_H_ColOrNot(false); // Set using the normal shell model interaction
    ReadH.PrepareV_Identical();
  }
}

void ReadWriteFiles::Read_InteractionFile_pn(Hamiltonian &ReadH)
{
  if (ReadH.H_IsCollective())
  {
    Read_Collective_Ham(2 * Proton, ReadH);  // Read Vpp
    Read_Collective_Ham(2 * Neutron, ReadH); // Read Vnn
    // Read Vpn, the particle-hole representation of Vpn is used
    Read_OSLO_Ham(Neutron + Proton, ReadH);
    // ReadH.Set_H_ColOrNot(true); // Set using the normal shell model interaction
    ReadH.PrepareV_pnSystem_v1();
  }
  else
  {
    Read_OSLO_Ham(2 * Proton, ReadH);       // Read Vpp
    Read_OSLO_Ham(2 * Neutron, ReadH);      // Read Vnn
    Read_OSLO_Ham(Neutron + Proton, ReadH); // Read Vpn
                                            // ReadH.Set_H_ColOrNot(false);            // Set using the normal shell model interaction
    ReadH.PrepareV_pnSystem_v1();
  }
}

void ReadWriteFiles::Read_InteractionFile_Mscheme(Hamiltonian &ReadH) // Read OSLO interaction
{
  Read_OSLO_Ham(2 * Proton, ReadH);       // Read Vpp
  Read_OSLO_Ham(2 * Neutron, ReadH);      // Read Vnn
  Read_OSLO_Ham(Neutron + Proton, ReadH); // Read Vpn
  ReadH.Prepare_MschemeH();
}

void ReadWriteFiles::Read_InteractionFile_Mscheme_Unrestricted_ForPhaffian(Hamiltonian &ReadH)
{
  Read_OSLO_Ham(2 * Proton, ReadH);       // Read Vpp
  Read_OSLO_Ham(2 * Neutron, ReadH);      // Read Vnn
  Read_OSLO_Ham(Neutron + Proton, ReadH); // Read Vpn
  ReadH.Prepare_MschemeH_Unrestricted_ForPhaffian();
}

void ReadWriteFiles::Read_InteractionFile_Mscheme_Unrestricted(Hamiltonian &ReadH)
{
  Read_OSLO_Ham(2 * Proton, ReadH);       // Read Vpp
  Read_OSLO_Ham(2 * Neutron, ReadH);      // Read Vnn
  Read_OSLO_Ham(Neutron + Proton, ReadH); // Read Vpn
  ReadH.Prepare_MschemeH_Unrestricted();
}

void ReadWriteFiles::Read_OSLO_Format_Identical(Hamiltonian &ReadH)
{
  Read_OSLO_Ham(2 * Proton, ReadH); // Read Vpp
  // Read_OSLO_Ham( 2 * Neutron, ReadH); // Read Vnn
  // Read_OSLO_Ham( Neutron + Proton, ReadH); // Read Vpn
  ReadH.Set_H_ColOrNot(false); // Set using the normal shell model interaction
  return;
}

void ReadWriteFiles::Read_OSLO_Format_pnSystem(Hamiltonian &ReadH)
{
  Read_OSLO_Ham(2 * Proton, ReadH);       // Read Vpp
  Read_OSLO_Ham(2 * Neutron, ReadH);      // Read Vnn
  Read_OSLO_Ham(Neutron + Proton, ReadH); // Read Vpn
  ReadH.Set_H_ColOrNot(false);            // Set using the normal shell model interaction
  return;
}

void ReadWriteFiles::Read_OSLO_Ham(int tz2, Hamiltonian &ReadH)
{
  if (tz2 != 2 and tz2 != -2 and tz2 != 0)
  {
    cout << "tz2 should be -2 0 2. Read_OSLO_Ham error!" << endl;
  }
  if (tz2 != 0)
  {
    ifstream input_file;
    string filename;
    vector<HamiltonianElements> *V_Iden;
    ModelSpace *ms;
    int Tz = -100;
    ms = ReadH.GetModelSpace();
    int ReadVnum;
    if (tz2 == 2 * Proton)
    {
      filename = ReadH.GetVppFilename();
      V_Iden = &ReadH.Vpp;
      Tz = Proton;
    }
    else if (tz2 == 2 * Neutron)
    {
      filename = ReadH.GetVnnFilename();
      V_Iden = &ReadH.Vnn;
      Tz = Neutron;
    }
    else
    {
      cout << "Tz2 error" << endl;
    }
    input_file.open(filename, std::ios_base::in);
    if (!input_file.is_open())
    {
      cerr << "Could not open the OSLO type Hamiltonian file - '" << filename << "'" << endl;
      exit(0);
    }
    input_file >> ReadVnum; // Read total number of MEs
    // cout << ReadVnum << endl;
    int n1, l1, j1, orbit_index[4], T2, J2;
    double V;
    for (size_t i = 0; i < ReadVnum; i++)
    {
      for (size_t loopOrbits = 0; loopOrbits < 4; loopOrbits++)
      {
        input_file >> n1;
        input_file >> l1;
        input_file >> j1;
        orbit_index[loopOrbits] = ms->FindOrbit(Tz, n1, l1, j1);
      }
      input_file >> T2;
      input_file >> J2;
      input_file >> V;
      // cout << orbit_index[0] << orbit_index[1] << orbit_index[2] << orbit_index[3] << T2 << J2 << V<<endl;
      HamiltonianElements tempele = HamiltonianElements(T2, orbit_index[0], orbit_index[1], orbit_index[2], orbit_index[3], J2 / 2, V);
      V_Iden->push_back(tempele);
    }
    input_file.close();
  }
  else
  {
    ifstream input_file;
    string filename;
    vector<HamiltonianElements> *V_pn;
    ModelSpace *ms;
    int Tz = 0;
    int ReadVnum;
    int Vpn_structure[] = {Proton, Neutron, Proton, Neutron};
    ms = ReadH.GetModelSpace();
    V_pn = &ReadH.Vpn;
    filename = ReadH.GetVpnFilename();
    input_file.open(filename, std::ios_base::in);
    if (!input_file.is_open())
    {
      cerr << "Could not open the OSLO type Hamiltonian file  - '" << filename << "'" << endl;
      exit(0);
    }
    input_file >> ReadVnum; // Read total number of MEs
    // cout << ReadVnum << endl;
    int n1, l1, j1, orbit_index[4], T2, J2;
    double V;
    for (size_t i = 0; i < ReadVnum; i++)
    {
      for (size_t loopOrbits = 0; loopOrbits < 4; loopOrbits++)
      {
        input_file >> n1;
        input_file >> l1;
        input_file >> j1;
        orbit_index[loopOrbits] = ms->FindOrbit(Vpn_structure[loopOrbits], n1, l1, j1);
      }
      input_file >> T2;
      input_file >> J2;
      input_file >> V;
      // cout << orbit_index[0] << orbit_index[1] << orbit_index[2] << orbit_index[3] << T2 << J2 << V<<endl;
      HamiltonianElements tempele = HamiltonianElements(T2, orbit_index[0], orbit_index[1], orbit_index[2], orbit_index[3], J2 / 2, V);
      V_pn->push_back(tempele);
    }
    input_file.close();
  }
  return;
}

void ReadWriteFiles::Read_Collective_Format_Identical(Hamiltonian &ReadH)
{
  Read_Collective_Ham(2 * Proton, ReadH); // Read Vpp
  // Read_Collective_Ham( 2 * Neutron, ReadH); // Read Vnn
  // Read_Collective_Ham( Neutron + Proton, ReadH); // Read Vpn
  ReadH.Set_H_ColOrNot(true); // Set using the normal shell model interaction
  return;
}

void ReadWriteFiles::Read_Collective_Format_pnSystem(Hamiltonian &ReadH)
{
  Read_Collective_Ham(2 * Proton, ReadH);  // Read Vpp
  Read_Collective_Ham(2 * Neutron, ReadH); // Read Vnn
  // Read_Collective_Ham( Neutron + Proton, ReadH); // Read Vpn
  ReadH.Set_H_ColOrNot(true); // Set using the normal shell model interaction
  return;
}

void ReadWriteFiles::Read_Collective_Ham(int tz2, Hamiltonian &ReadH)
{
  if (tz2 != 2 and tz2 != -2 and tz2 != 0)
  {
    cout << "tz2 should be -2 0 2." << __func__ << " error! " << tz2 << endl;
  }
  if (tz2 != 0)
  {
    ifstream input_file;
    string folderName, filename;
    vector<HamiltoaninColllectiveElements> *V_Iden;
    ModelSpace *ms;
    int Tz = -100;
    ms = ReadH.GetModelSpace();
    int ReadVnum;
    if (tz2 == 2 * Proton)
    {
      folderName = ReadH.GetVppFilename();
      filename = folderName + "list.dat";
      V_Iden = &ReadH.VCol_pp;
      Tz = Proton;
      ReadVnum = ms->GetProtonOrbitsNum() * ms->GetProtonOrbitsNum();
    }
    else if (tz2 == 2 * Neutron)
    {
      folderName = ReadH.GetVnnFilename();
      filename = folderName + "list.dat";
      V_Iden = &ReadH.VCol_nn;
      Tz = Neutron;
      ReadVnum = ms->GetNeutronOrbitsNum() * ms->GetNeutronOrbitsNum();
    }
    else
    {
      cout << "Tz2 error" << endl;
    }
    input_file.open(filename, std::ios_base::in);
    if (!input_file.is_open())
    {
      cerr << "Could not open the collective Hamiltonian file  - '" << filename << "'" << endl;
      exit(0);
    }
    // std::cout << filename << std::endl;
    std::string line;
    while (std::getline(input_file, line) and !line.empty())
    {
      string newfileName = folderName + line;
      // std::cout << newfileName << std::endl;
      ifstream Read_Vint_file;
      Read_Vint_file.open(newfileName, std::ios_base::in);
      int read_J, read_parity, read_sgn;
      double read_V;
      std::vector<double> pair_structure;
      Read_Vint_file >> read_J;
      Read_Vint_file >> read_parity;
      Read_Vint_file >> read_sgn;
      // std::cout<< read_J << read_parity << read_sgn << std::endl;
      for (size_t i = 0; i < ReadVnum; i++)
      {
        Read_Vint_file >> read_V;
        pair_structure.push_back(read_V);
        // std::cout<< pair_structure[i] << std::endl;
      }

      HamiltoaninColllectiveElements tempele = HamiltoaninColllectiveElements(read_J, read_parity, read_sgn, pair_structure);
      V_Iden->push_back(tempele);
      Read_Vint_file.close();
    }
    input_file.close();
  }
  else
  {
    std::cout << "The collective Vpn part will be implement" << std::endl;
    exit(0);
  }
}

void ReadWriteFiles::skip_comments(std::ifstream &in)
{
  size_t pos1, pos2, size_check = 8;
  std::streampos oldpos = in.tellg();
  for (std::string line; getline(in, line);)
  {
    std::string com = line.substr(0, size_check);
    pos1 = com.find('#');
    pos2 = com.find('!');
    if (pos1 > size_check and pos2 > size_check)
    {
      in.seekg(oldpos);
      break;
    }
    oldpos = in.tellg();
  }
}

#include <regex>
double ReadWriteFiles::skip_comments_Zerobody(std::ifstream &in)
{
  size_t pos1, pos2, size_check = 8;
  double ZeroBodyE = 0.;
  std::streampos oldpos = in.tellg();
  for (std::string line; getline(in, line);)
  {
    // Regular expression to match the floating number after the "Zero body term:" string
    std::regex re("Zero body term:\\s*([+-]?\\d*\\.\\d+|\\d+\\.\\d*)");

    // Search for the first match of the regular expression in the input text
    std::smatch match;
    if (std::regex_search(line, match, re))
    {
      // Extract the floating number from the first submatch
      ZeroBodyE = std::stod(match[1]);
    }

    std::string com = line.substr(0, size_check);
    pos1 = com.find('#');
    pos2 = com.find('!');
    if (pos1 > size_check and pos2 > size_check)
    {
      in.seekg(oldpos);
      break;
    }
    oldpos = in.tellg();
  }
  return ZeroBodyE;
}

// Read Tokyo format Snt file
void ReadWriteFiles::ReadTokyo(std::string filename, ModelSpace &ms, Hamiltonian &inputH)
{
  std::string line;
  std::ifstream infile;
  infile.open(filename);
  if (!infile.is_open())
  {
    // Failed to open the file
    std::cout << "Failed to open the file. " << filename << std::endl;
    exit(0);
  }

  if (!infile.good())
  {
    std::cerr << "************************************" << std::endl
              << "**    Trouble reading file  !!!   **" << filename << std::endl
              << "************************************" << std::endl;
    return;
  }

  // skip_comments(infile);
  ms.SetEnergyConstantShift(skip_comments_Zerobody(infile));
  int prtorb, ntnorb, pcore, ncore;
  infile >> prtorb >> ntnorb >> pcore >> ncore;
  ms.SetCoreProtonNum(pcore);
  ms.SetCoreNeutronNum(ncore);
  int numorb = prtorb + ntnorb;
  // std::cout << " " << prtorb << " " << ntnorb << " " << pcore << " " << ncore << std::endl;
  std::vector<Orbit> list_Orbit;
  int Orbit_map[numorb];
  for (int i = 0; i < numorb; i++)
  {
    int iorb, read_n, read_l, read_j, tz, read_parity;
    infile >> iorb >> read_n >> read_l >> read_j >> tz;
    read_parity = sgn(read_l);

    Orbit temp_orbit = Orbit(read_n, read_l, read_j, tz, read_parity);
    list_Orbit.push_back(temp_orbit);
    if (tz == Proton)
    {
      ms.Orbits_p.push_back(temp_orbit);
      Orbit_map[i] = ms.Orbits_p.size() - 1;
    }
    else
    {
      ms.Orbits_n.push_back(temp_orbit);
      Orbit_map[i] = ms.Orbits_n.size() - 1;
    }
    skip_comments(infile);
    // std::cout << i << " " << iorb << " " << read_n << " " << read_l << " " << read_j << " " << tz << std::endl;
  }

  skip_comments(infile);
  // Read a single line from the file
  std::getline(infile, line);
  if (line.empty())
  {
    skip_comments(infile);
    std::getline(infile, line);
  }
  skip_comments(infile);
  // Extract the numbers from the line
  std::istringstream iss(line);
  // Read the numbers into a vector
  std::vector<double> numbers;
  double num;
  skip_comments(infile);
  while (iss >> num)
  {
    numbers.push_back(num);
  }
  // Check the number of read numbers and do something with them
  if (numbers.size() == 1)
  {
    // std::cout << "Read one number: " << numbers[0] << std::endl;
    numorb = static_cast<int>(numbers[0]);
  }
  else if (numbers.size() == 2)
  {
    // std::cout << "Read two numbers: " << numbers[0] << ", " << numbers[1] << std::endl;
    numorb = static_cast<int>(numbers[0]);
  }
  else if (numbers.size() == 3)
  {
    // std::cout << "Read three numbers: " << numbers[0] << ", " << numbers[1] << ", " << numbers[2] << std::endl;
    numorb = static_cast<int>(numbers[0]);
    ms.Set_hw(numbers[2]);
  }
  else
  {
    std::cout << "Error: Unknown format! " << numbers.size() << std::endl;
  }
  // std::cout << "num SPE = " << numorb << std::endl;

  skip_comments(infile);
  for (int n = 0; n < numorb; n++)
  {
    int i, j;
    double h1;
    infile >> i >> j >> h1;
    // list_Orbit[i - 1].SetSPE(h1); // for valence calculation
    int tz = list_Orbit[i - 1].tz2;
    if (tz == Proton)
    {
      if (i == j)
        ms.Orbits_p[Orbit_map[i - 1]].SetSPE(h1); // for valence calculation
      OneBodyElement tempele = OneBodyElement(Orbit_map[i - 1], Orbit_map[j - 1], 0, Proton, h1);
      inputH.OBEs_p.push_back(tempele);
    }
    else
    {
      if (i == j)
        ms.Orbits_n[Orbit_map[i - 1]].SetSPE(h1); // for valence calculation
      OneBodyElement tempele = OneBodyElement(Orbit_map[i - 1], Orbit_map[j - 1], 0, Neutron, h1);
      inputH.OBEs_n.push_back(tempele);
    }
    // std::cout << i << " " << j << " " << h1 << "   " << list_Orbit[i - 1].tz2 << "   " << list_Orbit[i - 1].j2 << std::endl;
  }
  skip_comments(infile);
  getline(infile, line);

  /// read TBME
  skip_comments(infile);
  // Read a single line from the file
  std::getline(infile, line);
  // Extract the numbers from the line
  std::istringstream issTBME(line);
  // Read the numbers into a vector
  numbers.clear();
  int numTBME, method;
  skip_comments(infile);
  while (issTBME >> num)
  {
    numbers.push_back(num);
  }
  // Check the number of read numbers and do something with them
  if (numbers.size() == 2)
  {
    // std::cout << "Read one number: " << numbers[0] << std::endl;
    numTBME = static_cast<int>(numbers[0]);
    method = 0;
    inputH.SetMassDep(false);
  }
  else if (numbers.size() == 3)
  {
    // std::cout << "Read two numbers: " << numbers[0] << ", " << numbers[1] << std::endl;
    numTBME = static_cast<int>(numbers[0]);
    method = 0;
    inputH.SetMassDep(false);
  }
  else if (numbers.size() == 4)
  {
    // std::cout << "Read four numbers: " << numbers[0] << ", " << numbers[1] << ", " << numbers[2] << std::endl;
    numTBME = static_cast<int>(numbers[0]);
    method = static_cast<int>(numbers[1]);
    inputH.SetMassDep(method);
    ms.SetMassPowerFactor(-numbers[3]);
    ms.SetMassReferenceA(numbers[2]);
  }
  else
  {
    std::cout << "Error: Unknown format! TBMEs  " << numbers.size() << std::endl;
  }
  skip_comments(infile);
  for (int n = 0; n < numTBME; n++)
  {
    int i, j, k, l, jj;
    double tbme;
    infile >> i >> j >> k >> l >> jj >> tbme;
    int tz1 = list_Orbit[i - 1].tz2;
    int tz2 = list_Orbit[j - 1].tz2;
    HamiltonianElements tempele = HamiltonianElements(tz1 + tz2, Orbit_map[i - 1], Orbit_map[j - 1], Orbit_map[k - 1], Orbit_map[l - 1], jj, tbme);
    if (tz1 + tz2 == 0) // Vpn
    {
      inputH.Vpn.push_back(tempele);
    }
    else if (tz1 + tz2 == 2 * Proton)
    {
      inputH.Vpp.push_back(tempele);
    }
    else if (tz1 + tz2 == 2 * Neutron)
    {
      inputH.Vnn.push_back(tempele);
    }
  }
  infile.close();
}

void ReadWriteFiles::MPI_ReadMatrix(int dim, ComplexNum *ele, const std::string &filename)
{
  int totalnum, i, val;
  double *tempME;
  MPI_File fh;
  MPI_Status status;
  totalnum = dim * dim;
  tempME = (double *)mkl_malloc((2 * totalnum) * sizeof(double), 64);

  if (MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
  {
    fprintf(stderr, "Can not open %s file\n", filename.c_str());
    exit(0);
  }
  if (MPI_File_read(fh, &i, 1, MPI_INT, &status) != MPI_SUCCESS)
  {
    printf("File read error: 1\n");
    exit(0);
  }
  if (i != dim)
  {
    printf("Read number error !!  %d    %d\n", i, dim);
  }
  if (MPI_File_read(fh, tempME, 2 * totalnum, MPI_DOUBLE, &status) != MPI_SUCCESS)
  {
    printf("File read error: 1\n");
    exit(0);
  }
  MPI_File_close(&fh);
  /// convert to complex number
  for (i = 0; i < totalnum; i++)
  {
    ele[i] = ComplexNum(tempME[2 * i + 0], tempME[2 * i + 1]);
  }
  mkl_free(tempME);
  return;
}

void ReadWriteFiles::OutputME(int dim, ComplexNum *ele, string filename)
{
  int totalnum, i;
  double *tempME;
  FILE *fp;
  totalnum = dim * dim;
  tempME = (double *)mkl_malloc((2 * totalnum) * sizeof(double), 64);
  for (i = 0; i < totalnum; i++)
  {
    tempME[2 * i + 0] = ele[i].real();
    tempME[2 * i + 1] = ele[i].imag();
  }
  if ((fp = fopen(filename.c_str(), "wb")) == NULL)
  {
    printf("Can not open %s file.\n", filename.c_str());
    exit(0);
  }
  fwrite(&dim, sizeof(int), 1, fp);
  fwrite(tempME, sizeof(double), 2 * totalnum, fp);
  fclose(fp);
  mkl_free(tempME);
}

void ReadWriteFiles::SavePairStruc_DiffPairs_Iden(ModelSpace &ms, const std::vector<double> &x)
{

  if (ms.IsJbrokenPair())
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int isospin = Proton;
    int N = ms.GetPairNumber(isospin);
    int PairStr_num = ms.GetMSchemeNumberOfFreePara(isospin);
    int IsospinSP;
    if (isospin == Neutron)
    {
      IsospinSP = ms.GetMSchemeNumberOfFreePara(Proton) * N;
      if (!(ff = fopen(Save_Parameters_n.c_str(), "w")))
      {
        printf("Open para_p.dat file error!!\n");
        exit(0);
      }
    }
    else
    {
      IsospinSP = 0;
      if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
      {
        printf("Open para_n.dat file error!!\n");
        exit(0);
      }
    }

    fprintf(ff, " %d\n", PairStr_num * N);
    for (loopN = 1; loopN <= N; loopN++)
    {
      SP = IsospinSP + (loopN - 1) * PairStr_num;
      for (i = 0; i < PairStr_num; i++)
      {
        fprintf(ff, " %d	%.15f   %d \n", i, x[SP + i], loopN);
      }
    }
    fclose(ff);
  }
  else
  {

    int i, j, SP, loopN, count;
    FILE *ff;
    int N_p = ms.GetPairNumber(Proton);
    int PairStr_num_p = ms.GetTotal_NonCollectivePair_num_p();
    vector<CollectivePairs> CP_p = ms.GetCollectivePairVector(Proton);
    if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
    {
      printf("Open para_p.dat file error!!\n");
      exit(0);
    }
    fprintf(ff, " %d\n", PairStr_num_p * N_p);
    for (loopN = 1; loopN <= N_p; loopN++)
    {
      SP = PairStr_num_p * (loopN - 1);
      count = 0;
      for (i = 0; i < ms.GetCollectivePairNumber(Proton); i++)
      {
        for (j = 0; j < CP_p[i].GetNumberofNoncollectivePair(); j++)
        {
          fprintf(ff, " %d	%.15f   %d %d %d %d %d\n", i, x[SP + count], CP_p[i].J, CP_p[i].parity, CP_p[i].GetIndex_i(j), CP_p[i].GetIndex_j(j), loopN);
          count++;
        }
      }
    }
    fclose(ff);
  }
}

void ReadWriteFiles::SavePairStruc_SamePairs_Iden(ModelSpace &ms, const std::vector<double> &x)
{

  if (ms.IsJbrokenPair())
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int isospin = Proton;
    int N = ms.GetPairNumber(isospin);
    int PairStr_num = ms.GetMSchemeNumberOfFreePara(isospin);
    int IsospinSP;
    if (isospin == Neutron)
    {
      IsospinSP = ms.GetMSchemeNumberOfFreePara(Proton);
      if (!(ff = fopen(Save_Parameters_n.c_str(), "w")))
      {
        printf("Open para_p.dat file error!!\n");
        exit(0);
      }
    }
    else
    {
      IsospinSP = 0;
      if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
      {
        printf("Open para_n.dat file error!!\n");
        exit(0);
      }
    }

    fprintf(ff, " %d\n", PairStr_num);
    SP = IsospinSP;
    for (i = 0; i < PairStr_num; i++)
    {
      fprintf(ff, " %d	%.15f   %d \n", i, x[SP + i], loopN);
    }
    fclose(ff);
  }
  else
  {

    int i, j, SP, loopN, count;
    FILE *ff;
    int N_p = ms.GetPairNumber(Proton);
    int PairStr_num_p = ms.GetTotal_NonCollectivePair_num_p();
    vector<CollectivePairs> CP_p = ms.GetCollectivePairVector(Proton);
    if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
    {
      printf("Open para_p.dat file error!!\n");
      exit(0);
    }
    fprintf(ff, " %d\n", PairStr_num_p * N_p);
    for (loopN = 1; loopN <= N_p; loopN++)
    {
      SP = PairStr_num_p * (loopN - 1);
      count = 0;
      for (i = 0; i < ms.GetCollectivePairNumber(Proton); i++)
      {
        for (j = 0; j < CP_p[i].GetNumberofNoncollectivePair(); j++)
        {
          fprintf(ff, " %d	%.15f   %d %d %d %d %d\n", i, x[SP + count], CP_p[i].J, CP_p[i].parity, CP_p[i].GetIndex_i(j), CP_p[i].GetIndex_j(j), loopN);
          count++;
        }
      }
    }
    fclose(ff);
  }
}

void ReadWriteFiles::SavePairStruc_DiffPairs(int isospin, ModelSpace &ms, const std::vector<double> &x)
{
  if (ms.IsJbrokenPair())
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int N = ms.GetPairNumber(isospin);
    int PairStr_num = ms.GetMSchemeNumberOfFreePara(isospin);
    int IsospinSP;
    if (isospin == Neutron)
    {
      IsospinSP = ms.GetMSchemeNumberOfFreePara(Proton) * N;
      if (!(ff = fopen(Save_Parameters_n.c_str(), "w")))
      {
        printf("Open para_p.dat file error!!\n");
        exit(0);
      }
    }
    else
    {
      IsospinSP = 0;
      if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
      {
        printf("Open para_n.dat file error!!\n");
        exit(0);
      }
    }

    fprintf(ff, " %d\n", PairStr_num * N);
    for (loopN = 1; loopN <= N; loopN++)
    {
      SP = IsospinSP + (loopN - 1) * PairStr_num;
      for (i = 0; i < PairStr_num; i++)
      {
        fprintf(ff, " %d	%.15f   %d \n", i, x[SP + i], loopN);
      }
    }
    fclose(ff);
  }
  else
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int N = ms.GetPairNumber(isospin);
    int PairStr_num = ms.Get_NonCollecitvePairNumber(isospin);
    vector<CollectivePairs> CP = ms.GetCollectivePairVector(isospin);

    int IsospinSP;
    if (isospin == Neutron)
    {
      IsospinSP = ms.GetTotal_NonCollectivePair_num_p() * ms.GetProtonPairNum();
      if (!(ff = fopen(Save_Parameters_n.c_str(), "w")))
      {
        printf("Open para_p.dat file error!!\n");
        exit(0);
      }
    }
    else
    {
      IsospinSP = 0;
      if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
      {
        printf("Open para_n.dat file error!!\n");
        exit(0);
      }
    }

    fprintf(ff, " %d\n", PairStr_num * N);
    for (loopN = 1; loopN <= N; loopN++)
    {
      SP = PairStr_num * (loopN - 1) + IsospinSP;
      count = 0;
      for (i = 0; i < ms.GetCollectivePairNumber(isospin); i++)
      {
        for (j = 0; j < CP[i].GetNumberofNoncollectivePair(); j++)
        {
          fprintf(ff, " %d	%.15f   %d %d %d %d %d\n", i, x[SP + count], CP[i].J, CP[i].parity, CP[i].GetIndex_i(j), CP[i].GetIndex_j(j), loopN);
          count++;
        }
      }
    }
    fclose(ff);
  }
}

void ReadWriteFiles::SavePairStruc_SamePairs(int isospin, ModelSpace &ms, const std::vector<double> &x)
{
  if (ms.IsJbrokenPair())
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int N = ms.GetPairNumber(isospin);
    int PairStr_num = ms.GetMSchemeNumberOfFreePara(isospin);
    int IsospinSP;
    if (isospin == Neutron)
    {
      IsospinSP = ms.GetMSchemeNumberOfFreePara(Proton);
      if (!(ff = fopen(Save_Parameters_n.c_str(), "w")))
      {
        printf("Open para_p.dat file error!!\n");
        exit(0);
      }
    }
    else
    {
      IsospinSP = 0;
      if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
      {
        printf("Open para_n.dat file error!!\n");
        exit(0);
      }
    }

    fprintf(ff, " %d\n", PairStr_num);
    loopN = 1;
    SP = IsospinSP;
    count = 0;
    for (i = 0; i < PairStr_num; i++)
    {
      fprintf(ff, " %d	%.15f   %d \n", i, x[SP + count], loopN);
      count++;
    }
    fclose(ff);
  }
  else
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int N = ms.GetPairNumber(isospin);
    int PairStr_num = ms.Get_NonCollecitvePairNumber(isospin);
    vector<CollectivePairs> CP = ms.GetCollectivePairVector(isospin);

    int IsospinSP;
    if (isospin == Neutron)
    {
      IsospinSP = ms.GetTotal_NonCollectivePair_num_p();
      if (!(ff = fopen(Save_Parameters_n.c_str(), "w")))
      {
        printf("Open para_p.dat file error!!\n");
        exit(0);
      }
    }
    else
    {
      IsospinSP = 0;
      if (!(ff = fopen(Save_Parameters_p.c_str(), "w")))
      {
        printf("Open para_n.dat file error!!\n");
        exit(0);
      }
    }

    fprintf(ff, " %d\n", PairStr_num);
    // for (loopN = 1; loopN <= N; loopN++)
    //{
    loopN = 1;
    SP = IsospinSP;
    count = 0;
    for (i = 0; i < ms.GetCollectivePairNumber(isospin); i++)
    {
      for (j = 0; j < CP[i].GetNumberofNoncollectivePair(); j++)
      {
        fprintf(ff, " %d	%.15f   %d %d %d %d %d\n", i, x[SP + count], CP[i].J, CP[i].parity, CP[i].GetIndex_i(j), CP[i].GetIndex_j(j), loopN);
        count++;
      }
    }
    //}
    fclose(ff);
  }
}

void ReadWriteFiles::ReadME_vector(int dim, std::vector<ComplexNum> &ele, string filename)
{
  int totalnum, i, j, val;
  FILE *fp;
  totalnum = dim * dim;
  std::vector<double> tempME;
  tempME.resize(2 * totalnum);

  if ((fp = fopen(filename.c_str(), "rb")) == NULL)
  {
    printf("Can not open %s file.\n", filename.c_str());
    exit(0);
  }
  val = fread(&i, sizeof(int), 1, fp);
  if (i != dim)
  {
    printf("Read number error !!  %d    %d\n", i, dim);
  }
  val = fread(&tempME[0], sizeof(double), tempME.size(), fp);
  fclose(fp);
  val = 0;
  for (i = 0; i < dim * dim; i++)
  {
    ele[i] = ComplexNum(tempME[2 * i + 0], tempME[2 * i + 1]);
  }
  return;
}

void ReadWriteFiles::Output_GCM_points(string filename, ModelSpace &ms, const std::vector<double> &x, double value, int saved_number)
{
  if (ms.IsJbrokenPair())
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int Np = ms.GetPairNumber(Proton);
    int PairStr_num_p = ms.GetMSchemeNumberOfFreePara(Proton);

    int Nn = ms.GetPairNumber(Neutron);
    int PairStr_num_n = ms.GetMSchemeNumberOfFreePara(Neutron);

    if (!(ff = fopen(filename.c_str(), "w")))
    {
      printf("Open %s file error!!\n", filename.c_str());
      exit(0);
    }

    if (ms.GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
      fprintf(ff, " %d  %d  %lf %d\n", PairStr_num_p * Np, PairStr_num_n * Nn, value, saved_number);
      for (loopN = 1; loopN <= Np; loopN++)
      {
        SP = (loopN - 1) * PairStr_num_p;
        for (i = 0; i < PairStr_num_p; i++)
        {
          fprintf(ff, " %d	%.15f   %d  Proton\n", i, x[SP + i], loopN);
        }
      }
      for (loopN = 1; loopN <= Nn; loopN++)
      {
        SP = PairStr_num_p * Np + (loopN - 1) * PairStr_num_n;
        for (i = 0; i < PairStr_num_n; i++)
        {
          fprintf(ff, " %d	%.15f   %d  Neutron\n", i, x[SP + i], loopN);
        }
      }
      fclose(ff);
    }
    else // 1 identical pairs
    {
      loopN = 1;
      fprintf(ff, " %d  %d  %lf %d\n", PairStr_num_p, PairStr_num_n, value, saved_number);
      for (i = 0; i < PairStr_num_p; i++)
      {
        fprintf(ff, " %d	%.15f   %d  Proton\n", i, x[i], loopN);
      }

      SP = PairStr_num_p;
      for (i = 0; i < PairStr_num_n; i++)
      {
        fprintf(ff, " %d	%.15f   %d  Neutron\n", i, x[SP + i], loopN);
      }
      fclose(ff);
    }
  }
  else
  {
    int i, j, SP, loopN, count;
    FILE *ff;
    int Np = ms.GetPairNumber(Proton);
    int PairStr_num_p = ms.Get_NonCollecitvePairNumber(Proton);
    vector<CollectivePairs> CP_p = ms.GetCollectivePairVector(Proton);

    int Nn = ms.GetPairNumber(Neutron);
    int PairStr_num_n = ms.Get_NonCollecitvePairNumber(Neutron);
    vector<CollectivePairs> CP_n = ms.GetCollectivePairVector(Neutron);

    ////////////////////
    if (!(ff = fopen(filename.c_str(), "w")))
    {
      printf("Open %s file error!!\n", filename.c_str());
      exit(0);
    }
    if (ms.GetBasisType() == 0) // 0 different pairs; 1 identical pairs
    {
      fprintf(ff, " %d  %d  %lf %d\n", PairStr_num_p * Np, PairStr_num_n * Nn, value, saved_number);
      for (loopN = 1; loopN <= Np; loopN++)
      {
        SP = PairStr_num_p * (loopN - 1);
        count = 0;
        for (i = 0; i < ms.GetCollectivePairNumber(Proton); i++)
        {
          for (j = 0; j < CP_p[i].GetNumberofNoncollectivePair(); j++)
          {
            fprintf(ff, " %d	%.15f   %d %d %d %d %d Proton\n", i, x[SP + count], CP_p[i].J, CP_p[i].parity, CP_p[i].GetIndex_i(j), CP_p[i].GetIndex_j(j), loopN);
            count++;
          }
        }
      }
      for (loopN = 1; loopN <= Nn; loopN++)
      {
        SP = PairStr_num_p * Np + PairStr_num_n * (loopN - 1);
        count = 0;
        for (i = 0; i < ms.GetCollectivePairNumber(Neutron); i++)
        {
          for (j = 0; j < CP_n[i].GetNumberofNoncollectivePair(); j++)
          {
            fprintf(ff, " %d	%.15f   %d %d %d %d %d Neutron\n", i, x[SP + count], CP_n[i].J, CP_n[i].parity, CP_n[i].GetIndex_i(j), CP_n[i].GetIndex_j(j), loopN);
            count++;
          }
        }
      }
      fclose(ff);
    }
    else // 1 identical pairs
    {
      fprintf(ff, " %d  %d  %lf %d\n", PairStr_num_p, PairStr_num_n, value, saved_number);
      // SP = PairStr_num_p * (loopN - 1);
      count = 0;
      loopN = 1;
      for (i = 0; i < ms.GetCollectivePairNumber(Proton); i++)
      {
        for (j = 0; j < CP_p[i].GetNumberofNoncollectivePair(); j++)
        {
          fprintf(ff, " %d	%.15f   %d %d %d %d %d Proton\n", i, x[count], CP_p[i].J, CP_p[i].parity, CP_p[i].GetIndex_i(j), CP_p[i].GetIndex_j(j), loopN);
          count++;
        }
      }

      SP = PairStr_num_p;
      count = 0;
      for (i = 0; i < ms.GetCollectivePairNumber(Neutron); i++)
      {
        for (j = 0; j < CP_n[i].GetNumberofNoncollectivePair(); j++)
        {
          fprintf(ff, " %d	%.15f   %d %d %d %d %d Neutron\n", i, x[SP + count], CP_n[i].J, CP_n[i].parity, CP_n[i].GetIndex_i(j), CP_n[i].GetIndex_j(j), loopN);
          count++;
        }
      }
      fclose(ff);
    }
  }
}

vector<string> ReadWriteFiles::Get_all_files_names_within_folder(string folder)
{
  DIR *dr;
  struct dirent *en;
  vector<string> names;
  dr = opendir(folder.c_str()); // open all directory
  if (dr)
  {
    while ((en = readdir(dr)) != NULL)
    {
      names.push_back(en->d_name);
    }
    closedir(dr); // close all directory
  }
  return names;
}

void ReadWriteFiles::Read_GCM_points(ModelSpace &ms, string Filename, std::vector<double> &para_x, std::vector<double> &E)
{
  FILE *ff;
  int readnum_p, readnum_n, temp, myid, i;
  char cmd[OneLine], *chr;
  double value;

  para_x.clear();

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  ff = fopen(Filename.c_str(), "r"); // read proton part
  if (fscanf(ff, "  %d  %d %lf %d\n", &readnum_p, &readnum_n, &value, &temp) != 4)
  {
    printf("input file error!!\n");
    exit(0);
  }
  E.push_back(value);

  if (ms.GetBasisType() == 0) // 0 different pairs; 1 identical pairs
  {
    if (ms.GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
    {
      if (readnum_p != ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton))
      {
        if (myid == 0)
        {
          printf("Read part of Proton parameters! %d  %d  %d\n", readnum_p, ms.GetPairNumber(Proton), ms.Get_NonCollecitvePairNumber(Proton));
        }
        exit(0);
      }
      if (readnum_n != ms.GetPairNumber(Neutron) * ms.Get_NonCollecitvePairNumber(Neutron))
      {
        if (myid == 0)
        {
          printf("Read part of Neutron parameters! %d  %d  %d\n", readnum_n, ms.GetPairNumber(Neutron), ms.Get_NonCollecitvePairNumber(Neutron));
        }
        exit(0);
      }
    }
    else
    {
      if (readnum_p != ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum())
      {
        if (myid == 0)
        {
          printf("Read part of Proton parameters! %d  %d  %d\n", readnum_p, ms.GetPairNumber(Proton), ms.GetMSchemeNumberOfFreePara(Proton));
        }
        exit(0);
      }
      if (readnum_n != ms.GetPairNumber(Neutron) * ms.GetMSchemeNumberOfFreePara(Neutron))
      {
        if (myid == 0)
        {
          printf("Read part of Neutron parameters! %d  %d  %d\n", readnum_n, ms.GetPairNumber(Neutron), ms.GetMSchemeNumberOfFreePara(Neutron));
        }
        exit(0);
      }
    }
  }
  else
  {
    if (ms.GetPairType() == 0) // 0 J conserved pairs; 1 J borken pairs
    {
      if (readnum_p != ms.Get_NonCollecitvePairNumber(Proton))
      {
        if (myid == 0)
        {
          printf("Read part of Proton parameters! %d  %d\n", readnum_p, ms.Get_NonCollecitvePairNumber(Proton));
        }
        exit(0);
      }
      if (readnum_n != ms.Get_NonCollecitvePairNumber(Neutron))
      {
        if (myid == 0)
        {
          printf("Read part of Neutron parameters! %d  %d\n", readnum_n, ms.Get_NonCollecitvePairNumber(Neutron));
        }
        exit(0);
      }
    }
    else
    {
      if (readnum_p != ms.GetMSchemeNumberOfFreePara(Proton))
      {
        if (myid == 0)
        {
          printf("Read part of Proton parameters! %d   %d\n", readnum_p, ms.GetMSchemeNumberOfFreePara(Proton));
        }
        exit(0);
      }
      if (readnum_n != ms.GetMSchemeNumberOfFreePara(Neutron))
      {
        if (myid == 0)
        {
          printf("Read part of Neutron parameters! %d %d\n", readnum_n, ms.GetMSchemeNumberOfFreePara(Neutron));
        }
        exit(0);
      }
    }
  }

  for (i = 0; i < readnum_p + readnum_n; i++)
  {
    chr = fgets(cmd, OneLine, ff);
    // printf("%s\n",cmd);
    if (sscanf(cmd, " %d	%lf", &temp, &value) != 2)
    {
      printf("Reading input file error!! %d %d %f \n", i, temp, value);
      exit(0);
    }
    para_x.push_back(value);
  }
  fclose(ff);
  return;
}

void ReadWriteFiles::Read_GCM_HF_points(string Filename, std::vector<double> &para_x, std::vector<double> &E)
{
  FILE *ff;
  int N_p, dim_p, N_n, dim_n, myid, temp, i;
  char cmd[OneLine], *chr;
  double value;
  para_x.clear();
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  ff = fopen(Filename.c_str(), "r"); // read proton part
  if (fscanf(ff, "  %d  %d %d  %d %lf \n", &N_p, &dim_p, &N_n, &dim_n, &value) != 5)
  {
    printf("input file error!!\n");
    exit(0);
  }
  E.push_back(value);
  for (i = 0; i < N_p * dim_p + N_n * dim_n; i++)
  {
    chr = fgets(cmd, OneLine, ff);
    // printf("%s\n",cmd);
    if (sscanf(cmd, " %d	%lf", &temp, &value) != 2)
    {
      printf("Reading input file error!! %d %d %f \n", i, temp, value);
      exit(0);
    }
    para_x.push_back(value);
  }
  fclose(ff);
  return;
}

void ReadWriteFiles::SavePairStruc_MCMC(string filename, int isospin, int num_p, int num_n, const std::vector<double> &x)
{
  int i, SP, num;
  FILE *ff;
  string OutputFile = this->MCMC_output_path + "/" + filename;

  if (!(ff = fopen(OutputFile.c_str(), "w")))
  {
    printf("Open %s file error!!\n", OutputFile.c_str());
    exit(0);
  }

  if (isospin == Proton)
  {
    fprintf(ff, " %d\n", num_p);
    SP = 0;
    num = num_p;
  }
  else
  {
    fprintf(ff, " %d\n", num_n);
    SP = num_p;
    num = num_n;
  }
  for (i = 0; i < num; i++)
  {
    fprintf(ff, " %d	%.15f \n", i, x[SP + i]);
  }
  fclose(ff);
}

void ReadWriteFiles::Save_MCMC_GCMinput(string filename, int num_p, int num_n, int step, const std::vector<double> &x, double Einput)
{
  int i;
  FILE *ff;
  string OutputFile = filename;
  if (!(ff = fopen(OutputFile.c_str(), "w")))
  {
    printf("Open %s file error!!\n", OutputFile.c_str());
    exit(0);
  }
  fprintf(ff, " %d  %d  %.8f %d\n", num_p, num_n, Einput, step);
  for (i = 0; i < num_p; i++)
  {
    fprintf(ff, " %d	%.15f   Proton\n", i, x[i]);
  }
  for (i = 0; i < num_n; i++)
  {
    fprintf(ff, " %d	%.15f   Neutron\n", i, x[num_p + i]);
  }
  fclose(ff);
}

void ReadWriteFiles::Save_HF_Parameters(int N_p, int dim_p, int N_n, int dim_n, double *prt, string filename)
{
  FILE *fp;
  int total = N_p * dim_p + N_n * dim_n;
  if ((fp = fopen(filename.c_str(), "wb")) == NULL)
  {
    printf("Can not open %s file.\n", filename.c_str());
    exit(0);
  }
  fwrite(&N_p, sizeof(int), 1, fp);
  fwrite(&dim_p, sizeof(int), 1, fp);
  fwrite(&N_n, sizeof(int), 1, fp);
  fwrite(&dim_p, sizeof(int), 1, fp);
  fwrite(prt, sizeof(double), 2 * total, fp);
  fclose(fp);
}

void ReadWriteFiles::Save_HF_Parameters_TXT(int N_p, int dim_p, int N_n, int dim_n, double *prt, string filename)
{
  FILE *fp;
  int total = N_p * dim_p + N_n * dim_n;
  if ((fp = fopen(filename.c_str(), "w")) == NULL)
  {
    printf("Can not open %s file.\n", filename.c_str());
    exit(0);
  }
  fprintf(fp, " %d  %d  %d  %d\n", N_p, dim_p, N_n, dim_n);

  for (int i = 0; i < total; i++)
  {
    fprintf(fp, " %d	%.15f   \n", i, prt[i]);
  }
  fclose(fp);
}

void ReadWriteFiles::Save_HF_Parameters_TXT(int N_p, int dim_p, int N_n, int dim_n, double *prt, double EHF, string filename)
{
  FILE *fp;
  int total = N_p * dim_p + N_n * dim_n;
  if ((fp = fopen(filename.c_str(), "w")) == NULL)
  {
    printf("Can not open %s file.\n", filename.c_str());
    exit(0);
  }
  fprintf(fp, " %d  %d  %d  %d  %lf\n", N_p, dim_p, N_n, dim_n, EHF);

  for (int i = 0; i < total; i++)
  {
    fprintf(fp, " %d	%.15f   \n", i, prt[i]);
  }
  fclose(fp);
}

void ReadWriteFiles::Read_HF_Parameters_TXT(string Filename, double *para_x)
{
  FILE *ff;
  int N_p, N_n, dim_p, dim_n, temp;
  double E_cal;
  char cmd[OneLine], *chr;
  ff = fopen(Filename.c_str(), "r"); // read proton part
  if (fscanf(ff, "  %d  %d  %d  %d %lf\n", &N_p, &dim_p, &N_n, &dim_n, &E_cal) != 5)
  {
    printf("input file error!!\n");
    exit(0);
  }
  // std::cout << N_p << N_n << dim_p << dim_n << E_cal << std::endl;
  for (int i = 0; i < N_p * dim_p + N_n * dim_n; i++)
  {
    chr = fgets(cmd, OneLine, ff);
    // printf("%s\n",cmd);
    if (sscanf(cmd, " %d	%lf", &temp, &(para_x[i])) != 2)
    {
      printf("Reading input file error!! %d %d %f \n", i, temp, para_x[i]);
      exit(0);
    }
  }
  fclose(ff);
  return;
}

void ReadWriteFiles::Save_HF_For_NPSM_TXT(int N_p, int dim_p, int N_n, int dim_n, double *prt, double E, string filename)
{
  FILE *fp;
  int total = N_p * dim_p + N_n * dim_n;
  int total_p, total_n;
  total_p = (dim_p * dim_p - dim_p) / 2;
  total_n = (dim_n * dim_n - dim_n) / 2;

  if ((fp = fopen(filename.c_str(), "w")) == NULL)
  {
    printf("Can not open %s file.\n", filename.c_str());
    exit(0);
  }
  fprintf(fp, " %d  %d  %lf   1\n", total_p * N_p / 2, total_n * N_n / 2, E);
  int count = 0;
  double value;
  // Proton -------------------------------//
  for (int loop_N = 0; loop_N < N_p / 2; loop_N++)
  {
    count = 0;
    for (size_t i = 0; i < dim_p; i++)
    {
      for (size_t j = 0; j < i; j++)
      {
        // value = 0.5 * (prt[2 * loop_N * dim_p + i] * prt[2 * loop_N * dim_p + dim_p + j] - prt[2 * loop_N * dim_p + j] * prt[2 * loop_N * dim_p + dim_p + i]);
        // value = (prt[2 * loop_N * dim_p + i] * prt[2 * loop_N * dim_p + dim_p + j]);
        value = 0.5 * (prt[2 * loop_N * dim_p + i] * prt[2 * loop_N * dim_p + dim_p + j] - prt[2 * loop_N * dim_p + j] * prt[2 * loop_N * dim_p + dim_p + i]);
        fprintf(fp, " %d	%lf   %d  Proton\n", count, value, loop_N + 1);
        count++;
      }
    }
  }
  // Neutron -------------------------------//
  for (int loop_N = 0; loop_N < N_n / 2; loop_N++)
  {
    count = 0;
    for (size_t i = 0; i < dim_n; i++)
    {
      for (size_t j = 0; j < i; j++)
      {
        // value = 0.5 * (prt[N_p * dim_p + 2 * loop_N * dim_n + i] * prt[N_p * dim_p / 2 + 2 * loop_N * dim_n + dim_n + j] - prt[N_p * dim_p / 2 + 2 * loop_N * dim_n + j] * prt[N_p * dim_p / 2 + 2 * loop_N * dim_n + dim_n + i]);
        // value = (prt[N_p * dim_p + 2 * loop_N * dim_n + i] * prt[N_p * dim_p + 2 * loop_N * dim_n + dim_n + j]);
        value = 0.5 * (prt[N_p * dim_p + 2 * loop_N * dim_n + i] * prt[N_p * dim_p + 2 * loop_N * dim_n + dim_n + j] - prt[N_p * dim_p + 2 * loop_N * dim_n + j] * prt[N_p * dim_p + 2 * loop_N * dim_n + dim_n + i]);
        fprintf(fp, " %d	%lf   %d  Neutron\n", count, value, loop_N + 1);
        count++;
      }
    }
  }
  fclose(fp);
}

// Initial variation parameters
/*
void ReadWriteFiles::InitVariationPara_diff_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 2) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton))
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d  %d\n", readxnum_p, ms.GetPairNumber(Proton), ms.Get_NonCollecitvePairNumber(Proton));
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    ms.SetNumPara_p(readxnum_p);
    fclose(ff);
    ///------------------------
    for (i = 0; i < ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton); i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
    }
    std::cout << "Initialize parameters: " << readX.size() << std::endl;
  }
  else
  {
    srand(ms.GetRandomSeed());
    for (i = 0; i < ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton); i++)
    {
      sprintf(cmd, "x[%d]", i);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    ms.SetNumPara_p(ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton));
    std::cout << "Initialize parameters: " << ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton) << std::endl;
  }
  return;
}

void ReadWriteFiles::InitVariationPara_diff_Mscheme_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 2) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum())
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d\n", readxnum_p, ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum());
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    for (i = 0; i < readxnum_p; i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout<< i<<"  "<< readX[i] <<std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Identical nucleons parameters: " << readX.size() << std::endl;
  }
  else
  {
    readxnum_p = ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum();
    srand(ms.GetRandomSeed());
    for (i = 0; i < readxnum_p; i++)
    {
      sprintf(cmd, "x[%d]", i);
      //(*upar).add(cmd, 1., 0.1);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    if (myid == 0)
    {
      std::cout << "Initialize  Identical nucleons parameters: " << readxnum_p << std::endl;
    }
  }
  ms.SetNumPara_p(readxnum_p);
  ms.UsingJbrokenPair(); // Set using Rotational broken basis in inital stage
  return;
}

void ReadWriteFiles::InitVariationPara_SamePairs_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 2) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.Get_NonCollecitvePairNumber(Proton))
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d\n", readxnum_p, ms.Get_NonCollecitvePairNumber(Proton));
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    ms.SetNumPara_p(readxnum_p);
    fclose(ff);
    ///------------------------
    for (i = 0; i < ms.Get_NonCollecitvePairNumber(Proton); i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
    }
    std::cout << "Initialize parameters: " << readX.size() << std::endl;
  }
  else
  {
    srand(ms.GetRandomSeed());
    for (i = 0; i < ms.Get_NonCollecitvePairNumber(Proton); i++)
    {
      sprintf(cmd, "x[%d]", i);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    ms.SetNumPara_p(ms.Get_NonCollecitvePairNumber(Proton));
    std::cout << "Initialize parameters: " << ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton) << std::endl;
  }
  return;
}

void ReadWriteFiles::InitVariationPara_SamePairs_Mscheme_Iden(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 2) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.GetMSchemeNumberOfFreePara(Proton))
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d\n", readxnum_p, ms.GetMSchemeNumberOfFreePara(Proton));
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    for (i = 0; i < readxnum_p; i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout<< i<<"  "<< readX[i] <<std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Identical nucleons parameters: " << readX.size() << std::endl;
  }
  else
  {
    readxnum_p = ms.GetMSchemeNumberOfFreePara(Proton);
    srand(ms.GetRandomSeed());
    for (i = 0; i < readxnum_p; i++)
    {
      sprintf(cmd, "x[%d]", i);
      //(*upar).add(cmd, 1., 0.1);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    if (myid == 0)
    {
      std::cout << "Initialize  Identical nucleons parameters: " << readxnum_p << std::endl;
    }
  }
  ms.SetNumPara_p(readxnum_p);
  ms.UsingJbrokenPair(); // Set using Rotational broken basis in inital stage
  return;
}

void ReadWriteFiles::InitVariationPara_diff_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 3) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton))
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d  %d\n", readxnum_p, ms.GetPairNumber(Proton), ms.Get_NonCollecitvePairNumber(Proton));
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    for (i = 0; i < readxnum_p; i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout<< i<<"  "<< readX[i] <<std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Proton parameters: " << readX.size() << std::endl;

    ms.SetNumPara_p(readxnum_p);
    //-----------------------------------------------
    ff = fopen(argv[2], "r"); // read neutron part
    if (fscanf(ff, "%d\n", &readxnum_n) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_n != ms.GetPairNumber(Neutron) * ms.Get_NonCollecitvePairNumber(Neutron))
    {
      if (myid == 0)
      {
        printf("Read part of Neutron parameters! %d  %d  %d\n", readxnum_n, ms.GetPairNumber(Neutron), ms.Get_NonCollecitvePairNumber(Neutron));
        exit(0);
      }
    }
    readX.clear();
    for (i = 0; i < readxnum_n; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    readxnum_p = ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton);
    for (i = 0; i < readxnum_n; i++) // for neutron
    {
      sprintf(cmd, "x[%d]", i + readxnum_p);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout << i << "  " << readX[i] << std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Neutron parameters: " << readX.size() << std::endl;

    ms.SetNumPara_n(readxnum_n);
  }
  else
  {
    readxnum_p = ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton);
    readxnum_n = ms.GetPairNumber(Neutron) * ms.Get_NonCollecitvePairNumber(Neutron);
    ms.SetNumPara_p(readxnum_p);
    ms.SetNumPara_n(readxnum_n);
    srand(ms.GetRandomSeed());
    for (i = 0; i < readxnum_p + readxnum_n; i++)
    {
      sprintf(cmd, "x[%d]", i);
      //(*upar).add(cmd, 1., 0.1);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    if (myid == 0)
    {
      std::cout << "Initialize Proton parameters: " << readxnum_p << std::endl;
      std::cout << "Initialize Neutron parameters: " << readxnum_n << std::endl;
    }
  }
  return;
}

void ReadWriteFiles::InitVariationPara_diff_Mscheme_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 3) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum())
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d\n", readxnum_p, ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum());
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    for (i = 0; i < readxnum_p; i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout<< i<<"  "<< readX[i] <<std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Proton parameters: " << readX.size() << std::endl;

    //-----------------------------------------------
    ff = fopen(argv[2], "r"); // read neutron part
    if (fscanf(ff, "%d\n", &readxnum_n) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_n != ms.GetMSchemeNumberOfFreePara(Neutron) * ms.GetNeutronPairNum())
    {
      if (myid == 0)
      {
        printf("Read part of Neutron parameters! %d  %d\n", readxnum_n, ms.GetMSchemeNumberOfFreePara(Neutron) * ms.GetNeutronPairNum());
        exit(0);
      }
    }
    readX.clear();
    for (i = 0; i < readxnum_n; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    readxnum_p = ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum();
    for (i = 0; i < readxnum_n; i++) // for neutron
    {
      sprintf(cmd, "x[%d]", i + readxnum_p);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout << i << "  " << readX[i] << std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Neutron parameters: " << readX.size() << std::endl;

    ms.SetNumPara_p(readxnum_p);
    ms.SetNumPara_n(readxnum_n);
  }
  else
  {
    readxnum_p = ms.GetMSchemeNumberOfFreePara(Proton) * ms.GetProtonPairNum();
    readxnum_n = ms.GetMSchemeNumberOfFreePara(Neutron) * ms.GetNeutronPairNum();
    ms.SetNumPara_p(readxnum_p);
    ms.SetNumPara_n(readxnum_n);
    srand(ms.GetRandomSeed());
    for (i = 0; i < readxnum_p + readxnum_n; i++)
    {
      sprintf(cmd, "x[%d]", i);
      //(*upar).add(cmd, 1., 0.1);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    if (myid == 0)
    {
      std::cout << "Initialize Proton parameters: " << readxnum_p << std::endl;
      std::cout << "Initialize Neutron parameters: " << readxnum_n << std::endl;
    }
  }
  ms.UsingJbrokenPair(); // Set using Rotational broken basis in inital stage
  return;
}

void ReadWriteFiles::InitVariationPara_SamePairs_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 3) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.Get_NonCollecitvePairNumber(Proton))
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d\n", readxnum_p, ms.Get_NonCollecitvePairNumber(Proton));
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    for (i = 0; i < readxnum_p; i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout<< i<<"  "<< readX[i] <<std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Proton parameters: " << readX.size() << std::endl;

    //-----------------------------------------------
    ff = fopen(argv[2], "r"); // read neutron part
    if (fscanf(ff, "%d\n", &readxnum_n) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_n != ms.Get_NonCollecitvePairNumber(Neutron))
    {
      if (myid == 0)
      {
        printf("Read part of Neutron parameters! %d  %d\n", readxnum_n, ms.Get_NonCollecitvePairNumber(Neutron));
        exit(0);
      }
    }
    readX.clear();
    for (i = 0; i < readxnum_n; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    readxnum_p = ms.Get_NonCollecitvePairNumber(Proton);
    for (i = 0; i < readxnum_n; i++) // for neutron
    {
      sprintf(cmd, "x[%d]", i + readxnum_p);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout << i << "  " << readX[i] << std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Neutron parameters: " << readX.size() << std::endl;

    ms.SetNumPara_p(readxnum_p);
    ms.SetNumPara_n(readxnum_n);
  }
  else
  {
    readxnum_p = ms.Get_NonCollecitvePairNumber(Proton);
    readxnum_n = ms.Get_NonCollecitvePairNumber(Neutron);
    ms.SetNumPara_p(readxnum_p);
    ms.SetNumPara_n(readxnum_n);
    srand(ms.GetRandomSeed());
    for (i = 0; i < readxnum_p + readxnum_n; i++)
    {
      sprintf(cmd, "x[%d]", i);
      //(*upar).add(cmd, 1., 0.1);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    if (myid == 0)
    {
      std::cout << "Initialize Proton parameters: " << readxnum_p << std::endl;
      std::cout << "Initialize Neutron parameters: " << readxnum_n << std::endl;
    }
  }
  return;
}

void ReadWriteFiles::InitVariationPara_SamePairs_Mscheme_pn(ModelSpace &ms, MnUserParameters *upar, int argc, char *argv[])
{
  std::vector<double> readX;
  int readxnum_p, readxnum_n, i, j, myid;
  double value;
  char cmd[OneLine], *chr;
  FILE *ff;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc == 3) // argv[1] proton file  argv[2] neutron file
  {
    ff = fopen(argv[1], "r"); // read proton part
    if (fscanf(ff, "%d\n", &readxnum_p) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_p != ms.GetMSchemeNumberOfFreePara(Proton))
    {
      if (myid == 0)
      {
        printf("Read part of proton parameters! %d  %d\n", readxnum_p, ms.GetMSchemeNumberOfFreePara(Proton));
        exit(0);
      }
    }
    for (i = 0; i < readxnum_p; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    for (i = 0; i < readxnum_p; i++) // for proton
    {
      sprintf(cmd, "x[%d]", i);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout<< i<<"  "<< readX[i] <<std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Proton parameters: " << readX.size() << std::endl;

    //-----------------------------------------------
    ff = fopen(argv[2], "r"); // read neutron part
    if (fscanf(ff, "%d\n", &readxnum_n) != 1)
    {
      printf("input file error!!\n");
      exit(0);
    }
    if (readxnum_n != ms.GetMSchemeNumberOfFreePara(Neutron))
    {
      if (myid == 0)
      {
        printf("Read part of Neutron parameters! %d  %d\n", readxnum_n, ms.GetMSchemeNumberOfFreePara(Neutron));
        exit(0);
      }
    }
    readX.clear();
    for (i = 0; i < readxnum_n; i++)
    {
      chr = fgets(cmd, OneLine, ff);
      // printf("%s\n",cmd);
      if (sscanf(cmd, " %d	%lf", &j, &value) != 2)
      {
        printf("Reading input file error!! %d %d %f \n", i, j, value);
        exit(0);
      }
      readX.push_back(value);
    }
    fclose(ff);
    ///------------------------
    readxnum_p = ms.GetMSchemeNumberOfFreePara(Proton);
    for (i = 0; i < readxnum_n; i++) // for neutron
    {
      sprintf(cmd, "x[%d]", i + readxnum_p);
      (*upar).add(cmd, readX[i], readX[i] * 0.1);
      // std::cout << i << "  " << readX[i] << std::endl;
    }
    if (myid == 0)
      std::cout << "Initialize Neutron parameters: " << readX.size() << std::endl;

    ms.SetNumPara_p(readxnum_p);
    ms.SetNumPara_n(readxnum_n);
  }
  else
  {
    readxnum_p = ms.GetMSchemeNumberOfFreePara(Proton);
    readxnum_n = ms.GetMSchemeNumberOfFreePara(Neutron);
    ms.SetNumPara_p(readxnum_p);
    ms.SetNumPara_n(readxnum_n);
    srand(ms.GetRandomSeed());
    for (i = 0; i < readxnum_p + readxnum_n; i++)
    {
      sprintf(cmd, "x[%d]", i);
      //(*upar).add(cmd, 1., 0.1);
      value = (rand() % 1000) / 1000.;
      (*upar).add(cmd, value, value * 0.1);
    }
    if (myid == 0)
    {
      std::cout << "Initialize Proton parameters: " << readxnum_p << std::endl;
      std::cout << "Initialize Neutron parameters: " << readxnum_n << std::endl;
    }
  }
  ms.UsingJbrokenPair();
  return;
}

int ReadWriteFiles::OutputResults(FunctionMinimum min, ModelSpace &ms)
{
  int i, SP, loopN;
  FILE *ff;

  if (!(ff = fopen(Final_Report_path.c_str(), "w")))
  {
    std::cout << "Open " << Final_Report_path << " file error!!" << std::endl;
    exit(0);
  }
  fprintf(ff, "Projected 2I = %d    2M = %d subspace\n", ms.GetAMProjected_J(), ms.GetAMProjected_M());
  fprintf(ff, "Number of parameters:	proton part %d   neutron part %d\n", ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton), ms.GetPairNumber(Neutron) * ms.Get_NonCollecitvePairNumber(Neutron));
  fprintf(ff, "Number of protons: %d   neutrons: %d\n", 2 * ms.GetPairNumber(Proton), 2 * ms.GetPairNumber(Neutron));
  fprintf(ff, "Number function calls:	%d\n", min.userState().nfcn());
  fprintf(ff, "Final function value:	%f\n", min.userState().fval());
  fclose(ff);
  return 1;
}

int ReadWriteFiles::OutputResults_Iden(FunctionMinimum min, ModelSpace &ms)
{
  int i, SP, loopN;
  FILE *ff;

  if (!(ff = fopen(Final_Report_path.c_str(), "w")))
  {
    std::cout << "Open " << Final_Report_path << " file error!!" << std::endl;
    exit(0);
  }
  fprintf(ff, "Projected 2I = %d    2M = %d subspace\n", ms.GetAMProjected_J(), ms.GetAMProjected_M());
  fprintf(ff, "Number of parameters:	proton part %d\n", ms.GetPairNumber(Proton) * ms.Get_NonCollecitvePairNumber(Proton));
  fprintf(ff, "Number of protons: %d  \n", 2 * ms.GetPairNumber(Proton));
  fprintf(ff, "Number function calls:	%d\n", min.userState().nfcn());
  fprintf(ff, "Final function value:	%f\n", min.userState().fval());
  fclose(ff);
  return 1;
}
*/
