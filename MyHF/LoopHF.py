import os
import pandas as pd
import re
import numpy



def GenrateHF_input(ref): #developing
    with open("Input_HF.txt", 'w') as file:
        file.write("# number of valence protons and neutrons (exp.  4, 4  or 24Mg   Mg24)\n")
        file.write( "{}\n".format(ref) )
        file.write("///////////////////////////////////////////// # Hamiltonian File Kshell format\n")
        file.write( "Interaction/IMSRG2_{}_e14_hw12.snt\n".format(ref) )
        file.write("////////////////////////////////////////////  # variation parameters\n")
        file.write( "No  7.  0.                                    # Shape constrains for Q0 and Q2  \n" )
        file.close()


#---------------------------------------------------
# main program
def main():

    Pb_chain_WeCaled=['Pb172',
    'Pb173',
    'Pb174',
    'Pb175',
    'Pb176',
    'Pb177',
    'Pb178',
    'Pb179',
    'Pb180',
    'Pb181',
    'Pb182',
    'Pb183',
    'Pb184',
    'Pb185',
    'Pb186',
    'Pb187',
    'Pb188',
    'Pb189',
    'Pb190',
    'Pb191',
    'Pb192',
    'Pb193',
    'Pb194',
    'Pb195',
    'Pb196',
    'Pb197',
    'Pb198',
    'Pb199',
    'Pb200',
    'Pb201',
    'Pb202',
    'Pb203',
    'Pb204',
    'Pb205',
    'Pb206',
    'Pb207',
    'Pb208',
    'Pb209',
    'Pb210',
    'Pb211',
    'Pb212',
    'Pb213',
    'Pb214',
    'Pb215',
    'Pb216',
    'Pb217',
    'Pb218',
    'Pb219',
    'Pb220',
    'Pb221',
    'Pb222',
    'Pb223',
    'Pb224',
    'Pb225']

    for ele in Pb_chain_WeCaled:
        GenrateHF_input(ele)
        #print(ele )
        os.system("./HartreeFock.exe" )  

if __name__ == "__main__":
    main()