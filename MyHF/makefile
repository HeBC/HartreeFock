#--------------------------------------------------
# Hartree Fock
#--------------------------------------------------
FC = mpicxx
THEHOST = $(shell if [ `hostname|grep bebop` ]; then echo bebop; elif [ `hostname|grep crc` ]; then echo crc; else echo other; fi)

ifeq ($(THEHOST),other)  # default options. test on my laptop
  CFLAGS =   -O3 #-std=c++11
  LIBS = -lmkl_rt -lm -lnlopt -lgsl -lgslcblas    # include lnlopt to test code
endif

ifeq ($(THEHOST),bebop)  # Argonne bebop
  CFLAGS = -O3 
  LIBS = -m64  -Wl,--no-as-needed -lm -lgsl -lgslcblas -lmkl_intel_lp64  -lmkl_sequential -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl  #-lgomp
endif

ifeq ($(THEHOST),crc)   # Notre Dame CRC
  CFLAGS = -O3 -I/opt/crc/i/intel/19.0/mkl/include/ -fopenmp
  LIBS = -m64  -Wl,--no-as-needed -lm -lgsl -lgslcblas -lmkl_intel_lp64  -lmkl_sequential -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl -lgomp
endif


#----------------------------------------------------
all: HartreeFock.exe HF_Projection.exe

clean:
	/bin/rm -f *.o *.exe

OFILES = ModelSpace.o Hamiltonian.o AngMom.o ReadWriteFiles.o HFbasis.o CalHF.o GCM_Tools.o
#--------------------------------------------------
#   executable files
# HF
HF_Pfaffian.exe: $(OFILES) HF_Pfaffian.o 
	$(FC) -o $@ HF_Pfaffian.o $(CFLAGS) $(OFILES) $(LIBS) 


# HF
HartreeFock.exe: $(OFILES) HartreeFock.o 
	$(FC) -o $@ HartreeFock.o $(CFLAGS) $(OFILES) $(LIBS) 


# HF Projection
HF_Projection.exe: $(OFILES) HF_Projection.o 
	$(FC) -o $@ HF_Projection.o $(CFLAGS) $(OFILES) $(LIBS) 



#---- cpp file  ----
.cpp.o:
	$(FC) $(CFLAGS) -c $*.cpp


# debug
Testing.exe: $(OFILES) Testing.o 
	$(FC) -o $@ Testing.o $(CFLAGS) $(OFILES) $(LIBS) 

