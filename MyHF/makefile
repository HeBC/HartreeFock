#--------------------------------------------------
# Hartree Fock
#--------------------------------------------------
# The default makefile use MPI for projection code, OpenMP for Hartree-Fock code
FC = mpicxx
FGCC = g++
# FC = c++17
THEHOST = $(shell if [ `hostname|grep bebop` ]; then echo bebop; elif [ `hostname|grep crc` ]; then echo crc; else echo other; fi)


#----------------------------------------------------
# pybind 11
#----------------------------------------------------

# python version
PYTHON_MAJOR_VERSION = $(shell python -c "import sys; sys.stdout.write('{}'.format(sys.version_info.major))")
ifneq (,$(shell which python3)) # If it's there, default to python3
  PYTHON_MAJOR_VERSION = 3
endif
PYTHON_CONFIG = python$(PYTHON_MAJOR_VERSION)-config
ifeq (,$(shell which $(PYTHON_CONFIG) 2> /dev/null))
  PYTHON_CONFIG = python-config
endif
# Path to Pybind11 include folder
PYBIND11_INCLUDE_PATH = ./pybind11/include
PYTHON_CFLAGS = $(shell $(PYTHON_CONFIG) --cflags )
PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags )
PYTHONFLAGS =  $(PYTHON_CFLAGS) -Wl,--no-as-needed
PYTHON_COMMAND = python$(PYTHON_MAJOR_VERSION)
PYTHON_LDFLAGS += -undefined dynamic_lookup # This avoids trouble with missing symbols (see pybind11.readthedocs.io/en/stable/compiling.html )
PYTHONFLAGS +=  $(shell $(PYTHON_CONFIG) --cflags | sed -e 's/-arch i386//')
PYTHONFLAGS += -Wno-unused-value # suppress annoying warnings about stuff in the pybind11 library
PYTHONFLAGS += -Wno-deprecated-declarations

#-----------------------------------------------------------------
ifeq ($(THEHOST),other)  # default options. test on my laptop   
  CFLAGS =-O3 -fPIC -fopenmp -I/usr/include/mkl  #-Wall #-std=c++11  
  LIBS = -lmkl_rt -lm -lgsl -lgslcblas  # -lnlopt  include lnlopt to test code in HF_Pfaffian
endif

ifeq ($(THEHOST),bebop)  # Argonne bebop
  CFLAGS = -O3 -fPIC -fopenmp 
  LIBS = -m64  -Wl,--no-as-needed -lm -lgsl -lgslcblas -lmkl_intel_lp64  -lmkl_sequential -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl
endif

ifeq ($(THEHOST),crc)   # Notre Dame CRC
  CFLAGS = -O3 -fPIC -I/opt/crc/i/intel/19.0/mkl/include/ -fopenmp
  LIBS = -m64  -Wl,--no-as-needed -lm -lgsl -lgslcblas -lmkl_intel_lp64  -lmkl_sequential -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -ldl -lgomp
endif

#----------------------------------------------------
all: HartreeFock.exe HF_Projection.exe pyHFAndHFB.so 

clean:
	/bin/rm -f *.o *.exe *.so

OFILES = ModelSpace.o Hamiltonian.o AngMom.o ReadWriteFiles.o HFbasis.o Pfaffian_tools.o GCM_Tools.o
#--------------------------------------------------
#   executable files:
# HF
HF_Pfaffian.exe: $(OFILES) HF_Pfaffian.o 
	$(FC) -o $@ HF_Pfaffian.o $(CFLAGS) $(OFILES) $(LIBS) 


# HF
HartreeFock.exe: $(OFILES) HartreeFock.o 
	$(FC) -o $@ HartreeFock.o $(CFLAGS) $(OFILES) $(LIBS)


# HF Projection
HF_Projection.exe: $(OFILES) HF_Projection.o 
	$(FC) -o $@ HF_Projection.o $(CFLAGS) $(OFILES) $(LIBS) 

# pybind11
pyHFAndHFB.so: pyHFAndHFB.o $(OFILES) HartreeFock.o 
	$(FC) $(CFLAGS) $(PYTHONFLAGS) -I$(PYBIND11_INCLUDE_PATH) $(LIBS) -shared $(OFILES) HartreeFock.o  pyHFAndHFB.o -o $@


#---- cpp file  ----
.cpp.o:
	$(FC) $(CFLAGS) -c $*.cpp

pyHFAndHFB.o: pyHFandHFB.cpp
	$(FC) -c  $(CFLAGS) $(LIBS) -I$(PYBIND11_INCLUDE_PATH) $(PYTHONFLAGS)  $^ -o $@


# debug
Testing.exe: $(OFILES) Testing.o 
	$(FC) -o $@ Testing.o $(CFLAGS) $(OFILES) $(LIBS) 


HartreeFockBogoliubov.exe: $(OFILES) HartreeFockBogoliubov.o 
	$(FC) -o $@ HartreeFockBogoliubov.o $(CFLAGS) $(OFILES) $(LIBS)
 

HartreeFockBogoliubov_Real.exe: $(OFILES) HartreeFockBogoliubov_Real.o 
	$(FC) -o $@ HartreeFockBogoliubov_Real.o $(CFLAGS) $(OFILES) $(LIBS) 

