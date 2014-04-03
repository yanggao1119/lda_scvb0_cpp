CXX=/usr/local/bin/g++-4.7
EIGEN=eigen-eigen-6b38706d90a9
TCLAP=tclap-1.2.1/include
#DEIGEN_NO_DEBUG is imperative for good performance
CFLAGS=-DEIGEN_DONT_PARALLELIZE -DEIGEN_NO_DEBUG -DNDEBUG -O3 
OMP_FLAG=-fopenmp

SOURCE=lda_scvb0.cpp
EXE=lda_scvb0

#Setting the Architecture
all:
	$(CXX) $(SOURCE) -I$(EIGEN) -I$(TCLAP) $(OMP_FLAG) $(CFLAGS) -o $(EXE)

clean:
	rm -rf $(EXE)

rebuild: clean all
