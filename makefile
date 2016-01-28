all: TCLprop.o 
<<<<<<< HEAD
	g++ -fopenmp -std=c++11 -O3  -DOMP -DARMA_USE_BLAS -DARMA_USE_LAPACK TCLprop.o -o TCL -L/usr/local/Cellar/armadillo/4.100.2/lib -larmadillo -framework Accelerate -llapack -lblas  
#       g++ -fopenmp -O3 -DOMP -DARMA_USE_BLAS -DARMA_USE_LAPACK TCLprop.o -o TCL -framework Accelerate -llapack -lblas  
TCLprop.o: TCLprop.cpp TCL.h 
	g++ -std=c++11 -fopenmp -O3 -DARMA_USE_BLAS -DARMA_USE_LAPACK -I/usr/local/Cellar/armadillo/4.100.2/include -I . -c TCLprop.cpp 
=======
	g++ -g TCLprop.o -o TCL -O1 -L/afs/crc.nd.edu/group/parkhill/local/usr/local/lib -llapack -lblas 
TCLprop.o: TCLprop.cpp TCL.h 
	g++ -g -I/afs/crc.nd.edu/group/parkhill/local/usr/include -I . -c TCLprop.cpp 
>>>>>>> 68f92cfe0e866195d932a6912b35595060406857
clean: 
	rm -rf *.o 
	rm TCL 
