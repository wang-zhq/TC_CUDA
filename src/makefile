MACRO=-DDEBUG_DF
CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp $(MACRO)
NVCC=nvcc
NVCCFLAGS=--gpu-architecture=compute_70 --gpu-code=sm_70 -Xcompiler -O3 -std=c++11 $(MACRO)

CalcTriangle : CalcTriangleMain.o CalcTriangleCuda.o gpu.o
	$(CXX) $(CXXFLAGS) -o CalcTriangle CalcTriangleMain.o CalcTriangleCuda.o gpu.o -L /usr/local/cuda/lib64 -lcudart

CalcTriangleCuda.o : CalcTriangleCuda.cu
	$(NVCC) $(NVCCFLAGS) -dc CalcTriangleCuda.cu -o CalcTriangleCuda.o

gpu.o : CalcTriangleCuda.o
	$(NVCC) $(NVCCFLAGS) -dlink CalcTriangleCuda.o -o gpu.o

CalcTriangleMain.o : CalcTriangleMain.cpp mapfile.hpp
	$(CXX) $(CXXFLAGS) -c CalcTriangleMain.cpp -o CalcTriangleMain.o

.PHONY : clean
clean :
	rm CalcTriangle  *.o
