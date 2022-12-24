CXX = g++
NVCC = nvcc
CXXFLAGS = -Werror -Wextra -Wall -pedantic -std=c++17 -g
NVFLAGS = -Xcompiler -rdynamic -lineinfo
TRASH = main *.o
CUDA = /usr/local/cuda-11.3/lib64
CUFLAGS = -lcudart -lcurand -lcuda
BINCU = bin/cuda


all: build

.PHONY: build
build:
	$(NVCC) $(NVFLAGS) -o $(BINCU)/tensor_kern.o -c src/cuda/tensor_kern.cu
	$(NVCC) $(NVFLAGS) -o $(BINCU)/shape.o -c src/cuda/shape.cu
	$(NVCC) $(NVFLAGS) -o $(BINCU)/relu_activation.o -c src/cuda/relu_activation.cu

	$(CXX) $(CXXFLAGS) -c -I/usr/local/cuda-11.3/include -o bin/main.o main.cpp

	# $(CXX) $(CXXFLAGS) -o $(BINCU)/activations.o  $(BINCU)/relu_activation.o -L$(CUDA) $(CUFLAGS)
    
	# $(CXX) $(CXXFLAGS) -o $(BINCU)/tensor.o  $(BINCU)/shape.o $(BINCU)/tensor_kern.o -L$(CUDA) $(CUFLAGS)
    
	$(CXX) $(CXXFLAGS) -o main bin/main.o $(BINCU)/shape.o $(BINCU)/tensor_kern.o  $(BINCU)/relu_activation.o -L$(CUDA) $(CUFLAGS)

.PHONY: run
run: build
	./main

.PHONY: clean
clean:
	$(RM) $(TRASH)
