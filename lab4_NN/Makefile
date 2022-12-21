CXX = g++
NVCC = nvcc
CXXFLAGS = -Werror -Wextra -Wall -pedantic -std=c++17 -g
NVFLAGS = -Xcompiler -rdynamic -lineinfo
TRASH = main *.o

all: build

.PHONY: build
build:
	$(NVCC) $(NVFLAGS) -o bin/cuda/tensor_kern.o -c src/cuda/tensor_kern.cu

	$(CXX) $(CXXFLAGS) -c -I/usr/local/cuda-11.3/include  -o bin/main.o main.cpp

	$(CXX) $(CXXFLAGS) -o main bin/main.o  bin/cuda/tensor_kern.o -L/usr/local/cuda-11.3/lib64 -lcudart -lcurand -lcuda

.PHONY: run
run: build
	./main

.PHONY: clean
clean:
	$(RM) $(TRASH)
