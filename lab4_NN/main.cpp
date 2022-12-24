// Include C++ header files.
#include <chrono>
#include <iostream>
#include <ostream>

// Include local CUDA header files.

#include "include/relu_activation.h"

using namespace std::chrono;

int main() {

    Tensor<> A_cpu(1000, 1000), B_cpu(1000, 1000);
    Tensor<> A_gpu(1000, 1000, true), B_gpu(1000, 1000, true);

    // Only calculations
    auto t1 = steady_clock::now();
    A_cpu+B_cpu;
    auto t2 = steady_clock::now();
    auto elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A+B calculation (CPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    t1 = steady_clock::now();
    A_cpu.dot(B_cpu);
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A dot B calculation (CPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    t1 = steady_clock::now();
    5.*B_cpu;
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "a*B calculation (CPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    std::cout << std::endl;
    A_gpu+B_gpu; // for cuda init 
    // TODO: cuda tool

    t1 = steady_clock::now();
    A_gpu+B_gpu;
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A+B calculation (GPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    t1 = steady_clock::now();
    A_gpu.dot(B_gpu);
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A dot B calculation (GPU): " << elapsed.count( ) << " ms" << std::endl;

    t1 = steady_clock::now();
    5.*B_gpu;
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "a*B calculation (GPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    // ReLUActivation<> relu("test");
    
    // tensor3 = relu.forward(tensor3);
    // tensor3.display();

    return 0;
}