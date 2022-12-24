// Include C++ header files.
#include <iostream>

// Include local CUDA header files.

#include "include/relu_activation.h"


int main() {

    Tensor<> tensor1(2, 3), tensor2(2, 3), tensor3(2, 3);

    tensor1 = tensor1 + 1.;
    std::cout << "+\n"; 

    tensor2 = tensor2 + 1.;

    tensor3 = tensor1 + tensor2;

    tensor3.display();

    ReLUActivation<> relu("test");
    
    tensor3 = relu.forward(tensor3);
    tensor3.display();

    return 0;
}