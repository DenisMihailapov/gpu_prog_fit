#include "../../include/nn_exception.cuh"
#include "../../include/relu_activation.h"

template <class Scalar>
__global__ void reluActivationForward(Scalar *Z, Scalar *A, int Z_x_dim,
                                      int Z_y_dim) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < Z_x_dim * Z_y_dim) {
    A[index] = fmaxf(Z[index], 0);
  }
}

template <class Scalar>
__global__ void reluActivationBackprop(Scalar *Z, Scalar *dA, Scalar *dZ,
                                       int Z_x_dim, int Z_y_dim) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < Z_x_dim * Z_y_dim) {
    if (Z[index] > 0) {
      dZ[index] = dA[index];
    } else {
      dZ[index] = 0;
    }
  }
}

template <class Scalar>
__host__ void ReLUForwardKernelWrapper(Tensor<Scalar> &Z, Tensor<Scalar> &A) {
  
  A.display();
  dim3 block_size(256);
  dim3 num_of_blocks((Z.getHeight() * Z.getWidth() + block_size.x - 1) /
                     block_size.x);

  reluActivationForward<<<num_of_blocks, block_size>>>(
      Z.getArray(), A.getArray(), Z.getHeight(), Z.getWidth());
  NNException::throwIfDeviceErrorsOccurred(
      "Cannot perform ReLU forward propagation.");

  A.display();    
}

template <class Scalar>
__host__ void ReLUBackwardKernelWrapper(Tensor<Scalar> &dA, Tensor<Scalar> &dZ,
                                        Tensor<Scalar> &Z,
                                        double learning_rate) {

  dim3 block_size(256);
  dim3 num_of_blocks((Z.getHeight() * Z.getWidth() + block_size.x - 1) /
                     block_size.x);
  reluActivationBackprop<<<num_of_blocks, block_size>>>(
      Z.getArray(), dA.getArray(), dZ.getArray(), Z.getHeight(), Z.getWidth());
  NNException::throwIfDeviceErrorsOccurred(
      "Cannot perform ReLU back propagation");
}

template void ReLUForwardKernelWrapper(Tensor<float> &Z, Tensor<float> &A);
template void ReLUForwardKernelWrapper(Tensor<double> &Z, Tensor<double> &A);

template void ReLUBackwardKernelWrapper(Tensor<float> &dA, Tensor<float> &dZ,
                                        Tensor<float> &Z, double learning_rate);
template void ReLUBackwardKernelWrapper(Tensor<double> &dA, Tensor<double> &dZ,
                                        Tensor<double> &Z,
                                        double learning_rate);
