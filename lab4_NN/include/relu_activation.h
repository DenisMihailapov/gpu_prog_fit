#pragma once

#include "nn_layer.h"

template <class Scalar>
void ReLUForwardKernelWrapper(Tensor<Scalar> &Z, Tensor<Scalar> &A);

template <class Scalar>
void ReLUBackwardKernelWrapper(Tensor<Scalar> &dA, Tensor<Scalar> &dZ,
                               Tensor<Scalar> &Z, double learning_rate);



template <class Scalar = double> 
class ReLUActivation : public NNLayer {
private:
	Tensor<Scalar> A;

	Tensor<Scalar> Z;
	Tensor<Scalar> dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Tensor<Scalar>& forward(Tensor<Scalar>& Z);
	Tensor<Scalar>& backprop(Tensor<Scalar>& dA, double learning_rate = 0.01);
};

template <class Scalar>
ReLUActivation<Scalar>::ReLUActivation(std::string name) { this->name = name; }

template <class Scalar>
ReLUActivation<Scalar>::~ReLUActivation() {}

template <class Scalar>
Tensor<Scalar> & ReLUActivation<Scalar>::forward(Tensor<Scalar> &Z) {
  this->Z = Z; //save for backward
  A.init(Z.getShape(), Z.isOnGPU());
  ReLUForwardKernelWrapper(Z, A);
  return A;
}

template <class Scalar>
Tensor<Scalar> &ReLUActivation<Scalar>::backprop(Tensor<Scalar> &dA, double learning_rate) {
  dZ.init(Z.getShape());
  ReLUBackwardKernelWrapper(dA, dZ, Z);
  return dZ;
}