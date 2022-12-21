#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cstddef>

#include "tensor_kern.h"


template <class Scalar = double> 
class Tensor {
public:
  /* Constructors */
  Tensor<Scalar>(bool gpu = false);
  Tensor<Scalar>(size_t height, size_t width, bool gpu = false);
  Tensor<Scalar>(size_t height, size_t width, Scalar min, Scalar max,
                 bool gpu = false);

  /* Getters */
  size_t getWidth() const;
  size_t getHeight() const;
  //Shape shape = Shape(0, 0); //TODO

  Scalar get(size_t i, size_t j) const;
  void set(size_t i, size_t j, Scalar el);

  /* Operations */
  Tensor<Scalar> add(Scalar m) const;
  Tensor<Scalar> add(const Tensor<Scalar> &m) const;

  Tensor<Scalar> dot(Scalar m) const;
  Tensor<Scalar> dot(const Tensor<Scalar> &m) const;

  Tensor<Scalar> multiply(const Tensor<Scalar> &m) const;
  Tensor<Scalar> divide(Scalar m) const;
  Tensor<Scalar> inverse(Scalar m) const;
  Tensor<Scalar> exponential() const;
  Tensor<Scalar> power(unsigned n) const;

  Scalar sum() const;
  void apply(Scalar func(Scalar));
  Tensor<Scalar> getLine(unsigned n) const;
  Tensor<Scalar> getRow(unsigned n) const;
  bool eq(const Tensor<Scalar> &m) const;
  Tensor<Scalar> transpose() const;

  /* Operators */
  Tensor<Scalar> operator+(const Tensor<Scalar> &) const;
  Tensor<Scalar> operator*(const Tensor<Scalar> &) const;
  Tensor<Scalar> operator-(const Tensor<Scalar> &) const;
  Tensor<Scalar> operator%(const Tensor<Scalar> &) const;
  bool operator==(const Tensor<Scalar> &);

  /* Display */
  void display() const;

private:
  size_t height;
  size_t width;
  Scalar *array;
  bool gpu;
};


































template <class Scalar> Tensor<Scalar>::Tensor(bool gpu) {
  static_assert(std::is_same<Scalar, int>::value ||
                    std::is_same<Scalar, float>::value ||
                    std::is_same<Scalar, double>::value,
                "Type not allowed. Use <int>, <float> or <double>.");
  this->height = 0;
  this->width = 0;
  this->array = nullptr;
  this->gpu = gpu;
}

template <class Scalar>
Tensor<Scalar>::Tensor(size_t height, size_t width, bool gpu) {
  static_assert(std::is_same<Scalar, int>::value ||
                    std::is_same<Scalar, float>::value ||
                    std::is_same<Scalar, double>::value,
                "Type not allowed. Use <int>, <float> or <double>.");
  this->height = height;
  this->width = width;
  this->gpu = gpu;

  this->array = new Scalar[this->height * this->width];
  srand(time(NULL));
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      Scalar n = -1 + rand() / Scalar(RAND_MAX) * 2;
      this->set(i, j, n);
    }
  }
}

template <class Scalar>
Tensor<Scalar>::Tensor(size_t height, size_t width, Scalar min, Scalar max,
                       bool gpu) {
  static_assert(std::is_same<Scalar, int>::value ||
                    std::is_same<Scalar, float>::value ||
                    std::is_same<Scalar, double>::value,
                "Type not allowed. Use <int>, <float> or <double>.");
  this->height = height;
  this->width = width;
  this->gpu = gpu;

  this->array = new Scalar[this->height * this->width];
  srand(time(NULL));
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      Scalar n = min + (max - min) * (rand() / Scalar(RAND_MAX));
      this->set(i, j, n);
    }
  }
}

template <class Scalar> size_t Tensor<Scalar>::getHeight() const {
  return this->height;
}

template <class Scalar> size_t Tensor<Scalar>::getWidth() const {
  return this->width;
}

template <class Scalar> Scalar Tensor<Scalar>::get(size_t i, size_t j) const {
  if (i >= this->height || j >= this->width) {
    fprintf(stderr, "Can't subscrit at %li, %li. Shape = (%li, %li)\n", i, j,
            this->height, this->width);
    throw;
  }
  return this->array[i * this->width + j];
}

template <class Scalar>
void Tensor<Scalar>::set(size_t i, size_t j, Scalar el) {
  if (i >= this->height || j >= this->width) {
    fprintf(stderr, "Can't set element at %li, %li. Shape = (%li, %li)\n", i, j,
            this->height, this->width);
    throw;
  }
  this->array[i * this->width + j] = el;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::add(const Tensor<Scalar> &m) const {
  if (m.height != this->height || m.width != this->width) {
    fprintf(stderr,
            "Can't add element wise a matrix of shape (%li, %li) with a matrix "
            "of shape (%li, %li).\n",
            this->height, this->width, m.height, m.width);
    throw;
  }

  Tensor<Scalar> result(height, width, this->gpu);
  if (gpu) {
    Wrapper().add((Scalar *)&this->array[0], (Scalar *)&m.array[0],
                  (Scalar *)&result.array[0], this->width * this->height);
  } else {
    for (size_t i = 0; i < this->height; i++) {
      for (size_t j = 0; j < this->width; j++) {
        result.set(i, j, this->get(i, j) + m.get(i, j));
      }
    }
  }
  return result;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::add(Scalar m) const {

  if (gpu)
    std::cout << "add Scalar calc on CPU" << std::endl;
  Tensor<Scalar> result(height, width, this->gpu, this->gpu);
  for (size_t i = 0; i < this->height; i++) {
    for (size_t j = 0; j < this->width; j++) {
      result.set(i, j, this->get(i, j) + m);
    }
  }
  return result;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::dot(const Tensor<Scalar> &m) const {
  if (this->width != m.height) {
    fprintf(stderr,
            "Can't multiply a matrix of shape (%li, %li) with a matrix of "
            "shape (%li, %li).\n",
            this->height, this->width, m.height, m.width);
    throw;
  }
  Tensor<Scalar> result(this->height, m.width, this->gpu);
  std::cout << gpu << std::endl;
  if (gpu) {
    Wrapper().dot((Scalar *)&this->array[0], (Scalar *)&m.array[0],
                  (Scalar *)&result.array[0], this->height, m.width,
                  this->width);
  } else {
    Scalar val = 0;
    for (size_t i = 0; i < this->height; i++) {
      for (size_t j = 0; j < m.width; j++) {
        for (size_t h = 0; h < this->width; h++) {
          val += this->get(i, h) * m.get(h, j);
        }
        result.set(i, j, val);
        val = 0;
      }
    }
  }
  return result;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::dot(Scalar m) const {
  Tensor<Scalar> result(height, width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->height; i++)
      for (size_t j = 0; j < this->width; j++)
        result.set(i, j, this->get(i, j) * m);
  }
  return result;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::multiply(const Tensor<Scalar> &m) const {
  if (m.height != this->height || m.width != this->width) {
    fprintf(stderr,
            "Can't multiply element wise a matrix of shape (%li, %li) with a "
            "matrix of shape (%li, %li).\n",
            this->height, this->width, m.height, m.width);
    throw;
  }
  Tensor<Scalar> result(height, width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->height; i++)
      for (size_t j = 0; j < this->width; j++)
        result.set(i, j, this->get(i, j) * m.get(i, j));
  }
  return result;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::divide(Scalar m) const {
  Tensor<Scalar> result(height, width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->height; i++)
      for (size_t j = 0; j < this->width; j++)
        result.set(i, j, this->get(i, j) / m);
  }
  return result;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::inverse(Scalar m) const {
  Tensor<Scalar> result(height, width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->height; i++)
      for (size_t j = 0; j < this->width; j++)
        result.set(i, j, m / this->get(i, j));
  }
  return result;
}

template <class Scalar> Scalar Tensor<Scalar>::sum() const {
  Scalar result = 0;
  if (gpu)
    std::cout << "sum calc on CPU" << std::endl;

  for (size_t i = 0; i < this->height; i++) {
    for (size_t j = 0; j < this->width; j++) {
      result += this->get(i, j);
    }
  }

  return result;
}

template <class Scalar> void Tensor<Scalar>::apply(Scalar func(Scalar)) {
  if (gpu)
    std::cout << "apply calc on CPU" << std::endl;

  for (size_t i = 0; i < this->height; i++) {
    for (size_t j = 0; j < this->width; j++) {
      this->set(i, j, func(this->get(i, j)));
    }
  }
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::getLine(unsigned n) const {
  if (n >= this->height) {
    fprintf(stderr, "Can't subscrit line at %i.\n", n);
    throw;
  }

  Tensor<Scalar> result(1, width, this->gpu);
  for (size_t i = 0; i < this->width; i++)
    result.set(0, i, this->get(n, i));

  return result;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::getRow(unsigned n) const {
  if (n >= this->width) {
    fprintf(stderr, "Can't subscrit row at %i.\n", n);
    throw;
  }

  Tensor<Scalar> result(height, 1, this->gpu);
  for (size_t i = 0; i < this->height; i++)
    result.set(i, 0, this->get(i, n));

  return result;
}

template <class Scalar> bool Tensor<Scalar>::eq(const Tensor<Scalar> &m) const {
  for (size_t i = 0; i < this->height; i++)
    for (size_t j = 0; j < m.width; j++)
      if (this->get(i, j) != m.get(i, j))
        return false;

  return true;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::transpose() const {
  Tensor<Scalar> result(this->width, this->height, this->gpu);

  for (size_t i = 0; i < this->height; i++)
    for (size_t j = 0; j < this->width; j++)
      result.set(j, i, this->get(i, j));

  return result;
}

template <class Scalar> void Tensor<Scalar>::display() const {

  for (size_t i = 0; i < this->height; i++) {
    for (size_t j = 0; j < this->width; j++) {
      float n = (float)this->get(i, j);
      printf("%c%.*f  ", n >= 0 ? ' ' : '\0', 6, n);
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::operator+(const Tensor<Scalar> &m) const {
  return this->add(m);
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::operator*(const Tensor<Scalar> &m) const {
  return this->dot(m);
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::operator-(const Tensor<Scalar> &m) const {
  return this->sub(m);
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::operator%(const Tensor<Scalar> &m) const {
  return this->multiply(m);
}

template <class Scalar> bool Tensor<Scalar>::operator==(const Tensor<Scalar> &m) {
  return this->eq(m);
}

template <class Scalar>
Tensor<Scalar> operator+(const Tensor<Scalar> &m, Scalar n) {
  return m.add(n);
}

template <class Scalar>
Tensor<Scalar> operator+(Scalar n, const Tensor<Scalar> &m) {
  return m.add(n);
}

template <class Scalar>
Tensor<Scalar> operator-(const Tensor<Scalar> &m, Scalar n) {
  return m.add(-n);
}

template <class Scalar>
Tensor<Scalar> operator-(Scalar n, const Tensor<Scalar> &m) {
  return (Scalar)(-1) * m + n;
}

template <class Scalar>
Tensor<Scalar> operator*(const Tensor<Scalar> &m, Scalar n) {
  return m.dot(n);
}

template <class Scalar>
Tensor<Scalar> operator*(Scalar n, const Tensor<Scalar> &m) {
  return m.dot(n);
}

template <class Scalar>
Tensor<Scalar> operator/(const Tensor<Scalar> &m, Scalar n) {
  return m.divide(n);
}

template <class Scalar>
Tensor<Scalar> operator/(Scalar n, const Tensor<Scalar> &m) {
  return m.inverse(n);
}
