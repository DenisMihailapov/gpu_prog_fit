#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>

#include "shape.h"
#include "tensor_kern.h"

template <class Scalar = double> class Tensor {
public:
  /* Constructors */
  Tensor<Scalar>(bool gpu = false);
  void init(size_t height, size_t width, bool gpu = false, Scalar min = 0,
            Scalar max = 1, bool rand_init = false);
  void init(Shape shape, bool gpu = false, Scalar min = 0, Scalar max = 1,
            bool rand_init = false);

  Tensor<Scalar>(size_t height, size_t width, bool gpu = false, Scalar min = 0,
                 Scalar max = 1, bool rand_init = false);
  Tensor<Scalar>(Shape shape, bool gpu = false, Scalar min = 0, Scalar max = 1,
                 bool rand_init = false);

  /* Getters */
  size_t getWidth() const;
  size_t getHeight() const;
  // Shape shape = Shape(0, 0); //TODO

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
  Shape shape;
  Scalar *array;
  bool gpu;
};

template <class Scalar> Tensor<Scalar>::Tensor(bool gpu) {
  static_assert(std::is_same<Scalar, int>::value ||
                    std::is_same<Scalar, float>::value ||
                    std::is_same<Scalar, double>::value,
                "Type not allowed. Use <int>, <float> or <double>.");
  this->shape = Shape(0, 0);
  this->array = nullptr;
  this->gpu = gpu;
}

template <class Scalar>
void Tensor<Scalar>::init(size_t height, size_t width, bool gpu, Scalar min,
                          Scalar max, bool rand_init) {

  static_assert(std::is_same<Scalar, int>::value ||
                    std::is_same<Scalar, float>::value ||
                    std::is_same<Scalar, double>::value,
                "Type not allowed. Use <int>, <float> or <double>.");
  this->shape = Shape(height, width);
  this->gpu = gpu;

  this->array = new Scalar[height * width];

  if (rand_init)
    srand(time(NULL));

  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      if (rand_init)
        this->set(i, j, min + (max - min) * (rand() / Scalar(RAND_MAX)));
      else
        this->set(i, j, 0.);
    }
  }
}

template <class Scalar>
void Tensor<Scalar>::init(Shape shape, bool gpu, Scalar min, Scalar max,
                          bool rand_init) {
  this->init(shape.height, shape.width, gpu, min, max, rand_init);
}

template <class Scalar>
Tensor<Scalar>::Tensor(size_t height, size_t width, bool gpu, Scalar min,
                       Scalar max, bool rand_init) {
  this->init(height, width, gpu, min, max, rand_init);
}

template <class Scalar>
Tensor<Scalar>::Tensor(Shape shape, bool gpu, Scalar min, Scalar max,
                       bool rand_init) {
  this->init(shape.height, shape.width, gpu, min, max, rand_init);
}

template <class Scalar> size_t Tensor<Scalar>::getHeight() const {
  return this->shape.height;
}

template <class Scalar> size_t Tensor<Scalar>::getWidth() const {
  return this->shape.width;
}

template <class Scalar> Scalar Tensor<Scalar>::get(size_t i, size_t j) const {
  if (i >= this->shape.height || j >= this->shape.width) {
    fprintf(stderr, "Can't subscrit at %li, %li. Shape = (%li, %li)\n", i, j,
            this->shape.height, this->shape.width);
    throw;
  }
  return this->array[i * this->shape.width + j];
}

template <class Scalar>
void Tensor<Scalar>::set(size_t i, size_t j, Scalar el) {
  if (i >= this->shape.height || j >= this->shape.width) {
    fprintf(stderr, "Can't set element at %li, %li. Shape = (%li, %li)\n", i, j,
            this->shape.height, this->shape.width);
    throw;
  }
  this->array[i * this->shape.width + j] = el;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::add(const Tensor<Scalar> &m) const {
  if (m.shape.height != this->shape.height ||
      m.shape.width != this->shape.width) {
    fprintf(stderr,
            "Can't add element wise a matrix of shape (%li, %li) with a matrix "
            "of shape (%li, %li).\n",
            this->shape.height, this->shape.width, m.shape.height,
            m.shape.width);
    throw;
  }

  Tensor<Scalar> result(shape.height, shape.width, this->gpu);
  if (gpu) {
    Wrapper().add((Scalar *)&this->array[0], (Scalar *)&m.array[0],
                  (Scalar *)&result.array[0],
                  this->shape.width * this->shape.height);
  } else {
    for (size_t i = 0; i < this->shape.height; i++) {
      for (size_t j = 0; j < this->shape.width; j++) {
        result.set(i, j, this->get(i, j) + m.get(i, j));
      }
    }
  }
  return result;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::add(Scalar m) const {

  if (gpu)
    std::cout << "add Scalar calc on CPU" << std::endl;

  Tensor<Scalar> result(shape, this->gpu);

  for (size_t i = 0; i < this->shape.height; i++) {
    for (size_t j = 0; j < this->shape.width; j++) {
      result.set(i, j, this->get(i, j) + m);
    }
  }
  return result;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::dot(const Tensor<Scalar> &m) const {
  if (this->shape.width != m.shape.height) {
    fprintf(stderr,
            "Can't multiply a matrix of shape (%li, %li) with a matrix of "
            "shape (%li, %li).\n",
            this->shape.height, this->shape.width, m.shape.height,
            m.shape.width);
    throw;
  }
  Tensor<Scalar> result(this->shape.height, m.shape.width, this->gpu);
  std::cout << gpu << std::endl;
  if (gpu) {
    Wrapper().dot((Scalar *)&this->array[0], (Scalar *)&m.array[0],
                  (Scalar *)&result.array[0], this->shape.height, m.shape.width,
                  this->shape.width);
  } else {
    Scalar val = 0;
    for (size_t i = 0; i < this->shape.height; i++) {
      for (size_t j = 0; j < m.shape.width; j++) {
        for (size_t h = 0; h < this->shape.width; h++) {
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
  Tensor<Scalar> result(shape.height, shape.width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->shape.height; i++)
      for (size_t j = 0; j < this->shape.width; j++)
        result.set(i, j, this->get(i, j) * m);
  }
  return result;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::multiply(const Tensor<Scalar> &m) const {
  if (m.shape.height != this->shape.height ||
      m.shape.width != this->shape.width) {
    fprintf(stderr,
            "Can't multiply element wise a matrix of shape (%li, %li) with a "
            "matrix of shape (%li, %li).\n",
            this->shape.height, this->shape.width, m.shape.height,
            m.shape.width);
    throw;
  }
  Tensor<Scalar> result(shape.height, shape.width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->shape.height; i++)
      for (size_t j = 0; j < this->shape.width; j++)
        result.set(i, j, this->get(i, j) * m.get(i, j));
  }
  return result;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::divide(Scalar m) const {
  Tensor<Scalar> result(shape.height, shape.width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->shape.height; i++)
      for (size_t j = 0; j < this->shape.width; j++)
        result.set(i, j, this->get(i, j) / m);
  }
  return result;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::inverse(Scalar m) const {
  Tensor<Scalar> result(shape.height, shape.width, this->gpu);
  if (gpu) {
    throw;
  } else {
    for (size_t i = 0; i < this->shape.height; i++)
      for (size_t j = 0; j < this->shape.width; j++)
        result.set(i, j, m / this->get(i, j));
  }
  return result;
}

template <class Scalar> Scalar Tensor<Scalar>::sum() const {
  Scalar result = 0;
  if (gpu)
    std::cout << "sum calc on CPU" << std::endl;

  for (size_t i = 0; i < this->shape.height; i++) {
    for (size_t j = 0; j < this->shape.width; j++) {
      result += this->get(i, j);
    }
  }

  return result;
}

template <class Scalar> void Tensor<Scalar>::apply(Scalar func(Scalar)) {
  if (gpu)
    std::cout << "apply calc on CPU" << std::endl;

  for (size_t i = 0; i < this->shape.height; i++) {
    for (size_t j = 0; j < this->shape.width; j++) {
      this->set(i, j, func(this->get(i, j)));
    }
  }
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::getLine(unsigned n) const {
  if (n >= this->shape.height) {
    fprintf(stderr, "Can't subscrit line at %i.\n", n);
    throw;
  }

  Tensor<Scalar> result(1, shape.width, this->gpu);
  for (size_t i = 0; i < this->shape.width; i++)
    result.set(0, i, this->get(n, i));

  return result;
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::getRow(unsigned n) const {
  if (n >= this->shape.width) {
    fprintf(stderr, "Can't subscrit row at %i.\n", n);
    throw;
  }

  Tensor<Scalar> result(shape.height, 1, this->gpu);
  for (size_t i = 0; i < this->shape.height; i++)
    result.set(i, 0, this->get(i, n));

  return result;
}

template <class Scalar> bool Tensor<Scalar>::eq(const Tensor<Scalar> &m) const {
  for (size_t i = 0; i < this->shape.height; i++)
    for (size_t j = 0; j < m.width; j++)
      if (this->get(i, j) != m.get(i, j))
        return false;

  return true;
}

template <class Scalar> Tensor<Scalar> Tensor<Scalar>::transpose() const {
  Tensor<Scalar> result(this->shape.width, this->shape.height, this->gpu);

  for (size_t i = 0; i < this->shape.height; i++)
    for (size_t j = 0; j < this->shape.width; j++)
      result.set(j, i, this->get(i, j));

  return result;
}

template <class Scalar> void Tensor<Scalar>::display() const {

  for (size_t i = 0; i < this->shape.height; i++) {
    for (size_t j = 0; j < this->shape.width; j++) {
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
  return this->add(m);
}

template <class Scalar>
Tensor<Scalar> Tensor<Scalar>::operator%(const Tensor<Scalar> &m) const {
  return this->multiply(m);
}

template <class Scalar>
bool Tensor<Scalar>::operator==(const Tensor<Scalar> &m) {
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
