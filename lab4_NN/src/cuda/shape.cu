#include "../../include/shape.h"
#include <iostream>


Shape::Shape(size_t x, size_t y) :
	x(x), y(y)
{ }

void Shape::print(){ std::cout << "(" << x << ", " << y << ")" << std::endl; }