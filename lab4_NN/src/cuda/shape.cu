#include "../../include/shape.h"
#include <iostream>


Shape::Shape(size_t height, size_t width) :
	height(height), width(width)
{ }

void Shape::print(){ std::cout << "(" << height << ", " << width << ")" << std::endl; }