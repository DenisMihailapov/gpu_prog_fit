#pragma once

#include <cstddef>

struct Shape {
	size_t x, y;

	Shape(size_t x = 1, size_t y = 1);
	void print();
	
};
