#pragma once

#include <cstddef>

struct Shape {
	size_t height;
    size_t width;

	Shape(size_t height = 1, size_t width = 1);
	void print();
	
};
