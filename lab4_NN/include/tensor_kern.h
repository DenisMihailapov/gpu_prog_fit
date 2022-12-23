#include <iostream>


template <class Number> void addKernelWrapper(Number* m1, Number* m2, Number* m3, size_t size);
template <class Number> void dotKernelWrapper(Number* m1, Number* m2, Number* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns);


class Wrapper {
public:
    template <class Number> void add(Number* m1, Number* m2, Number* m3, size_t size);
    template <class Number> void dot(Number* m1, Number* m2, Number* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns);
};
