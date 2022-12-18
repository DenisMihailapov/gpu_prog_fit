# Отчёт Лаб 3: Уравнение теплопроводности (CUDA)

## Задание

Реализовать решение уравнение теплопроводности (пятиточечный шаблон) в двумерной области на  равномерных сетках (128^2, 256^2, 512^2, 1024^2). Граничные условия – линейная интерполяция между углами области. Значение в углах – 10, 20, 30, 20. Ограничить точность – 10^-6 и максимальное число итераций – 10^6.

Параметры (точность, размер сетки, количество итераций) должны задаваться через параметры командной строки.

Вывод программы - количество итераций и достигнутое значение ошиибки.

Перенести программу на GPU используя CUDA. Операцию редукции (подсчет максимальной ошибки) реализовать с использованием библиотеки CUB.
Сравнить скорость работы для разных размеров сеток на графическом процессоре с  предыдущей реализацией на OpenACC.

## Результат

### Реализованные функции (heatCUDA.cu)

```C++

//Функции для отслеживания ошибок cuda
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T, const char* const, const char* const, const int);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const, const int);


//Функция для перевода 2D индексов в 1D
int __host__ __device__ ind2D(const size_t, const size_t, const size_t);

//Итерационный шаг (gpu)
__global__ void step_estimate(REAL*, REAL*, size_t);

//Функция для возведения массива в квадрат (gpu)
__global__ void pow_2(REAL*, REAL*  size_t)

//Функция для вычисления разности двух массивов (gpu)
__global__ void sub(REAL*, REAL*, REAL*, size_t)

//Функция для вычисления копирования значений массива (gpu)
__global__ void copy(REAL*, REAL*, size_t);

//Функция для вычисления суммы элементов масива (cub gpu)
REAL gpu_sum(REAL*  T, size_t num_items)

// Функция для инициализации граничных условий
void init_border(uint, REAL**, REAL, REAL, REAL, REAL);

//Функция для подчёта суммы элементов массива (cpu)
REAL sum_cpu(REAL*, uint)

//Функция для вычисления L2 нормы (cpu)
REAL norm_cpu(REAL*, uint)

//Вывод 2D массива в консоль
void print2D(REAL*, size_t, size_t)

```

### Примеры запуска программы

Размер массива: N^2 (N = 128, 256, 512, 1024)

Количество блоков для редукции вычислений: 32

Количество элементов в блоке: N/32

#### Эксперимент 1

```bash

Input params(tol = 9.00e-06, N = 128, max_iter = 100000)
count of blocks: 32, block size: 4

iter: 0
||new_T|| (gpu): 118.169083 
diff (||subT||): 225.150773 


iter: 100
||new_T|| (gpu): 1.151170 
diff (||subT||): 0.432954 


iter: 200
||new_T|| (gpu): 0.626997 
diff (||subT||): 0.183571 


iter: 300
||new_T|| (gpu): 0.447548 
diff (||subT||): 0.111816 


iter: 400
||new_T|| (gpu): 0.354223 
diff (||subT||): 0.078864 


iter: 500
||new_T|| (gpu): 0.296140 
diff (||subT||): 0.060244 


iter: 600
||new_T|| (gpu): 0.256135 
diff (||subT||): 0.048395 


iter: 700
||new_T|| (gpu): 0.226721 
diff (||subT||): 0.040254 


iter: 800
||new_T|| (gpu): 0.204085 
diff (||subT||): 0.034355 


iter: 900
||new_T|| (gpu): 0.186070 
diff (||subT||): 0.029907

...

iter: 43500
||new_T|| (gpu): 0.000198 
diff (||subT||): 0.000010 


iter: 43600
||new_T|| (gpu): 0.000195 
diff (||subT||): 0.000010 


iter: 43700
||new_T|| (gpu): 0.000193 
diff (||subT||): 0.000009 


iter: 43800
||new_T|| (gpu): 0.000190 
diff (||subT||): 0.000009 


iter: 43900
||new_T|| (gpu): 0.000187 
diff (||subT||): 0.000009 


iter: 44000
||new_T|| (gpu): 0.000184 
diff (||subT||): 0.000009

```

#### Эксперимент 2

```bash

Input params(tol = 9.00e-06, N = 25tt56, max_iter = 100000)
count of blocks: 32, block size: 8

iter: 0
||new_T|| (gpu): 166.826030 
diff (||subT||): 320.226990 


iter: 100
||new_T|| (gpu): 1.258760 
diff (||subT||): 0.592696 


iter: 200
||new_T|| (gpu): 0.653113 
diff (||subT||): 0.248538 


iter: 300
||new_T|| (gpu): 0.458491 
diff (||subT||): 0.150255 


iter: 400
||new_T|| (gpu): 0.360029 
diff (||subT||): 0.105347 


iter: 500
||new_T|| (gpu): 0.299653 
diff (||subT||): 0.080071 


iter: 600
||new_T|| (gpu): 0.258447 
diff (||subT||): 0.064036 


iter: 700
||new_T|| (gpu): 0.228332 
diff (||subT||): 0.053036 


iter: 800
||new_T|| (gpu): 0.205251 
diff (||subT||): 0.045063 


iter: 900
||new_T|| (gpu): 0.186931 
diff (||subT||): 0.039043 

...

iter: 99500
||new_T|| (gpu): 0.000993 
diff (||subT||): 0.000034 


iter: 99600
||new_T|| (gpu): 0.000990 
diff (||subT||): 0.000034 


iter: 99700
||new_T|| (gpu): 0.000986 
diff (||subT||): 0.000034 


iter: 99800
||new_T|| (gpu): 0.000982 
diff (||subT||): 0.000034 


iter: 99900
||new_T|| (gpu): 0.000979 
diff (||subT||): 0.000034 


iter: 100000
||new_T|| (gpu): 0.000975 
diff (||subT||): 0.000034

```

#### Эксперимент 3

```bash
Input params(tol = 9.00e-06, N = 512, max_iter = 100000)
count of blocks: 32, block size: 16

iter: 0
||new_T|| (gpu): 235.721065 
diff (||subT||): 451.926859 


iter: 100
||new_T|| (gpu): 1.450776 
diff (||subT||): 0.825414 


iter: 200
||new_T|| (gpu): 0.702936 
diff (||subT||): 0.344108 


iter: 300
||new_T|| (gpu): 0.480046 
diff (||subT||): 0.207218 


iter: 400
||new_T|| (gpu): 0.371728 
diff (||subT||): 0.144839 


iter: 500
||new_T|| (gpu): 0.306879 
diff (||subT||): 0.109802 


iter: 600
||new_T|| (gpu): 0.263298 
diff (||subT||): 0.087613 


iter: 700
||new_T|| (gpu): 0.231785 
diff (||subT||): 0.072415 


iter: 800
||new_T|| (gpu): 0.207817 
diff (||subT||): 0.061414 


iter: 900
||new_T|| (gpu): 0.188902 
diff (||subT||): 0.053117 

...

iter: 99500
||new_T|| (gpu): 0.005503 
diff (||subT||): 0.000142 


iter: 99600
||new_T|| (gpu): 0.005498 
diff (||subT||): 0.000142 


iter: 99700
||new_T|| (gpu): 0.005492 
diff (||subT||): 0.000142 


iter: 99800
||new_T|| (gpu): 0.005487 
diff (||subT||): 0.000142 


iter: 99900
||new_T|| (gpu): 0.005482 
diff (||subT||): 0.000142 


iter: 100000
||new_T|| (gpu): 0.005477 
diff (||subT||): 0.000141

```

#### Эксперимент 4

```bash
Input params(tol = 9.00e-06, N = 1024, max_iter = 100000)
count of blocks: 32, block size: 32

iter: 0
||new_T|| (gpu): 333.213398 
diff (||subT||): 630.225409 


iter: 100
||new_T|| (gpu): 1.773734 
diff (||subT||): 1.160725 


iter: 200
||new_T|| (gpu): 0.793463 
diff (||subT||): 0.482395 


iter: 300
||new_T|| (gpu): 0.520673 
diff (||subT||): 0.289710 


iter: 400
||new_T|| (gpu): 0.394253 
diff (||subT||): 0.202030 


iter: 500
||new_T|| (gpu): 0.320992 
diff (||subT||): 0.152859 


iter: 600
||new_T|| (gpu): 0.272877 
diff (||subT||): 0.121762 


iter: 700
||new_T|| (gpu): 0.238664 
diff (||subT||): 0.100490 


iter: 800
||new_T|| (gpu): 0.212970 
diff (||subT||): 0.085111 


iter: 900
||new_T|| (gpu): 0.192889 
diff (||subT||): 0.073524 

...

iter: 99500
||new_T|| (gpu): 0.005151 
diff (||subT||): 0.000276 


iter: 99600
||new_T|| (gpu): 0.005148 
diff (||subT||): 0.000276 


iter: 99700
||new_T|| (gpu): 0.005144 
diff (||subT||): 0.000276 


iter: 99800
||new_T|| (gpu): 0.005140 
diff (||subT||): 0.000275 


iter: 99900
||new_T|| (gpu): 0.005136 
diff (||subT||): 0.000275 


iter: 100000
||new_T|| (gpu): 0.005133 
diff (||subT||): 0.000275 

```

### Сводная табличка

Количество блоков: 32

tol: 9e-06, max_iter: 1e05

|  N    |  block size  |  iter   |
|:-----:|:------------:|:-------:|
| 128   |  4           |  44000  |
| 256   |  8           |  100000 |
| 512   |  16          |  100000 |
| 1024  |  32          |  100000 |

tol: 1e-06, max_iter: 1e06

|  N    |  block size  |   iter   |
|:-----:|:------------:|:--------:|
| 128   |  4           |  59000   |
| 256   |  8           |  194900  |
| 512   |  16          |  624500  |
| 1024  |  32          |  1000000 |

Эксперименты показали, что

1. Разные запуски при одинаковом N дают разное количество итераций (в таблице укащзано средне нескольких запусков).
2. Чем больше N, тем больше итераций необходимо.
3. При большом N, скорость схождения алгоритма (уменьшение `||subT||`), значительно замедляется с увеличением счётчика итераций.
4. Количество итераций и занчения итераций (`||subT||`, `||new_T||`)  не зависит от `BLOCKSNUM`.
