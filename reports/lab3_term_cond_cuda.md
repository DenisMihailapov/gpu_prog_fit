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

//Функция для вычисления квадрата L2 нормы (gpu)
__global__ void square_sum(REAL*  T, REAL *v_norm, size_t N)

//Функция для вычисления разности двух массивов (gpu)
__global__ void sub(REAL*, REAL*, REAL*, size_t)

//Функция для вычисления копирования значений массива (gpu)
__global__ void copy(REAL*, REAL*, size_t);


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

Размер массива: N^2 (N = 128, 256, 512)

Количество блоков для редукции вычислений: 16

Количество элементов в блоке: N/16

#### Эксперимент 1

```bash

Input params(tol = 9.00e-06, N = 128, max_iter = 100000)

count of blocks: 16, block size: 8

iter: 0
||new_T|| (gpu): 81.745304 
diff (||subT||): 74.639380 


iter: 1
||new_T|| (gpu): 116.956750 
diff (||subT||): 28.131078 


iter: 2
||new_T|| (gpu): 94.466105 
diff (||subT||): 20.512399 


iter: 3
||new_T|| (gpu): 93.312672 
diff (||subT||): 9.460607 


iter: 4
||new_T|| (gpu): 81.714596 
diff (||subT||): 9.849178 


iter: 5
||new_T|| (gpu): 79.584790 
diff (||subT||): 6.151183 


iter: 6
||new_T|| (gpu): 67.715824 
diff (||subT||): 4.605171 


iter: 7
||new_T|| (gpu): 70.749755 
diff (||subT||): 13.128720 


iter: 8
||new_T|| (gpu): 53.361139 
diff (||subT||): 10.687530 


iter: 9
||new_T|| (gpu): 62.205571 
diff (||subT||): 11.079595 

iter: 10
||new_T|| (gpu): 61.782954 
diff (||subT||): 9.978603

...

iter: 29267
||new_T|| (gpu): 0.047655 
diff (||subT||): 0.000009 


iter: 29268
||new_T|| (gpu): 0.047656 
diff (||subT||): 0.000009 


iter: 29269
||new_T|| (gpu): 0.047639 
diff (||subT||): 0.000009 


iter: 29270
||new_T|| (gpu): 0.047649 
diff (||subT||): 0.000009 


iter: 29271
||new_T|| (gpu): 0.047639 
diff (||subT||): 0.000009 


iter: 29272
||new_T|| (gpu): 0.047619 
diff (||subT||): 0.000009

```

#### Эксперимент 2

```bash

Input params(tol = 9.00e-06, N = 256, max_iter = 100000)

count of blocks: 16, block size: 16

iter: 0
||new_T|| (gpu): 68.305548 
diff (||subT||): 36.118319 


iter: 1
||new_T|| (gpu): 55.375855 
diff (||subT||): 6.946569 


iter: 2
||new_T|| (gpu): 15.483102 
diff (||subT||): 8.677144 


iter: 3
||new_T|| (gpu): 41.018221 
diff (||subT||): 3.413683 


iter: 4
||new_T|| (gpu): 8.321993 
diff (||subT||): 1.625429 


iter: 5
||new_T|| (gpu): 7.655138 
diff (||subT||): 1.198946 


iter: 6
||new_T|| (gpu): 17.759334 
diff (||subT||): 0.923873 


iter: 7
||new_T|| (gpu): 18.059189 
diff (||subT||): 1.944265 


iter: 8
||new_T|| (gpu): 40.832225 
diff (||subT||): 1.716922 


iter: 9
||new_T|| (gpu): 5.536848 
diff (||subT||): 1.436230 


iter: 10
||new_T|| (gpu): 4.978793 
diff (||subT||): 0.487646

...

iter: 95017
||new_T|| (gpu): 0.066184 
diff (||subT||): 0.000012 


iter: 95018
||new_T|| (gpu): 0.066197 
diff (||subT||): 0.000012 


iter: 95019
||new_T|| (gpu): 0.066184 
diff (||subT||): 0.000012 


iter: 95020
||new_T|| (gpu): 0.066176 
diff (||subT||): 0.000012 


iter: 95021
||new_T|| (gpu): 0.066172 
diff (||subT||): 0.000012 


iter: 95022
||new_T|| (gpu): 0.066176 
diff (||subT||): 0.000012 


iter: 95023
||new_T|| (gpu): 0.066172 
diff (||subT||): 0.000012 


iter: 95024
||new_T|| (gpu): 0.066164 
diff (||subT||): 0.000012 


iter: 95025
||new_T|| (gpu): 0.066173 
diff (||subT||): 0.000008 

```

#### Эксперимент 3

```bash
Input params(tol = 9.00e-06, N = 512, max_iter = 1000000)

count of blocks: 16, block size: 32

iter: 0
||new_T|| (gpu): 124.402978 
diff (||subT||): 80.557407 


iter: 1
||new_T|| (gpu): 180.348172 
diff (||subT||): 46.102836 


iter: 2
||new_T|| (gpu): 55.141376 
diff (||subT||): 19.186920 


iter: 3
||new_T|| (gpu): 138.382716 
diff (||subT||): 13.836824 


iter: 4
||new_T|| (gpu): 113.476969 
diff (||subT||): 8.718369 


iter: 5
||new_T|| (gpu): 104.577520 
diff (||subT||): 7.207953 


iter: 6
||new_T|| (gpu): 88.564922 
diff (||subT||): 4.620734 


iter: 7
||new_T|| (gpu): 114.094248 
diff (||subT||): 3.198661 


iter: 8
||new_T|| (gpu): 111.199870 
diff (||subT||): 6.191944 


iter: 9
||new_T|| (gpu): 93.001759 
diff (||subT||): 4.502291 


iter: 10
||new_T|| (gpu): 80.206183 
diff (||subT||): 3.442309 

...

iter: 91960
||new_T|| (gpu): 0.726904 
diff (||subT||): 0.000009 


iter: 91961
||new_T|| (gpu): 0.726902 
diff (||subT||): 0.000009 


iter: 91962
||new_T|| (gpu): 0.726717 
diff (||subT||): 0.000009 


iter: 91963
||new_T|| (gpu): 0.726868 
diff (||subT||): 0.000009 


iter: 91964
||new_T|| (gpu): 0.726820 
diff (||subT||): 0.000009 


iter: 91965
||new_T|| (gpu): 0.726598 
diff (||subT||): 0.000009 


iter: 91966
||new_T|| (gpu): 0.726760 
diff (||subT||): 0.000009 

```

### Сводная табличка

Количество блоков: 16

|  N    |  block size  | iter   |
|:-----:|:------------:|:------:|
| 128   |   8          |  29272 |
| 256   |  16          |  75220 |
| 512   |  32          |  90866 |

Эксперименты показали, что

1. Разные запуски при одинаковом N дают разное количество итераций (в таблице укащзано средне нескольких запусков).
2. Чем больше N, тем больше итераций необходимо.
3. При большом N, скорость схождения алгоритма (уменьшение ||subT||), значительно замедляется с увеличением счётчика итераций.
4. Для дольшего N требуется большее количество блоков, иначе происходит `CUDA Runtime Error: invalid configuration argument`. Так для N=512 пеобходимо количество блоков не менее 16.
