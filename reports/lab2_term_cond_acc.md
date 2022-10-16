# Отчёт Лаб 2: Уравнение теплопроводности (OpenACC)

## Задание

Реализовать решение уравнение теплопроводности (пятиточечный шаблон) в двумерной области на  равномерных сетках (128^2, 256^2, 512^2). Граничные условия – линейная интерполяция между углами области. Значение в углах – 10, 20, 30, 20. Ограничить точность – 10^-6 и максимальное число итераций – 10^6.

Параметры (точность, размер сетки, количество итераций) должны задаваться через параметры командной строки.

Вывод программы - количество итераций и достигнутое значение ошибки.

Перенести программу на GPU используя директивы OpenACC. Сравнить скорость работы для разных размеров сеток на центральном и графическом процессоре.

Произвести профилирование программы с использованием NsightSystems. Произвести оптимизацию кода.

## Результат

### Реализованные функции

```C

// Функция для инициализации граничных условий
void init_border(uint, REAL**, REAL, REAL, REAL, REAL);

void init_full(uint, REAL**, REAL, REAL, REAL, REAL);

// Функции работы с памятью
REAL **allocate2DArray(int, int)

void free2DArray(REAL**, int, int)

// Функции метрик
REAL mse_norm(int, int, REAL**)

REAL diff(int, int, REAL**, REAL**)
   
```

### Вывод профилировщика

tol: 0.000009, N: 128, max_iter: 100000

```bash
$ pgcc -acc -Minfo=accel -o term_cond term_cond.c -lm && PGI_ACC_TIME=1 ./term_cond 9e-6 128 1e5
"term_cond_acc.c", line 95: warning: variable "v" was declared but never referenced
    REAL v = 0.0, t;
         ^

"term_cond_acc.c", line 95: warning: variable "t" was declared but never referenced
    REAL v = 0.0, t;
                  ^

mse_norm:
     14, Generating implicit copy(v) [if not already present]
         Generating implicit copyin(a[:nx][:ny]) [if not already present]
     16, Loop is parallelizable
     17, Loop is parallelizable
         Generating Tesla code
         16, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x collapsed-innermost */
             Generating reduction(+:v)
         17,   /* blockIdx.x threadIdx.x auto-collapsed */
diff:
     26, Generating implicit copyin(a[:nx][:ny]) [if not already present]
         Generating implicit copy(v) [if not already present]
         Generating implicit copyin(b[:nx][:ny]) [if not already present]
     28, Loop is parallelizable
     29, Loop is parallelizable
         Generating Tesla code
         28, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x collapsed-innermost */
             Generating reduction(+:v)
         29,   /* blockIdx.x threadIdx.x auto-collapsed */
main:
    101, Generating create(error,new_T[:N][:N]) [if not already present]
         Generating copy(T[:N][:N]) [if not already present]
    103, Generating Tesla code
        108, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        109, #pragma acc loop seq
    109, Complex loop carried dependence of T->->,new_T->-> prevents parallelization
    119, Generating Tesla code
        124, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        125, #pragma acc loop seq
    125, Complex loop carried dependence of new_T->->,T->-> prevents parallelization
tol: 0.000009, N: 128, max_iter: 100000
iter: 100
norm T: 0.029565 
udiff: 0.000296 (toi 0.000009)

iter: 200
norm T: 0.017646 
udiff: 0.000089 (toi 0.000009)

iter: 300
norm T: 0.013080 
udiff: 0.000044 (toi 0.000009)

iter: 400
norm T: 0.010587 
udiff: 0.000027 (toi 0.000009)

iter: 500
norm T: 0.008992 
udiff: 0.000019 (toi 0.000009)

iter: 600
norm T: 0.007874 
udiff: 0.000014 (toi 0.000009)

iter: 700
norm T: 0.007044 
udiff: 0.000010 (toi 0.000009)


Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  mse_norm  NVIDIA  devicenum=0
    time(us): 16,729
    14: compute region reached 764 times
        17: kernel launched 764 times
            grid: [128]  block: [128]
             device time(us): total=3,836 max=7 min=5 avg=5
            elapsed time(us): total=20,031 max=132 min=25 avg=26
        17: reduction kernel launched 764 times
            grid: [1]  block: [256]
             device time(us): total=3,071 max=5 min=4 avg=4
            elapsed time(us): total=16,798 max=125 min=21 avg=21
    14: data region reached 1528 times
        14: data copyin transfers: 764
             device time(us): total=3,139 max=8 min=4 avg=4
        18: data copyout transfers: 764
             device time(us): total=6,683 max=24 min=8 avg=8
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  diff  NVIDIA  devicenum=0
    time(us): 17,909
    26: compute region reached 765 times
        29: kernel launched 765 times
            grid: [128]  block: [128]
             device time(us): total=3,856 max=7 min=5 avg=5
            elapsed time(us): total=19,951 max=39 min=25 avg=26
        29: reduction kernel launched 765 times
            grid: [1]  block: [256]
             device time(us): total=3,076 max=5 min=4 avg=4
            elapsed time(us): total=16,949 max=125 min=21 avg=22
    26: data region reached 1530 times
        26: data copyin transfers: 1021
             device time(us): total=4,295 max=12 min=3 avg=4
        31: data copyout transfers: 765
             device time(us): total=6,170 max=12 min=8 avg=8
        42: kernel launched 256 times
            grid: [1]  block: [128]
             device time(us): total=512 max=2 min=2 avg=2
            elapsed time(us): total=5,359 max=35 min=20 avg=20
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  main  NVIDIA  devicenum=0
    time(us): 1,518,118
    101: data region reached 1528 times
        42: kernel launched 195584 times
            grid: [1]  block: [128]
             device time(us): total=395,362 max=5 min=2 avg=2
            elapsed time(us): total=4,123,169 max=1,419 min=20 avg=21
        101: data copyin transfers: 97792
             device time(us): total=461,884 max=117 min=4 avg=4
        128: data copyout transfers: 97792
             device time(us): total=592,852 max=127 min=5 avg=6
    103: compute region reached 764 times
        103: kernel launched 764 times
            grid: [1]  block: [128]
             device time(us): total=43,422 max=69 min=56 avg=56
            elapsed time(us): total=60,302 max=182 min=77 avg=78
    119: compute region reached 764 times
        119: kernel launched 764 times
            grid: [1]  block: [128]
             device time(us): total=24,598 max=39 min=32 avg=32
            elapsed time(us): total=41,110 max=142 min=52 avg=53

```


tol: 0.000009, N: 256, max_iter: 100000

```bash
$ pgcc -acc -Minfo=accel -o term_cond term_cond.c -lm && PGI_ACC_TIME=1 ./term_cond 9e-6 256 1e5
"term_cond_acc.c", line 95: warning: variable "v" was declared but never referenced
    REAL v = 0.0, t;
         ^

"term_cond_acc.c", line 95: warning: variable "t" was declared but never referenced
    REAL v = 0.0, t;
                  ^

mse_norm:
     14, Generating implicit copy(v) [if not already present]
         Generating implicit copyin(a[:nx][:ny]) [if not already present]
     16, Loop is parallelizable
     17, Loop is parallelizable
         Generating Tesla code
         16, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x collapsed-innermost */
             Generating reduction(+:v)
         17,   /* blockIdx.x threadIdx.x auto-collapsed */
diff:
     26, Generating implicit copyin(a[:nx][:ny]) [if not already present]
         Generating implicit copy(v) [if not already present]
         Generating implicit copyin(b[:nx][:ny]) [if not already present]
     28, Loop is parallelizable
     29, Loop is parallelizable
         Generating Tesla code
         28, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x collapsed-innermost */
             Generating reduction(+:v)
         29,   /* blockIdx.x threadIdx.x auto-collapsed */
main:
    101, Generating create(error,new_T[:N][:N]) [if not already present]
         Generating copy(T[:N][:N]) [if not already present]
    103, Generating Tesla code
        108, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        109, #pragma acc loop seq
    109, Complex loop carried dependence of T->->,new_T->-> prevents parallelization
    119, Generating Tesla code
        124, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        125, #pragma acc loop seq
    125, Complex loop carried dependence of new_T->->,T->-> prevents parallelization
tol: 0.000009, N: 256, max_iter: 100000
iter: 100
norm T: 0.020734 
udiff: 0.000204 (toi 0.000009)

iter: 200
norm T: 0.012327 
udiff: 0.000061 (toi 0.000009)

iter: 300
norm T: 0.009111 
udiff: 0.000030 (toi 0.000009)

iter: 400
norm T: 0.007357 
udiff: 0.000018 (toi 0.000009)

iter: 500
norm T: 0.006235 
udiff: 0.000012 (toi 0.000009)

iter: 600
norm T: 0.005448 
udiff: 0.000009 (toi 0.000009)


Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  mse_norm  NVIDIA  devicenum=0
    time(us): 16,121
    14: compute region reached 605 times
        17: kernel launched 605 times
            grid: [512]  block: [128]
             device time(us): total=5,459 max=12 min=8 avg=9
            elapsed time(us): total=18,448 max=103 min=29 avg=30
        17: reduction kernel launched 605 times
            grid: [1]  block: [256]
             device time(us): total=2,427 max=5 min=4 avg=4
            elapsed time(us): total=13,392 max=37 min=21 avg=22
    14: data region reached 1210 times
        14: data copyin transfers: 605
             device time(us): total=2,602 max=8 min=4 avg=4
        18: data copyout transfers: 605
             device time(us): total=5,633 max=22 min=8 avg=9
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  diff  NVIDIA  devicenum=0
    time(us): 19,546
    26: compute region reached 606 times
        29: kernel launched 606 times
            grid: [512]  block: [128]
             device time(us): total=6,074 max=12 min=9 avg=10
            elapsed time(us): total=18,935 max=46 min=30 avg=31
        29: reduction kernel launched 606 times
            grid: [1]  block: [256]
             device time(us): total=2,431 max=5 min=4 avg=4
            elapsed time(us): total=13,528 max=128 min=21 avg=22
    26: data region reached 1212 times
        26: data copyin transfers: 1118
             device time(us): total=4,894 max=14 min=4 avg=4
        31: data copyout transfers: 606
             device time(us): total=5,123 max=24 min=8 avg=8
        42: kernel launched 512 times
            grid: [1]  block: [128]
             device time(us): total=1,024 max=2 min=2 avg=2
            elapsed time(us): total=10,669 max=42 min=20 avg=20
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  main  NVIDIA  devicenum=0
    time(us): 2,425,661
    101: data region reached 1210 times
        42: kernel launched 309760 times
            grid: [1]  block: [128]
             device time(us): total=623,396 max=4 min=2 avg=2
            elapsed time(us): total=6,462,672 max=1,428 min=19 avg=20
        101: data copyin transfers: 154880
             device time(us): total=757,811 max=122 min=4 avg=4
        128: data copyout transfers: 154880
             device time(us): total=939,799 max=128 min=5 avg=6
    103: compute region reached 605 times
        103: kernel launched 605 times
            grid: [2]  block: [128]
             device time(us): total=67,649 max=135 min=111 avg=111
            elapsed time(us): total=80,980 max=194 min=132 avg=133
    119: compute region reached 605 times
        119: kernel launched 605 times
            grid: [2]  block: [128]
             device time(us): total=37,006 max=75 min=61 avg=61
            elapsed time(us): total=50,521 max=199 min=82 avg=83
```


### Осмысление

#### Подробности 

Я рассмотрел вариант с размером сетки 128х128.

```txt
Есть вопрос, как использовать функции и процедуры правитьно. То есть передавать туда данные именно с GPU, чтоб они не копировались каждый раз. 

1. Комприлятор мне выдал такие предупреждения

"term_cond_acc.c", line 95: warning: variable "v" was declared but never referenced
    REAL v = 0.0, t;
         ^

"term_cond_acc.c", line 95: warning: variable "t" was declared but never referenced
    REAL v = 0.0, t;
                  ^

2. Вывод профелирования времени для функции mse_norm

14: data region reached 1528 times # 1528/764 = 2
    14: data copyin transfers: 764 # вот здесь проблема с постоянным копированием
          device time(us): total=3,139 max=8 min=4 avg=4
    18: data copyout transfers: 764
          device time(us): total=6,683 max=24 min=8 avg=8
```

#### Вывод для всех функций, где происходит распаралеливание

```bash
mse_norm:
     14, Generating implicit copy(v) [if not already present]
         Generating implicit copyin(a[:nx][:ny]) [if not already present] # копируем данные если их нет на GPU
     16, Loop is parallelizable
     17, Loop is parallelizable
         Generating Tesla code
         16, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x collapsed-innermost */
             Generating reduction(+:v)  # проводим редкуцию суммирования
         17,   /* blockIdx.x threadIdx.x auto-collapsed */
diff:
     26, Generating implicit copyin(a[:nx][:ny]) [if not already present]
         Generating implicit copy(v) [if not already present]
         Generating implicit copyin(b[:nx][:ny]) [if not already present]
     28, Loop is parallelizable
     29, Loop is parallelizable
         Generating Tesla code
         28, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x collapsed-innermost */
             Generating reduction(+:v)
         29,   /* blockIdx.x threadIdx.x auto-collapsed */
main:
    101, Generating create(error,new_T[:N][:N]) [if not already present]
         Generating copy(T[:N][:N]) [if not already present]
    103, Generating Tesla code  # вызов функции `mse_norm`
        108, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        109, #pragma acc loop seq 
          # в этой строчке производитя подсчёт нового массива значений `T`
    109, Complex loop carried dependence of T->->,new_T->-> prevents parallelization
    119, Generating Tesla code // вызов функции `diff` 
        124, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        125, #pragma acc loop seq
    125, Complex loop carried dependence of new_T->->,T->-> prevents parallelization
```

#### Вывод затраченного времени:

```bash
Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  mse_norm  NVIDIA  devicenum=0
    time(us): 16,729
    14: compute region reached 764 times # 764 это количество итераций
        17: kernel launched 764 times # здесь производится подсчёт умножений a[i][j] * a[i][j]
            grid: [128]  block: [128] # кажется не зависимо и параллельно
             device time(us): total=3,836 max=7 min=5 avg=5
            elapsed time(us): total=20,031 max=132 min=25 avg=26
        17: reduction kernel launched 764 times
            grid: [1]  block: [256] # здесь уже запускается суммирование
             device time(us): total=3,071 max=5 min=4 avg=4
            elapsed time(us): total=16,798 max=125 min=21 avg=21
    14: data region reached 1528 times # 1528/764 = 2
        14: data copyin transfers: 764 # вот здесь проблема с постоянным копированием
             device time(us): total=3,139 max=8 min=4 avg=4
        18: data copyout transfers: 764
             device time(us): total=6,683 max=24 min=8 avg=8

/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  diff  NVIDIA  devicenum=0
    time(us): 17,909
    26: compute region reached 765 times
        29: kernel launched 765 times
            grid: [128]  block: [128]
             device time(us): total=3,856 max=7 min=5 avg=5
            elapsed time(us): total=19,951 max=39 min=25 avg=26
        29: reduction kernel launched 765 times
            grid: [1]  block: [256]
             device time(us): total=3,076 max=5 min=4 avg=4
            elapsed time(us): total=16,949 max=125 min=21 avg=22
    26: data region reached 1530 times
        26: data copyin transfers: 1021
             device time(us): total=4,295 max=12 min=3 avg=4
        31: data copyout transfers: 765
             device time(us): total=6,170 max=12 min=8 avg=8
        42: kernel launched 256 times
            grid: [1]  block: [128]
             device time(us): total=512 max=2 min=2 avg=2
            elapsed time(us): total=5,359 max=35 min=20 avg=20

/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  main  NVIDIA  devicenum=0
    time(us): 1,518,118
    101: data region reached 1528 times
        42: kernel launched 195584 times
            grid: [1]  block: [128]
             device time(us): total=395,362 max=5 min=2 avg=2
            elapsed time(us): total=4,123,169 max=1,419 min=20 avg=21
        101: data copyin transfers: 97792
             device time(us): total=461,884 max=117 min=4 avg=4
        128: data copyout transfers: 97792
             device time(us): total=592,852 max=127 min=5 avg=6
    103: compute region reached 764 times
        103: kernel launched 764 times
            grid: [1]  block: [128]
             device time(us): total=43,422 max=69 min=56 avg=56
            elapsed time(us): total=60,302 max=182 min=77 avg=78
    119: compute region reached 764 times
        119: kernel launched 764 times
            grid: [1]  block: [128]
             device time(us): total=24,598 max=39 min=32 avg=32
            elapsed time(us): total=41,110 max=142 min=52 avg=53

```