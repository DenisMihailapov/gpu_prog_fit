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
    102, Generating create(error,new_T[:N][:N]) [if not already present]
         Generating copy(T[:N][:N]) [if not already present]
    103, Generating Tesla code
        108, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        109, #pragma acc loop seq
    109, Complex loop carried dependence of T->->,new_T->-> prevents parallelization
    118, Generating Tesla code
        123, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        124, #pragma acc loop seq
    124, Complex loop carried dependence of new_T->->,T->-> prevents parallelization

Input params(tol = 9.00e-06, N = 128, max_iter = 100000)
START calculation...

END calculation
iter: 746
norm T: 0.006305 
udiff: 0.000009 (toi 0.000009)


Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  mse_norm  NVIDIA  devicenum=0
    time(us): 17,264
    14: compute region reached 746 times
        17: kernel launched 746 times
            grid: [128]  block: [128]
             device time(us): total=4,476 max=11 min=5 avg=6
            elapsed time(us): total=20,384 max=139 min=25 avg=27
        17: reduction kernel launched 746 times
            grid: [1]  block: [256]
             device time(us): total=3,501 max=9 min=4 avg=4
            elapsed time(us): total=17,114 max=137 min=22 avg=22
    14: data region reached 1492 times
        14: data copyin transfers: 746
             device time(us): total=3,147 max=133 min=3 avg=4
        18: data copyout transfers: 746
             device time(us): total=6,140 max=119 min=8 avg=8
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  diff  NVIDIA  devicenum=0
    time(us): 17,277
    26: compute region reached 746 times
        29: kernel launched 746 times
            grid: [128]  block: [128]
             device time(us): total=4,489 max=12 min=6 avg=6
            elapsed time(us): total=20,822 max=152 min=26 avg=27
        29: reduction kernel launched 746 times
            grid: [1]  block: [256]
             device time(us): total=3,501 max=9 min=4 avg=4
            elapsed time(us): total=17,124 max=146 min=21 avg=22
    26: data region reached 1492 times
        26: data copyin transfers: 746
             device time(us): total=3,000 max=7 min=3 avg=4
        31: data copyout transfers: 746
             device time(us): total=6,287 max=139 min=8 avg=8
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  main  NVIDIA  devicenum=0
    time(us): 81,447
    102: data region reached 2 times
        42: kernel launched 256 times
            grid: [1]  block: [128]
             device time(us): total=769 max=4 min=3 avg=3
            elapsed time(us): total=6,353 max=454 min=21 avg=24
        102: data copyin transfers: 128
             device time(us): total=634 max=14 min=4 avg=4
        131: data copyout transfers: 128
             device time(us): total=989 max=16 min=7 avg=7
    103: compute region reached 746 times
        103: kernel launched 746 times
            grid: [1]  block: [128]
             device time(us): total=50,627 max=150 min=64 avg=67
            elapsed time(us): total=67,321 max=201 min=85 avg=90
    118: compute region reached 746 times
        118: kernel launched 746 times
            grid: [1]  block: [128]
             device time(us): total=28,428 max=74 min=36 avg=38
            elapsed time(us): total=45,049 max=169 min=57 avg=60
```


tol: 0.000009, N: 256, max_iter: 100000

```bash
$ pgcc -acc -Minfo=accel -o term_cond term_cond.c -lm && PGI_ACC_TIME=1 ./term_cond 9e-6 256 1e5
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
    102, Generating create(error,new_T[:N][:N]) [if not already present]
         Generating copy(T[:N][:N]) [if not already present]
    103, Generating Tesla code
        108, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        109, #pragma acc loop seq
    109, Complex loop carried dependence of T->->,new_T->-> prevents parallelization
    118, Generating Tesla code
        123, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        124, #pragma acc loop seq
    124, Complex loop carried dependence of new_T->->,T->-> prevents parallelization

Input params(tol = 9.00e-06, N = 256, max_iter = 100000)
START calculation...

END calculation
iter: 597
norm T: 0.005313 
udiff: 0.000009 (toi 0.000009)


Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  mse_norm  NVIDIA  devicenum=0
    time(us): 16,767
    14: compute region reached 597 times
        17: kernel launched 597 times
            grid: [512]  block: [128]
             device time(us): total=6,540 max=12 min=10 avg=10
            elapsed time(us): total=19,529 max=150 min=31 avg=32
        17: reduction kernel launched 597 times
            grid: [1]  block: [256]
             device time(us): total=2,986 max=6 min=5 avg=5
            elapsed time(us): total=14,286 max=141 min=22 avg=23
    14: data region reached 1194 times
        14: data copyin transfers: 597
             device time(us): total=2,412 max=12 min=4 avg=4
        18: data copyout transfers: 597
             device time(us): total=4,829 max=23 min=8 avg=8
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  diff  NVIDIA  devicenum=0
    time(us): 17,590
    26: compute region reached 597 times
        29: kernel launched 597 times
            grid: [512]  block: [128]
             device time(us): total=7,148 max=12 min=11 avg=11
            elapsed time(us): total=20,135 max=155 min=32 avg=33
        29: reduction kernel launched 597 times
            grid: [1]  block: [256]
             device time(us): total=2,985 max=5 min=5 avg=5
            elapsed time(us): total=14,028 max=136 min=22 avg=23
    26: data region reached 1194 times
        26: data copyin transfers: 597
             device time(us): total=2,531 max=109 min=4 avg=4
        31: data copyout transfers: 597
             device time(us): total=4,926 max=16 min=8 avg=8
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  main  NVIDIA  devicenum=0
    time(us): 129,649
    102: data region reached 2 times
        42: kernel launched 512 times
            grid: [1]  block: [128]
             device time(us): total=1,537 max=4 min=3 avg=3
            elapsed time(us): total=12,221 max=461 min=21 avg=23
        102: data copyin transfers: 256
             device time(us): total=1,292 max=14 min=4 avg=5
        131: data copyout transfers: 256
             device time(us): total=1,963 max=19 min=7 avg=7
    103: compute region reached 597 times
        103: kernel launched 597 times
            grid: [2]  block: [128]
             device time(us): total=80,673 max=136 min=135 avg=135
            elapsed time(us): total=94,590 max=271 min=156 avg=158
    118: compute region reached 597 times
        118: kernel launched 597 times
            grid: [2]  block: [128]
             device time(us): total=44,184 max=75 min=74 avg=74
            elapsed time(us): total=58,887 max=223 min=96 avg=98
```

### Осмысление

#### Подробности

Я рассмотрел вариант с размером сетки 128х128.

```txt
Есть вопрос, как использовать функции и процедуры правитьно. То есть передавать туда данные именно с GPU, чтоб они не копировались каждый раз. 

 Вынес #pragma acc data copy(T) copy(new_T) ... вне do ... while стало лучше
  main  NVIDIA  devicenum=0
    time(us): 81,447
    102: data region reached 2 times
        42: kernel launched 256 times
 

 Вывод профелирования времени для функции mse_norm

14: data region reached 1492 times # 1492/764 ~ 2
    14: data copyin transfers: 746  # вот здесь проблема с постоянным копированием
          device time(us): total=3,147 max=133 min=3 avg=4
    18: data copyout transfers: 746
          device time(us): total=6,140 max=119 min=8 avg=8
```

#### Вывод для всех функций, где происходит распараллеливание

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
    102, Generating create(error,new_T[:N][:N]) [if not already present]
         Generating copy(T[:N][:N]) [if not already present]
    103, Generating Tesla code  # вызов функции `mse_norm`
        108, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        109, #pragma acc loop seq 
          # в этой строчке производитя подсчёт нового массива значений `T`
    109, Complex loop carried dependence of T->->,new_T->-> prevents parallelization
    118, Generating Tesla code // вызов функции `diff` 
        123, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
        124, #pragma acc loop seq
    124, Complex loop carried dependence of new_T->->,T->-> prevents parallelization
```

#### Анализ затраченного времени

```bash
Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  mse_norm  NVIDIA  devicenum=0
    time(us): 17,264
    14: compute region reached 746 times # 764 это количество итераций
        17: kernel launched 746 times # здесь производится подсчёт умножений a[i][j] * a[i][j]
            grid: [128]  block: [128]
             device time(us): total=4,476 max=11 min=5 avg=6
            elapsed time(us): total=20,384 max=139 min=25 avg=27
        17: reduction kernel launched 746 times
            grid: [1]  block: [256] # здесь уже запускается суммирование
             device time(us): total=3,501 max=9 min=4 avg=4
            elapsed time(us): total=17,114 max=137 min=22 avg=22
    14: data region reached 1492 times # 1492/764 ~ 2
        14: data copyin transfers: 746  # вот здесь проблема с постоянным копированием
             device time(us): total=3,147 max=133 min=3 avg=4
        18: data copyout transfers: 746
             device time(us): total=6,140 max=119 min=8 avg=8

/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  diff  NVIDIA  devicenum=0
    time(us): 17,277
    26: compute region reached 746 times
        29: kernel launched 746 times
            grid: [128]  block: [128]
             device time(us): total=4,489 max=12 min=6 avg=6
            elapsed time(us): total=20,822 max=152 min=26 avg=27
        29: reduction kernel launched 746 times
            grid: [1]  block: [256] 
             device time(us): total=3,501 max=9 min=4 avg=4
            elapsed time(us): total=17,124 max=146 min=21 avg=22
    26: data region reached 1492 times
        26: data copyin transfers: 746
             device time(us): total=3,000 max=7 min=3 avg=4
        31: data copyout transfers: 746
             device time(us): total=6,287 max=139 min=8 avg=8
/home/students/d.mikhailapov/gpu_prog_fit/lab2&3_thermal_conductivity/term_cond_acc.c
  main  NVIDIA  devicenum=0
    time(us): 81,447 # Обработка основного цикла
    102: data region reached 2 times
        42: kernel launched 256 times  # это отностися к allocate2DArray, но почему?
            grid: [1]  block: [128]
             device time(us): total=769 max=4 min=3 avg=3
            elapsed time(us): total=6,353 max=454 min=21 avg=24
        102: data copyin transfers: 128  # это не посредственно запуск do ... while 
             device time(us): total=634 max=14 min=4 avg=4
        131: data copyout transfers: 128 # это выход из do ... while 
             device time(us): total=989 max=16 min=7 avg=7
    103: compute region reached 746 times
        103: kernel launched 746 times # запуск `mse_norm`
            grid: [1]  block: [128]
             device time(us): total=50,627 max=150 min=64 avg=67
            elapsed time(us): total=67,321 max=201 min=85 avg=90
    118: compute region reached 746 times
        118: kernel launched 746 times # запуск `diff`
            grid: [1]  block: [128]
             device time(us): total=28,428 max=74 min=36 avg=38
            elapsed time(us): total=45,049 max=169 min=57 avg=60

```