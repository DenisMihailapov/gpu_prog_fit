# Отчёт Лаб 1: Вычисление суммы синусов

## Задание

Заполнить на графическом процессоре массив типа float/double значениями синуса (один период на всю длину массива). Размер массива - 10^7 элементов.

Для заполненного массива на графическом процессоре посчитать сумму всех элементов массива.

Сравнить со значением, вычисленном на центральном процессоре.
Сравнить разультат для массивов float и double.

## Результат

### Таблица результатов суммирования

|       | float    |  double  |
|:-----:|:--------:|:--------:|
| cpu   | -0.03410 |  0.00000 |
| gpu   |  0.00470 |  0.00000 |

### Accelerator Kernel Timing data

#### float

```bash
main:
     23, Generating copy(sum_sin) [if not already present]
         Generating create(S[:]) [if not already present]
     26, Loop is parallelizable
         Generating Tesla code
         26, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
     30, Loop is parallelizable
         Generating Tesla code
         30, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
             Generating reduction(+:sum_sin)
Сумма: 0.00470

Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab1_sin/calc_sin.c
  main  NVIDIA  devicenum=0
    time(us): 388
    23: compute region reached 1 time
        26: kernel launched 1 time
            grid: [7813]  block: [128]
             device time(us): total=278 max=278 min=278 avg=278
            elapsed time(us): total=422 max=422 min=422 avg=422
    23: data region reached 2 times
        23: data copyin transfers: 1
             device time(us): total=8 max=8 min=8 avg=8
        31: data copyout transfers: 1
             device time(us): total=39 max=39 min=39 avg=39
    27: compute region reached 1 time
        30: kernel launched 1 time
            grid: [7813]  block: [128]
             device time(us): total=50 max=50 min=50 avg=50
            elapsed time(us): total=78 max=78 min=78 avg=78
        30: reduction kernel launched 1 time
            grid: [1]  block: [256]
             device time(us): total=13 max=13 min=13 avg=13
            elapsed time(us): total=33 max=33 min=33 avg=33
```

#### double

```bash
main:
     23, Generating copy(sum_sin) [if not already present]
         Generating create(S[:]) [if not already present]
     26, Loop is parallelizable
         Generating Tesla code
         26, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
     30, Loop is parallelizable
         Generating Tesla code
         30, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
             Generating reduction(+:sum_sin)
Сумма: 0.00000

Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/lab1_sin/calc_sin.c
  main  NVIDIA  devicenum=0
    time(us): 408
    23: compute region reached 1 time
        26: kernel launched 1 time
            grid: [7813]  block: [128]
             device time(us): total=268 max=268 min=268 avg=268
            elapsed time(us): total=421 max=421 min=421 avg=421
    23: data region reached 2 times
        23: data copyin transfers: 1
             device time(us): total=8 max=8 min=8 avg=8
        31: data copyout transfers: 1
             device time(us): total=32 max=32 min=32 avg=32
    27: compute region reached 1 time
        30: kernel launched 1 time
            grid: [7813]  block: [128]
             device time(us): total=84 max=84 min=84 avg=84
            elapsed time(us): total=111 max=111 min=111 avg=111
        30: reduction kernel launched 1 time
            grid: [1]  block: [256]
             device time(us): total=16 max=16 min=16 avg=16
            elapsed time(us): total=35 max=35 min=35 avg=35
```

#### Осмысление

Компилятор написал о том, что

- Сгенерировались данные на GPU `copy(sum_sin)` и `create(S[:])`
- Распаралелелись два цикла: На строке 26 и 30

  ```bash
  time(us): ... // Написал время выполнение кода
  23: compute region reached 1 time // Сгенерировали данные на GPU
      26: kernel launched 1 time // Цикл заполненеия синусов
      ...
  23: data region reached 2 times // Снова заходим на GPU
      23: data copyin transfers: 1 // Необходимо скопировать только `sum_sin` (`S[:]` уже на GPU)
      ...
      32: data copyout transfers: 1 // Забираем `sum_sin` с GPU 
      ...
  27: compute region reached 1 time // Вычисляем сумму
      30: kernel launched 1 time
          grid: [7813]  block: [128] // Разбиваем массив на 7813 блока по по 128 нитей (7813*128 ~ 10^6)
          ...
      30: reduction kernel launched 1 time
          grid: [1]  block: [256]   // ??

  ```

#### Сравнение float и double

- Разная сумма: 0.00470 (float) и 0.00000 (double)
- Разные time(us): 388 и 408 (и в целом работа с float быстрее, чем с double)

#### Дополнительные вопросы

- Как выполняется параллельное суммирование
  
  Массив разбивается на блоки по несколько элементов (размер блока равен количеству нитей в блоке gpu).
  далее производится суммирование в рамках каждого блока.

- Почему в программе два цикла, а ядер (по выводу профилировщика) три

- Почему сумма всех элементов не равна нулю
  
  Один для заполнения, второй для суммирования, третий для редукции

  Точности вычисления элементов не достаточно и происходят округления, что даёт суммарную ошибку.  
