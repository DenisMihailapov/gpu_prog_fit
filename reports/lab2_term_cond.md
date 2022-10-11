# Отчёт Лаб 2: Уравнение теплопроводности

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
main:
     98, Generating create(new_T[:N][:N],error,T[:N][:N]) [if not already present]
    103, Generating present(new_T[:N][:N],T[:N][:N])
tol: 0.000009, N: 128, max_iter: 100000
udiff: 0.000000 (toi 0.000009)
itold: 746

Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/thermal_conductivity/term_cond.c
  main  NVIDIA  devicenum=0
    time(us): 769
    98: data region reached 2 times
        42: kernel launched 256 times
            grid: [1]  block: [128]
             device time(us): total=769 max=4 min=3 avg=3
            elapsed time(us): total=6,543 max=438 min=21 avg=25
    103: data region reached 1492 times

```


tol: 0.000009, N: 256, max_iter: 100000

```bash
$ pgcc -acc -Minfo=accel -o term_cond term_cond.c -lm && PGI_ACC_TIME=1 ./term_cond 9e-6 256 1e5
main:
     98, Generating create(new_T[:N][:N],error,T[:N][:N]) [if not already present]
    103, Generating present(new_T[:N][:N],T[:N][:N])
tol: 0.000009, N: 256, max_iter: 100000
udiff: 0.000000 (toi 0.000009)
itold: 597

Accelerator Kernel Timing data
/home/students/d.mikhailapov/gpu_prog_fit/thermal_conductivity/term_cond.c
  main  NVIDIA  devicenum=0
    time(us): 1,537
    98: data region reached 2 times
        42: kernel launched 512 times
            grid: [1]  block: [128]
             device time(us): total=1,537 max=4 min=3 avg=3
            elapsed time(us): total=12,264 max=432 min=22 avg=23
    103: data region reached 1194 times
```



#### Осмысление

Почему то нет информации о блоках "Compute a new estimate." (102-114) и "Save the current estimate." (121-126), хотя там стоит  acc loop independent. (может они специально никак не отмечаются в профелировании)

Вывод `42: kernel launched 256 times` не понятный.

...