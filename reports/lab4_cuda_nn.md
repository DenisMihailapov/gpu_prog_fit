# Отчёт Лаб 4: Нейронная сеть на CUDA

## Задание

Для нейронной сети, описаной ниже реализовать на CUDA программу реализующую обучение этой сети. Для обучения: входные данные - заданая матрица из элементов типа int,  на выходе значение от 0 до 1. Можно реализовать распознавание сайлика или конкретной цифры.

``` python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32**2, 16**2) # входной слой
        self.fc2 = nn.Linear(16**2, 4**2) # скрытый слой
        self.fc3 = nn.Linear(4 ** 2,1) # скрытый слой
# прямое распространение информации
    def forward(self, x):
          sigmoid = nn.Sigmoid()
          x = sigmoid(self.fc1(x))
          x = sigmoid(self.fc2(x))
          x = sigmoid(self.fc3(x))
          return x

```

Для проверки корректности работы программы использовать реализацию на Python.
Сравнить скорость обучения нейронной сети на Torch, PyTorch, собственной реализацией.

При реализации на CUDA по возможности использовать библиотеку cuDNN

## Результат

### Структура проекта

```bash
lab4_NN/
├── bin // папка с бинарниками (обновления не коммитятся)
│   ├── cuda
│   │   ├── ...
│   └── ...
├── include  // заголовочнве файлы (все в куче)
│   ├── nn_exception.cuh  // обрабодчик ошибок
│   ├── relu_activation.h  // функция ReLu (пока не работает до конца)
│   ├── shape.h  // клаcc размера тензора (можно добавить функционала)
│   ├── tensor.h  // cpp код тенсора
│   └── tensor_kern.h  // ядра для суммы и произведения тензоров
├── main.cpp
├── Makefile
├── README.md
└── src  // исходники (пока без cpp)
    └── cuda  // код для cuda ядер
        ├── relu_activation.cu
        ├── shape.cu
        └── tensor_kern.cu


```

Я не буду описывать все функции в коде (это долго, по крайней мере сейчас).
Лучше приведу пример использования и эксперименты.

### Примеры запуска программы

`main.cpp`

```C++
// Include C++ header files.
#include <chrono>
#include <iostream>
#include <ostream>

// Include local CUDA header files.

#include "include/relu_activation.h"

using namespace std::chrono;

int main() {

    Tensor<> A_cpu(1000, 1000), B_cpu(1000, 1000);
    Tensor<> A_gpu(1000, 1000, true), B_gpu(1000, 1000, true);

    // Only calculations
    auto t1 = steady_clock::now();
    A_cpu+B_cpu;
    auto t2 = steady_clock::now();
    auto elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A+B calculation (CPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    t1 = steady_clock::now();
    A_cpu.dot(B_cpu);
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A dot B calculation (CPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    t1 = steady_clock::now();
    5.*B_cpu;
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "a*B calculation (CPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    std::cout << std::endl;
    A_gpu+B_gpu; // for cuda init 
    // TODO: cuda tool

    t1 = steady_clock::now();
    A_gpu+B_gpu;
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A+B calculation (GPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    t1 = steady_clock::now();
    A_gpu.dot(B_gpu);
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "A dot B calculation (GPU): " << elapsed.count( ) << " ms" << std::endl;

    t1 = steady_clock::now();
    5.*B_gpu;
    t2 = steady_clock::now();
    elapsed = duration_cast<milliseconds>(t2 - t1);
    std::cout << "a*B calculation (GPU): " << elapsed.count( ) << " ms" << std::endl;
    t1 = t2;

    // ReLUActivation<> relu("test");
    
    // tensor3 = relu.forward(tensor3);
    // tensor3.display();

    return 0;
}
```

Здесь мы замеряем время выполнения остновых операций Тензора, которые нужны для нейронки (сумма и произведение матриц):

Размеры совпадают и равны 1000x1000.
`A.shape == B.shape == Shape(1000, 1000)`

```bash

A+B calculation (CPU): 36 ms
A dot B calculation (CPU): 21541 ms
a*B calculation (CPU): 28 ms

A+B calculation (GPU): 20 ms
A dot B calculation (GPU): 43 ms
No implemented. // умножения на скаляр для gpu нока нет
terminate called without an active exception
Aborted (core dumped)

```

Из вывода можно увидеть, что вычисление матричного учножения на `CPU` (`dot`) весьма долгое и его использование весьма оправдано. Издержки копирования на видеокарту не значительны, а прирост существенный.

`a*B` для `GPU` пока нет, вывел пример вывода.

Коректность вычислений

```bash
A
 1.000000   1.000000   1.000000   1.000000  
 1.000000   1.000000   1.000000   1.000000  
 1.000000   1.000000   1.000000   1.000000  
 1.000000   1.000000   1.000000   1.000000  

B
 2.000000   2.000000   2.000000   2.000000  
 2.000000   2.000000   2.000000   2.000000  
 2.000000   2.000000   2.000000   2.000000  
 2.000000   2.000000   2.000000   2.000000  


A+B
 3.000000   3.000000   3.000000   3.000000  
 3.000000   3.000000   3.000000   3.000000  
 3.000000   3.000000   3.000000   3.000000  
 3.000000   3.000000   3.000000   3.000000  

A dot B
 8.000000   8.000000   8.000000   8.000000  
 8.000000   8.000000   8.000000   8.000000  
 8.000000   8.000000   8.000000   8.000000  
 8.000000   8.000000   8.000000   8.000000  

5*B
 10.000000   10.000000   10.000000   10.000000  
 10.000000   10.000000   10.000000   10.000000  
 10.000000   10.000000   10.000000   10.000000  
 10.000000   10.000000   10.000000   10.000000  


A+B calculation (GPU)
 3.000000   3.000000   3.000000   3.000000  
 3.000000   3.000000   3.000000   3.000000  
 3.000000   3.000000   3.000000   3.000000  
 3.000000   3.000000   3.000000   3.000000  

A dot B calculation (GPU)
 8.000000   8.000000   8.000000   8.000000  
 8.000000   8.000000   8.000000   8.000000  
 8.000000   8.000000   8.000000   8.000000  
 8.000000   8.000000   8.000000   8.000000  

```

Так же есть код для ReLu (`ReLUActivation`), однако она не доконца протестирована и не работает как надо. Поэтому пока в отчёте нет.

### Планы

1. Дописать ReLu и добваить Сигмоиду
2. Добавить Линейный слой и протестировать градиенты.
3. Добавить загрузку простых данных.
