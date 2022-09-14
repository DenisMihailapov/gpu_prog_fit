#include <stdio.h>
#include <math.h>

 
#define PI 3.14159265      // число ПИ
#define uint unsigned int 


// pgcc -acc -Minfo=accel -o calc_sin calc_sin.c -lm && PGI_ACC_TIME=1 ./calc_sin
// PGI_ACC_NOTIFY = 63 (<b1b2b3b4>)
// PGI_ACC_DEBUG = 1

// nsys profile -t openacc, cuda -- графическая утилита для профилорования (указывать надо програмку типа a.out)



int main()
{
  const uint N = pow(10, 6);
  double S[N], sum_sin = 0.0;

  #pragma acc data create(S) copy(sum_sin) // create(S) создать массив на gpu и не копировать на процессор
   {
   
    #pragma acc kernels 
    for(int i = 0; i < N; i++)
      S[i] = sin(2*PI*i/N);
    
    sum_sin = 0.0;
    #pragma acc kernels loop reduction(+:sum_sin)
    for(int i = 0; i < N; i++) sum_sin += S[i];
   }

  printf("Сумма: %2.3f\n", sum_sin);

  return 0;
}