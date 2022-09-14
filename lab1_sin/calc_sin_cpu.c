#include <stdio.h>
#include <math.h>

 
#define PI 3.14159265      // число ПИ
#define uint unsigned int 


// pgcc -o calc_sin_cpu calc_sin_cpu.c -lm && PGI_ACC_TIME=1 ./calc_sin_cpu
// PGI_ACC_NOTIFY = 63 (<b1b2b3b4>)
// PGI_ACC_DEBUG = 1

// nsys profile -t openacc, cuda -- графическая утилита для профилорования (указывать надо програмку типа a.out)


int main(int argc, char* argv[])
{
  const uint N = pow(10, 6);

  double S[N], sum_sin = 0.0;


  for(int i = 0; i < N; i++)
    S[i] = sin(2*PI*i/N);
    
  for(int i = 0; i < N; i++) sum_sin += S[i];

  printf("Сумма: %2.5f\n", sum_sin);

  return 0;
}
