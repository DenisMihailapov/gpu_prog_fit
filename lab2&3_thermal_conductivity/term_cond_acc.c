#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// pgcc -acc -Minfo=accel -o term_cond term_cond_acc.c -lm && PGI_ACC_TIME=1 ./term_cond 9e-6 128 1e5

#define PI 3.141592
#define REAL double
#define SHIFT 1
#define freq_print 10

REAL mse_norm(int nx, int ny, REAL **a) {

  REAL v = 0.0;
  #pragma acc kernels loop reduction(+:v)
  for (int j = 0 + SHIFT; j < ny - SHIFT; j++)
    for (int i = 0 + SHIFT; i < nx - SHIFT; i++)
      v += a[i][j] * a[i][j];


  return sqrt(v / (REAL)((nx - 2*SHIFT) * (ny - 2*SHIFT)));
}

REAL diff(int nx, int ny, REAL **a, REAL **b) {

  REAL v = 0.0, t;
  #pragma acc kernels loop reduction(+:v)
  for (int j = 0 + SHIFT; j < ny - SHIFT; j++)
    for (int i = 0 + SHIFT; i < nx - SHIFT; i++) {
      t = a[i][j] - b[i][j];
      v += t * t;
    }

  return sqrt(v / (REAL)(nx * ny));
}

//****************************************************************************80

void print2D(REAL **array, size_t row, size_t col) {
  for (size_t j = 0 + SHIFT; j < col - SHIFT; j++) {
    for (size_t i = 0 + SHIFT; i < row - SHIFT; i++)
      printf("%.3f\t", array[i][j]);

    printf("\n");
  }
}

REAL **allocate2DArray(int row, int col)
{
  REAL ** ptr = (REAL **) malloc(sizeof(REAL *)*row);
  for(int i = 0; i < row; i++)
        ptr[i] = (REAL *) malloc(sizeof(REAL)*col);

  return ptr;
}

void free2DArray(REAL **ptr, int row, int col)
{
  for(int i = 0; i < row; i++)
    free(ptr[i]);
  free(ptr);
}

void init_full(uint N, REAL **T, REAL left_top, REAL right_top, REAL left_bottom, REAL right_bottom){
  
  int n = N - 1 - 2*SHIFT;
  for(uint i = 0 + SHIFT; i < N - SHIFT; i++) {
    T[i][0] = left_top + i*(right_top - left_top) / n;
    T[i][N - 1] = left_bottom + i*(right_bottom - left_bottom) / n;

    for(uint j = 0 + SHIFT; j < N - SHIFT; j++)
      T[i][j] = T[i][0] + j*(T[i][N - 1] - T[i][0]) / n;
  }
}

void init_border(uint N, REAL **T, REAL left_top, REAL right_top, REAL left_bottom, REAL right_bottom){
  
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      T[i][j] = 0.0;
  
  int true_N = N - 2;
  for (uint i = 0; i < true_N; i++) { 
    T[i + SHIFT][SHIFT] = left_top + i * (right_top - left_top) / (true_N - SHIFT);
    T[i + SHIFT][true_N + 1 - SHIFT] = left_bottom + i * (right_bottom - left_bottom) / (true_N - SHIFT);

    T[SHIFT][i + SHIFT] = left_top + i * (left_bottom - left_top) / (true_N - SHIFT);
    T[true_N + 1 - SHIFT][i + SHIFT] = right_top + i * (right_bottom - right_top) / (true_N - SHIFT);
  }
}


int main(int argc, char *argv[]) {


  REAL tol = atof(argv[1]);
  const uint N = atof(argv[2]) + 2*SHIFT;
  const uint max_iter = atof(argv[3]);

  printf("\nInput params(tol = %2.2e, N = %d, max_iter = %d)\n", tol, N - 2*SHIFT, max_iter);


  REAL **T = allocate2DArray(N, N);
  REAL **new_T = allocate2DArray(N, N);

  init_border(N, T, 10, 20, 20, 30);


  int iter = 0; 
  REAL error = 10000., normT = 0.;

  printf("START calculation...\n");
  //#pragma acc data copy(T[:N][:N]) create(new_T[:N][:N]) create(error)
  do {
    normT = mse_norm(N, N, T);
    if(iter % freq_print == 0){
      print2D(T, N, N);
      printf("\n");
    }
    //
    // Compute a new estimate.
    //#pragma acc parallel loop independent
    for (int j = 0 + SHIFT; j < N - SHIFT; j++)
      for (int i = 0 + SHIFT; i < N - SHIFT; i++)
          new_T[i][j] = 0.25 * (
            T[i - 1][j] + T[i][j + 1] + 
            T[i][j - 1] + T[i + 1][j]
            );


    //
    // Check for convergence.
    error = diff(N, N, new_T, T);  

    //
    // Save the current estimate.
    //#pragma acc parallel loop independent
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        T[i][j] = new_T[i][j];
    
    if(iter % freq_print == 0){
        printf("iter: %d\n", iter);
        printf("||new_T||: %f \n", mse_norm(N, N, new_T));

        printf("diff (||subT||): %f \n\n", error);
        printf("\n");
    }
    
    //
    // Do iteration.
    iter++;

  }while(error >= tol && iter <= max_iter);
  
  
  printf("\nEND calculation\n");
  printf("iter: %d\n", iter);
  printf("norm T: %f \n", normT);
  printf("udiff: %f (toi %f)\n\n", error, tol);
  

  //
  // Terminate.
  free2DArray(T, N, N);  
  free2DArray(new_T, N, N);  
  
  return 0;
}
