#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// pgcc -acc -Minfo=accel -o term_cond term_cond.c -lm && PGI_ACC_TIME=1 ./term_cond 9e-6 128 1e5

#define PI 3.141592
#define REAL double

REAL mse_norm(int nx, int ny, REAL **a) {

  REAL v = 0.0;

  for (int j = 0; j < ny; j++) 
    for (int i = 0; i < nx; i++)
      v += a[i][j] * a[i][j];


  return sqrt(v / (REAL)(nx * ny));
}

REAL diff(int nx, int ny, REAL **a, REAL **b) {

  REAL v = 0.0, t;

  for (int j = 0; j < ny; j++) 
    for (int i = 0; i < nx; i++){
      t = a[i][j] - b[i][j];  v += t * t;
    }

  return sqrt(v / (REAL)(nx * ny));
}

//****************************************************************************80

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
    
    for(uint i = 0; i < N; i++){
        T[i][0] = left_top + i*(right_top - left_top) / (N - 1);
        T[i][N - 1] = left_bottom + i*(right_bottom - left_bottom) / (N - 1);

        for(uint j = 0; j < N; j++)
            T[i][j] = T[i][0] + j*(T[i][N - 1] - T[i][0]) / (N - 1);
    }
}

void init_border(uint N, REAL **T, REAL left_top, REAL right_top, REAL left_bottom, REAL right_bottom){
    
    for (int j = 1; j < N - 1; j++) 
      for (int i = 1; i < N - 1; i++) 
        T[i][j] = 0.0;

    for(uint i = 0; i < N; i++){
        T[i][0] = left_top + i*(right_top - left_top) / (N - 1);
        T[i][N - 1] = left_bottom + i*(right_bottom - left_bottom) / (N - 1);

        T[0][i] = left_bottom + i*(left_bottom - left_top) / (N - 1);
        T[N - 1][i] = right_top + i*(right_bottom - right_top) / (N - 1);
    }
}


int main(int argc, char *argv[]) {

  REAL tol = atof(argv[1]);
  const uint N = atof(argv[2]);
  const uint max_iter = atof(argv[3]);

  printf("tol: %f, N: %d, max_iter: %d\n", tol, N, max_iter);

  REAL **T = allocate2DArray(N, N);
  REAL **new_T = allocate2DArray(N, N);

  init_border(N, T, 10, 20, 20, 30);

  REAL error = 10000.;
  int itold = 0; 
  
  #pragma acc data create(T[:N][:N]) create(new_T[:N][:N]) create(error)
  do {

    //
    // Compute a new estimate.
    #pragma acc data present(T[:N][:N], new_T[:N][:N])
    {
      #pragma acc loop independent
      for (int j = 1; j < N - 1; j++){
        #pragma acc loop independent
        for (int i = 1; i < N - 1; i++){
            new_T[i][j] = 0.25*(
              T[i - 1][j] + T[i][j + 1] +
              T[i][j - 1] + T[i + 1][j]
              );
        }
      }
    }
    //
    // Check for convergence.
    error = diff(N, N, new_T, T);    

    //
    // Save the current estimate.
    #pragma acc loop independent
    for (int j = 0; j < N; j++){
      #pragma acc loop independent
      for (int i = 0; i < N; i++)
        T[i][j] = new_T[i][j];
    }
    //
    // Do iteration.
    itold++;

  }while(error >= tol && itold <= max_iter);

  error = diff(N, N, new_T, T);
  
  printf("udiff: %f (toi %f)\n", error, tol);
  printf("itold: %d\n", itold);


  //
  // Terminate.
  //
  free2DArray(T, N, N);  
  free2DArray(new_T, N, N);  
  

  return 0;
}
