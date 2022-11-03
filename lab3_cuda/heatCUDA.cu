#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define REAL float

int __host__ __device__ getIndex(const int i, const int j, const int width)
{
    return i*width + j;
}

__global__ void substr_kernel(const REAL* Un, const REAL* Unp1, int N, REAL *diff_U)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((0 < i && i < N - 1) && (0 < j && j < N - 1))
    {
        int k = getIndex(i, j, N);
        diff_U[k] = pow(Un[k] - Unp1[k], 2.0); 
    }

}
__global__ void evolve_kernel(const REAL* Un, REAL* Unp1, 
                              const int nx, const int ny, 
                              const REAL dx2, const REAL dy2, const REAL aTimesDt)
{

    __shared__ REAL s_Un[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    int s_i = threadIdx.x + 1;
    int s_j = threadIdx.y + 1;
    int s_ny = BLOCK_SIZE_Y + 2;

    // Load data into shared memory
    // Central square
    s_Un[getIndex(s_i, s_j, s_ny)] = Un[getIndex(i, j, ny)];
    // Top border
    if (s_i == 1 && s_j !=1 && i != 0)
        s_Un[getIndex(0, s_j, s_ny)] = Un[getIndex(blockIdx.x*blockDim.x - 1, j, ny)];

    // Bottom border
    if (s_i == BLOCK_SIZE_X && s_j != BLOCK_SIZE_Y && i != nx - 1)
        s_Un[getIndex(BLOCK_SIZE_X + 1, s_j, s_ny)] = Un[getIndex((blockIdx.x + 1)*blockDim.x, j, ny)];

    // Left border
    if (s_i != 1 && s_j == 1 && j != 0)
        s_Un[getIndex(s_i, 0, s_ny)] = Un[getIndex(i, blockIdx.y*blockDim.y - 1, ny)];
    
    // Right border
    if (s_i != BLOCK_SIZE_X && s_j == BLOCK_SIZE_Y && j != ny - 1)
        s_Un[getIndex(s_i, BLOCK_SIZE_Y + 1, s_ny)] = Un[getIndex(i, (blockIdx.y + 1)*blockDim.y, ny)];
    
    // Make sure all the data is loaded before computing
    __syncthreads();

    if (i > 0 && i < nx - 1)
        if (j > 0 && j < ny - 1)
        {
            REAL uij = s_Un[getIndex(s_i, s_j, s_ny)];
            REAL uim1j = s_Un[getIndex(s_i-1, s_j, s_ny)];
            REAL uijm1 = s_Un[getIndex(s_i, s_j-1, s_ny)];
            REAL uip1j = s_Un[getIndex(s_i+1, s_j, s_ny)];
            REAL uijp1 = s_Un[getIndex(s_i, s_j+1, s_ny)];

            // Explicit scheme
            Unp1[getIndex(i, j, ny)] = uij + aTimesDt * ( (uim1j - 2.0*uij + uip1j)/dx2 + (uijm1 - 2.0*uij + uijp1)/dy2 );
        }
}

int main(int argc, char *argv[])
{
    printf("Calc");
    REAL tol = atof(argv[1]);
    const uint N = atof(argv[2]);
    const uint max_iter = atof(argv[3]);

    int GRID_SIZE = N*N;

    const REAL a = 0.5;     // Diffusion constant
    const REAL dx = 0.01;   // Horizontal grid spacing 
    const REAL dy = 0.01;   // Vertical grid spacing

    const REAL dx2 = dx*dx;
    const REAL dy2 = dy*dy;

    const REAL dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step
    const int outputEvery = max_iter/10;                 // How frequently to write output image


    // Allocate two sets of data for current and next timesteps
    REAL* h_Un   = (REAL*)calloc(GRID_SIZE, sizeof(REAL));
    REAL* h_diff_U   = (REAL*)calloc(GRID_SIZE, sizeof(REAL));

    // Initializing the data with a pattern of disk of radius of 1/6 of the width
    REAL radius2 = (N/6.) * (N/6.);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            // Distance of point i, j from the origin
            if ((i - N/2.) * (i - N/2.) + (j - N/2.)*(j - N/2.) < radius2)
                h_Un[getIndex(i, j, N)] = 65.;
            else
                h_Un[getIndex(i, j, N)] = 5.;

    REAL* d_Un;
    REAL* d_Unp1;
    REAL* diff_U;
    REAL error;
    
    cudaMalloc((void**)&d_Un, GRID_SIZE*sizeof(REAL));
    cudaMalloc((void**)&d_Unp1, GRID_SIZE*sizeof(REAL));
    cudaMalloc((void**)&diff_U, GRID_SIZE*sizeof(REAL));

    cudaMemcpy(d_Un, h_Un, GRID_SIZE*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Unp1, h_Un, GRID_SIZE*sizeof(REAL), cudaMemcpyHostToDevice);

    dim3 numBlocks(N/BLOCK_SIZE_X + 1, N/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Timing
    clock_t start = clock();

    // Main loop
    for (int n = 0; n <= max_iter; n++)
    {
        printf("Iter %d", n);
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_Un, d_Unp1, N, N, dx2, dy2, a*dt);

        // Write the output if neededs
        if (n % outputEvery == 0)
        {
            cudaMemcpy(h_Un, d_Un, GRID_SIZE*sizeof(REAL), cudaMemcpyDeviceToHost);
            cudaError_t errorCode = cudaGetLastError();
            if (errorCode != cudaSuccess)
            {
                printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                exit(0);
            }


            //substr_kernel<<<numBlocks, threadsPerBlock>>>(d_Un, d_Unp1, N, diff_U);
            
            for (int i = 0; i < N; i++)
              for (int j = 0; j < N; j++)
                printf("Calc error %f\n", diff_U[getIndex(i, j, N)]);
            // char filename[64];
            // sprintf(filename, "heat_%04d.png", n);
            // save_png(h_Un, N, N, filename, 'c');
        }

        std::swap(d_Un, d_Unp1);
    }

    // Timing
    clock_t finish = clock();
    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(h_Un);
    cudaFree(d_Un);
    cudaFree(d_Unp1);
    
    return 0;
}


/*
 * write the given temperature u matrix to rgb values
 * and write the resulting image to file f
 */
void write_image(FILE * f, REAL *u, unsigned sizex, unsigned sizey) {
    /* RGB table */
    unsigned char r[1024], g[1024], b[1024];
    int i, j, k;

    REAL min, max;

    j = 1023;

    /* prepare RGB table */
    for (i = 0; i < 256; i++) {
        r[j] = 255;
        g[j] = i;
        b[j] = 0;
        j--;
    }
    for (i = 0; i < 256; i++) {
        r[j] = 255 - i;
        g[j] = 255;
        b[j] = 0;
        j--;
    }
    for (i = 0; i < 256; i++) {
        r[j] = 0;
        g[j] = 255;
        b[j] = i;
        j--;
    }
    for (i = 0; i < 256; i++) {
        r[j] = 0;
        g[j] = 255 - i;
        b[j] = 255;
        j--;
    }

    min = DBL_MAX;
    max = -DBL_MAX;

    /* find minimum and maximum */
    for (i = 0; i < sizey; i++) {
        for (j = 0; j < sizex; j++) {
            if (u[i * sizex + j] > max)
                max = u[i * sizex + j];
            if (u[i * sizex + j] < min)
                min = u[i * sizex + j];
        }
    }

    fprintf(f, "P3\n");
    fprintf(f, "%u %u\n", sizex, sizey);
    fprintf(f, "%u\n", 255);

    for (i = 0; i < sizey; i++) {
        for (j = 0; j < sizex; j++) {
            k = (int) (1024.0 * (u[i * sizex + j] - min) / (max - min));
            fprintf(f, "%d %d %d  ", r[k], g[k], b[k]);
        }
        fprintf(f, "\n");
    }
}