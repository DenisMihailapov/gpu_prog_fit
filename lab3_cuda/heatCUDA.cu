

#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

 //////////////////////////////////////////////////////////////////

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
		printf("CUDA Runtime Error at: %s:%d\n", file, line);
		printf("%s %s\n", cudaGetErrorString(err), func);
		exit(err);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
       	printf("CUDA Runtime Error at: %s:%d\n", file, line);
		printf("%s\n", cudaGetErrorString(err));
    }
}

 //////////////////////////////////////////////////////////////////


// nvcc -o heatCUDA heatCUDA.cu  && ./heatCUDA 9e-6 128 1e5


#define REAL float

int __host__ __device__ ind2D(const size_t i, const size_t j, const size_t width)
{
    return i*width + j;
}

__global__ void step_estimate(REAL*  T, REAL* new_T, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (0 < i < N && 0 < j < N)

		new_T[ind2D(i, j, N)] = 0.25*(
            T[ind2D(i - 1, j, N)] + T[ind2D(i, j + 1, N)] +
            T[ind2D(i, j - 1, N)] + T[ind2D(i + 1, j, N)]
            );

}

__global__ void square_sum(REAL*  T, REAL *v_norm, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if ( i < N && j < N)
	{   
		size_t ind = ind2D(i, j, N);
		v_norm[0] += T[ind]*T[ind];
	}	

}

__global__ void diff(REAL*  T, REAL*  nT, REAL *v_norm, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    REAL dif;

	if ( i < N && j < N)
	{
		size_t ind = ind2D(i, j, N);
        dif = T[ind] - nT[ind];
		v_norm[0] += dif*dif;
	}

}

__global__ void copy(REAL*  T, REAL*  nT, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if ( i < N && j < N)
	{   
		size_t ind = ind2D(i, j, N);
        T[ind] = nT[ind];
	}
}


void init_border(REAL *T, uint N, REAL left_top, REAL right_top, REAL left_bottom, REAL right_bottom){
    
    for (int j = 1; j < N - 1; j++) 
      for (int i = 1; i < N - 1; i++) 
        T[ind2D(i, j, N)] = 0.0;

    for(uint i = 0; i < N; i++){
        T[ind2D(i, 0, N)] = left_top + i*(right_top - left_top) / (N - 1);
        T[ind2D(i, N - 1, N)] = left_bottom + i*(right_bottom - left_bottom) / (N - 1);

        T[ind2D(0, i, N)] = left_top + i*(left_bottom - left_top) / (N - 1);
        T[ind2D(N - 1, i, N)] = right_top + i*(right_bottom - right_top) / (N - 1);
    }
}

REAL norm(REAL *T, uint N){
    
    REAL n = 0;
    for (int ind = 0; ind < N*N ; ind++)
        n += std::pow(T[ind], 2);

    return std::sqrt(n);
}

void print2D(REAL *array, size_t row, size_t col){
	for (size_t j = 0; j < col; j++) {
        for (size_t i = 0; i < row; i++) {
            printf("%.4f ", array[ind2D(i, j, row)]);
        }
        printf("\n");
    }
}

void swap(REAL* &a, REAL* &b){
  REAL *temp = a;
  a = b;
  b = temp;
}

int main(int argc, char *argv[]){

    //Setup
	REAL tol = atof(argv[1]);
    uint N = atof(argv[2]), iter = 0;
    uint max_iter = atof(argv[3]);
	size_t FULL_MEM_SIZE = N * N * sizeof(REAL);

    printf("\nInput params(tol = %2.2e, N = %d, max_iter = %d)\n", tol, N, max_iter);
    
    
    //Alloc CPU memory and init
    REAL *T = (REAL *)std::malloc(FULL_MEM_SIZE);
	REAL *nT = (REAL *)std::malloc(FULL_MEM_SIZE);
    REAL norm_dT, normT, diffT;
    
	init_border(T, N, 1, 9, 9, 18);
    print2D(T, N, N);


    //Alloc GPU memory and init
	REAL *dev_T, *dev_nT, *dev_normT, *dev_normT_2, *dev_diffT;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_normT, sizeof(REAL)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_normT_2, sizeof(REAL)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_diffT, sizeof(REAL)));

	CHECK_CUDA_ERROR(cudaMalloc(&dev_T, FULL_MEM_SIZE));
	CHECK_CUDA_ERROR(cudaMalloc(&dev_nT, FULL_MEM_SIZE));
	 
     //Set init data on GPU
	CHECK_CUDA_ERROR(cudaMemcpy(dev_T, T, FULL_MEM_SIZE, cudaMemcpyHostToDevice));

    
    //CUDA grids and blocks
	dim3 BS(N/4., N/4.); 
	dim3 GS(ceil(N/(float)BS.x), ceil(N/(float)BS.x));
	printf("threads: %d, block size: %d\n\n", BS.x, GS.x);

    do 
    {
        printf("iter: %d\n", iter);

        //
        // Compute a new estimate.
        step_estimate<<<GS, BS>>>(dev_T, dev_nT, N);

        
        //Print new estimate (for debug)
        CHECK_CUDA_ERROR(cudaMemcpy(nT,  dev_nT, FULL_MEM_SIZE, cudaMemcpyDeviceToHost));
        print2D(nT, N, N);
        printf("\n");
        normT = norm(nT, N);
        printf("norm nT: %f \n", normT);
        
        // Calculate norm of estimate on GPU
        square_sum<<<GS, BS>>>(dev_nT, dev_normT_2, N);
        CHECK_CUDA_ERROR(cudaMemcpy(&norm_dT,  &dev_normT_2[0], sizeof(REAL), cudaMemcpyDeviceToHost));
        norm_dT = std::sqrt(norm_dT);
        printf("norm dev_nT: %f \n", norm_dT);

        if (normT != norm_dT)
          printf("err normT != norm_dT\n");
        

        //
        // Check for convergence.
        diff<<<GS, BS>>>(dev_T, dev_nT, dev_diffT, N);  
        CHECK_CUDA_ERROR(cudaMemcpy(&diffT, &dev_diffT[0], sizeof(REAL), cudaMemcpyDeviceToHost));
        printf("diff: %f \n", diffT);

        //
        // Save the current estimate.
        copy<<<GS, BS>>>(dev_T, dev_nT, N);
                
        //
        // Do iteration.
        iter++;
        printf("\n");

  }while(diffT >= tol &&  iter <= max_iter);

 
	 //4. Copy the array ‘c’ from GPU to CPU
	CHECK_CUDA_ERROR(cudaMemcpy(nT,  dev_nT, FULL_MEM_SIZE, cudaMemcpyDeviceToHost));
 
	 //Display the result on the CPU
	print2D(nT, N, N);

 
	 //5. Release the memory allocated on the GPU. Purpose to avoid memory leaks
	cudaFree(dev_T); std::free(T);
	cudaFree(dev_nT); std::free(nT);
	CHECK_LAST_CUDA_ERROR();
 
	return 0;
}
