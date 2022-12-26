#include <cstddef>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <stdio.h>

#define BLOCKSNUM  2
#define freq_print 10

#define REAL double
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


// nvcc -o heatCUDA heatCUDA.cu  && ./heatCUDA 9e-6 256 1e5


size_t __host__ __device__ ind2D(const size_t i, const size_t j, const size_t width){
    
    return i*width + j;
}

REAL __host__ __device__ get_valid_T(REAL*  T, long i, const long j, const long width){
    
    if(i < 0 || width <= i || j < 0 || width <= j) return 0.; 
    return T[ind2D(i, j, width)];

}

__global__ void step_estimate(REAL*  T, REAL* new_T, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    
    new_T[ind2D(i, j, N)] = 0.25*(
        get_valid_T(T, i - 1, j, N) + get_valid_T(T, i, j + 1, N) +
        get_valid_T(T, i, j - 1, N) + get_valid_T(T, i + 1, j, N)
        );

}

__global__ void pow_2(REAL*  T, REAL* squareT, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	size_t ind = ind2D(i, j, N);
    squareT[ind] = T[ind]*T[ind];

}

__global__ void sub(REAL*  T, REAL*  nT, REAL* subT, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (0 < i < N && 0 < j < N)
	{
		size_t ind = ind2D(i, j, N);
        subT[ind] = T[ind] - nT[ind];
	}

}

__global__ void copy(REAL*  T, REAL*  nT, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	size_t ind = ind2D(i, j, N);
    T[ind] = nT[ind];

}

REAL gpu_sum(REAL*  T, size_t num_items){

     // Allocate device output array
    REAL *d_out = NULL;
    cudaMalloc((void**)&d_out, sizeof(REAL));

    void   *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, T, d_out, num_items));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, T, d_out, num_items));
    
    REAL h_sum;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_sum, d_out, sizeof(REAL), cudaMemcpyDeviceToHost));

    cudaFree(d_out); cudaFree(d_temp_storage);

    return h_sum;
}


void init_border(REAL *T, size_t N, REAL left_top, REAL right_top, REAL left_bottom, REAL right_bottom){
    
    for (int j = 1; j < N - 1; j++) 
      for (int i = 1; i < N - 1; i++) 
        T[ind2D(i, j, N)] = 0.0;

    for(size_t i = 0; i < N; i++){
        T[ind2D(i, 0, N)] = left_top + i*(right_top - left_top) / (N - 1);
        T[ind2D(i, N - 1, N)] = left_bottom + i*(right_bottom - left_bottom) / (N - 1);

        T[ind2D(0, i, N)] = left_top + i*(left_bottom - left_top) / (N - 1);
        T[ind2D(N - 1, i, N)] = right_top + i*(right_bottom - right_top) / (N - 1);
    }
}


REAL sum_cpu(REAL *T, size_t N){
    
    REAL s = 0;
    for (int ind = 0; ind < N ; ind++)
        s += T[ind];

    return s;
}

REAL norm_cpu(REAL *T, size_t N){
    
    REAL n = 0;
    for (int ind = 0; ind < N*N ; ind++)
        n += std::pow(T[ind], 2);

    return std::sqrt(n);
}

void print2D(REAL *array, size_t row, size_t col){
	for (size_t j = 0; j < col; j+=1) {
        for (size_t i = 0; i < row; i+=1) {
            printf("%.3f\t", array[ind2D(i, j, row)]);
       }
        printf("\n");
    }
}


int main(int argc, char *argv[]){

    //Setup
	REAL tol = atof(argv[1]);
    const size_t N = atof(argv[2]);
    size_t max_iter = atof(argv[3]), iter = 0;
	size_t FULL_MEM_SIZE = N * N * sizeof(REAL);

    printf("\nInput params(tol = %2.2e, N = %zu, max_iter = %zu)\n", tol, N, max_iter);
    
    
    //Alloc CPU memory and init
    REAL *T = (REAL *)std::malloc(FULL_MEM_SIZE);
	REAL *new_T = (REAL *)std::malloc(FULL_MEM_SIZE);
    REAL *subT = (REAL *)std::malloc(FULL_MEM_SIZE);

    REAL norm_nT_gpu, step_diff;
    
	init_border(T, N, 10, 20, 20, 30);
    //print2D(T, N, N);


    //Alloc GPU memory and init
	REAL *dev_T, *dev_nT, *dev_subT, *squareT;

	CHECK_CUDA_ERROR(cudaMalloc(&dev_T, FULL_MEM_SIZE));
	CHECK_CUDA_ERROR(cudaMalloc(&dev_nT, FULL_MEM_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_subT, FULL_MEM_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&squareT, FULL_MEM_SIZE));
	 
     //Set init data on GPU
	CHECK_CUDA_ERROR(cudaMemcpy(dev_T, T, FULL_MEM_SIZE, cudaMemcpyHostToDevice));

    
    //CUDA grids and blocks
    dim3 GS(BLOCKSNUM, BLOCKSNUM); // count of blocks
	dim3 BS(ceil(N/(float)BLOCKSNUM), ceil(N/(float)BLOCKSNUM)); // block size
	printf("count of blocks: %d, block size: %d\n\n", GS.x, BS.x);

    do 
    {
        if(iter % freq_print == 0){
            CHECK_CUDA_ERROR(cudaMemcpy(new_T,  dev_T, FULL_MEM_SIZE, cudaMemcpyDeviceToHost));
            print2D(new_T, N, N);
            printf("\n");
        }
        //
        // Compute a new estimate.
        step_estimate<<<GS, BS>>>(dev_T, dev_nT, N);

        // Calculate norm of estimate on GPU
        pow_2<<<GS, BS>>>(dev_nT, squareT, N);
        norm_nT_gpu = std::sqrt(gpu_sum(squareT, N));

        ///////////////////////////////////////////////////////////////////////  

        //
        // Check for convergence.
        sub<<<GS, BS>>>(dev_T, dev_nT, dev_subT, N);
        pow_2<<<GS, BS>>>(dev_subT, squareT, N);
        step_diff = std::sqrt(gpu_sum(squareT, N * N));

        //
        // Save the current estimate.
        copy<<<GS, BS>>>(dev_T, dev_nT, N);
        //
        // print information
        if(iter % freq_print == 0){
            printf("iter: %zu\n", iter);
            printf("||new_T|| (gpu): %f \n", norm_nT_gpu);

            // printf("\n\nsubT = T - new_T\n");
            // CHECK_CUDA_ERROR(cudaMemcpy(new_T,  dev_nT, FULL_MEM_SIZE, cudaMemcpyDeviceToHost));
            // print2D(new_T, N, N);
            // printf("%d\n", N);

            printf("diff (||subT||): %f \n\n", step_diff);
            printf("\n");
        }


        //
        // Do iteration.
        iter++;
        //if(iter%(10*freq_print) == 0) break;

  }while(step_diff >= tol &&  iter <= max_iter);

 
	 //4. Copy the array from GPU to CPU
	CHECK_CUDA_ERROR(cudaMemcpy(new_T,  dev_nT, FULL_MEM_SIZE, cudaMemcpyDeviceToHost));
 
	//Display the result on the CPU
	//print2D(new_T, N, N);

 
	 //5. Release the memory allocated on the GPU. Purpose to avoid memory leaks
	cudaFree(dev_T); std::free(T);
	cudaFree(dev_nT); std::free(new_T);
	CHECK_LAST_CUDA_ERROR();
 
	return 0;
}
