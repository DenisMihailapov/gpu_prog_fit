#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <stdio.h>

#define BLOCKSIZE  1024
#define BLOCKSNUM  32

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


// nvcc -o heatCUDA heatCUDA.cu  && ./heatCUDA 9e-6 1e5


int __host__ __device__ ind2D(const size_t i, const size_t j, const size_t width)
{
    return i*width + j;
}

__global__ void step_estimate(REAL*  T, REAL* new_T, size_t N){

	//Calculate the data at the index
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (0 < i < N - 1 && 0 < j < N - 1)
		new_T[ind2D(i, j, N)] = 0.25*(
            T[ind2D(i - 1, j, N)] + T[ind2D(i, j + 1, N)] +
            T[ind2D(i, j - 1, N)] + T[ind2D(i + 1, j, N)]
            );

}

__global__ void square_sum(REAL*  T, REAL *reduced){

    //Calculate the data at the index
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    
    typedef cub::BlockReduce<REAL, BLOCKSIZE> BlockReduceT; 
    
    // --- Allocate temporary storage in shared memory 
    __shared__ typename BlockReduceT::TempStorage temp_storage; 

    size_t ind = ind2D(i, j, BLOCKSIZE);   
    REAL result = BlockReduceT(temp_storage).Sum(T[ind]*T[ind]);

    // --- Update block reduction value
    if(threadIdx.x == 0) reduced[blockIdx.x] = result;

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


REAL sum_cpu(REAL *T, uint N){
    
    REAL s = 0;
    for (int ind = 0; ind < N ; ind++)
        s += T[ind];

    return s;
}

REAL norm_cpu(REAL *T, uint N){
    
    REAL n = 0;
    for (int ind = 0; ind < N*N ; ind++)
        n += std::pow(T[ind], 2);

    return std::sqrt(n);
}

void print2D(REAL *array, size_t row, size_t col){
	for (size_t j = 0; j < col; j+=(col/BLOCKSNUM)/2) {
        for (size_t i = 0; i < row; i+=(row/BLOCKSNUM)/2) {
            printf("%.4f ", array[ind2D(i, j, row)]);
       }
        printf("\n");
    }
}


int main(int argc, char *argv[]){

    //Setup
	REAL tol = atof(argv[1]);
    const uint N = BLOCKSIZE;
    uint max_iter = atof(argv[2]), iter = 0;
	size_t FULL_MEM_SIZE = N * N * sizeof(REAL);
    size_t REDUCE_MEM_SIZE = N * sizeof(REAL);

    printf("\nInput params(tol = %2.2e, N = %d, max_iter = %d)\n", tol, N, max_iter);
    
    
    //Alloc CPU memory and init
    REAL *T = (REAL *)std::malloc(FULL_MEM_SIZE);
	REAL *new_T = (REAL *)std::malloc(FULL_MEM_SIZE);
    REAL *subT = (REAL *)std::malloc(FULL_MEM_SIZE);
    REAL *norm_nT_gpu = (REAL *)std::malloc(REDUCE_MEM_SIZE);
    REAL *step_diff = (REAL *)std::malloc(REDUCE_MEM_SIZE);
    
	init_border(T, N, 10, 20, 30, 20);
    //print2D(T, N, N);


    //Alloc GPU memory and init
	REAL *dev_T, *dev_nT, *norm_dev_nT, *dev_subT, *dev_step_diff;
    CHECK_CUDA_ERROR(cudaMalloc(&norm_dev_nT, REDUCE_MEM_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_step_diff, REDUCE_MEM_SIZE));

	CHECK_CUDA_ERROR(cudaMalloc(&dev_T, FULL_MEM_SIZE));
	CHECK_CUDA_ERROR(cudaMalloc(&dev_nT, FULL_MEM_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_subT, FULL_MEM_SIZE));
	 
     //Set init data on GPU
	CHECK_CUDA_ERROR(cudaMemcpy(dev_T, T, FULL_MEM_SIZE, cudaMemcpyHostToDevice));

    
    //CUDA grids and blocks
    dim3 GS(BLOCKSNUM, BLOCKSNUM); // count of blocks
	dim3 BS(ceil(N/(float)BLOCKSNUM), ceil(N/(float)BLOCKSNUM)); // block size
	printf("count of blocks: %d, block size: %d\n\n", GS.x, BS.x);

    do 
    {
        printf("iter: %d\n", iter);

        //
        // Compute a new estimate.
        step_estimate<<<GS, BS>>>(dev_T, dev_nT, N);
        
        // Calculate norm of estimate on GPU
        square_sum<<<GS, BS>>>(dev_nT, norm_dev_nT);
        CHECK_CUDA_ERROR(cudaMemcpy(norm_nT_gpu,  norm_dev_nT, REDUCE_MEM_SIZE, cudaMemcpyDeviceToHost));
        norm_nT_gpu[0] = std::sqrt(sum_cpu(norm_nT_gpu, N));
        printf("||new_T|| (gpu): %f \n", norm_nT_gpu[0]);

        ///////////////////////////////////////////////////////////////////////  

        //
        // Check for convergence.
        sub<<<GS, BS>>>(dev_T, dev_nT, dev_subT, N);
        CHECK_CUDA_ERROR(cudaMemcpy(subT,  dev_subT, FULL_MEM_SIZE, cudaMemcpyDeviceToHost));
        // printf("\n\nsubT = T - new_T\n");
        // print2D(subT, N, N);
        // printf("\n");

        square_sum<<<GS, BS>>>(dev_subT, dev_step_diff);
        CHECK_CUDA_ERROR(cudaMemcpy(step_diff,  dev_step_diff, REDUCE_MEM_SIZE, cudaMemcpyDeviceToHost));
        step_diff[0] = std::sqrt(sum_cpu(step_diff, N));
        printf("diff (||subT||): %f \n\n", step_diff[0]);

        //
        // Save the current estimate.
        copy<<<GS, BS>>>(dev_T, dev_nT, N);
                
        //
        // Do iteration.
        iter++;
        printf("\n");

  }while(step_diff[0] >= tol &&  iter <= max_iter);

 
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
