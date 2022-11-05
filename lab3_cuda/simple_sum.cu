
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


// nvcc -o simple_sum simple_sum.cu  && ./simple_sum 15


#define REAL float

int __host__ __device__ ind2D(const size_t i, const size_t j, const size_t width)
{
    return i + j*width;
}

__global__ void add2D(REAL* a, REAL* b, REAL* c, size_t N){

	//Calculate the data at the index
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N)
	{   
		size_t ind = ind2D(i, j, N);
		c[ind] = a[ind] + b[ind];
	}	

}

void init_full(size_t N, REAL *array, REAL left_top, REAL right_top, REAL left_bottom, REAL right_bottom){
    
    for(size_t i = 0; i < N; i++){
        array[ind2D(i, 0, N)] = left_top + i*(right_top - left_top) / N;
        array[ind2D(i, N - 1, N)] = left_bottom + i*(right_bottom - left_bottom) / N;

        for(size_t j = 0; j < N; j++)
            array[ind2D(i, j, N)] = 
			    array[ind2D(i, 0, N)] + 
				j*(array[ind2D(i, N - 1, N)] - array[ind2D(i, 0, N)]) / N;
    }
}

void print2D(REAL *array, size_t row, size_t col){
	for (size_t j = 0; j < col; j++) {
        for (size_t i = 0; i < row; i++) {
            printf("%.4f ", array[ind2D(i, j, row)]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]){

	size_t N = atof(argv[1]);
	size_t FULL_MEM_SIZE = N * N * sizeof(REAL);
    
	//0. Allocate and Initialize memory on the CPU
    REAL *a = (REAL *)std::malloc(FULL_MEM_SIZE);
	REAL *b = (REAL *)std::malloc(FULL_MEM_SIZE);
	REAL *c = (REAL *)std::malloc(FULL_MEM_SIZE);
    
	init_full(N, a, 1, 9, 9, 18);
	init_full(N, b, 0, 8, 8, 16);

	 //1. Allocate memory on the GPU
	REAL *dev_a, *dev_b, *dev_c;
	CHECK_CUDA_ERROR(cudaMalloc(&dev_a, FULL_MEM_SIZE));
	CHECK_CUDA_ERROR(cudaMalloc(&dev_b, FULL_MEM_SIZE));
	CHECK_CUDA_ERROR(cudaMalloc(&dev_c, FULL_MEM_SIZE));
 
	 //2. Copy the arrays ‘a’ and ‘b’ to the GPU
	CHECK_CUDA_ERROR(cudaMemcpy(dev_a, a, FULL_MEM_SIZE, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(dev_b, b, FULL_MEM_SIZE, cudaMemcpyHostToDevice));
 
	 //3. Execute the kernel
	dim3 BS(N/4., N/4.); 
	dim3 GS(ceil(N/(float)BS.x), ceil(N/(float)BS.x));
	printf("threads: %d, block size: %d\n\n", BS.x, GS.x);

	add2D<<<GS, BS>>>(dev_a, dev_b, dev_c, N);
 
	 //4. Copy the array ‘c’ from GPU to CPU
	CHECK_CUDA_ERROR(cudaMemcpy(c,  dev_c, FULL_MEM_SIZE, cudaMemcpyDeviceToHost));
 
	 //Display the result on the CPU
	print2D(a, N, N);
	printf("\n");
	printf("    | \n");
	printf(" ---|---\n");
	printf("    | \n");
	printf("\n");
	print2D(b, N, N);
	printf("\n");
	printf("   ||\n");
	printf("\n");
	print2D(c, N, N);
 
	 //5. Release the memory allocated on the GPU. Purpose to avoid memory leaks
	cudaFree(dev_a); std::free(a);
	cudaFree(dev_b); std::free(b);
	cudaFree(dev_c); std::free(c);
	CHECK_LAST_CUDA_ERROR();
 
	return 0;
}
