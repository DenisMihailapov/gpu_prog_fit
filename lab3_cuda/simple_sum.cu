
#include <cstdlib>
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
#define uint unsigned int 

__global__ void add(REAL* a, REAL* b, REAL* c, uint N){

	//Calculate the data at the index
	uint index = blockIdx.x *blockDim.x + threadIdx.x;

	if(index < N)
		c[index] = a[index] + b[index];

}

int main(int argc, char *argv[]){

	uint N = atof(argv[1]);
	uint MEM_ARRAY_SIZE = N * sizeof(REAL);
    
	//0. Allocate and Initialize memory on the CPU
    REAL *a = (REAL *) std::malloc(MEM_ARRAY_SIZE);
	REAL *b = (REAL *) std::malloc(MEM_ARRAY_SIZE);
	REAL *c = (REAL *) std::malloc(MEM_ARRAY_SIZE);
    
	for (int i = 0; i< N; i++) {
		a[i] = -i; b[i] = i * i;
	}
	 //1. Allocate memory on the GPU
	REAL *dev_a, *dev_b, *dev_c;
	CHECK_CUDA_ERROR(cudaMalloc(&dev_a, N * sizeof(REAL)));
	CHECK_CUDA_ERROR(cudaMalloc(&dev_b, N * sizeof(REAL)));
	CHECK_CUDA_ERROR(cudaMalloc(&dev_c, N * sizeof(REAL)));
 
	 //2. Copy the arrays ‘a’ and ‘b’ to the GPU
	CHECK_CUDA_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
 
	 //3. Execute the kernel
	dim3 BS(N);
	dim3 GS(ceil(N/(float)BS.x));

	add<<<BS, GS>>>(dev_a,dev_b,dev_c, N);
 
	 //4. Copy the array ‘c’ from GPU to CPU
	CHECK_CUDA_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
 
	 //Display the result on the CPU
	for (int i = 0; i<N; i++) {
		printf("%.4f + %.4f = %.4f\n", a[i], b[i], c[i]);
	}
 
	 //5. Release the memory allocated on the GPU. Purpose to avoid memory leaks
	cudaFree(dev_a); std::free(a);
	cudaFree(dev_b); std::free(b);
	cudaFree(dev_c); std::free(c);
	CHECK_LAST_CUDA_ERROR();
 
	return 0;
}
