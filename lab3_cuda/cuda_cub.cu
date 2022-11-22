#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <iostream>

#define BLOCKSIZE  4

const int N = 64;

void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }



/**************************/
/* BLOCK REDUCTION KERNEL */
/**************************/
__global__ void sum(const float * __restrict__ indata, float * __restrict__ outdata) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Specialize BlockReduce for type float. 
    typedef cub::BlockReduce<float, BLOCKSIZE> BlockReduceT; 

    // --- Allocate temporary storage in shared memory 
    __shared__ typename BlockReduceT::TempStorage temp_storage; 

    float result;
    if(tid < N) result = BlockReduceT(temp_storage).Sum(indata[tid]);

    // --- Update block reduction value
    if(threadIdx.x == 0) outdata[blockIdx.x] = result;

    return;  
}

/********/
/* MAIN */
/********/
int main() {

    // --- Allocate host side space for 
    float *h_data       = (float *)malloc(N * sizeof(float));
    float *h_result     = (float *)malloc((N / BLOCKSIZE) * sizeof(float));

    float *d_data;      gpuErrchk(cudaMalloc(&d_data, N * sizeof(float)));
    float *d_result;    gpuErrchk(cudaMalloc(&d_result, (N / BLOCKSIZE) * sizeof(float)));

    for (int i = 0.; i < N; i++) h_data[i] = i;
    std::cout << "input: ";
    uint S = 0;
    for(int i = 0; i < N; i++){
        
        std::cout << h_data[i] << " ";
        S += h_data[i];

    }
    std::cout << std::endl;
    std::cout << S << std::endl; S = 0;

    gpuErrchk(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    sum<<<(N / BLOCKSIZE), BLOCKSIZE>>>(d_data, d_result);
    gpuErrchk(cudaMemcpy(h_result, d_result, (N / BLOCKSIZE) * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_result, d_result, (N / BLOCKSIZE) * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "output: ";
    for(int i = 0; i < (N / BLOCKSIZE); i++){
        
        std::cout << h_result[i] << " ";
        S += h_result[i];
    }
    std::cout << std::endl;
    std::cout << S << std::endl;
    

    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_result));

    return 0;
}