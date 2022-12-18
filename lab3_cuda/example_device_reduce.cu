
/******************************************************************************
 * Simple example of DeviceReduce::Sum().
 *
 * Sums an array of int keys.
 *
 * To compile using the command line:
 *   nvcc -o cub_sum  example_device_reduce.cu && ./cub_sum
 *
 ******************************************************************************/
// Ensure printing of CUDA runtime errors to console
#include <stdio.h>
#include <cub/device/device_reduce.cuh>


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------
void DisplayResults(int *array, size_t N){
    for (size_t i = 0; i < N; i++) {
        printf("%d ", array[i]);
    }

}


void Initialize(
    int   *h_in,
    int     num_items)
{
    for (int i = 0; i < num_items; ++i)
        h_in[i] = i*i;

    printf("Input:\n");
    DisplayResults(h_in, num_items);
    printf("\n\n");

}

int sum(int *h_in, int num_items)
{
    int h_ref = h_in[0];
    for (int i = 1; i < num_items; ++i)
       h_ref += h_in[i];

    return h_ref;   
}

int gpu_sum(int *d_in,  int num_items){
    // Allocate device output array
    int *d_out = NULL;
    cudaMalloc((void**)&d_out, sizeof(int));

    // Request and allocate temporary storage
    void   *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    
    int h_actual;
    cudaMemcpy(&h_actual, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\tsum: %d\n", h_actual);

    cudaFree(d_out);
    cudaFree(d_temp_storage);


    return h_actual;

}
//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv)
{
    int num_items = 15;
   

    printf("cub::DeviceReduce::Sum() %d items (%d-byte elements)\n",
        num_items, (int) sizeof(int));
    fflush(stdout);

    // Allocate host arrays
    int* h_in = new int[num_items];
    int  h_reference, h_actual;

    // Initialize problem and solution
    Initialize(h_in, num_items);
    h_reference = sum(h_in, num_items);


    // Allocate problem device arrays
    int *d_in = NULL;
    cudaMalloc((void**)&d_in, sizeof(int) * num_items);

    // Initialize device input
    cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice);

    h_actual = gpu_sum(d_in, num_items);

    // Check for correctness (and display results, if specified)
    printf("\t%s", h_reference == h_actual ? "PASS" : "FAIL");

    // Cleanup
    delete[] h_in;
    cudaFree(d_in);
    printf("\n\n");
    return 0;
}