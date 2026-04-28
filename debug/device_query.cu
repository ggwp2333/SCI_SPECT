// nvcc -o device_query device_query.cu
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device 0: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("\n--- Threads ---\n");
    printf("Max threads per block:          %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM:             %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size:                      %d\n", prop.warpSize);
    printf("Max block dimensions:           (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions:            (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("\n--- Multiprocessors ---\n");
    printf("SM count:                       %d\n", prop.multiProcessorCount);
    printf("Max blocks per SM:              %d\n", prop.maxBlocksPerMultiProcessor);

    printf("\n--- Memory ---\n");
    printf("Global memory:                  %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Shared memory per block:        %zu bytes\n", prop.sharedMemPerBlock);
    printf("Shared memory per SM:           %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Max dynamic shared memory/block:%zu bytes\n", prop.reservedSharedMemPerBlock > 0 ? prop.sharedMemPerBlock - prop.reservedSharedMemPerBlock : prop.sharedMemPerBlock);
    printf("Registers per block:            %d\n", prop.regsPerBlock);
    printf("Registers per SM:               %d\n", prop.regsPerMultiprocessor);
    printf("L2 cache size:                  %d bytes\n", prop.l2CacheSize);

    printf("\n--- Clock ---\n");
    printf("Clock rate:                     %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory clock rate:              %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory bus width:               %d bits\n", prop.memoryBusWidth);

    return 0;
}
