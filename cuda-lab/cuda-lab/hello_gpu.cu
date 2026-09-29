#include "cuda_check.cuh"

__global__ void hello(int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        printf("block=%u local=%u global=%d\n", blockIdx.x, threadIdx.x, i);
    }
}

int main()
{
    const int n = 17;
    const int threads = 16;
    const int blocks = (n + threads - 1) / threads;

    hello<<<blocks, threads>>>(n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Host: the kernel has finished.\n");
    return 0;
}
