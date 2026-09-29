#include "cuda_check.cuh"
#include <vector>

constexpr int BLOCK = 256; // Harus berupa pangkat dua

__global__ void block_sum(const int *a, int *partial, int n)
{
    __shared__ int values[BLOCK];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    values[t] = (i < n) ? a[i] : 0;
    __syncthreads();

    for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
        if (t < offset) {
            values[t] += values[t + offset];
        }
        __syncthreads();
    }

    if (t == 0) {
        partial[blockIdx.x] = values[0];
    }
}

int main()
{
    const int n = 1000003;
    const int blocks = (n + BLOCK - 1) / BLOCK;
    std::vector<int> a(n), partial(blocks);
    long long expected = 0;

    for (int i = 0; i < n; ++i) {
        a[i] = i % 7;
        expected += a[i];
    }

    int *d_a = nullptr, *d_partial = nullptr;
    std::size_t bytes = static_cast<std::size_t>(n) * sizeof(int);
    std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(int);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_partial), partial_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));

    block_sum<<<blocks, BLOCK>>>(d_a, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, partial_bytes, cudaMemcpyDeviceToHost));

    long long result = 0;
    for (int x : partial) {
        result += x;
    }

    std::printf("expected=%lld result=%lld match=%s\n", expected, result, (expected == result) ? "yes" : "no");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_partial));
    return result != expected;
}
