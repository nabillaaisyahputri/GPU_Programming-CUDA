#include "cuda_check.cuh"
#include <vector>
#include <chrono>

__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int n = 1000003;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 97) * 0.25f;
        b[i] = (i % 31) * 0.5f;
    }

    // Serial reference and CPU timer
    using Clock = std::chrono::steady_clock;
    std::vector<float> reference(n);
    auto cpu_start = Clock::now();
    for (int i = 0; i < n; ++i) {
        reference[i] = a[i] + b[i];
    }
    double cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - cpu_start).count();

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up runtime and kernel
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed operation including H2D transfers and kernel execution
    auto wall_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    double operation_ms = std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    std::printf("CPU=%.6f ms | Kernel=%.6f ms | Operation=%.6f ms\n", cpu_ms, kernel_ms, operation_ms);
    if (kernel_ms > 0.0f && operation_ms > 0.0f) {
        std::printf("Resident-data speedup=%.3f | Operation speedup=%.3f\n", cpu_ms / kernel_ms, cpu_ms / operation_ms);
    }

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        double expected = reference[i];
        if (!close_enough(c[i], expected)) ++errors;
    }
    std::printf("Errors=%d\n", errors);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors != 0;
}
