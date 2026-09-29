#include "cuda_check.cuh"
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdio>

__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = 2.0f * a[i] + b[i];
    }
}

// Fungsi helper untuk CPU timing
double benchmark_cpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        c[i] = 2.0f * a[i] + b[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

int main() {
    // 4 Ukuran data yang berbeda
    int sizes[] = {100000, 1000000, 5000000, 10000000};
    const int runs = 5;
    const int threads = 256;

    std::printf("N | CPU Median (ms) | Kernel Median (ms) | Total Op Median (ms) | Kernel Speedup | Op Speedup\n");
    std::printf("---|---|---|---|---|---\n");

    for (int n : sizes) {
        std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);

        for (int i = 0; i < n; ++i) {
            h_a[i] = (i % 97) * 0.25f;
            h_b[i] = (i % 31) * 0.5f;
        }

        // 1. Ukur CPU
        std::vector<double> cpu_times;
        for (int r = 0; r < runs; ++r) {
            cpu_times.push_back(benchmark_cpu(h_a, h_b, h_c_cpu, n));
        }
        std::sort(cpu_times.begin(), cpu_times.end());
        double cpu_median = cpu_times[runs / 2];

        // Alokasi GPU
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        std::vector<double> kernel_times;
        std::vector<double> total_op_times;

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        int blocks = (n + threads - 1) / threads;

        for (int r = 0; r < runs; ++r) {
            // Waktu Total Operasi (H2D + Kernel + D2H)
            auto total_start = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

            // Waktu Kernel murni menggunakan CUDA Event
            cudaEventRecord(start_event, 0);
            vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
            cudaEventRecord(stop_event, 0);

            CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

            auto total_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> total_elapsed = total_end - total_start;
            total_op_times.push_back(total_elapsed.count());

            cudaEventSynchronize(stop_event);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start_event, stop_event);
            kernel_times.push_back(static_cast<double>(milliseconds));
        }

        std::sort(kernel_times.begin(), kernel_times.end());
        std::sort(total_op_times.begin(), total_op_times.end());
        double kernel_median = kernel_times[runs / 2];
        double total_op_median = total_op_times[runs / 2];

        double kernel_speedup = cpu_median / kernel_median;
        double op_speedup = cpu_median / total_op_median;

        std::printf("%d | %.3f | %.3f | %.3f | %.2fx | %.2fx\n",
                    n, cpu_median, kernel_median, total_op_median, kernel_speedup, op_speedup);

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    return 0;
}
