#include "cuda_check.cuh"
#include <vector>
#include <cmath>

__global__ void stencil_kernel(const float *in, float *out, int height, int width)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < height && c < width) {
        int idx = r * width + c;
        // Salin sel batas (boundary) apa adanya
        if (r == 0 || r == height - 1 || c == 0 || c == width - 1) {
            out[idx] = in[idx];
        } else {
            // Hitung rata-rata 5-titik untuk sel interior
            float sum = in[idx]                      // sel pusat
                      + in[(r - 1) * width + c]      // atas
                      + in[(r + 1) * width + c]      // bawah
                      + in[r * width + (c - 1)]      // kiri
                      + in[r * width + (c + 1)];     // kanan
            out[idx] = sum / 5.0f;
        }
    }
}

int main()
{
    // Menggunakan dimensi ganjil/tidak beraturan sesuai anjuran modul (H=257, W=259)
    const int height = 257;
    const int width = 259;
    const int n = height * width;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> h_in(n), h_out_gpu(n), h_out_cpu(n);

    // Inisialisasi data input
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            h_in[r * width + c] = static_cast<float>((r * 13 + c * 7) % 100) * 0.1f;
        }
    }

    // Referensi CPU
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * width + c;
            if (r == 0 || r == height - 1 || c == 0 || c == width - 1) {
                h_out_cpu[idx] = h_in[idx];
            } else {
                float sum = h_in[idx]
                          + h_in[(r - 1) * width + c]
                          + h_in[(r + 1) * width + c]
                          + h_in[r * width + (c - 1)]
                          + h_in[r * width + (c + 1)];
                h_out_cpu[idx] = sum / 5.0f;
            }
        }
    }

    // Alokasi memori GPU
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out), bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Konfigurasi Grid dan Block 2D
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Eksekusi Kernel
    stencil_kernel<<<grid, block>>>(d_in, d_out, height, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Validasi Hasil GPU vs CPU
    int errors = 0;
    double max_error = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::fabs(h_out_gpu[i] - h_out_cpu[i]);
        if (err > max_error) max_error = err;
        if (!close_enough(h_out_gpu[i], h_out_cpu[i], 1e-4, 1e-4)) {
            ++errors;
        }
    }

    std::printf("Stencil 2D Test -> Height=%d, Width=%d, Errors=%d, Max Error=%.3e\n",
                height, width, errors, max_error);

    // Bersihkan memori perangkat
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return errors != 0;
}
