#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

inline void cuda_check(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: CUDA error: %s\n",
                     file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

inline bool close_enough(double got, double expected, double atol = 1e-5, double rtol = 1e-5)
{
    return std::isfinite(got) && std::isfinite(expected) &&
           std::fabs(got - expected) <= atol + rtol * std::fabs(expected);
}
