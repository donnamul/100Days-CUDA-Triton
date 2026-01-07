#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024
__global__ void vec_scale(float *vec, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vec[idx] *= scale;
    }
}

int main() {
    int n = NUM_DATA;
    std::vector<float> h_vec(n);
    for (int i = 0; i < n; i++) {
        h_vec[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    std::vector<float> h_ref(n);
    for (int i = 0; i < n; i++) {
        h_ref[i] = 2.0f * h_vec[i];
    }

    float *d_vec;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vec_scale<<<blocksPerGrid, threadsPerBlock>>>(d_vec, 2.0f, n);

    cudaMemcpy(h_vec.data(), d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);

    float max_error = 0.0f;
    float max_abs_value = 0.0f;
    for (int i = 0; i < n; i++) {
        max_error = std::max(max_error, std::abs(h_vec[i] - h_ref[i]));
        max_abs_value = std::max(max_abs_value, std::max(std::abs(h_vec[i]), std::abs(h_ref[i])));
    }
    std::cout << "Max error: " << max_error << std::endl;

    float max_error_percentage = 0.0f;
    if (max_abs_value > 1e-12f) {
        max_error_percentage = (max_error / max_abs_value) * 100.0f;
    }
    std::cout << "Max error percentage: " << max_error_percentage << "%" << std::endl;

    return 0;
}