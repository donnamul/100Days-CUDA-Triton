#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024
__global__ void dropout_kernel(float *vec, float *mask, float p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = vec[idx];
        float m = mask[idx];
        vec[idx] = (m > p) ? x / (1 - p) : 0.0f;
    }
}

int main() {
    int n = NUM_DATA;
    float p = 0.5f;
    std::vector<float> h_vec(n), h_mask(n);
    for (int i = 0; i < n; i++) {
        h_vec[i] = 20.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 10.0f;
    }
    for (int i = 0; i < n; i++) {
        h_mask[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    std::vector<float> h_ref(n);
    for (int i = 0; i < n; i++) {
        h_ref[i] = h_mask[i] > p ? h_vec[i] / (1 - p) : 0.0f;
    }

    float *d_vec, *d_mask;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMalloc(&d_mask, n * sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_mask, p, n);

    cudaMemcpy(h_vec.data(), d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_mask);

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