#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024
__global__ void sigmoid_kernel(float *vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vec[idx] = 1.0f/(1.0f + expf(-vec[idx]));
    }
}

int main() {
    int n = NUM_DATA;
    std::vector<float> h_vec(n);

    for (int i = 0; i < n; i++) {
        h_vec[i] = 20.0f * (static_cast<float>(rand()) / RAND_MAX) - 10.0f;
    }

    std::vector<float> h_ref(n);
    for (int i = 0; i < n; i++) {
        h_ref[i] = 1.0f/(1.0f + std::exp(-h_vec[i]));
    }

    float *d_vec;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    cudaDeviceSynchronize();

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