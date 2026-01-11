#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024
__global__ void Leaky_Relu(float *vec, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = vec[idx];
        vec[idx] = (v > 0.f) ? v : alpha * v;
    }
}

int main() {
    int n = NUM_DATA;
    std::vector<float> h_vec(n);
    float alpha = 0.01f;

    for (int i = 0; i < n; i++) {
        h_vec[i] = 200.0f * (static_cast<float>(rand()) / RAND_MAX) - 100.0f;
    }

    std::vector<float> h_ref(n);
    for (int i = 0; i < n; i++) {
        if(h_vec[i] > 0){
            h_ref[i] = h_vec[i];
        }
        else{
            h_ref[i] = alpha * h_vec[i];
        }
    }

    float *d_vec;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    Leaky_Relu<<<blocksPerGrid, threadsPerBlock>>>(d_vec, alpha, n);

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