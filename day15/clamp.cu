#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024 * 1024
__global__ void clamp_kernel(float *vec, float min_val, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = vec[idx];
        vec[idx] = (x < min_val) ? min_val : (x > max_val) ? max_val : x;
    }
}

int main() {
    int n = NUM_DATA;
    std::vector<float> h_vec(n);
    for (int i = 0; i < n; i++) {
        h_vec[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    float min_val = 20.0f;
    float max_val = 80.0f;
    
    std::vector<float> h_ref(n);
    for (int i = 0; i < n; i++) {
        h_ref[i] = (h_vec[i] < min_val) ? min_val : (h_vec[i] > max_val) ? max_val : h_vec[i];
    }


    float *d_vec;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    clamp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, min_val, max_val, n);
    cudaMemcpy(h_vec.data(), d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vec);

    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        max_error = std::max(max_error, std::abs(h_vec[i] - h_ref[i]));
    }
    std::cout << "Max error: " << max_error << std::endl;

    return 0;
}