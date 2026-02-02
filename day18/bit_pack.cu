#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024

__global__ void pack_int4_kernel(uint8_t *vec, uint32_t *result, int n) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n * 4) {
        result[idx / 4] = static_cast<uint32_t>((vec[idx] << 0) | (vec[idx + 1] << 8) | (vec[idx + 2] << 16) | (vec[idx + 3] << 24));
    }
}

int main() {
    int n = NUM_DATA;
    std::vector<uint8_t> h_vec(n * 4);
    for (int i = 0; i < n * 4; i++) {
        h_vec[i] = static_cast<uint8_t>((rand()) % 256);
    }

    std::vector<uint32_t> h_ref(n);
    std::vector<uint32_t> h_result(n);
    for (int i = 0; i < n; i++) {
        h_ref[i] = static_cast<uint32_t>((h_vec[i * 4] << 0) | (h_vec[i * 4 + 1] << 8) | (h_vec[i * 4 + 2] << 16) | (h_vec[i * 4 + 3] << 24));
    }   

    uint8_t *d_vec;
    uint32_t *d_result;
    cudaMalloc(&d_vec, n * 4 * sizeof(uint8_t));
    cudaMalloc(&d_result, n * sizeof(uint32_t));
    cudaMemcpy(d_vec, h_vec.data(), n * 4 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    pack_int4_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_result, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result.data(), d_result, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_result);

    int max_error = 0;
    for (int i = 0; i < n; i++) {
        max_error = std::max(max_error, std::abs(static_cast<int>(h_result[i] - h_ref[i])));
    }
    std::cout << "Max error: " << max_error << std::endl;

    return 0;
}