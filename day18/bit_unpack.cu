#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024

__global__ void unpack_int4_kernel(uint32_t *vec, uint8_t *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx * 4] = static_cast<uint8_t>((vec[idx] >> 0) & 0x000000FF);
        result[idx * 4 + 1] = static_cast<uint8_t>((vec[idx] >> 8) & 0x000000FF);
        result[idx * 4 + 2] = static_cast<uint8_t>((vec[idx] >> 16) & 0x000000FF);
        result[idx * 4 + 3] = static_cast<uint8_t>((vec[idx] >> 24) & 0x000000FF);
    }
}

int main() {
    int n = NUM_DATA;
    std::vector<uint32_t> h_vec(n);
    for (int i = 0; i < n; i++) {
        h_vec[i] = static_cast<uint32_t>(rand()) % 256;
    }

    std::vector<uint8_t> h_ref(n * 4);
    std::vector<uint8_t> h_result(n * 4);
    for (int i = 0; i < n; i++) {
        h_ref[i * 4] = static_cast<uint8_t>((h_vec[i] >> 0) & 0x000000FF);
        h_ref[i * 4 + 1] = static_cast<uint8_t>((h_vec[i] >> 8) & 0x000000FF);
        h_ref[i * 4 + 2] = static_cast<uint8_t>((h_vec[i] >> 16) & 0x000000FF);
        h_ref[i * 4 + 3] = static_cast<uint8_t>((h_vec[i] >> 24) & 0x000000FF);
    }   

    uint32_t *d_vec;
    uint8_t *d_result;
    cudaMalloc(&d_vec, n * sizeof(uint32_t));
    cudaMalloc(&d_result, n * 4 * sizeof(uint8_t));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    unpack_int4_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_result, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result.data(), d_result, n * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_result);

    int max_error = 0;
    for (int i = 0; i < n * 4; i++) {
        max_error = std::max(max_error, std::abs(static_cast<int>(h_result[i]) - static_cast<int>(h_ref[i])));
    }
    std::cout << "Max error: " << max_error << std::endl;

    
    return 0;
}