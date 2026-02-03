#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024

// Clamp function for CUDA
__device__ __forceinline__ float clamp(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, val));
}

__global__ void int8_quant_kernel(float *x, int8_t *y, float scale, int zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] / scale + zero_point;
        val = roundf(val);
        val = clamp(val, -128.0f, 127.0f);
        y[idx] = static_cast<int8_t>(val);
    }
}

__global__ void int8_dequant_kernel(int8_t *x, float *y, float scale, int zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = static_cast<float>(x[idx] - zero_point) * scale;
    }
}

int main() {
    int n = NUM_DATA;
    std::vector<float> h_x(n);
    for (int i = 0; i < n; i++) {
        h_x[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    float x_min = *std::min_element(h_x.begin(), h_x.end());
    float x_max = *std::max_element(h_x.begin(), h_x.end());
    float scale = (x_max - x_min) / 255.0f;
    int zero_point = int(round(-x_min / scale - 128));
    zero_point = std::max(-128, std::min(127, zero_point));

    std::vector<int8_t> h_ref_temp(n);
    std::vector<float> h_ref(n);
    float float_val;
    for (int i = 0; i < n; i++) {
        float_val = h_x[i] / scale + zero_point;
        float_val = roundf(float_val);
        float_val = std::max(-128.0f, std::min(127.0f, float_val));
        h_ref_temp[i] = static_cast<int8_t>(float_val);
        //dequant
        float_val = static_cast<float>(h_ref_temp[i] - zero_point) * scale;
        h_ref[i] = float_val;
    }

    float *d_x;
    int8_t *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(int8_t));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int8_quant_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, scale, zero_point, n);
    cudaDeviceSynchronize();
    int8_dequant_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_x, scale, zero_point, n);
    cudaDeviceSynchronize();
    std::vector<float> h_result(n);
    cudaMemcpy(h_result.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        max_error = std::max(max_error, std::abs(h_result[i] - h_ref[i]));
    }
    std::cout << "Max error: " << max_error << std::endl;
    return 0;
}