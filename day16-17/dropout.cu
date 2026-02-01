#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 1024

__global__ void fixed_dropout_kernel(float *input, float *mask, float *output, float p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float m = mask[idx];
        output[idx] = (m > 0.5f) ? x / (1.0f - p) : 0.0f;
    }
}

__device__ float simple_rand(unsigned int seed, unsigned int idx) {
    unsigned int state = seed ^ (idx * 0x9e3779b9);
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFF);
}

__global__ void rng_dropout_kernel(float *input, float *output, float p, unsigned int seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float random = simple_rand(seed, idx);
        output[idx] = (random > p) ? x / (1.0f - p) : 0.0f;
    }
}

int main() {
    int n = NUM_DATA;
    float p = 0.5f;
    unsigned int seed = 123;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = 20.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 10.0f;
    }
    
    std::cout << "============================================================" << std::endl;
    std::cout << "Fixed Dropout Test" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    std::vector<float> h_mask(n);
    for (int i = 0; i < n; i++) {
        float rand_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mask[i] = (rand_val > p) ? 1.0f : 0.0f;
    }
    
    std::vector<float> h_fixed_ref(n);
    for (int i = 0; i < n; i++) {
        h_fixed_ref[i] = (h_mask[i] > 0.5f) ? h_input[i] / (1.0f - p) : 0.0f;
    }
    
    float *d_input, *d_mask, *d_fixed_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_mask, n * sizeof(float));
    cudaMalloc(&d_fixed_output, n * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    fixed_dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_mask, d_fixed_output, p, n);
    
    std::vector<float> h_fixed_output(n);
    cudaMemcpy(h_fixed_output.data(), d_fixed_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float fixed_max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        fixed_max_error = std::max(fixed_max_error, std::abs(h_fixed_output[i] - h_fixed_ref[i]));
    }
    std::cout << "Max error: " << fixed_max_error << std::endl;
    
    cudaFree(d_mask);
    cudaFree(d_fixed_output);
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "RNG Dropout Seed Reproducibility Test" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    float *d_rng_output1, *d_rng_output2, *d_rng_output3;
    cudaMalloc(&d_rng_output1, n * sizeof(float));
    cudaMalloc(&d_rng_output2, n * sizeof(float));
    cudaMalloc(&d_rng_output3, n * sizeof(float));
    
    rng_dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_rng_output1, p, seed, n);
    rng_dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_rng_output2, p, seed, n);
    rng_dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_rng_output3, p, seed + 1, n);
    
    std::vector<float> h_rng_output1(n), h_rng_output2(n), h_rng_output3(n);
    cudaMemcpy(h_rng_output1.data(), d_rng_output1, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rng_output2.data(), d_rng_output2, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rng_output3.data(), d_rng_output3, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float same_seed_diff = 0.0f;
    float diff_seed_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        same_seed_diff = std::max(same_seed_diff, std::abs(h_rng_output1[i] - h_rng_output2[i]));
        diff_seed_diff = std::max(diff_seed_diff, std::abs(h_rng_output1[i] - h_rng_output3[i]));
    }
    
    std::cout << "Max difference with same seed (123): " << same_seed_diff << std::endl;
    std::cout << "  → " << (same_seed_diff == 0.0f ? "✓ Reproducible!" : "✗ Not reproducible") << std::endl;
    
    std::cout << "\nMax difference with different seeds (123 vs 124): " << diff_seed_diff << std::endl;
    std::cout << "  → " << (diff_seed_diff > 0.0f ? "✓ Different outputs as expected" : "✗ Unexpectedly identical outputs (issue)") << std::endl;
    
    cudaFree(d_input);
    cudaFree(d_rng_output1);
    cudaFree(d_rng_output2);
    cudaFree(d_rng_output3);
    
    return 0;
}
