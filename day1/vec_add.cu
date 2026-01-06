#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vec_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n;
    std::cout << "Enter the size of the vector: ";
    std::cin >> n;
    int bytes = n * sizeof(float);
    
    std::vector<float> h_a(n), h_b(n), h_c(n);

    for (int i = 0; i < n; i++) {
        h_a[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_b[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vec_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
     

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        max_error = std::max(max_error, std::abs(h_c[i] - (h_a[i] + h_b[i])));
    }
    std::cout << "Max error: " << max_error << std::endl;

    float max_error_percentage = (max_error / std::max(std::abs(h_a[0]), std::abs(h_b[0]))) * 100.0f;
    std::cout << "Max error percentage: " << max_error_percentage << "%" << std::endl;

    return 0;
}