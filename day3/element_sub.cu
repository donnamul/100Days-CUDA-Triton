#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void element_sub_2d(const float *a, const float *b, float *c,
                               int row, int col) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (i < row && j < col) {
        int idx = i * col + j; // row-major flatten
        c[idx] = a[idx] - b[idx];
    }
}

int main() {
    int col = 1024;
    int row = 1024;
    int elements = row * col;

    std::vector<float> h_a(elements), h_b(elements), h_c(elements);

    for (int i = 0; i < elements; i++) {
        h_a[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_b[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 2D launch
    dim3 threadsPerBlock(16, 16);  // or (32, 8), (256,1) etc.
    dim3 blocksPerGrid(
        (col + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (row + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    element_sub_2d<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, row, col);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    float max_error = 0.0f;
    for (int i = 0; i < elements; i++) {
        max_error = std::max(max_error, std::fabs(h_c[i] - (h_a[i] - h_b[i])));
    }

    std::cout << "Max error: " << max_error << std::endl;
    return 0;
}
