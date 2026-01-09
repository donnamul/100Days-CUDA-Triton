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


__global__ void fused_axpby(const float a, const float *x, const float b, const float *y, float*z, int row, int col){
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (i < row && j < col) {
        int idx = i * col + j; // row-major flatten
        z[idx] = a*x[idx] + b*y[idx];
    }
}

int main() {
    int col = 1024;
    int row = 1024;
    int elements = row * col;

    std::vector<float> h_x(elements), h_y(elements), h_z(elements);

    for (int i = 0; i < elements; i++) {
        h_x[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_y[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    // 2D launch
    dim3 threadsPerBlock(16, 16);  // or (32, 8), (256,1) etc.
    dim3 blocksPerGrid(
        (col + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (row + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    fused_axpby<<<blocksPerGrid, threadsPerBlock>>>(2.0f, d_x, 3.0f, d_y, d_z, row, col);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    float max_error = 0.0f;
    for (int i = 0; i < elements; i++) {
        max_error = std::max(max_error, std::fabs(h_z[i] - (2.0f*h_x[i] + 3.0f*h_y[i])));
    }

    std::cout << "Max error: " << max_error << std::endl;
    return 0;
}
