#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define H 64
#define W 128

__global__ void transpose_strided_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int h, int w,
    int in_stride0, int in_stride1,
    int out_stride0, int out_stride1
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < h && j < w) {
        float v = in[i * in_stride0 + j * in_stride1];
        out[j * out_stride0 + i * out_stride1] = v;
    }
}

int main() {
    const int h = H;
    const int w = W;

    std::vector<float> h_in(h * w);
    for (int i = 0; i < h * w; i++) {
        h_in[i] = 20.0f * (static_cast<float>(rand()) / RAND_MAX) - 10.0f;
    }

    std::vector<float> h_ref(w * h, 0.0f);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            h_ref[j * h + i] = h_in[i * w + j];
        }
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, h * w * sizeof(float));
    cudaMalloc(&d_out, w * h * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), h * w * sizeof(float), cudaMemcpyHostToDevice);

    int in_stride0 = w;
    int in_stride1 = 1;
    int out_stride0 = h;
    int out_stride1 = 1;

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    transpose_strided_kernel<<<grid, block>>>(
        d_in, d_out, h, w,
        in_stride0, in_stride1,
        out_stride0, out_stride1
    );

    cudaGetLastError();
    cudaDeviceSynchronize();

    std::vector<float> h_out(w * h);
    cudaMemcpy(h_out.data(), d_out, w * h * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    float max_error = 0.0f;
    float max_abs_value = 0.0f;
    for (int i = 0; i < w * h; i++) {
        float diff = std::abs(h_out[i] - h_ref[i]);
        max_error = std::max(max_error, diff);
        max_abs_value = std::max(max_abs_value, std::max(std::abs(h_out[i]), std::abs(h_ref[i])));
    }

    std::cout << "Max error: " << max_error << "\n";
    float max_error_percentage = 0.0f;
    if (max_abs_value > 1e-12f) {
        max_error_percentage = (max_error / max_abs_value) * 100.0f;
    }
    std::cout << "Max error percentage: " << max_error_percentage << "%\n";

    return 0;
}
