#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define NUM_DATA 4096
__global__ void l2_norm_kernel(const float *vec, float *result, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 1단계: 각 스레드가 grid-stride loop으로 local sum 계산
    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float val = vec[i];
        local_sum += val * val;
    }
    
    // 2단계: local sum을 shared memory에 저장
    sdata[tid] = local_sum;
    __syncthreads();
    
    // 3단계: 블록 내 reduction (병렬로 합산)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 4단계: 각 블록의 결과를 atomic add로 최종 합산
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}



int main() {
    int n = NUM_DATA;
    std::vector<float> h_vec(n);
    for (int i = 0; i < n; i++) {
        h_vec[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float h_ref;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += std::pow(h_vec[i], 2);
    }
    h_ref = std::sqrt(sum);
    std::cout << "Reference L2 norm: " << h_ref << std::endl;

    float *d_vec, *d_result;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    l2_norm_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_vec, d_result, n);

    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    result = std::sqrt(result);
    std::cout << "GPU L2 norm: " << result << std::endl;

    cudaFree(d_vec);
    cudaFree(d_result);

    return 0;
}