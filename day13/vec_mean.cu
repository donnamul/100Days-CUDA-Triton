#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

__global__ void sum_kernel(float *x, int n, float *block_sums) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1단계: 전역 메모리에서 shared memory로 데이터 로드
    sdata[tid] = (idx < n) ? x[idx] : 0.0f;
    __syncthreads();
    
    // 2단계: 블록 내 병렬 리덕션
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 3단계: 블록의 합을 저장
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_kernel(float *block_sums, int num_blocks, int n, float *result) {
    int idx = threadIdx.x;
    
    if (idx < num_blocks) {
        atomicAdd(result, block_sums[idx]);
    }
    if (threadIdx.x == num_blocks - 1) {
        *result /= n;
    }
}

int main() {
    int n = 1024;
    std::vector<float> h_vec(n);
    for (int i = 0; i < n; i++) {
        h_vec[i] = 100.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    // 참조 평균 계산 (검증용)
    float ref_mean = 0.0f;
    for (int i = 0; i < n; i++) {
        ref_mean += h_vec[i];
    }
    ref_mean /= n;
    
    float *d_vec, *d_block_sums, *d_result;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));
    
    // 결과를 저장할 메모리 할당 및 0으로 초기화
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));
    
    // 1단계: 블록 내 리덕션으로 각 블록의 합 계산
    sum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_vec, n, d_block_sums);
    
    // 2단계: 블록 간 atomic add로 최종 합 계산
    reduce_kernel<<<1, blocksPerGrid>>>(d_block_sums, blocksPerGrid, n, d_result);
    cudaDeviceSynchronize();
    
    // 결과 복사 (이미 평균 계산됨)
    float mean = 0.0f;
    cudaMemcpy(&mean, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_vec);
    cudaFree(d_block_sums);
    cudaFree(d_result);
    
    // 결과 검증
    float error = std::abs(mean - ref_mean);
    std::cout << "Reference mean: " << ref_mean << std::endl;
    std::cout << "GPU mean: " << mean << std::endl;
    std::cout << "Error: " << error << std::endl;
    
    return 0;
}