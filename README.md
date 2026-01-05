# GPU Kernel Challenge
CUDA, Triton, + (?)



# 🚀 CUDA & Triton 100-Day Challenge

This challenge focuses on implementing high-performance kernels in both **CUDA (C++)** and **Triton (Python)**. Every day requires a code PR that passes functional verification and performance benchmarks on **LeetGPU**.

## 🟢 Phase 1: Element-wise Operations & Memory Foundation (Day 01–20)
| Day | Topic | LeetGPU Task / Implementation Goal |
|:---:|:---|:---|
| 01 | Vector Add | `y = a + b` (Basic Memory I/O) |
| 02 | Vector Scale | `y = alpha * x` (Scalar Broadcasting) |
| 03 | Element-wise Sub | `y = a - b` |
| 04 | Fused AXPBY | `y = alpha * x + beta * y` (Load/Store Optimization) |
| 05 | ReLU | `max(0, x)` |
| 06 | Leaky ReLU | `x < 0 ? alpha * x : x` |
| 07 | Sigmoid | Numerical stability for `1 / (1 + exp(-x))` |
| 08 | GeLU | GELU Approximation Kernel |
| 09 | SiLU (Swish) | `x * sigmoid(x)` |
| 10 | Tanh | Hyperbolic Tangent |
| 11 | Boundary Handling | Handling non-aligned sizes: `N != block_size * K` |
| 12 | 2D Strided Load | Row-major/Col-major layout conversion |
| 13 | Vector Mean | 1D Vector average |
| 14 | L2 Norm | `sqrt(sum(x^2))` |
| 15 | Clamp/Clip | Limit `x` in range `[min, max]` |
| 16 | Dropout (Fixed) | Apply dropout using an input mask |
| 17 | Dropout (RNG) | On-the-fly Philox RNG mask generation |
| 18 | Bitwise Ops | Bit shifting for low-bit quantization |
| 19 | Int8 Quant | `FP32 -> Int8` Scaling & Clipping |
| 20 | Dequantization | `Int8 -> FP32` Recovery |



## 🟡 Phase 2: Reductions & Parallel Algorithms (Day 21–40)
| Day | Topic | LeetGPU Task / Implementation Goal |
|:---:|:---|:---|
| 21 | Warp-level Sum | CUDA Warp Shuffle / Triton `tl.reduce` |
| 22 | Block-level Sum | Shared Memory-based reduction |
| 23 | Global Reduction | `atomicAdd` for cross-block global sum |
| 24 | Block-wise Max | Find max value within a block |
| 25 | Row-wise Sum | 2D Matrix horizontal sum |
| 26 | Col-wise Sum | 2D Matrix vertical sum |
| 27 | Row-wise Max | 2D Matrix horizontal max |
| 28 | Variance | Row-wise `Var(x)` |
| 29 | Std Deviation | Row-wise `sigma` calculation |
| 30 | Naive Softmax | Basic formula (Watch for Overflow) |
| 31 | Safe Softmax | Numerical stability via Max subtraction |
| 32 | Online Softmax (Max)| Online algorithm: Max tracking loop |
| 33 | Online Softmax (Final)| 1-pass Online Softmax integration |
| 34 | Log-Softmax | `log(softmax(x))` |
| 35 | Inclusive Scan | Kogge-Stone or Hillis-Steele Prefix Sum |
| 36 | Exclusive Scan | Prefix sum excluding current element |
| 37 | Top-k (k=1) | Optimized `argmax` search |
| 38 | Histogram | Counting value distributions (Atomic) |
| 39 | Min-Max Normalizer | `x_norm = (x - min) / (max - min)` |
| 40 | Dot Product | Vector-Vector inner product reduction |



## 🟠 Phase 3: Normalization & Memory Layout (Day 41–60)
| Day | Topic | LeetGPU Task / Implementation Goal |
|:---:|:---|:---|
| 41 | LayerNorm (FWD) | Normalization based on `mu` and `sigma` |
| 42 | RMSNorm (FWD) | Root Mean Square Norm with weight `gamma` |
| 43 | BatchNorm (FWD) | Channel-wise stats normalization |
| 44 | GroupNorm (FWD) | Group-wise normalization |
| 45 | InstanceNorm (FWD)| Instance-wise normalization |
| 46 | LayerNorm (BWD) | Input gradient `dx` calculation |
| 47 | LayerNorm (Weight BWD)| `d_gamma`, `d_beta` weight gradient calculation |
| 48 | RMSNorm (BWD) | RMSNorm Backward kernel |
| 49 | Naive Transpose | Basic matrix transpose |
| 50 | Tiled Transpose | Shared Memory bank conflict resolution |
| 51 | Padding | Padding arbitrary matrices for Tiled ops |
| 52 | Unpadding | Removing padding and extracting data |
| 53 | Concatenation | Joining two tensors along a specific dimension |
| 54 | Chunk / Split | Equal division of tensors |
| 55 | Gather | Data collection via Index Tensor |
| 56 | Scatter | Data distribution via Index Tensor |
| 57 | Diagonal Extract | Vectorizing matrix diagonal elements |
| 58 | Triu / Tril | Upper/Lower triangular masking |
| 59 | Roll | N-dimensional circular shift |
| 60 | Fused Norm + Act | LayerNorm + GeLU integrated kernel |

## 🔴 Phase 4: Matrix Multiplication (GEMM) (Day 61–80)
| Day | Topic | LeetGPU Task / Implementation Goal |
|:---:|:---|:---|
| 61 | Naive MatMul | Basic triple-loop matrix multiplication |
| 62 | Tiled MatMul | Shared Memory Tiling (Basic) |
| 63 | Vectorized GEMM | CUDA `float4` / Triton vector load optimization |
| 64 | Rectangular GEMM | Handling `M, N, K` variable sizes |
| 65 | Batched GEMM (BMM)| Batch matrix-matrix multiplication |
| 66 | FP16 GEMM | Half-precision acceleration |
| 67 | BF16 GEMM | Bfloat16 support |
| 68 | TF32 GEMM | TensorFloat-32 acceleration |
| 69 | Triton Autotuning | `Config(num_warps, block_size)` optimization |
| 70 | GEMM + Bias | `C = AB + bias` Fusion |
| 71 | GEMM + Activation | `C = ReLU(AB)` Fusion |
| 72 | GEMM + Residual | `C = AB + D` Fusion |
| 73 | GEMM BWD (dA) | Weight gradient for linear layer |
| 74 | GEMM BWD (dB) | Input gradient for linear layer |
| 75 | Grouped GEMM | Handling different matrix sizes in one kernel |
| 76 | GEMV (Row-major) | Matrix-Vector multiplication |
| 77 | GEMV (Col-major) | Column-major Matrix-Vector multiplication |
| 78 | L2 Cache Blocking | Tiling order optimization for L2 hit rate |
| 79 | Split-K GEMM | Parallelizing along the K-dimension |
| 80 | W8A16 MatMul | Int8 Weight & FP16 Activation product |



## 🟣 Phase 5: LLM & SOTA Specialized Kernels (Day 81–100)
| Day | Topic | LeetGPU Task / Implementation Goal |
|:---:|:---|:---|
| 81 | RoPE (FWD) | Rotary Positional Embedding Forward |
| 82 | RoPE (BWD) | Rotary Positional Embedding Backward |
| 83 | Sinusoidal Pos | Standard Transformer Position Encoding |
| 84 | Scaled Dot-Product| `(Q * K^T) / sqrt(d_k)` Score calculation |
| 85 | Attention Masking | Padding & Causal Masking addition |
| 86 | Softmax Fusion | Attention-specific masked Softmax |
| 87 | Self-Attention FWD | Integrated Q, K, V processing |
| 88 | Cross-Attention FWD | Encoder-Decoder Attention structure |
| 89 | KV Cache Write | Updating cache with new tokens |
| 90 | KV Cache Read | Paged load & RoPE integration |
| 91 | FlashAttn Tiling | Block-wise score loop structure |
| 92 | FlashAttn Online | Online Softmax block update logic |
| 93 | FlashAttn (FWD) | Full Forward Pass integration |
| 94 | FlashAttn (BWD-dQ) | Attention Backward `dQ` calculation |
| 95 | FlashAttn (BWD-dK/V)| Attention Backward `dK`, `dV` calculation |
| 96 | Group Query Attn | GQA optimization |
| 97 | Paged Attention | VLLM-style non-contiguous cache access |
| 98 | MoE Gating | Top-k Softmax for Expert selection |
| 99 | MoE Dispatch | Data routing to experts kernel |
| 100 | DeepSeek MoE Fused| **Fused MoE Grouped GEMM (Router + Expert integration)** |



#⚠️
This repository is a personal learning project (100-day challenge).
Code may be incomplete or experimental.
