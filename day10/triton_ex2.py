"""
Triton Matrix Multiplication (GEMM) with Autotuning

이 파일은 Triton을 사용하여 고성능 행렬 곱셈(GEMM) 커널을 구현하고,
autotuning을 통해 최적의 성능을 자동으로 찾는 예제입니다.

주요 기능:
- FP16 및 FP8 입력 지원
- Autotuning을 통한 자동 최적화
- cuBLAS와의 성능 비교 벤치마크
- Leaky ReLU 활성화 함수 fusion 지원
"""

import torch
import triton
import triton.language as tl

# FP8 지원 여부 확인 (전역 변수로 사용)
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")


# ============================================================================
# Autotune Configuration
# ============================================================================

def get_autotune_config():
    """
    Autotuning을 위한 다양한 설정(config) 목록을 반환합니다.
    
    각 Config는 다음을 포함합니다:
    - BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: 타일 크기
    - GROUP_SIZE_M: L2 캐시 최적화를 위한 그룹 크기
    - num_stages: 파이프라인 스테이지 수
    - num_warps: 워프 수 (CUDA 워프 = 32 스레드)
    
    Triton은 이 설정들을 모두 시도하고 가장 빠른 것을 선택합니다.
    """
    return [
        # FP16 입력에 최적화된 설정들
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # FP8 입력에 최적화된 설정들 (더 큰 BLOCK_SIZE_K 사용)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


# ============================================================================
# Matrix Multiplication Kernel
# ============================================================================

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],  # 이 값들이 변경되면 autotune 재실행
)
@triton.jit
def matmul_kernel(
        # 포인터: 행렬의 첫 번째 요소를 가리킴
        a_ptr, b_ptr, c_ptr,
        # 행렬 차원: A는 (M, K), B는 (K, N), C는 (M, N)
        M, N, K,
        # Stride: 각 차원에서 1칸 이동할 때 포인터가 증가하는 바이트 수
        # stride_am: A의 행 방향 stride, stride_ak: A의 열 방향 stride
        stride_am, stride_ak,
        stride_bk, stride_bn,  # B의 stride
        stride_cm, stride_cn,  # C의 stride
        # 메타파라미터: 컴파일 타임 상수 (autotune에 의해 자동 선택됨)
        BLOCK_SIZE_M: tl.constexpr,  # M 방향 타일 크기
        BLOCK_SIZE_N: tl.constexpr,  # N 방향 타일 크기
        BLOCK_SIZE_K: tl.constexpr,  # K 방향 타일 크기
        GROUP_SIZE_M: tl.constexpr,  # L2 캐시 최적화를 위한 그룹 크기
        ACTIVATION: tl.constexpr     # 활성화 함수 (예: "leaky_relu")
):
    """
    행렬 곱셈 커널: C = A @ B
    
    이 커널은 타일링(tiling) 기법을 사용하여:
    1. 큰 행렬을 작은 블록으로 나눔
    2. 각 블록을 병렬로 계산
    3. L2 캐시 재사용을 최적화 (GROUP_SIZE_M 사용)
    4. FP32로 누적하여 정확도 향상
    """
    # ========================================================================
    # 1. Program ID를 블록 좌표로 변환 (L2 캐시 최적화)
    # ========================================================================
    # 각 program은 C 행렬의 하나의 블록을 담당합니다.
    # GROUP_SIZE_M을 사용하여 인접한 M 블록들을 그룹화하여 L2 캐시 재사용을 향상시킵니다.
    
    pid = tl.program_id(axis=0)  # 1D program ID
    
    # 각 차원의 program 수 계산
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M 방향 블록 수
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N 방향 블록 수
    
    # 그룹화된 인덱싱 (L2 캐시 최적화)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ========================================================================
    # 2. 정수 경계 가정 (컴파일러 최적화를 위한 힌트)
    # ========================================================================
    # 주석: 일부 Triton 버전에서는 tl.assume이 지원되지 않을 수 있습니다.
    # 이 가정들은 선택적 최적화 힌트이며, 없어도 기능에는 문제가 없습니다.
    # 필요시 다음 코드의 주석을 해제하세요:
    # tl.assume(pid_m >= 0)
    # tl.assume(pid_n >= 0)
    # tl.assume(stride_am > 0)
    # tl.assume(stride_ak > 0)
    # tl.assume(stride_bn > 0)
    # tl.assume(stride_bk > 0)
    # tl.assume(stride_cm > 0)
    # tl.assume(stride_cn > 0)

    # ========================================================================
    # 3. A와 B 행렬의 포인터 초기화
    # ========================================================================
    # 각 program은 [BLOCK_SIZE_M, BLOCK_SIZE_K] 크기의 A 블록과
    # [BLOCK_SIZE_K, BLOCK_SIZE_N] 크기의 B 블록을 처리합니다.
    
    # 오프셋 계산: 각 블록의 시작 위치
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 포인터 계산: broadcasting을 사용하여 2D 포인터 그리드 생성
    # a_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_K] 크기의 포인터 배열
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N] 크기의 포인터 배열
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ========================================================================
    # 4. 행렬 곱셈 계산 (K 차원을 따라 누적)
    # ========================================================================
    # FP32로 누적하여 정확도를 높입니다 (FP16보다 정밀함)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # K 차원을 BLOCK_SIZE_K 크기의 청크로 나누어 처리
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # A와 B의 다음 블록 로드 (경계 체크 포함)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 행렬 곱셈 누적: accumulator += a @ b
        accumulator = tl.dot(a, b, accumulator)
        
        # 다음 K 블록으로 포인터 이동
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # ========================================================================
    # 5. 활성화 함수 적용 (선택적)
    # ========================================================================
    # FP32 상태에서 활성화 함수를 적용하면 정확도가 더 좋습니다.
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    
    # FP16로 변환하여 저장
    c = accumulator.to(tl.float16)

    # ========================================================================
    # 6. 결과를 C 행렬에 저장
    # ========================================================================
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# Activation Functions
# ============================================================================

@triton.jit
def leaky_relu(x):
    """
    Leaky ReLU 활성화 함수: f(x) = max(0.01x, x)
    
    Triton 커널 내에서 사용할 수 있도록 @triton.jit으로 데코레이트됨.
    """
    return tl.where(x >= 0, x, 0.01 * x)


# ============================================================================
# Wrapper Function
# ============================================================================

def matmul(a, b, activation=""):
    """
    행렬 곱셈 래퍼 함수
    
    Args:
        a: torch.Tensor, shape (M, K)
        b: torch.Tensor, shape (K, N)
        activation: str, 활성화 함수 이름 (예: "leaky_relu")
    
    Returns:
        torch.Tensor, shape (M, N)
    
    이 함수는:
    1. 입력 검증
    2. 출력 텐서 할당
    3. 커널 실행 (autotune이 최적 설정 자동 선택)
    """
    # 입력 검증
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    # 차원 추출
    M, K = a.shape
    K, N = b.shape
    
    # 출력 텐서 할당
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 그리드 크기 계산 (람다 함수 사용하여 autotune의 메타파라미터에 접근)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    # 커널 실행 (autotune이 자동으로 최적 설정 선택)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  # A의 stride
        b.stride(0), b.stride(1),  # B의 stride
        c.stride(0), c.stride(1),  # C의 stride
        ACTIVATION=activation
    )
    return c


# ============================================================================
# Unit Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Triton Matrix Multiplication - Unit Tests")
    print("=" * 70)
    
    # ========================================================================
    # FP16 입력 테스트
    # ========================================================================
    print("\n[Test 1] FP16 Input Test")
    print("-" * 70)
    torch.manual_seed(0)
    a = torch.rand((512, 512), device="cuda", dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device="cuda", dtype=torch.float16) - 0.5
    
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    
    print(f"Triton output shape: {triton_output.shape}")
    print(f"Torch output shape: {torch_output.shape}")
    
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match!")
    else:
        max_error = (triton_output - torch_output).abs().max().item()
        print(f"❌ Triton and Torch differ (max error: {max_error})")
    
    # ========================================================================
    # FP8 입력 테스트 (지원되는 경우)
    # ========================================================================
    if TORCH_HAS_FP8:
        print("\n[Test 2] FP8 Input Test")
        print("-" * 70)
        torch.manual_seed(0)
        a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        a = a.to(torch.float8_e5m2)
        # 효율성을 위해 B를 전치
        b = b.T
        b = b.to(torch.float8_e5m2)
        
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        
        print(f"Triton output shape: {triton_output.shape}")
        print(f"Torch output shape: {torch_output.shape}")
        
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print("✅ Triton and Torch match!")
        else:
            max_error = (triton_output - torch_output).abs().max().item()
            print(f"❌ Triton and Torch differ (max error: {max_error})")
    else:
        print("\n[Test 2] FP8 Input Test")
        print("-" * 70)
        print("⚠️  FP8 not supported in this PyTorch version. Skipping FP8 test.")


# ============================================================================
# Performance Benchmark
# ============================================================================

def setup_benchmark():
    """
    벤치마크 설정을 구성합니다.
    cuBLAS와 Triton의 성능을 비교합니다.
    """
    ref_lib = 'cuBLAS'
    
    configs = []
    for fp8_inputs in [False, True]:
        # FP8이 지원되지 않으면 FP8 벤치마크 건너뛰기
        if fp8_inputs and not TORCH_HAS_FP8:
            continue
        
        configs.append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],  # x축 변수
                x_vals=[128 * i for i in range(2, 33)],  # 256부터 4096까지
                line_arg="provider",  # 각 라인을 구분하는 인자
                # FP8의 경우 cuBLAS 비교 불가 (torch.matmul이 FP8 미지원)
                line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],
                line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",  # Tera Floating Point Operations Per Second
                plot_name="matmul-performance-" + ("fp16" if not fp8_inputs else "fp8"),
                args={"fp8_inputs": fp8_inputs},
            ))
    return configs


@triton.testing.perf_report(setup_benchmark())
def benchmark(M, N, K, provider, fp8_inputs):
    """
    성능 벤치마크 함수
    
    각 (M, N, K) 조합에 대해:
    - cuBLAS (torch.matmul) 성능 측정
    - Triton 커널 성능 측정
    - TFLOPS로 변환하여 반환
    """
    # 테스트 데이터 생성
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    # FP8 변환 (요청된 경우)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    
    # 성능 측정
    quantiles = [0.5, 0.2, 0.8]  # 중앙값, 최소값(20%), 최대값(80%)
    
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    
    # TFLOPS 계산: 2 * M * N * K (행렬 곱셈 연산 수) / 시간
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# 벤치마크 실행 (주석 해제하여 실행)
# benchmark.run(show_plots=True, print_data=True)
