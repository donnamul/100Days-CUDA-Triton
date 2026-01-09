import torch
import triton
import triton.language as tl

@triton.jit
def fused_axpby_kernel(a, x_ptr, b, y_ptr, z_ptr,
                         M, N,
                         stride_xm, stride_xn,
                         stride_ym, stride_yn,
                         stride_zm, stride_zn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cols

    # 2D pointer arithmetic
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    tl.store(z_ptrs, a*x + b*y, mask=mask)

def fused_axpby_triton(a: float, x: torch.Tensor, b: float, y: torch.Tensor, BLOCK_M: int, BLOCK_N: int) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == y.shape

    M, N = x.shape
    z = torch.empty_like(x)
    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()
    stride_zm, stride_zn = z.stride()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_axpby_kernel[grid](
        a, x, b, y, z,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        stride_zm, stride_zn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return z

if __name__ == "__main__":
    shape = (1024, 1024)
    a = 2.0
    b = 3.0
    x = torch.rand(shape, device="cuda", dtype=torch.float32) * 100.0
    y = torch.rand(shape, device="cuda", dtype=torch.float32) * 100.0

    z = fused_axpby_triton(a, x, b, y, BLOCK_M = 32, BLOCK_N = 32)
    ref = a * x + b * y
    max_error = (z - ref).abs().max().item()
    print("Max error:", max_error)
