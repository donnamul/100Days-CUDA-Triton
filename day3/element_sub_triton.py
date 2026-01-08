import torch
import triton
import triton.language as tl

@triton.jit
def element_sub_kernel(a_ptr, b_ptr, c_ptr,
                         M, N,
                         stride_am, stride_an,
                         stride_bm, stride_bn,
                         stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cols

    # 2D pointer arithmetic
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    tl.store(c_ptrs, a - b, mask=mask)


def element_sub_triton(a: torch.Tensor, b: torch.Tensor, BLOCK_M: int, BLOCK_N: int) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.shape == b.shape

    M, N = a.shape
    c = torch.empty_like(a)
    stride_am, stride_an = a.stride()
    stride_bm, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    element_sub_kernel[grid](
        a, b, c,
        M, N,
        stride_am, stride_an,
        stride_bm, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return c


if __name__ == "__main__":
    shape = (1024, 1024)
    a = torch.rand(shape, device="cuda", dtype=torch.float32) * 100.0
    b = torch.rand(shape, device="cuda", dtype=torch.float32) * 100.0

    c = element_sub_triton(a, b, BLOCK_M = 32, BLOCK_N = 32)
    ref = a-b
    max_error = (c - ref).abs().max().item()
    print("Max error:", max_error)