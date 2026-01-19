import torch
import triton
import triton.language as tl

H = 64
W = 128

@triton.jit
def transpose2d_strided_kernel(
    x_ptr, y_ptr,
    H: tl.constexpr, W: tl.constexpr,
    x_s0, x_s1,
    y_s0, y_s1,
    BM: tl.constexpr, BN: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    i = pid_m * BM + tl.arange(0, BM)[:, None]
    j = pid_n * BN + tl.arange(0, BN)[None, :]

    mask = (i < H) & (j < W)

    x = tl.load(x_ptr + i * x_s0 + j * x_s1, mask=mask, other=0.0)

    tl.store(y_ptr + j * y_s0 + i * y_s1, x, mask=mask)


def transpose_triton(x: torch.Tensor, BM: int = 32, BN: int = 32) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    assert x.ndim == 2

    H, W = x.shape
    y = torch.empty((W, H), device=x.device, dtype=x.dtype)

    x_s0, x_s1 = x.stride()
    y_s0, y_s1 = y.stride()

    grid = (triton.cdiv(H, BM), triton.cdiv(W, BN))
    transpose2d_strided_kernel[grid](
        x, y,
        H=H, W=W,
        x_s0=x_s0, x_s1=x_s1,
        y_s0=y_s0, y_s1=y_s1,
        BM=BM, BN=BN,
        num_warps=4
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    x = (torch.rand((H, W), device="cuda", dtype=torch.float32) * 20.0) - 10.0

    ref = x.t().contiguous()

    y = transpose_triton(x, BM=32, BN=32)

    max_error = (y - ref).abs().max().item()
    print("Max error:", max_error)
