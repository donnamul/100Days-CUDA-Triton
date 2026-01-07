import torch
import triton
import triton.language as tl

NUM_DATA = 1024

@triton.jit
def vec_scale_kernel(x_ptr, y_ptr, alpha, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.full((), alpha, dtype=tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, a * x, mask=mask)

def vec_scale_triton(x: torch.Tensor, alpha: float, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    # assert x.numel() == NUM_DATA

    n = x.numel()
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, block),)
    vec_scale_kernel[grid](x, y, alpha, n, BLOCK=block)
    return y

if __name__ == "__main__":
    x = torch.rand(NUM_DATA, device="cuda", dtype=torch.float32) * 100.0
    alpha = 2.0

    y = vec_scale_triton(x, alpha, block=1024)

    ref = alpha * x
    max_error = (y - ref).abs().max().item()
    print("Max error:", max_error)