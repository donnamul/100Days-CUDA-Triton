import torch
import triton
import triton.language as tl

NUM_DATA = 1024 * 1024
@triton.jit
def clamp_kernel(x_ptr, min_val, max_val, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x < min_val, min_val, tl.where(x > max_val, max_val, x))
    tl.store(y_ptr + offsets, y, mask=mask)

def clamp_triton(x: torch.Tensor, min_val: float, max_val: float, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    

    n = x.numel()
    y = torch.empty_like(x)

    grid = (triton.cdiv(n, block),)
    clamp_kernel[grid](x, min_val, max_val, y, n, BLOCK=block)
    return y

if __name__ == "__main__":
    x = torch.rand(NUM_DATA, device="cuda", dtype=torch.float32) * 100.0
    min_val = 20.0
    max_val = 80.0

    y = clamp_triton(x, min_val, max_val, block=1024)
    ref = torch.clamp(x, min_val, max_val)
    max_error = (y - ref).abs().max().item()
    print("Max error:", max_error)
    print("Results match:", torch.allclose(y, ref, rtol=1e-5, atol=1e-8))