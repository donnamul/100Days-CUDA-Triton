import torch
import triton
import triton.language as tl

NUM_DATA = 1024

@triton.jit
def Leaky_Relu_kernel(x_ptr, alpha, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0.0, x, alpha * x)
    tl.store(x_ptr + offsets, y, mask=mask)

def LeakyRelu_triton(x: torch.Tensor, alpha: float,  block: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    # assert x.numel() == NUM_DATA

    n = x.numel()
    grid = (triton.cdiv(n, block),)
    Leaky_Relu_kernel[grid](x, alpha, n, BLOCK=block)
    return x

if __name__ == "__main__":
    x = torch.rand(NUM_DATA, device="cuda", dtype=torch.float32) * 100.0
    alpha = 0.01
    ref = torch.where(x > 0.0, x, alpha * x)

    LeakyRelu_triton(x, alpha, block=1024)

    max_error = (x - ref).abs().max().item()
    print("Max error:", max_error)