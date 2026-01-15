import torch
import triton
import triton.language as tl

NUM_DATA = 1024

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def Gelu_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # GeLU tanh approximation:
    # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)  # sqrt(2/pi) constant
    x = 0.5 * x * (1.0 + tanh(inner))
    tl.store(x_ptr + offsets, x, mask=mask)

def Gelu_triton(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    # assert x.numel() == NUM_DATA

    n = x.numel()
    grid = (triton.cdiv(n, block),)
    Gelu_kernel[grid](x, n, BLOCK=block)
    return x

if __name__ == "__main__":
    x = torch.rand(NUM_DATA, device="cuda", dtype=torch.float32) * 100.0
    ref = torch.nn.functional.gelu(x)
    Gelu_triton(x, block=1024)

    max_error = (x - ref).abs().max().item()
    print("Max error:", max_error)