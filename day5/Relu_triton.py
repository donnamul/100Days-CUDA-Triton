import torch
import triton
import triton.language as tl

NUM_DATA = 1024

@triton.jit
def Relu_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(x_ptr + offsets, y, mask=mask)

def Relu_triton(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    # assert x.numel() == NUM_DATA

    n = x.numel()
    grid = (triton.cdiv(n, block),)
    Relu_kernel[grid](x, n, BLOCK=block)
    return x

if __name__ == "__main__":
    x = torch.rand(NUM_DATA, device="cuda", dtype=torch.float32) * 100.0
    ref = torch.relu(x)

    Relu_triton(x, block=1024)

    max_error = (x - ref).abs().max().item()
    print("Max error:", max_error)