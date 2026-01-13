import torch
import triton
import triton.language as tl

NUM_DATA = 1024

@triton.jit
def sigmoid_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sigmoid(x)
    tl.store(x_ptr + offsets, y, mask=mask)

def sigmoid_triton(x: torch.Tensor,  block: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    # assert x.numel() == NUM_DATA

    n = x.numel()
    grid = (triton.cdiv(n, block),)
    sigmoid_kernel[grid](x, n, BLOCK=block)
    return x

if __name__ == "__main__":
    x = torch.rand(NUM_DATA, device="cuda", dtype=torch.float32) * 100.0
    ref = torch.sigmoid(x)

    sigmoid_triton(x, block=1024)

    is_close = torch.allclose(x, ref, rtol=1e-5, atol=1e-8)
    max_error = (x - ref).abs().max().item()
    print("Max error:", max_error)
    print("Results match:", is_close)