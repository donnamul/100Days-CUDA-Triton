import torch

import triton
import triton.language as tl

NUM_DATA = 1024

@triton.jit
def vec_sum_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(x, axis=0)           # program(=block) 내부 reduction
    if tl.program_id(0) == pid:              # 그냥 명시적으로 한 번만
        tl.atomic_add(out_ptr, block_sum)


def vec_mean_triton(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32

    n = x.numel()
    out = torch.zeros(1, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(n, block),)
    vec_sum_kernel[grid](x, out, n, BLOCK=block)
    return out / n

if __name__ == "__main__":
    x = torch.rand(NUM_DATA, device="cuda", dtype=torch.float32) * 100.0
    ref = x.mean()
    out = vec_mean_triton(x, block=1024)
    print(f"Reference mean: {ref}")
    print(f"Triton mean: {out}")
    print(f"Error: {out - ref}")