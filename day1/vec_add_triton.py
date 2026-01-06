import torch
import triton
import triton.language as tl

@triton.jit
def vec_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)

def vec_add_triton(a: torch.Tensor, b: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.numel() == b.numel()

    n = a.numel()
    c = torch.empty_like(a)

    grid = (triton.cdiv(n, block),)
    vec_add_kernel[grid](a, b, c, n, BLOCK=block)
    return c

def main():
    n = int(input("Enter the size of the vector: "))

    a = torch.rand(n, device="cuda", dtype=torch.float32) * 100.0
    b = torch.rand(n, device="cuda", dtype=torch.float32) * 100.0

    c = vec_add_triton(a, b, block=1024)

    ref = a + b
    max_error = (c - ref).abs().max().item()
    print("Max error:", max_error)

    # denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
    # max_err_pct = (((c - ref).abs() / denom) * 100.0).max().item()
    # print("Max error percentage:", max_err_pct, "%")

if __name__ == "__main__":
    main()
