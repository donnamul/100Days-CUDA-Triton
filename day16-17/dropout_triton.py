# | 16 | Dropout (Fixed) | Apply dropout using an input mask |
# | 17 | Dropout (RNG) | On-the-fly Philox RNG mask generation |
import torch
import triton
import triton.language as tl

NUM_DATA = 1024

@triton.jit
def fixed_dropout_kernel(x_ptr, mask_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mask_vec = tl.load(mask_ptr + offsets, mask=mask)
    output = tl.where(mask_vec, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def rng_dropout_kernel(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def rng_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    rng_dropout_kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

def fixed_dropout(x, mask, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    fixed_dropout_kernel[grid](x, mask, output, n_elements, p, BLOCK_SIZE=1024)
    return output

if __name__ == "__main__":
    p = 0.5
    seed = 123
    x = torch.randn(NUM_DATA, device="cuda")
    
    # Fixed dropout 테스트
    mask_bool = torch.rand(NUM_DATA, device="cuda") > p
    mask = mask_bool.to(torch.int32)
    fixed_output = fixed_dropout(x, mask, p)
    fixed_ref = torch.where(mask_bool, x / (1 - p), 0.0)
    fixed_max_error = (fixed_output - fixed_ref).abs().max().item()
    
    print("=" * 60)
    print("Fixed Dropout Test")
    print("=" * 60)
    print(f"max error: {fixed_max_error}")
    print(f"input: {x[:10].tolist()}...")
    print(f"keep mask: {mask[:10].tolist()}...")
    print(f"output: {fixed_output[:10].tolist()}...")
    
    print("\n" + "=" * 60)
    print("RNG Dropout Seed Reproducibility Test")
    print("=" * 60)
    
    rng_output1 = rng_dropout(x, p, seed)
    rng_output2 = rng_dropout(x, p, seed)
    
    rng_output3 = rng_dropout(x, p, seed + 1)
    
    same_seed_diff = (rng_output1 - rng_output2).abs().max().item()
    diff_seed_diff = (rng_output1 - rng_output3).abs().max().item()
    
    print(f"Max difference with same seed (123): {same_seed_diff}")
    print(f"  → {'✓ Reproducible!' if same_seed_diff == 0 else '✗ Not reproducible'}")
    
    print(f"\nMax difference with different seeds (123 vs 124): {diff_seed_diff}")
    print(f"  → {'✓ Different outputs as expected' if diff_seed_diff > 0 else '✗ Unexpectedly identical outputs (issue)'}")
    
    print(f"\nRNG output 1 (seed=123): {rng_output1[:10].tolist()}...")
    print(f"RNG output 2 (seed=123): {rng_output2[:10].tolist()}...")
    print(f"RNG output 3 (seed=124): {rng_output3[:10].tolist()}...")