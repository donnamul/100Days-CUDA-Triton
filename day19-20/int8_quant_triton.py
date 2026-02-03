import torch
import triton
import triton.language as tl

@triton.jit
def int8_quant_kernel(
    x_ptr,
    y_ptr,
    scale,
    zero_point,
    N,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    q_fp = x / scale + zero_point
    q_fp = tl.math.floor(q_fp + 0.5)
    q_fp = tl.clamp(q_fp, -128.0, 127.0)
    y = q_fp.to(tl.int8)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def int8_dequant_kernel(
    x_ptr,
    y_ptr,
    scale,
    zero_point,
    N,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    q = tl.load(x_ptr + offsets, mask=mask, other=0)

    q_float = q.to(tl.float32)
    x = (q_float - zero_point) * scale
    tl.store(y_ptr + offsets, x, mask=mask)


def int8_quant(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    assert x.is_cuda
    assert x.dtype == torch.float32
    x_min = x.min().item()
    x_max = x.max().item()
    scale = (x_max - x_min) / 255.0
    if scale == 0:
        scale = 1.0
    zero_point = int(round(-x_min / scale - 128))
    zero_point = max(-128, min(127, zero_point))
    y = torch.empty_like(x, dtype=torch.int8)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    int8_quant_kernel[grid](
        x, y,
        scale, float(zero_point),
        N, BLOCK_SIZE
    )
    
    return y, scale, zero_point


def int8_dequant(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    assert q.is_cuda
    assert q.dtype == torch.int8
    y = torch.empty_like(q, dtype=torch.float32)
    N = q.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    int8_dequant_kernel[grid](
        q, y,
        scale, float(zero_point),
        N, BLOCK_SIZE
    )
    
    return y


if __name__ == "__main__":
    print("=" * 50)
    print("INT8 Quantization Test")
    print("=" * 50)
    
    # Create test tensor
    x = torch.randn(10240, device="cuda", dtype=torch.float32) * 100
    
    print(f"\nOriginal tensor:")
    print(f"  Shape: {x.shape}")
    print(f"  Dtype: {x.dtype}")
    print(f"  Min: {x.min().item():.4f}")
    print(f"  Max: {x.max().item():.4f}")
    print(f"  Mean: {x.mean().item():.4f}")
    
    # Quantize
    q, scale, zero_point = int8_quant(x)
    
    print(f"\nQuantized tensor:")
    print(f"  Shape: {q.shape}")
    print(f"  Dtype: {q.dtype}")
    print(f"  Scale: {scale:.6f}")
    print(f"  Zero point: {zero_point}")
    print(f"  Min: {q.min().item()}")
    print(f"  Max: {q.max().item()}")
    
    # Dequantize
    x_dequant = int8_dequant(q, scale, zero_point)
    
    print(f"\nDequantized tensor:")
    print(f"  Shape: {x_dequant.shape}")
    print(f"  Dtype: {x_dequant.dtype}")
    print(f"  Min: {x_dequant.min().item():.4f}")
    print(f"  Max: {x_dequant.max().item():.4f}")
    print(f"  Mean: {x_dequant.mean().item():.4f}")
    
    # Calculate error
    error = torch.abs(x - x_dequant).mean().item()
    relative_error = (error / torch.abs(x).mean().item()) * 100
    
    print(f"\nQuantization Error:")
    print(f"  Absolute error: {error:.6f}")
    print(f"  Relative error: {relative_error:.4f}%")
    
    # Verify correctness with PyTorch's quantization
    print(f"\n" + "=" * 50)
    print("Verification with PyTorch quantization")
    print("=" * 50)
    
    # PyTorch quantization (for comparison)
    q_pytorch = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    x_dequant_pytorch = q_pytorch.dequantize()
    
    pytorch_error = torch.abs(x - x_dequant_pytorch).mean().item()
    triton_error = torch.abs(x - x_dequant).mean().item()
    
    print(f"\nPyTorch error: {pytorch_error:.6f}")
    print(f"Triton error: {triton_error:.6f}")
    print(f"Match: {torch.allclose(x_dequant, x_dequant_pytorch, rtol=1e-3, atol=1e-3)}")