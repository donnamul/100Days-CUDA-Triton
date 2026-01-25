import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    X_ptrs = X + pid * stride
    Y_ptrs = Y + pid * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X_ptrs + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_ptrs + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
        # Write mean / rstd
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X_ptrs + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y_ptrs + cols, y, mask=mask)

def layer_norm_forward_triton(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float = 1e-5, block: int = 1024) -> torch.Tensor:
    assert X.is_cuda and W.is_cuda and B.is_cuda
    assert X.dtype == torch.float32 and W.dtype == torch.float32 and B.dtype == torch.float32

    N = X.shape[1]
    Y = torch.empty_like(X)
    Mean = torch.empty_like(X[:, :1])
    Rstd = torch.empty_like(X[:, :1])
    stride = X.stride(0)
    grid = (X.shape[0],)
    _layer_norm_fwd_fused[grid](X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE=block)
    return Y

if __name__ == "__main__":
    X = torch.rand(1024, 1024, device="cuda", dtype=torch.float32)
    W = torch.rand(1024, device="cuda", dtype=torch.float32)
    B = torch.rand(1024, device="cuda", dtype=torch.float32)
    eps = 1e-5
    block = 1024
    Y = layer_norm_forward_triton(X, W, B, eps, block)
    ref = torch.nn.functional.layer_norm(X, (1024,), W, B, eps=eps)
    max_error = (Y - ref).abs().max().item()
    print("Max error:", max_error)
    print("Results match:", torch.allclose(Y, ref, rtol=1e-5, atol=1e-6))