import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(x_ptr, y_ptr, input_row_stride, output_row_stride, n_row, n_col, BLOCK: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in range(row_start, n_row, row_step, num_stages = num_stages):
        row_start_ptr = x_ptr + row_idx * input_row_stride
        col_offset = tl.arange(0, BLOCK)
        input_ptrs = row_start_ptr + col_offset
        mask = col_offset < n_col
        
        x = tl.load(input_ptrs, mask=mask)
        max_val = tl.max(x)
        exp_x = tl.exp(x - max_val)
        sum_exp_x = tl.sum(exp_x)
        y = exp_x / sum_exp_x

        tl.store(y_ptr + row_idx * output_row_stride + col_offset, y, mask=mask)
        
def fused_softmax_triton(x: torch.Tensor, y: torch.Tensor, input_row_stride: int, output_row_stride: int, n_row: int, n_col: int, BLOCK: int, num_stages: int) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == y.shape

    grid = (triton.cdiv(n_row, BLOCK),)
    fused_softmax_kernel[grid](x, y, input_row_stride, output_row_stride, n_row, n_col, BLOCK=BLOCK, num_stages=num_stages)
    return y

if __name__ == "__main__":
    n_row = 1024
    n_col = 1024
    BLOCK = 1024
    num_stages = 1
    x = torch.rand(n_row, n_col, device="cuda", dtype=torch.float32) * 100.0
    y = torch.rand(n_row, n_col, device="cuda", dtype=torch.float32) * 100.0
    input_row_stride = n_col
    output_row_stride = n_col
    fused_softmax_triton(x, y, input_row_stride, output_row_stride, n_row, n_col, BLOCK=BLOCK, num_stages=num_stages)
    ref = torch.softmax(x, dim=1)
    max_error = (y - ref).abs().max().item()
    print("Max error:", max_error)