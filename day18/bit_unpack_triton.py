import torch
import triton
import triton.language as tl

@triton.jit
def bit_unpack_kernel(vec_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # 각 프로그램이 처리할 uint32 인덱스
    vec_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vec_mask = vec_offsets < n
    
    # uint32 값 로드
    vec_data = tl.load(vec_ptr + vec_offsets, mask=vec_mask, other=0)
    
    # 4개의 uint8 추출
    byte0 = (vec_data >> 0) & 0xFF
    byte1 = (vec_data >> 8) & 0xFF
    byte2 = (vec_data >> 16) & 0xFF
    byte3 = (vec_data >> 24) & 0xFF
    
    # 결과에 저장 (각 uint32 → 4개의 uint8)
    base_offset = vec_offsets * 4
    tl.store(result_ptr + base_offset + 0, byte0, mask=vec_mask)
    tl.store(result_ptr + base_offset + 1, byte1, mask=vec_mask)
    tl.store(result_ptr + base_offset + 2, byte2, mask=vec_mask)
    tl.store(result_ptr + base_offset + 3, byte3, mask=vec_mask)

def bit_unpack_triton(vec: torch.Tensor, n: int) -> torch.Tensor:
    assert vec.is_cuda
    assert vec.dtype == torch.uint32
    assert vec.numel() == n

    result = torch.empty((n * 4,), device="cuda", dtype=torch.uint8)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    bit_unpack_kernel[grid](vec, result, n, BLOCK_SIZE)
    return result

if __name__ == "__main__":
    n = 1024
    vec = torch.randint(0, 2**32, (n,), device="cuda", dtype=torch.uint32)
    
    # CPU에서 레퍼런스 계산 (PyTorch의 uint32 CUDA 비트 연산 미지원)
    vec_cpu = vec.cpu().numpy()
    ref = torch.empty((n * 4,), dtype=torch.uint8)
    for i in range(n):
        ref[i * 4] = (vec_cpu[i] >> 0) & 0xFF
        ref[i * 4 + 1] = (vec_cpu[i] >> 8) & 0xFF
        ref[i * 4 + 2] = (vec_cpu[i] >> 16) & 0xFF
        ref[i * 4 + 3] = (vec_cpu[i] >> 24) & 0xFF
    ref = ref.cuda()
    
    # Triton 커널 실행
    result = bit_unpack_triton(vec, n)
    
    # 검증
    max_error = (result.int() - ref.int()).abs().max().item()
    print(f"Max error: {max_error}")
    print(f"Results match: {torch.equal(result, ref)}")