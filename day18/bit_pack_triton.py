import torch
import triton
import triton.language as tl

@triton.jit
def bit_pack_kernel(vec_ptr, result_ptr, n_result, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # 각 프로그램이 처리할 uint32 인덱스 (4개의 uint8 → 1개의 uint32)
    result_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    result_mask = result_offsets < n_result
    
    # 4개의 연속된 uint8 로드
    vec_base = result_offsets * 4
    vec_mask0 = vec_base < n_result * 4
    vec_mask1 = vec_base + 1 < n_result * 4
    vec_mask2 = vec_base + 2 < n_result * 4
    vec_mask3 = vec_base + 3 < n_result * 4
    byte0 = tl.load(vec_ptr + vec_base + 0, mask=vec_mask0, other=0).to(tl.uint32)
    byte1 = tl.load(vec_ptr + vec_base + 1, mask=vec_mask1, other=0).to(tl.uint32)
    byte2 = tl.load(vec_ptr + vec_base + 2, mask=vec_mask2, other=0).to(tl.uint32)
    byte3 = tl.load(vec_ptr + vec_base + 3, mask=vec_mask3, other=0).to(tl.uint32)
    
    # 4개의 uint8을 하나의 uint32로 패킹
    packed = (byte0 << 0) | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    
    # 결과 저장
    tl.store(result_ptr + result_offsets, packed, mask=result_mask)

def bit_pack_triton(vec: torch.Tensor, n: int) -> torch.Tensor:
    assert vec.is_cuda
    assert vec.dtype == torch.uint8
    assert vec.numel() == n * 4

    result = torch.empty((n,), device="cuda", dtype=torch.uint32)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    bit_pack_kernel[grid](vec, result, n, BLOCK_SIZE)
    return result

if __name__ == "__main__":
    n = 1024
    vec = torch.randint(0, 256, (n * 4,), device="cuda", dtype=torch.uint8)
    
    # CPU에서 레퍼런스 계산 (PyTorch의 uint32 CUDA 비트 연산 미지원)
    vec_cpu = vec.cpu().numpy()
    ref = torch.empty((n,), dtype=torch.uint32)
    for i in range(n):
        # 4개의 연속된 uint8을 하나의 uint32로 패킹
        byte0 = int(vec_cpu[i * 4 + 0])
        byte1 = int(vec_cpu[i * 4 + 1])
        byte2 = int(vec_cpu[i * 4 + 2])
        byte3 = int(vec_cpu[i * 4 + 3])
        ref[i] = (byte0 << 0) | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    ref = ref.cuda()
    
    # Triton 커널 실행
    result = bit_pack_triton(vec, n)
    
    # 검증
    matches = torch.equal(result, ref)
    print(f"Results match: {matches}")
    if not matches:
        diff = (result != ref).sum().item()
        print(f"Number of mismatches: {diff}")
        # 첫 몇 개 값 출력
        print(f"First 5 results: {result[:5]}")
        print(f"First 5 refs:    {ref[:5]}")