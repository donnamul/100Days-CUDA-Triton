

## 1. `tl.where` vs `torch.where`

* **`torch.where`**

  * PyTorch / Python 레벨
  * `torch.Tensor` 대상으로 동작
  * 모델 코드, reference 계산에 사용

  ```python
  y = torch.where(x > 0, x, alpha * x)
  ```

* **`tl.where`**

  * Triton **커널 내부 전용**
  * `tl.tensor`(벡터) 대상 element-wise select
  * Python `if / else` 대신 사용

  ```python
  y = tl.where(x > 0, x, alpha * x)
  ```

> Triton 커널에서는 Python 조건문(`if`) 사용 불가 → 반드시 `tl.where`

---

## 2. `tl.constexpr`를 쓰는 이유 & 써야 하는 상황

* `tl.constexpr` = **컴파일 타임 상수**
* 값이 바뀌면 **커널이 새로 specialize(JIT 컴파일)** 됨
* 컴파일러가 미리 알면 **성능 최적화 가능**

### 써야 하는 경우 ✅

* 커널 구조를 바꾸는 값

  * `BLOCK`
  * `num_warps`
  * `num_stages`

### 보통 쓰지 않는 경우 ❌

* 입력마다 바뀌는 값

  * `n_elements` (텐서 길이)
  * `alpha` (Leaky ReLU slope)

→ 이런 값은 **런타임 인자**로 넘기고 `mask` / 연산으로 처리

> tl.constexpr는 “컴파일 타임에 더 강한 최적화를 걸고 싶은 항목”에 쓰지만,
그 대가로 커널 specialization(갯수 증가)이 생긴다.
---
