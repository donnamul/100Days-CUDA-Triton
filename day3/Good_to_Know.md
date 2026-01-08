---

# GPU Indexing Cheat Sheet

### (Iteration Space → Memory Address Mapping)

---

## 0. 핵심 원리 

> **Memory is always 1D.**
> **Execution (grid / program) defines the iteration space.**
> **Indexing maps iteration space → linear address via strides.**

---

## 1. Triton Program Indexing

### 1.1 1D Linear Indexing (Flat Tensor)

```python
pid = tl.program_id(0)                    # program index
offs = pid * BLOCK + tl.arange(0, BLOCK)  # vector offsets
ptrs = base + offs                        # linear pointers
mask = offs < N                           # boundary
```

**Meaning**

* Iteration space: **1D**
* Memory: **contiguous 1D**
* Equivalent to CUDA 1D thread indexing

---

### 1.2 2D Tiled Indexing (Row / Col + Stride)

```python
pid_m = tl.program_id(0)                 # tile row id
pid_n = tl.program_id(1)                 # tile col id

offs_m = pid_m * BM + tl.arange(0, BM)   # row indices (BM,)
offs_n = pid_n * BN + tl.arange(0, BN)   # col indices (BN,)

ptrs = base \
     + offs_m[:, None] * stride_m \
     + offs_n[None, :] * stride_n         # (BM, BN)

mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
```

**Meaning**

* Iteration space: **2D tiles**
* Memory: **1D linear with strides**
* `None` = **add dimension for broadcasting**
* Produces a **2D pointer grid**

---

### 🔑 Triton Mental Model

```text
tl.program_id  →  tile
tl.arange      →  vector lanes
broadcasting  →  implicit nested loops
```

---

## 2. CUDA Thread Indexing

### 2.1 1D Thread Indexing

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    ptr = base + idx;
}
```

**Meaning**

* Iteration space: **1D threads**
* Memory: **1D linear**
* Scalar version of Triton 1D

---

### 2.2 2D Thread Indexing (Row-Major)

```cpp
int j = blockIdx.x * blockDim.x + threadIdx.x; // col
int i = blockIdx.y * blockDim.y + threadIdx.y; // row

if (i < M && j < N) {
    int idx = i * ld + j;   // ld = leading dimension
    ptr = base + idx;
}
```

**Meaning**

* Iteration space: **2D threads**
* Memory: **1D linear via flattening**
* Explicit `(i, j) → linear index`

---

## 3. Triton ↔ CUDA 1:1 Mapping

| Concept            | Triton                    | CUDA                                  |
| ------------------ | ------------------------- | ------------------------------------- |
| Execution unit     | program                   | thread                                |
| Program / block ID | `tl.program_id()`         | `blockIdx.{x,y}`                      |
| Lane ID            | `tl.arange()`             | `threadIdx.x`                         |
| Tile               | program                   | thread block                          |
| 1D offset          | `pid*B + arange`          | `blockIdx.x*blockDim.x + threadIdx.x` |
| 2D coords          | `offs_m, offs_n`          | `i, j`                                |
| Stride address     | `i*stride_m + j*stride_n` | `i*ld + j`                            |
| Boundary           | `mask`                    | `if (cond)`                           |

---

## 4. Shape & Stride Intuition (중요)

```text
shape  = logical tensor view
stride = how to walk memory
```

Row-major `(M, N)`:

```python
stride_m = N
stride_n = 1
```

Column-major:

```python
stride_m = 1
stride_n = M
```

Same indexing code, **different layout**.

---

## 5. One-Line Summary

> **CUDA expresses iteration with scalar threads.**
> **Triton expresses iteration with vectorized tiles.**
> **Both map N-D iteration space to 1-D memory using strides.**

---