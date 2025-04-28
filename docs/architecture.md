# Architecture Overview

```mermaid
graph TB
  subgraph Quantization
    Q1[Quantizer]
    Q2[LAQOptimizer]
  end

  subgraph FixedPoint
    F1[FixedPoint<T>]
  end

  subgraph TensorOps
    T1[Tensor]
    T2[Element-wise Ops]
    T3[GEMM]
  end

  subgraph Memory
    M1[PoolAllocator]
  end

  subgraph SIMD
    S1[neon_quant]
    S2[neon_gemm]
  end

  subgraph Simulator
    C1[Crossbar Read/Write]
    C2[Energy & Error Model]
  end

  Q1 --> T1
  T1 --> T3
  T3 --> Simulator
  F1 -.-> Q1
  M1 --> T1
  S1 --> Q1
  S2 --> T3
