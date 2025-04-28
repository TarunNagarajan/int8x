# INT8 Quantization Engine

## Project Overview
The **INT8 Quantization Engine** is a software project designed to efficiently quantize machine learning models from FP32 (floating-point 32-bit) to INT8 (8-bit integer) for optimized deployment on resource-constrained devices. This engine aims to bridge the gap between high-performance model inference and hardware limitations, especially in edge devices like smartphones and embedded systems. The project will cover key topics such as quantization theory, model compression, memory management, and distributed quantization techniques.

# Module Outline

This document describes the high-level components of the **Hardware-Aware Quantization & Mapping Engine**.

## 1. Quantization (src/quant/)
- **Quantizer**  
  - FP32 → INT8 conversion  
  - Scale & zero-point calculation (per-tensor & per-channel)  
  - Rounding modes & error metrics
- **LAQOptimizer**  
  - Loss-aware scale tuning post-quantization  

## 2. Fixed-Point Arithmetic (src/fixed/)
- **FixedPoint<T>**  
  - Q7, Q15, Q31 formats  
  - Basic ops: +, –, × (with saturation)  
  - Conversion to/from float  

## 3. Tensor Operations (src/tensor/)
- **Tensor**  
  - Shape, stride, memory management  
- **Element-wise Ops**  
  - add, mul, relu  
- **GEMM**  
  - Naïve & tiled INT8 matrix multiply  
  - NEON-accelerated inner loops  

## 4. Memory Management (src/alloc/)
- **PoolAllocator**  
  - Free-list with alignment support  
- **(later) Stack/Arena allocators**  

## 5. SIMD Kernels (src/simd/)
- **neon_quant**  
  - Vectorized quant/dequant routines  
- **neon_gemm**  
  - Inner-loop NEON kernels for GEMM  

## 6. Compute-in-Memory Simulator (src/sim/)
- **Simulator**  
  - Crossbar read/write API  
  - Energy & error modeling (pJ/op, noise, drift)  

## 7. Research Extensions (src/research/)
- **DynamicQuantController**  
- **ErrorCompensator**  
- **SmartMapper**  
- **Pruner**
