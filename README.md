# int8x

---

##  Module Breakdown

1. **Core Quantization Engine**  
   - FP32→INT8 conversion, scaling & zero‑point  
2. **Fixed‑Point Arithmetic & Precision Management**  
   - Q7/Q15/Q31 operations & conversions  
3. **Model Compression & Sparsity**  
   - Pruning, Quantization‑Aware Training (QAT)  
4. **Tensor Ops & GEMM**  
   - Element‑wise ops, INT8 General Matrix Multiply  
5. **Distributed Quantization**  
   - Scale quantization across threads/nodes  
6. **Transfer Learning with Quantization**  
   - Quantize & fine‑tune pre‑trained models  
7. **ML Framework Integration**  
   - PyTorch & TensorFlow Lite adapters  
8. **Performance Benchmarking & Validation**  
   - Latency, throughput, memory & accuracy metrics    

---

##  Phase 1 (Days 1–15) Roadmap

| Day  | Task                                                                   |
|------|------------------------------------------------------------------------|
| 1    | Project scoping; finalize name, goals, deliverables                    |
| 2    | Research C++ pointers, smart pointers, memory pitfalls                  |
| 3    | Study fixed‑point arithmetic (Q7/Q15/Q31)                              |
| 4    | Design FP32→INT8 conversion pseudocode & flowcharts                     |
| 5    | Research ARM Cortex‑A53 & NEON SIMD concepts                            |
| 6    | Setup dev environment (VSCode, CMake, GitHub repo init)                 |
| 7    | Outline modules in `docs/module‑outline.md`                             |
| 8    | Draft `README.md`, `.gitignore`; commit initial structure               |
| 9    | Initialize `devlog.md`, `research‑notes.md`; log Days 1–8                |
| 10   | Populate research notes on memory allocators                            |
| 11   | Populate research notes on fixed‑point theory                           |
| 12   | Sketch C++ API for quantizer & fixed‑point modules                      |
| 13   | Draft unit‑test plan in `tests/`                                         |
| 14   | Review & refine Phase 1 deliverables                                    |
| 15   | Finalize docs & plan Phase 2 milestones                                 |


## 🧑‍💻 Quick Start

```bash
git clone https://github.com/TarunNagarajan/int8x.git
cd int8x
mkdir build && cd build
cmake ..
make -j$(nproc)

