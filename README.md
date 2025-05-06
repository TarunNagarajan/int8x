# Quantization & Mapping Engine for CiM Accelerators  

## Days 1–40: Core Quantization, Fixed-Point, Tensor/GEMM & Basic Mapping

| Day | Task                                                                                                 |
|----:|------------------------------------------------------------------------------------------------------|
|  1  | Initialize repo & folder structure; install C++ toolchain, CMake & Git; set up your editor           |
|  2  | Draft `docs/module-outline.md`; sketch module flow: Quant → Fixed → Tensor → GEMM → Mapping → Sim     |
|  3  | Learn Fixed-Point (Q7/Q15/Q31); note formulas in `docs/fixed_point.md`                               |
|  4  | Code `FixedPoint` add/sub; write `tests/test_fixed_add_sub.cpp`                                      |
|  5  | Extend to mul + saturation; write `tests/test_fixed_mul.cpp`                                         |
|  6  | Read PTQ basics; draft `docs/quant_primer.md` (scale, zero-point, rounding)                          |
|  7  | Build scalar `Quantizer::quantize(float)→int8_t`; test in `tests/test_quant_scalar.cpp`              |
|  8  | Vectorize quant: `vector<float>→vector<int8_t>`; test & bench                                        |
|  9  | Create `Tensor` (shape + data pointer); test ctor in `tests/test_tensor_ctor.cpp`                    |
| 10  | Implement `Tensor::add()` and `Tensor::mul()`; test in `tests/test_tensor_ops.cpp`                    |
| 11  | Naïve INT8 GEMM (3-nested loops); test correctness on 2×2, 4×4 matrices                               |
| 12  | Benchmark GEMM vs FP32 (using `std::chrono`); log ms in `docs/gemm_bench.md`                         |
| 13  | Add simple 8×8 tiling to GEMM; test & benchmark tiling                                               |
| 14  | Integrate Quant→GEMM into a single pipeline; write `tests/test_pipeline.cpp`                         |
| 15  | Log end-to-end latency (ms/inference) to console or file                                             |
| 16  | Draft `docs/mapping_basics.md`: row-major vs blocked tensor layouts                                  |
| 17  | Implement blocked layout helper function; test for correct indexing                                  |
| 18  | Compare blocked vs row-major performance in your pipeline; record results                            |
| 19  | Add a toy CiM error model: inject Gaussian noise into accumulators in `Simulator.cpp`                |
| 20  | Simulate full pipeline with noise; measure & log accuracy drop                                       |
| 21  | Document mapping vs error trade-off in `docs/mapping_results.md`                                      |
| 22  | Stub a simple pool allocator; swap tensor allocations to call it; ensure build succeeds               |
| 23  | Flesh out `PoolAllocator` (free list + reset); write `tests/test_allocator.cpp`                       |
| 24  | Replace all `new`/`delete` in `Tensor` with your pool allocator; run full test suite                  |
| 25  | Add a cost model (cycles + bytes) per layer; log cost metrics during inference                        |
| 26  | Use cost model to choose optimal block sizes; document decisions in `docs/block_tuning.md`            |
| 27  | Cross-compile & run the pipeline on Raspberry Pi 3; record latency & memory in `docs/edge_benchmarks.md` |
| 28  | Cross-compile & run on Jetson Nano; record & compare performance metrics                               |
| 29  | Add a basic thread-pool; parallelize GEMM; write `tests/test_threadpool_gemm.cpp`                      |
| 30  | Benchmark single-threaded vs multi-threaded GEMM; log speedup                                         |
| 31  | Implement per-channel quantization; compare error vs per-tensor quant                                  |
| 32  | Test per-channel quant on TinyCNN; log results in `docs/quant_comparison.md`                          |
| 33  | Build a CLI (`main.cpp`) to run quant→gemm→map in one command                                         |
| 34  | Create `examples/run_demo.sh` script for end-to-end demo                                              |
| 35  | Learn profiling tools (`perf`, `valgrind`); profile your pipeline                                     |
| 36  | Identify top 2 hotspots; sketch micro-optimizations (e.g. loop unrolling, pointer arithmetic)          |
| 37  | Implement one hotspot optimization manually; re-benchmark                                              |
| 38  | Add a simple energy proxy (pJ = cycles × constant); log energy per inference                           |
| 39  | Plot accuracy vs energy trade-off in `docs/energy_accuracy.md`                                         |
| 40  | Draft `docs/thesis_outline.md` with sections: Motivation, Method, Results, Conclusion                  |
| **41** | **ONNX Runtime Execution Provider**: stub `src/onnx_ep/YourEngineEP.cpp`, register provider         |
| **42** | Implement model import & tensor conversion in EP; write `examples/onnx_ep_demo.cpp`                |
| **43** | Benchmark ONNX EP vs native pipeline on Pi3; log in `docs/onnx_ep_bench.md`                          |
| **44** | Optimize EP data binding & initialization time                                                     |
| **45** | Add support for quantized Conv2D in your EP; test correctness                                      |
| **46** | **MLIR Codegen Pass**: stub `src/mlir/QuantGemmDialect.cpp` & pass pipeline                         |
| **47** | Lower a simple quant+GEMM op to MLIR dialect; JIT-compile & run on-device                           |
| **48** | Benchmark JIT vs static: measure cold-start & hot-path overhead; document                           |
| **49** | Fuse quant+gemm lowering in MLIR pass; validate on single-layer model                               |
| **50** | Integrate MLIR codegen into CMake build; add `tests/test_mlir.cpp`                                  |
| **51** | **DVFS Controller**: implement `src/power/DvfsController.cpp` using `/sys/devices/.../cpufreq`       |
| **52** | Automate CPU-freq sweep script; record latency & power at 4 governors                               |
| **53** | Analyze freq vs accuracy/latency trade-off; summarize in `docs/dvfs_tradeoff.md`                    |
| **54** | Add dynamic freq adjustment during inference based on model stage                                    |
| **55** | **Autotuner for GEMM/Quant**: build simple grid search over tile sizes & quant scales               |
| **56** | Run autotuner on-device for 50 configurations; log best in `docs/autotune_results.md`                |
| **57** | Bake tuned parameters into default build; confirm speedup & stability                               |
| **58** | **Hardware-Aware NAS (HA-NAS)**: stub `src/ha_nas/NasController.cpp`, define search space            |
| **59** | Run HA-NAS loop (e.g. tile size, quant bits) on TinyCNN for 30 trials; collect metrics              |
| **60** | Select top-3 configs; integrate into `examples/ha_nas_demo.cpp` and benchmark                        |
| **61** | **Dynamic Quant Controller**: implement histogram-based scale selection at runtime                   |
| **62** | Compare dynamic vs static quant on TinyCNN; log accuracy & timing in `docs/dynamic_quant.md`         |
| **63** | **Error Compensator**: implement 1×1 conv correction layer; embed in pipeline                        |
| **64** | Train compensator offline on one layer; export weights; measure MSE reduction                        |
| **65** | Benchmark quant+comp vs quant alone end-to-end; record results                                       |
| **66** | **Low-Rank Weight Compression**: apply SVD to one dense layer; quantize & test                       |
| **67** | Benchmark low-rank+quant vs quant only (speed & error)                                               |
| **68** | **Mixed-Precision Quant**: allow per-layer FP16/INT8 choice; implement in `Quantizer`                |
| **69** | Test mixed-precision on a small CNN; plot accuracy vs latency in `docs/mixed_precision.md`           |
| **70** | **Spiking Quant Simulation**: discretize activations to spike trains; simulate simple MLP            |
| **71** | Measure spiking MLP latency vs int8 pipeline; record energy proxy                                    |
| **72** | **TFLite Delegate**: stub `src/tflite_delegate/Delegate.cpp`; register with TFLite C API             |
| **73** | Implement quantized Conv2D in delegate; test on a TFLite model                                        |
| **74** | Benchmark delegate vs ONNX EP vs native for the same model; summarize                                 |
| **75** | **Federated Quant Experiment**: simulate 5 clients sharing scale stats, measure convergence speed     |
| **76** | **Reinforcement-Learning Tuner**: train small agent to pick quant scales for minimal error            |
| **77** | Integrate RL tuner into pipeline; benchmark overhead vs static                                        |
| **78** | **RTOS Port**: compile engine under FreeRTOS (ARM M4 target); run pipeline task & log worst latency   |
| **79** | **Autotune+DVFS Combo**: run autotuner under different CPU frequencies; identify best pair            |
| **80** | **Ablation Study**: disable each advanced module one at a time; measure impact on speed & error       |
| **81** | Aggregate all advanced results; generate plots in `docs/advanced_summary.md`                           |
| **82** | Refine any code hotspots identified during advanced modules                                           |
| **83** | Ensure 100% unit-test coverage across all new code                                                    |
| **84** | Cross-compile final advanced build for x86_64, armhf, and M4; fix any portability issues              |
| **85** | Run full advanced pipeline on all platforms; collect unified metrics                                   |
| **86** | Write short “advanced modules” README section describing each novel component                          |
| **87** | Peer-review advanced modules with a mentor or colleague; log feedback                                   |
| **88** | Implement top 3 peer suggestions for robustness or clarity                                             |
| **89** | Final performance regressions; confirm no slowdowns                                                    |
| **90** | Tag `v1.0-research` with all advanced modules merged                                                   |
| **91–100** | **Buffer** for any spillover on advanced tasks, final micro-optimizations, last-minute bug fixes    |


## 🛠️ Project Structure Overview

| Folder/Module         | Description |
|------------------------|-------------|
| `src/quant/`           | Core quantization techniques and calibration logic |
| `src/fixed/`           | Fixed-point math, Qm.n arithmetic |
| `src/tensor/`          | Basic tensor layout, memory strides, and INT8 GEMM |
| `src/alloc/`           | Custom memory allocator and buffer management |
| `src/simd/`            | Optimized SIMD kernels |
| `examples/`            | Python/C++ hybrid test cases for inference |
| `benchmarks/`          | Benchmarking pipelines for real models |
| `tests/`               | Unit and integration tests for all modules |

