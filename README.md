# Hardware-Aware Quantization & Mapping Engine for CiM Accelerators  

| Day   | Task                                                                                                                   |
|-----: |------------------------------------------------------------------------------------------------------------------------|
| **1**   | Init repo (`README.md`, `.gitignore`, `CMakeLists.txt`); provision RPi3 (OS update, build-tools, SSH)                 |
| **2**   | Draft `docs/module-outline.md`; sketch architecture; start `docs/devlog.md`                                          |
| **3**   | Research C++ allocators; stub `src/alloc/PoolAllocator.h/.cpp`; write `tests/test_allocator.cpp`                     |
| **4**   | Complete PoolAllocator (free-list + alignment); bench vs `malloc`; profile cache                                     |
| **5**   | Read Sze Ch1–2; extract efficiency notes to `docs/efficiency_summary.md`; draft `docs/fixed_point_design.md`         |
| **6**   | Implement `FixedPoint<T>` (ctor + `toFloat()`); add `operator+/-`; unit-test add/sub                                 |
| **7**   | Add `operator*` with 64-bit intermediate + saturation; unit-test mul; bench vs FP32                                  |
| **8**   | Draft PTQ primer (`docs/quantization_primer.md`); sketch `include/quant/Quantizer.h` API                             |
| **9**   | Implement `Quantizer::computeScaleZeroPoint()`; write `tests/test_quant_scale.cpp`                                  |
| **10**  | Extend to float→int8 conversion + rounding; test `tests/test_quant_single.cpp`; validate vs NumPy                  |
| **11**  | Batch quant API (`vector<float>→vector<int8_t>`); optimize loops; bench `benchmarks/quant_bench.cpp`                |
| **12**  | Design per-channel quant (`docs/quant_per_channel.md`); stub `Quantizer::perChannelQuantize()`                      |
| **13**  | Implement per-channel quant; optimize layout; test & bench per-channel error                                         |
| **14**  | Research NEON intrinsics; draft `docs/neon_reference.md`; stub `src/simd/neon_quant.cpp`; update CMake for NEON     |
| **15**  | NEON-accelerated quant loop (8→int8); validate vs scalar; bench speedup                                              |
| **16**  | Create `Tensor` class (shape+stride); implement ctor; write `tests/test_tensor_ctor.cpp`                              |
| **17**  | Implement `Tensor::add()`, `mul()`, `relu()`; optimize memory; write `tests/test_tensor_ops.cpp`                      |
| **18**  | Draft `docs/gemm_design.md` (tiling/blocking); stub `include/tensor/Gemm.h`; document API                            |
| **19**  | Implement naive INT8 GEMM; test `tests/test_gemm_naive.cpp`                                                          |
| **20**  | Benchmark naive GEMM vs FP32; log flops & bandwidth                                                                  |
| **21**  | Add 8×8 tiled GEMM; inline loops; test & bench                                                                      |
| **22**  | Optimize tiled GEMM indexing; profile cache misses; document performance                                             |
| **23**  | Integrate NEON in GEMM; test & bench NEON vs tiled scalar                                                            |
| **24**  | Research mixed-precision (INT4/INT8); draft `docs/mixed_precision.md`; stub INT4 path in `Quantizer`                |
| **25**  | Implement INT4 conversion & bit-pack; unit-test & bench error                                                         |
| **26**  | Draft LAQ design (`docs/laq_design.md`); stub `src/quant/LAQOptimizer.h`; define hyperparams                         |
| **27**  | Implement LAQ iteration; log MSE reduction; unit-test                                                                |
| **28**  | Integrate LAQ into pipeline; toggle via config; bench LAQ vs static                                                  |
| **29**  | Document quant & LAQ API; generate Doxygen; write `examples/laq_demo.cpp`                                             |
| **30**  | Survey CiM models (ReRAM/PCM/FeFET); draft `docs/cim_models.md`; stub `include/sim/Simulator.h`                      |
| **31**  | Implement crossbar read/write (`src/sim/Simulator.cpp`); test & bench sim                                             |
| **32**  | Integrate Quant+GEMM+Simulator in `src/Engine.cpp`; pipeline test; bench full-pipeline latency                       |
| **33**  | Add energy proxy (pJ/op) to Simulator; calibrate; log energy per inference                                            |
| **34**  | Stub `DynamicQuantController`; instrument activation histograms                                                      |
| **35**  | Train offline MLP for dynamic scale; export weights; integrate into C++                                               |
| **36**  | Complete `DynamicQuantController::adjustParams()`; unit-test; bench dynamic vs static                                 |
| **37**  | Write `examples/run_dynamic_quant.cpp`; batch tests; log & summarize                                                  |
| **38**  | Stub `ErrorCompensator` (1×1 conv); define config                                                                      |
| **39**  | Implement compensator; unit-test; embed weights; measure error correction                                              |
| **40**  | Integrate compensator; write demo; bench end-to-end gain                                                               |
| **41**  | Stub `SmartMapper`; define Monte Carlo error API                                                                       |
| **42**  | Implement error estimator; log distributions                                                                            |
| **43**  | Implement `SmartMapper::mapWeights()`; test mapping; bench overhead                                                     |
| **44**  | Integrate SmartMapper; toggle; bench smart vs random; record results                                                   |
| **45**  | Research block-sparse pruning; draft `docs/pruning_design.md`; stub `Pruner`                                           |
| **46**  | Implement block pruning; integrate into GEMM; unit-test                                                                |
| **47**  | Benchmark sparse vs dense; log speed & accuracy                                                                         |
| **48**  | Add JSON config loader; parse device & quant settings                                                                   |
| **49**  | Integrate config loader; bench parsing overhead                                                                         |
| **50**  | Configure CI: build/test/bench on push                                                                                  |
| **51**  | Run 1-hour inference stress test on RPi3; log CPU temp & memory                                                         |
| **52**  | Fix memory leaks (Valgrind); enable ASan/UBSan; fix issues                                                              |
| **53**  | Run clang-tidy; apply static analysis fixes                                                                              |
| **54**  | Optimize build (precompiled headers); refactor CMakeLists                                                               |
| **55**  | Scalar fallback for non-NEON targets                                                                                     |
| **56**  | Integrate tensor memory pooling; bench alloc speed                                                                       |
| **57**  | Final GEMM & quant loop tuning                                                                                            |
| **58**  | Audit & reach 100% unit-test coverage; tag `coverage-complete`                                                           |
| **59**  | **New: Profiler Module** – add `src/profiler/PerfCounter.h/.cpp`; collect cycles, cache-misses on key kernels            |
| **60**  | Integrate profiler into CI; log `/benchmarks/perf_logs/*.json`                                                          |
| **61**  | **New: DVFS Controller** – add `src/power/DvfsController.h/.cpp`; read/write `/sys/devices/system/cpu/cpu*/cpufreq`      |
| **62**  | Benchmark latency/energy trade-offs under different CPU freq settings                                                     |
| **63**  | **New: MLIR Codegen Prototype** – stub `src/mlir_codegen/MLIRPass.cpp`; lower quant ops to custom loop IR               |
| **64**  | JIT-compile IR on device; run basic quant kernel; measure JIT overhead                                                   |
| **65**  | **New: ONNX Runtime EP** – stub `src/onnx_ep/YourEngineEP.cpp`; build & register Execution Provider                       |
| **66**  | Test ONNX EP on a simple model (`examples/onnx_ep_demo.cpp`); bench end-to-end                                           |
| **67**  | **New: HA-NAS Module** – stub `src/ha_nas/NasController.h/.cpp`; implement random search over tiny CNN cells            |
| **68**  | Hook HA-NAS to engine latency & accuracy feedback; run first 50 candidates                                                 |
| **69**  | Analyze HA-NAS results; select top-3 cells; integrate into `examples/ha_nas_demo.cpp`                                     |
| **70**  | Bench inference of NAS-discovered cell vs baseline model                                                                 |
| **71**  | Merge all four new modules into `main`; ensure tests pass                                                                |
| **72**  | Stress-test new modules under 1-hr loop; log stability in `docs/stability.md`                                             |
| **73**  | Final performance tuning (profiler-driven)                                                                                 |
| **74**  | Final memory/energy regression checks                                                                                      |
| **75**  | Tag `v2.0-features` release; freeze code                                                                                   |
| **76**  | Containerize engine + new modules; write Dockerfile; build & test                                                         |
| **77**  | Validate reproducibility in container; document in `docs/deployment.md`                                                    |
| **78**  | Generate new API docs (Doxygen)                                                                                             |
| **79**  | Clean up examples & README to include new modules                                                                          |
| **80**  | Plot & publish extended benchmarks (`scripts/plot_extended_results.py`)                                                    |
|  81  | Cross-compile engine on Jetson Nano; fix any build issues                                                              |
|  82  | Benchmark full-pipeline latency + throughput on Jetson Nano; log in `docs/jetson_bench.md`                              |
|  83  | Cross-compile for Raspberry Pi Zero W; resolve toolchain quirks                                                         |
|  84  | Benchmark on Pi Zero W; compare performance vs Pi 3; document in `docs/pi_zero_bench.md`                               |
|  85  | Integrate a thread-pool (`src/thread/ThreadPool.h/.cpp`); stub basic API                                                |
|  86  | Parallelize GEMM & quant loops using ThreadPool; write `tests/test_threadpool_gemm.cpp`                                 |
|  87  | Profile multi-threaded vs single-threaded performance; tune thread affinity with `sched_setaffinity()`                 |
|  88  | Add power-meter integration script (`scripts/read_power.sh`) using I²C or onboard sensor                               |
|  89  | Run energy-per-inference measurements on Pi 3 & Jetson Nano; log in `docs/energy_results.md`                            |
|  90  | Summarize energy vs throughput trade-offs in `docs/energy_tradeoff.md`                                                  |
|  91  | Implement DVFS controller (`src/power/DvfsController.h/.cpp`); read/write `/sys/devices/system/cpu/cpu*/cpufreq`        |
|  92  | Automate CPU-freq sweep tests; record latency & energy at each frequency                                               |
|  93  | Plot DVFS results in `docs/dvfs_performance.md`; identify optimal governor settings                                     |
|  94  | Extend CI to run builds/tests on ARM and x86_64; add multi-arch GitHub Actions workflows                                |
|  95  | Add Windows build support (MinGW/CMake); resolve Windows-specific code paths                                           |
|  96  | Benchmark pipeline on Windows x64; log in `docs/windows_bench.md`                                                       |
|  97  | Document platform-specific optimizations in `docs/platform_tuning.md`                                                   |
|  98  | Stub ONNX Runtime Execution Provider (`src/onnx_ep/YourEngineEP.cpp`); register EP in CMake                              |
|  99  | Implement model import & quant execution in EP; write `examples/onnx_ep_demo.cpp`                                        |
| 100  | Benchmark ONNX EP inference vs direct engine on Pi 3; log in `docs/onnx_ep_bench.md`                                     |
| 101  | Optimize EP initialization time; measure cold-start vs warm-start latency                                                |
| 102  | Add CI job to build & test ONNX EP; ensure all tests pass                                                                |
| 103  | Stub MLIR codegen pass (`src/mlir_codegen/MLIRPass.cpp`); define custom dialect                                          |
| 104  | Lower a simple quant+GEMM kernel to MLIR, JIT-compile, run on device                                                      |
| 105  | Measure JIT vs static kernel performance; document in `docs/mlir_bench.md`                                               |
| 106  | Write tests for MLIR-generated kernels (`tests/test_mlir_kernels.cpp`)                                                   |
| 107  | Integrate MLIR pass into CMake build; ensure reproducibility                                                             |
| 108  | Stub HA-NAS controller (`src/ha_nas/NasController.h/.cpp`); implement random search loop                                  |
| 109  | Hook NAS loop to engine latency & accuracy feedback; run first 100 candidates                                             |
| 110  | Aggregate NAS results; select top-3 architectures; document in `docs/ha_nas_results.md`                                   |
| 111  | Integrate chosen NAS cells into `examples/ha_nas_demo.cpp`; test end-to-end                                                |
| 112  | Benchmark NAS-discovered models vs baseline; log speed & accuracy                                                        |
| 113  | Implement dynamic memory pooling improvements for variable model sizes                                                    |
| 114  | Benchmark pooling vs malloc/new for NAS models; document results                                                          |
| 115  | Add tiered quantization support (layer-wise bitwidth config); stub in `Quantizer`                                        |
| 116  | Implement tiered quantization; write tests & bench mixed-bit GEMM                                                         |
| 117  | Profile memory fragmentation & defragmentation performance                                                                |
| 118  | Integrate continuous memory defragment in PoolAllocator; test stability                                                   |
| 119  | Add end-to-end multi-model benchmark suite (`benchmarks/multi_model_bench.cpp`)                                           |
| 120  | Run multi-model benchmarks (ResNet-Mini, TinyCNN, NAS cell); log in `docs/multi_model_bench.md`                          |
| 121  | Implement early-exit inference (branchy networks) support; stub API in `Engine`                                           |
| 122  | Integrate a simple branchy-CNN example; test accuracy vs latency trade-off                                                |
| 123  | Benchmark early-exit models; document threshold selection effects                                                         |
| 124  | Add support for half-precision (FP16) fallback in `Quantizer`; write tests                                                |
| 125  | Benchmark FP16 vs INT8 paths; log accuracy & throughput                                                                  |
| 126  | Integrate a CLI performance summary command (`engine --report`)                                                           |
| 127  | Implement JSON report export; test on sample runs                                                                         |
| 128  | Update `README.md` with usage for new CLI & modules                                                                       |
| 129  | Conduct cross-platform regression tests on all devices; fix any regressions                                                |
| 130  | Final code cleanup: remove debug logs, enforce style with clang-format                                                     |
| 131  | Run final build & test suite; tag commit `v2.0-tech_complete`                                                               |
| 132  | Containerize full engine + all modules (Dockerfile); build & test container                                                |
| 133  | Validate container on clean VM; document in `docs/deployment.md`                                                          |
| 134  | Generate Doxygen API docs; host on GitHub Pages                                                                             |
| 135  | Clean up `examples/`; ensure each demo builds/runs; add quick-start snippets                                               |
| 136  | Develop `scripts/plot_all_results.py`; generate consolidated plots                                                         |
| 137  | Run final hyperparam sweeps for dynamic quant, compensator & mapping; log in `docs/final_sweep.md`                         |
| 138  | Aggregate final sweep data; identify optimal settings for paper inclusion                                                  |
| 139  | Merge all polished code & docs into `main`; prepare for v3.0 roadmap                                                       |
| 140  | Archive code snapshot (`docs/archive_snapshot.md`); ensure full reproducibility                                             |
| 141   | Plan & script final hyperparam experiments across all new modules                   |
| 142   | Run combined accuracy vs throughput vs energy experiments on selected models         |
| 143   | Aggregate & analyze all results; draft `docs/experiment_results.md`                  |
| 144   | Generate publication-quality figures & tables                                        |
| 145   | Verify reproducibility of all experiments                                            |
| 146   | Finalize `docs/experiment_results.md`; prepare summary tables                        |
| 147   | Draft slides & talking points for each module’s impact                               |
| 148   | Peer-review experiment writeups; incorporate feedback                                 |
| 149   | Sanity-check experiments on a fresh Pi 3 image                                       |
| 150   | Celebrate completion & transition to paper drafting                                  |
