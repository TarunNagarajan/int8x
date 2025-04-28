# Hardware-Aware Quantization & Mapping Engine for CiM Accelerators  

| Day  | Task                                                                                                             |
|-----:|------------------------------------------------------------------------------------------------------------------|
| 1    | Initialize GitHub repo (`README.md`, `.gitignore`, `CMakeLists.txt`); provision RPi3 (OS, build tools, SSH)     |
| 2    | Draft `docs/module-outline.md`; sketch architecture; start `docs/devlog.md`                                     |
| 3    | Research C++ allocators; stub `src/alloc/PoolAllocator.h/.cpp`; write `tests/test_allocator.cpp`               |
| 4    | Complete PoolAllocator (free-list + alignment); bench vs `malloc`; profile cache (`valgrind --tool=cachegrind`) |
| 5    | Read Sze Ch 1–2; extract efficiency notes to `docs/efficiency_summary.md`; draft `docs/fixed_point_design.md`   |
| 6    | Implement `FixedPoint<T>` (ctor + `toFloat()`); add `operator+`/`-`; write `tests/test_fixed_add_sub.cpp`       |
| 7    | Add `operator*` with 64-bit intermediate + saturation; test `tests/test_fixed_mul.cpp`; bench vs FP32           |
| 8    | Draft PTQ primer (`docs/quantization_primer.md`); sketch `include/quant/Quantizer.h` API                        |
| 9    | Implement `Quantizer::computeScaleZeroPoint()`; write `tests/test_quant_scale.cpp`                             |
| 10   | Extend to float→int8 conversion w/ rounding; test `tests/test_quant_single.cpp`; validate vs NumPy             |
| 11   | Batch quant API (`vector<float>→vector<int8_t>`); optimize pointer loops; add `benchmarks/quant_bench.cpp`      |
| 12   | Design per-channel quant (`docs/quant_per_channel.md`); stub `Quantizer::perChannelQuantize()`                 |
| 13   | Implement per-channel quant; optimize layout; test `tests/test_quant_per_channel.cpp`; bench error               |
| 14   | Research ARM NEON intrinsics; draft `docs/neon_reference.md`; stub `src/simd/neon_quant.cpp`; update CMake flags|
| 15   | NEON-accelerated quant loop (8 floats→8 int8); validate vs scalar; bench speedup                                |
| 16   | Create `include/tensor/Tensor.h` (shape+stride); implement ctor; write `tests/test_tensor_ctor.cpp`             |
| 17   | Implement `Tensor::add()`, `mul()`, `relu()`; optimize contiguous memory; test `tests/test_tensor_ops.cpp`      |
| 18   | Draft `docs/gemm_design.md` (tiling/blocking pseudocode); stub `include/tensor/Gemm.h`; document API            |
| 19   | Implement naive INT8 GEMM (`src/tensor/Gemm.cpp`); test `tests/test_gemm_naive.cpp`                             |
| 20   | Benchmark naive GEMM vs FP32 (`benchmarks/gemm_naive_bench.cpp`); log flops & bandwidth                         |
| 21   | Add 8×8 tiled GEMM; inline loops; test `tests/test_gemm_tiled.cpp`; bench vs naive                              |
| 22   | Optimize tiled GEMM indexing (pointer-only); profile cache misses; log `docs/gemm_performance.md`               |
| 23   | Integrate NEON in GEMM inner loops; test `tests/test_gemm_neon.cpp`; benchmark NEON vs tiled scalar            |
| 24   | Research mixed-precision (INT4/INT8); draft `docs/mixed_precision.md`; stub INT4 path in `Quantizer`           |
| 25   | Implement INT4 conversion + bit-pack/unpack; test `tests/test_quant_int4.cpp`; bench INT4 vs INT8 MSE           |
| 26   | Draft LAQ design (`docs/laq_design.md`); stub `src/quant/LAQOptimizer.h`; define hyperparams                    |
| 27   | Implement one LAQ iteration; log MSE in `docs/laq_logs.md`; test `tests/test_laq.cpp`; plot via `plot_laq.py`   |
| 28   | Integrate LAQ into pipeline; toggle in config; bench LAQ vs static quant                                        |
| 29   | Document quant & LAQ API (`docs/api_reference.md`); Doxygen; write `examples/laq_demo.cpp`                       |
| 30   | Survey CiM models (RRAM/PCM/FeFET); draft `docs/cim_models.md`; stub `include/sim/Simulator.h`                  |
| 31   | Implement crossbar read/write (`src/sim/Simulator.cpp`); test `tests/test_sim_basic.cpp`; bench sim             |
| 32   | Integrate Quant+GEMM+Simulator in `src/Engine.cpp`; test `tests/test_pipeline.cpp`; bench full-pipeline latency   |
| 33   | Add energy proxy (pJ/op) in Simulator; calibrate; log in `docs/energy_report.md`                                |
| 34   | Stub `DynamicQuantController`; instrument histogram collection in quantizer                                      |
| 35   | Train offline MLP (Python) for hist→scale; export weights; integrate MLP inference in C++                        |
| 36   | Complete `DynamicQuantController::adjustParams()`; test `tests/test_dynamic_quant.cpp`; bench dynamic vs static  |
| 37   | Write `examples/run_dynamic_quant.cpp`; batch inputs; log accuracy/latency; summarize in `docs/dynamic_results.md`|
| 38   | Stub `ErrorCompensator` (1×1 conv) in `src/research/error_comp/Compensator.h`; define config parameters          |
| 39   | Implement compensator; write `tests/test_compensator.cpp`; embed trained weights; measure error reduction         |
| 40   | Integrate compensator into pipeline; create `examples/run_error_comp.cpp`; bench end-to-end accuracy gain        |
| 41   | Stub `SmartMapper` in `src/research/cim_mapping/SmartMapper.h`; define Monte Carlo error-estimator API            |
| 42   | Implement error estimator per block; log in `docs/cim_mapping_stats.md`                                          |
| 43   | Implement `SmartMapper::mapWeights()`; test `tests/test_smart_mapper.cpp`; bench mapping overhead                |
| 44   | Integrate SmartMapper; toggle in config; bench smart vs random mapping; record in `docs/mapping_results.md`      |
| 45   | Research block-sparse pruning; draft `docs/pruning_design.md`; stub `src/research/pruning/Pruner.h`             |
| 46   | Implement block pruning; integrate into GEMM skip-zeros; test `tests/test_pruner.cpp`                            |
| 47   | Benchmark sparse vs dense inference; log speed & accuracy in `docs/pruning_results.md`                          |
| 48   | Add JSON config loader (`src/ConfigParser.cpp`); parse device & quant settings                                   |
| 49   | Integrate config loader; bench parsing overhead                                                                 |
| 50   | Configure CI: build/test/bench on push                                                                          |
| 51   | Run 1 hr continuous inference on RPi3; log CPU temp & memory (`docs/stability.md`)                              |
| 52   | Fix memory leaks; run Valgrind Memcheck; resolve issues                                                          |
| 53   | Enable ASan/UBSan builds; fix detected errors                                                                    |
| 54   | Run clang-tidy static analysis; apply fixes                                                                      |
| 55   | Optimize build speed (precompiled headers); refactor CMakeLists                                                 |
| 56   | Implement scalar fallback path for non-NEON targets                                                             |
| 57   | Integrate tensor memory pooling; benchmark alloc speed                                                           |
| 58   | Finalize GEMM & quant loop tuning for max throughput                                                            |
| 59   | Audit unit-test coverage; add missing tests                                                                      |
| 60   | Achieve 100% coverage; tag `coverage-complete`                                                                   |
| 61   | Develop inline-ASM microkernels for quant loops; test `tests/test_asm_kernels.cpp`                              |
| 62   | Bench ASM vs NEON kernels; profile cycle counts                                                                 |
| 63   | Merge ASM kernels; ensure all tests pass                                                                         |
| 64   | Cross-compile for x86_64 & armhf; test builds                                                                     |
| 65   | Benchmark on Jetson Nano; compare to RPi3                                                                        |
| 66   | Optimize memory layout & cache for multiple architectures                                                        |
| 67   | Aggregate multi-device metrics via `scripts/aggregate_results.py`                                                |
| 68   | Analyze & select best config per device; update defaults                                                         |
| 69   | Integrate optimal configs; bench final pipeline                                                                  |
| 70   | Tag `v1.0-technical` release; freeze code for experiments                                                         |
| 71   | Containerize engine; write Dockerfile; build & test                                                              |
| 72   | Validate container reproducibility; document steps                                                               |
| 73   | Final perf profiling & hot-loop tuning                                                                            |
| 74   | Remove debug/log code; ensure code clarity                                                                       |
| 75   | Prepare final release branch; tag `v1.0-final`                                                                   |
| 76   | Generate & publish Doxygen API docs                                                                               |
| 77   | Clean up examples; ensure demos build & run; add usage in README                                                  |
| 78   | Implement result-plotting scripts; generate base plots                                                            |
| 79   | Develop hyperparam sweep harness                                                                                 |
| 80   | Run initial sweeps for dynamic quant, compensator & mapping                                                       |
| 81   | Aggregate sweep data; identify promising ranges                                                                   |
| 82   | Refine sweeps on narrowed ranges                                                                                 |
| 83   | Re-benchmark with refined settings; log in `docs/optimal_results.md`                                              |
| 84   | Conduct final memory & performance regression checks                                                              |
| 85   | Merge refined settings; tag `v2.0-optimal`                                                                        |
| 86   | Final codebase refactor: naming, formatting, remove dead code                                                     |
| 87   | Update CI: add hyperparam sweep jobs                                                                              |
| 88   | Prepare reproducible experiment manifest (`docs/experiments_manifest.md`)                                         |
| 89   | Validate all configs in fresh environments                                                                        |
| 90   | Archive code & docs snapshot for reproducibility                                                                  |

### Last 10 days: Deep Experimentation & Wrap-Up

| Day    | Task                                                                                                    |
|-------:|---------------------------------------------------------------------------------------------------------|
| **91** | Plan & script hyperparam sweep: dynamic quant bins vs accuracy                                          |
| **92** | Sweep: LAQ iterations vs MSE reduction                                                                   |
| **93** | Sweep: compensator kernel size vs latency/accuracy                                                      |
| **94** | Sweep: mapping tile configurations vs error resilience                                                  |
| **95** | Combined experiment: static vs dyn vs LAQ vs comp vs mapping                                            |
| **96** | End-to-end benchmarks: accuracy, latency, memory, energy on 5 sample models                              |
| **97** | Aggregate & analyze all results; draft `docs/experiment_results.md`                                      |
| **98** | Generate publication-quality figures & tables                                                           |
| **99** | Final review & reproducibility check of all experiments                                                 |
| **100**| **(Optional buffer)** polish any missing detail, finalize logs, prepare for paper writing (Days 101–150) |
