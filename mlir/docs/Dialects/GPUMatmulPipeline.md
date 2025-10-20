# GPU Matmul Lowering Pipeline

This document describes the modular pipeline for lowering Linalg matrix multiplication operations to GPU-accelerated code targeting SPIR-V and DXIL backends with cooperative matrix support.

## Overview

The pipeline follows a modular design:
```
Linalg → GPU Dialect → LLVM IR → SPIR-V/DXIL
```

Each stage is designed to be composable and replaceable, allowing for different backend targets while maintaining a common transformation path.

## Pipeline Stages

### 1. Linalg to GPU Transformation

**Input**: `linalg.matmul` operations on tensors
**Output**: Tiled and mapped GPU operations

This stage performs:
- Tiling the matmul into submatrices suitable for cooperative matrix operations (typically 16x16, 8x16, or 16x8)
- Mapping iterations to GPU grid, blocks, and threads
- Optional vectorization for efficient memory access

Example transformation:
```mlir
// Input
%result = linalg.matmul ins(%A, %B: tensor<64x64xf16>, tensor<64x64xf16>)
                        outs(%C: tensor<64x64xf16>) -> tensor<64x64xf16>

// After tiling (16x16x16 tiles for cooperative matrix)
scf.for %i = ... step 16 {
  scf.for %j = ... step 16 {
    scf.for %k = ... step 16 {
      %tile = linalg.matmul ins(%A_tile, %B_tile)
                            outs(%C_tile) -> tensor<16x16xf16>
    }
  }
}
```

### 2. GPU Subgroup MMA Operations

**Input**: Tiled matmul operations
**Output**: `gpu.subgroup_mma_*` operations

The tiled matmul operations are lowered to GPU subgroup MMA operations:
- `gpu.subgroup_mma_load_matrix`: Load matrix fragments
- `gpu.subgroup_mma_compute`: Perform matrix multiply-accumulate
- `gpu.subgroup_mma_store_matrix`: Store results back to memory

These operations abstract hardware-specific matrix operations (WMMA, DPAS, cooperativeMatrixKHR).

### 3. GPU to SPIR-V/LLVM

**Input**: `gpu.subgroup_mma_*` operations
**Output**: SPIR-V or LLVM IR with target-specific intrinsics

For SPIR-V targets:
```mlir
gpu.subgroup_mma_load_matrix  → spirv.KHR.CooperativeMatrixLoad
gpu.subgroup_mma_compute      → spirv.KHR.CooperativeMatrixMulAdd
gpu.subgroup_mma_store_matrix → spirv.KHR.CooperativeMatrixStore
```

For NVIDIA (NVVM):
```mlir
gpu.subgroup_mma_load_matrix  → nvvm.wmma.load
gpu.subgroup_mma_compute      → nvvm.wmma.mma
gpu.subgroup_mma_store_matrix → nvvm.wmma.store
```

### 4. Software Fallback

For platforms without hardware MMA support, the pipeline includes a software fallback path that:
1. Decomposes cooperative matrix operations into vector operations
2. Uses standard load/store and vector arithmetic
3. Maintains correctness at the cost of performance

## Using the Pipeline

### Command-line Usage

To run the complete pipeline on MLIR source:

```bash
mlir-opt input.mlir \
  -transform-interpreter \
  -gpu-lower-to-spirv-pipeline="enable-cooperative-matrix=true target-env=vulkan1.3" \
  -o output.mlir
```

For NVIDIA targets:
```bash
mlir-opt input.mlir \
  -transform-interpreter \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80" \
  -o output.mlir
```

### Programmatic Usage

```c++
#include "mlir/Dialect/GPU/Pipelines/Passes.h"

OpPassManager pm(module.getContext());
gpu::GPUToSPIRVPipelineOptions options;
options.enableCooperativeMatrix = true;
options.targetEnv = "vulkan1.3";
gpu::buildLowerToSPIRVPassPipeline(pm, options);
```

## Pipeline Options

### GPU to SPIR-V Pipeline

- `enable-cooperative-matrix`: Enable SPIR-V cooperative matrix operations (default: true)
- `target-env`: Target SPIR-V environment (vulkan1.2, vulkan1.3, opencl)

### Tiling Configuration

Cooperative matrix operations typically support specific tile sizes:
- **NVIDIA Tensor Cores**: 16x16x16 (f16), 8x8x4 (f32)
- **AMD CDNA**: 16x16x16 (f16), 16x16x4 (f32)
- **Intel DPAS**: 8x16x16 (various types)

Choose tile sizes based on target hardware and data types.

## Performance Considerations

1. **Memory Coalescing**: Ensure contiguous memory access patterns within subgroups
2. **Occupancy**: Balance tile sizes with register usage and shared memory
3. **Data Types**: Use f16 for maximum throughput on most hardware
4. **Cooperative Matrix Support**: Verify target device capabilities

## Testing

The pipeline includes comprehensive tests:

1. **Unit Tests**: Individual transformation passes
   - `mlir/test/Dialect/Linalg/transform-op-tile-matmul-to-gpu-mma.mlir`

2. **Integration Tests**: Full pipeline execution
   - `mlir/test/Integration/Dialect/Linalg/GPU/linalg-matmul-to-gpu-mma.mlir`
   - `mlir/test/Integration/Dialect/Linalg/GPU/matmul-cooperative-matrix-spirv.mlir`

3. **Conversion Tests**: Backend-specific lowering
   - `mlir/test/Conversion/GPUToSPIRV/wmma-ops-to-spirv-khr-coop-matrix.mlir`

## Benchmarking

A GEMM micro-benchmark is provided to compare performance:

```bash
# Build the benchmark
cd mlir/benchmark
mkdir build && cd build
cmake .. -DMLIR_ENABLE_CUDA=ON
make gemm-benchmark

# Run with different configurations
./gemm-benchmark --size=1024 --backend=cuda --use-mma=true
./gemm-benchmark --size=1024 --backend=vulkan --use-cooperative-matrix=true
```

## Future Work

- Support for mixed-precision matmul (e.g., int8 x int8 -> int32)
- Dynamic tile size selection based on hardware capabilities
- DXIL backend integration for DirectX targets
- Quantization support for ML inference workloads
- Integration with distributed execution frameworks

## References

- [SPIR-V Cooperative Matrix Extension](https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#_cooperative_matrix)
- [NVIDIA WMMA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Vulkan Cooperative Matrix](https://www.khronos.org/blog/vulkan-cooperative-matrix-extension)
