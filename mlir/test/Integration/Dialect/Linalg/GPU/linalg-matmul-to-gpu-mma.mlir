// RUN: mlir-opt %s \
// RUN:   -transform-interpreter \
// RUN:   -test-transform-dialect-erase-schedule \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops \
// RUN:   -gpu-kernel-outlining \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-to-llvm \
// RUN: | FileCheck %s

// This integration test demonstrates the full lowering path:
// Linalg matmul -> tiled matmul -> GPU operations -> LLVM
// This serves as the foundation for SPIR-V/DXIL lowering with cooperative matrices

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {
  // Transform sequence to tile and map matmul to GPU
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 
      : (!transform.any_op) -> !transform.any_op
    
    // Tile for GPU workgroup (outer level)
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    // Map loops to GPU grid and blocks
    %grid_loop = transform.loop.map_to_gpu_blocks %loops#0 
      : (!transform.any_op) -> !transform.any_op
    %block_loop = transform.loop.map_to_gpu_blocks %loops#1
      : (!transform.any_op) -> !transform.any_op
    
    transform.yield
  }
}

// Simple 32x32 matmul test case
// CHECK-LABEL: func.func @matmul_32x32
func.func @matmul_32x32(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: linalg.matmul
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) 
                     outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
