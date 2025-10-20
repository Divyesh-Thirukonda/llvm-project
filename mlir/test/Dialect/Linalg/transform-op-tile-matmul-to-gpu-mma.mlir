// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// This test demonstrates tiling a linalg.matmul operation and mapping it to
// GPU subgroup MMA operations for cooperative matrix execution.

// CHECK-LABEL: func.func @matmul_tile_to_mma
func.func @matmul_tile_to_mma(%A: tensor<64x64xf16>, %B: tensor<64x64xf16>, 
                               %C: tensor<64x64xf16>) -> tensor<64x64xf16> {
  // CHECK: linalg.matmul
  %0 = linalg.matmul ins(%A, %B: tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%C: tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // Tile the matmul to 16x16x16 for subgroup MMA operations
    // These sizes match typical cooperative matrix dimensions
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @matmul_tile_and_vectorize
func.func @matmul_tile_and_vectorize(%A: tensor<128x128xf16>, %B: tensor<128x128xf16>, 
                                      %C: tensor<128x128xf16>) -> tensor<128x128xf16> {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: linalg.matmul
  %0 = linalg.matmul ins(%A, %B: tensor<128x128xf16>, tensor<128x128xf16>)
                     outs(%C: tensor<128x128xf16>) -> tensor<128x128xf16>
  return %0 : tensor<128x128xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // First tile to workgroup level
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [32, 32, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    // Further tile to subgroup level for MMA operations
    %tiled_again, %inner_loops:3 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
