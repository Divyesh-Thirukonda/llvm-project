// RUN: mlir-opt %s \
// RUN:   -convert-gpu-to-spirv \
// RUN:   -split-input-file \
// RUN: | FileCheck %s

// This test demonstrates the lowering of GPU subgroup MMA operations to
// SPIR-V cooperative matrix operations (SPV_KHR_cooperative_matrix).
// This is a key component of the Linalg → GPU → SPIR-V pipeline.

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
    [Shader, CooperativeMatrixKHR, Float16],
    [SPV_KHR_storage_buffer_storage_class, SPV_KHR_cooperative_matrix]>,
    #spirv.resource_limits<>>} {

  // CHECK-LABEL: spirv.module @{{.*}} Logical GLSL450
  gpu.module @cooperative_matmul {
    // CHECK-LABEL: spirv.func @matmul_16x16xf16_cooperative
    // Test case demonstrating GPU MMA operations mapped to SPIR-V cooperative matrix
    gpu.func @matmul_16x16xf16_cooperative(
      %arg0: memref<16x16xf16, #spirv.storage_class<StorageBuffer>>,
      %arg1: memref<16x16xf16, #spirv.storage_class<StorageBuffer>>,
      %arg2: memref<16x16xf16, #spirv.storage_class<StorageBuffer>>
    ) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
      %c0 = arith.constant 0 : index
      
      // Load matrix A (16x16xf16) into MMA fragment
      // CHECK: spirv.KHR.CooperativeMatrixLoad
      // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
      %A = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] 
        {leadDimension = 16 : index} 
        : memref<16x16xf16, #spirv.storage_class<StorageBuffer>> 
        -> !gpu.mma_matrix<16x16xf16, "AOp">
      
      // Load matrix B (16x16xf16) into MMA fragment
      // CHECK: spirv.KHR.CooperativeMatrixLoad
      // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
      %B = gpu.subgroup_mma_load_matrix %arg1[%c0, %c0] 
        {leadDimension = 16 : index} 
        : memref<16x16xf16, #spirv.storage_class<StorageBuffer>> 
        -> !gpu.mma_matrix<16x16xf16, "BOp">
      
      // Load accumulator matrix C (16x16xf16)
      // CHECK: spirv.KHR.CooperativeMatrixLoad
      // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %C = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0] 
        {leadDimension = 16 : index} 
        : memref<16x16xf16, #spirv.storage_class<StorageBuffer>> 
        -> !gpu.mma_matrix<16x16xf16, "COp">
      
      // Perform matrix multiply-accumulate: D = A * B + C
      // CHECK: spirv.KHR.CooperativeMatrixMulAdd
      // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
      // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
      // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %D = gpu.subgroup_mma_compute %A, %B, %C 
        : !gpu.mma_matrix<16x16xf16, "AOp">, 
          !gpu.mma_matrix<16x16xf16, "BOp"> 
        -> !gpu.mma_matrix<16x16xf16, "COp">
      
      // Store result back to memory
      // CHECK: spirv.KHR.CooperativeMatrixStore
      gpu.subgroup_mma_store_matrix %D, %arg2[%c0, %c0] 
        {leadDimension = 16 : index} 
        : !gpu.mma_matrix<16x16xf16, "COp">, 
          memref<16x16xf16, #spirv.storage_class<StorageBuffer>>
      
      gpu.return
    }
  }
}
