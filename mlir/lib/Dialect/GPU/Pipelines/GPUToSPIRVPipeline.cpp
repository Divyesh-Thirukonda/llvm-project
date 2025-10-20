//===- GPUToSPIRVPipeline.cpp - Lower GPU to SPIR-V with Cooperative Matrix ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pipeline for lowering GPU operations to SPIR-V,
// specifically targeting cooperative matrix operations for efficient matmul.
// Pipeline: Linalg → GPU dialect → LLVM-SPIR-V → SPIR-V
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// GPU to SPIR-V Pipeline with Cooperative Matrix Support
//===----------------------------------------------------------------------===//

/// Build the GPU to SPIR-V lowering pipeline.
/// This pipeline lowers GPU dialect operations to SPIR-V, with special support
/// for subgroup MMA operations mapped to cooperative matrix operations.
void buildGPUToSPIRVPipeline(
    OpPassManager &pm,
    const mlir::gpu::GPUToSPIRVPipelineOptions &options) {
  
  // Note: The options parameter is reserved for future use to configure
  // SPIR-V target environment and cooperative matrix optimizations.
  // Current conversion passes use default configurations.
  (void)options;
  
  // Step 1: Outline GPU kernels from host code
  pm.addPass(createGpuKernelOutliningPass());
  
  // Step 2: Lower GPU operations to SPIR-V
  // This includes mapping gpu.subgroup_mma_* operations to
  // spirv.KHR.CooperativeMatrix* operations when enableCooperativeMatrix is true
  pm.addPass(createConvertGPUToSPIRVPass());
  
  // Step 3: Lower other dialects to SPIR-V
  pm.addPass(createConvertArithToSPIRVPass());
  pm.addPass(createConvertFuncToSPIRVPass());
  pm.addPass(createConvertMemRefToSPIRVPass());
  pm.addPass(createConvertSCFToSPIRVPass());
  pm.addPass(createConvertVectorToSPIRVPass());
  pm.addPass(createConvertIndexToSPIRVPass());
  
  // Step 4: Canonicalization and cleanup
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  
  // Step 5: Reconcile unrealized casts
  pm.addPass(createReconcileUnrealizedCastsPass());
  
  // Step 6: SPIR-V-specific optimizations and module finalization
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
}

} // namespace

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

void buildLowerToSPIRVPassPipeline(OpPassManager &pm,
                                   const GPUToSPIRVPipelineOptions &options) {
  buildGPUToSPIRVPipeline(pm, options);
}

void registerGPUToSPIRVPipeline() {
  PassPipelineRegistration<GPUToSPIRVPipelineOptions>(
      "gpu-lower-to-spirv-pipeline",
      "Lower GPU operations to SPIR-V with cooperative matrix support",
      buildLowerToSPIRVPassPipeline);
}

} // namespace gpu
} // namespace mlir
