add_mlir_dialect_library(MLIRGPUPipelines
  GPUToNVVMPipeline.cpp
  GPUToSPIRVPipeline.cpp
  GPUToXeVMPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRAffineToStandard
  MLIRArithToSPIRV
  MLIRFuncToSPIRV
  MLIRGPUToLLVMSPV
  MLIRGPUToNVVMTransforms
  MLIRGPUToSPIRV
  MLIRIndexToLLVM
  MLIRIndexToSPIRV
  MLIRMathToLLVM
  MLIRMathToXeVM
  MLIRMemRefToSPIRV
  MLIRNVGPUToNVVM
  MLIRNVVMToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSCFToSPIRV
  MLIRSPIRVDialect
  MLIRSPIRVTransforms
  MLIRVectorToSCF
  MLIRVectorToSPIRV
  MLIRXeGPUTransforms
  MLIRXeGPUToXeVM
  MLIRXeVMToLLVM
)
