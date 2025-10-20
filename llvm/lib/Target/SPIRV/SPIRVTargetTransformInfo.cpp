//===- SPIRVTargetTransformInfo.cpp - SPIR-V specific TTI -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVTargetTransformInfo.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

using namespace llvm;

bool llvm::SPIRVTTIImpl::collectFlatAddressOperands(
    SmallVectorImpl<int> &OpIndexes, Intrinsic::ID IID) const {
  switch (IID) {
  case Intrinsic::spv_generic_cast_to_ptr_explicit:
    OpIndexes.push_back(0);
    return true;
  default:
    return false;
  }
}

Value *llvm::SPIRVTTIImpl::rewriteIntrinsicWithAddressSpace(IntrinsicInst *II,
                                                            Value *OldV,
                                                            Value *NewV) const {
  auto IntrID = II->getIntrinsicID();
  switch (IntrID) {
  case Intrinsic::spv_generic_cast_to_ptr_explicit: {
    unsigned NewAS = NewV->getType()->getPointerAddressSpace();
    unsigned DstAS = II->getType()->getPointerAddressSpace();
    return NewAS == DstAS ? NewV
                          : ConstantPointerNull::get(
                                PointerType::get(NewV->getContext(), DstAS));
  }
  default:
    return nullptr;
  }
}

bool llvm::SPIRVTTIImpl::isReconvergencePoint(const Value *V) const {
  // Subgroup intrinsics that force reconvergence across the subgroup
  const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V);
  if (!Intrinsic)
    return false;

  // SPIR-V subgroup operations that force reconvergence
  // Note: SPIR-V has extensive subgroup support that could be added here
  switch (Intrinsic->getIntrinsicID()) {
  // These are basic subgroup ballot/broadcast operations that force reconvergence
  case Intrinsic::spv_thread_id_in_group:
  case Intrinsic::spv_subgroup_size:
  case Intrinsic::spv_subgroup_id:
  case Intrinsic::spv_subgroup_local_invocation_id:
    // These operations query subgroup properties and force reconvergence
    return true;
  default:
    return false;
  }
}
