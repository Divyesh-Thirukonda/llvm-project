//===- DirectXTargetTransformInfo.cpp - DirectX TTI ---------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#include "DirectXTargetTransformInfo.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"

using namespace llvm;

bool DirectXTTIImpl::isTargetIntrinsicWithScalarOpAtArg(
    Intrinsic::ID ID, unsigned ScalarOpdIdx) const {
  switch (ID) {
  case Intrinsic::dx_wave_readlane:
    return ScalarOpdIdx == 1;
  default:
    return false;
  }
}

bool DirectXTTIImpl::isTargetIntrinsicWithOverloadTypeAtArg(Intrinsic::ID ID,
                                                            int OpdIdx) const {
  switch (ID) {
  case Intrinsic::dx_asdouble:
  case Intrinsic::dx_isinf:
  case Intrinsic::dx_isnan:
  case Intrinsic::dx_firstbitlow:
  case Intrinsic::dx_firstbituhigh:
  case Intrinsic::dx_firstbitshigh:
    return OpdIdx == 0;
  default:
    return OpdIdx == -1;
  }
}

bool DirectXTTIImpl::isTargetIntrinsicTriviallyScalarizable(
    Intrinsic::ID ID) const {
  switch (ID) {
  case Intrinsic::dx_asdouble:
  case Intrinsic::dx_firstbitlow:
  case Intrinsic::dx_firstbitshigh:
  case Intrinsic::dx_firstbituhigh:
  case Intrinsic::dx_frac:
  case Intrinsic::dx_isinf:
  case Intrinsic::dx_isnan:
  case Intrinsic::dx_rsqrt:
  case Intrinsic::dx_saturate:
  case Intrinsic::dx_splitdouble:
  case Intrinsic::dx_wave_readlane:
  case Intrinsic::dx_wave_reduce_max:
  case Intrinsic::dx_wave_reduce_sum:
  case Intrinsic::dx_wave_reduce_umax:
  case Intrinsic::dx_wave_reduce_usum:
  case Intrinsic::dx_imad:
  case Intrinsic::dx_umad:
    return true;
  default:
    return false;
  }
}

bool DirectXTTIImpl::isReconvergencePoint(const Value *V) const {
  // Wave intrinsics that force reconvergence across the wave
  const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V);
  if (!Intrinsic)
    return false;

  // DirectX wave operations that force reconvergence
  switch (Intrinsic->getIntrinsicID()) {
  case Intrinsic::dx_wave_readlane:
  case Intrinsic::dx_wave_reduce_max:
  case Intrinsic::dx_wave_reduce_umax:
  case Intrinsic::dx_wave_reduce_sum:
  case Intrinsic::dx_wave_reduce_usum:
  case Intrinsic::dx_wave_all:
  case Intrinsic::dx_wave_any:
  case Intrinsic::dx_wave_active_countbits:
  case Intrinsic::dx_wave_is_first_lane:
    // These operations force all lanes to participate and reconverge
    return true;
  default:
    return false;
  }
}
