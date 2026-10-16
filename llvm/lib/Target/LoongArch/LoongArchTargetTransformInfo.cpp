//===-- LoongArchTargetTransformInfo.cpp - LoongArch specific TTI ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// LoongArch target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "LoongArchTargetTransformInfo.h"

using namespace llvm;

#define DEBUG_TYPE "loongarchtti"

TypeSize LoongArchTTIImpl::getRegisterBitWidth(
    TargetTransformInfo::RegisterKind K) const {
  TypeSize DefSize = TargetTransformInfoImplBase::getRegisterBitWidth(K);
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    return TypeSize::getFixed(ST->is64Bit() ? 64 : 32);
  case TargetTransformInfo::RGK_FixedWidthVector:
    if (ST->hasExtLASX())
      return TypeSize::getFixed(256);
    if (ST->hasExtLSX())
      return TypeSize::getFixed(128);
    [[fallthrough]];
  case TargetTransformInfo::RGK_ScalableVector:
    return DefSize;
  }

  llvm_unreachable("Unsupported register kind");
}

unsigned LoongArchTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  switch (ClassID) {
  case LoongArchRegisterClass::GPRRC:
    // 30 = 32 GPRs - r0 (zero register) - r21 (non-allocatable)
    return 30;
  case LoongArchRegisterClass::FPRRC:
    return ST->hasBasicF() ? 32 : 0;
  case LoongArchRegisterClass::VRRC:
    return ST->hasExtLSX() ? 32 : 0;
  }
  llvm_unreachable("unknown register class");
}

unsigned LoongArchTTIImpl::getRegisterClassForType(bool Vector,
                                                   Type *Ty) const {
  if (Vector)
    return LoongArchRegisterClass::VRRC;
  if (!Ty)
    return LoongArchRegisterClass::GPRRC;

  Type *ScalarTy = Ty->getScalarType();
  if ((ScalarTy->isFloatTy() && ST->hasBasicF()) ||
      (ScalarTy->isDoubleTy() && ST->hasBasicD())) {
    return LoongArchRegisterClass::FPRRC;
  }

  return LoongArchRegisterClass::GPRRC;
}

unsigned
LoongArchTTIImpl::getMaxInterleaveFactor(ElementCount VF,
                                         bool HasUnorderedReductions) const {
  return ST->getMaxInterleaveFactor();
}

const char *LoongArchTTIImpl::getRegisterClassName(unsigned ClassID) const {
  switch (ClassID) {
  case LoongArchRegisterClass::GPRRC:
    return "LoongArch::GPRRC";
  case LoongArchRegisterClass::FPRRC:
    return "LoongArch::FPRRC";
  case LoongArchRegisterClass::VRRC:
    return "LoongArch::VRRC";
  }
  llvm_unreachable("unknown register class");
}

TargetTransformInfo::PopcntSupportKind
LoongArchTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasExtLSX() ? TTI::PSK_FastHardware : TTI::PSK_Software;
}

unsigned LoongArchTTIImpl::getCacheLineSize() const { return 64; }

unsigned LoongArchTTIImpl::getPrefetchDistance() const { return 200; }

bool LoongArchTTIImpl::enableWritePrefetching() const { return true; }

bool LoongArchTTIImpl::shouldExpandReduction(const IntrinsicInst *II) const {
  switch (II->getIntrinsicID()) {
  default:
    return true;
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_and:
  case Intrinsic::vector_reduce_or:
  case Intrinsic::vector_reduce_smax:
  case Intrinsic::vector_reduce_smin:
  case Intrinsic::vector_reduce_umax:
  case Intrinsic::vector_reduce_umin:
  case Intrinsic::vector_reduce_xor:
    return false;
  }
}

// Masked load/store are supported for all legal fixed-width vector types:
// the backend lowers them into a page-safety check + full-width vector
// access + bitsel blend (fast path) or a per-lane predicated sequence
// (cold path). Claiming legality lets the loop vectorizer widen
// conditionally-executed accesses and if-converted stores instead of
// leaving such loops scalar.
static bool isLegalMaskedMemOp(Type *ScalarTy) {
  if (ScalarTy->isPointerTy())
    return true;
  if (IntegerType *IntTy = dyn_cast<IntegerType>(ScalarTy))
    return llvm::isPowerOf2_32(IntTy->getBitWidth()) &&
           IntTy->getBitWidth() >= 8 && IntTy->getBitWidth() <= 64;
  return ScalarTy->isFloatTy() || ScalarTy->isDoubleTy();
}

// The loop vectorizer queries with the scalar type; the
// ScalarizeMaskedMemIntrin pass queries with the vector type. Accept both.
static bool isLegalMaskedMemOpTy(Type *Ty) {
  if (auto *VecTy = dyn_cast<VectorType>(Ty))
    return !VecTy->isScalableTy() &&
           isLegalMaskedMemOp(VecTy->getElementType());
  return isLegalMaskedMemOp(Ty);
}

bool LoongArchTTIImpl::isLegalMaskedLoad(Type *DataType, Align Alignment,
                                         unsigned AddressSpace,
                                         TTI::MaskKind MaskKind) const {
  return ST->hasExtLSX() && isLegalMaskedMemOpTy(DataType);
}

bool LoongArchTTIImpl::isLegalMaskedStore(Type *DataType, Align Alignment,
                                          unsigned AddressSpace,
                                          TTI::MaskKind MaskKind) const {
  return ST->hasExtLSX() && isLegalMaskedMemOpTy(DataType);
}

InstructionCost LoongArchTTIImpl::getMemIntrinsicInstrCost(
    const MemIntrinsicCostAttributes &MICA,
    TTI::TargetCostKind CostKind) const {
  // Masked load/store lowering instruction counts (see
  // LoongArchISelLowering.cpp emitMaskedLoadPseudo/emitMaskedStorePseudo):
  //   load:  add/xor/srli + branch + vld + bitsel           ~ 4
  //   store: add/xor/srli + branch + vld + bitsel + vst     ~ 5
  // This keeps masked widening cheaper than scalarizing (which costs O(VF)
  // predicated scalar accesses) while still more expensive than an
  // unconditional vector access.
  switch (MICA.getID()) {
  case Intrinsic::masked_load:
    return InstructionCost(4);
  case Intrinsic::masked_store:
    return InstructionCost(5);
  default:
    return BaseT::getMemIntrinsicInstrCost(MICA, CostKind);
  }
}

LoongArchTTIImpl::TTI::MemCmpExpansionOptions
LoongArchTTIImpl::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  TTI::MemCmpExpansionOptions Options;

  if (!ST->hasUAL())
    return Options;

  Options.MaxNumLoads = TLI->getMaxExpandSizeMemcmp(OptSize);
  Options.NumLoadsPerBlock = IsZeroCmp ? Options.MaxNumLoads : 1;
  Options.AllowOverlappingLoads = true;

  // TODO: Support for vectors.
  if (ST->is64Bit()) {
    Options.LoadSizes = {8, 4, 2, 1};
    Options.AllowedTailExpansions = {3, 5, 6};
  } else {
    Options.LoadSizes = {4, 2, 1};
    Options.AllowedTailExpansions = {3};
  }

  return Options;
}
