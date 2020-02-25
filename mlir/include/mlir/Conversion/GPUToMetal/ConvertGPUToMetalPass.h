//===- ConvertGPUToMetalPass.h - GPU to Metal conversion pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file declares a pass to convert GPU dialect ops to to Metal runtime
// calls.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOMETAL_CONVERTGPUTOMETALPASS_H
#define MLIR_CONVERSION_GPUTOMETAL_CONVERTGPUTOMETALPASS_H

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OpPassBase;

std::unique_ptr<OpPassBase<ModuleOp>>
createConvertGpuLaunchFuncToMetalCallsPass();

} // namespace mlir
#endif // MLIR_CONVERSION_GPUTOMETAL_CONVERTGPUTOMETALPASS_H
