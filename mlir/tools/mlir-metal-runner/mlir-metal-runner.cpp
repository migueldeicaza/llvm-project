//===- mlir-metal-runner.cpp - MLIR Metal Execution Driver --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the Metal by
// translating MLIR GPU module to SPIR-V and host part to LLVM IR before
// JIT-compiling and executing the latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToMetal/ConvertGPUToMetalPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/JitRunner.h"

using namespace mlir;

static LogicalResult runMLIRPasses(ModuleOp module) {
  PassManager passManager(module.getContext());
  applyPassManagerCLOptions(passManager);

  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  passManager.addPass(createConvertGPUToSPIRVPass());
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createLowerABIAttributesPass());
  passManager.addPass(createConvertGpuLaunchFuncToMetalCallsPass());
  passManager.addPass(createLowerToLLVMPass());
  return passManager.run(module);
}

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();
  return mlir::JitRunnerMain(argc, argv, &runMLIRPasses);
}
