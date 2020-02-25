//===- metal-runtime-wrappers.cpp - MLIR Metal runner wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C runtime wrappers around the MetalRuntime.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <numeric>

#include "MetalRuntime.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

namespace {

// TODO(denis0x0D): This static machinery should be replaced by `initMetal` and
// `deinitMetal` to be more explicit and to avoid static initialization and
// destruction.
class MetalRuntimeManager;
static llvm::ManagedStatic<MetalRuntimeManager> vkRuntimeManager;

class MetalRuntimeManager {
  public:
    MetalRuntimeManager() = default;
    MetalRuntimeManager(const MetalRuntimeManager &) = delete;
    MetalRuntimeManager operator=(const MetalRuntimeManager &) = delete;
    ~MetalRuntimeManager() = default;

    void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                         const MetalHostMemoryBuffer &memBuffer) {
      std::lock_guard<std::mutex> lock(mutex);
      metalRuntime.setResourceData(setIndex, bindIndex, memBuffer);
    }

    void setEntryPoint(const char *entryPoint) {
      std::lock_guard<std::mutex> lock(mutex);
      metalRuntime.setEntryPoint(entryPoint);
    }

    void setNumWorkGroups(NumWorkGroups numWorkGroups) {
      std::lock_guard<std::mutex> lock(mutex);
      metalRuntime.setNumWorkGroups(numWorkGroups);
    }

    void setShaderModule(uint8_t *shader, uint32_t size) {
      std::lock_guard<std::mutex> lock(mutex);
      metalRuntime.setShaderModule(shader, size);
    }

    void runOnMetal() {
      std::lock_guard<std::mutex> lock(mutex);
      if (failed(metalRuntime.initRuntime()) || failed(metalRuntime.run()) ||
          failed(metalRuntime.updateHostMemoryBuffers()) ||
          failed(metalRuntime.destroy())) {
        llvm::errs() << "runOnMetal failed";
      }
    }

  private:
    MetalRuntime metalRuntime;
    std::mutex mutex;
};

} // namespace

extern "C" {
/// Fills the given memref with the given value.
/// Binds the given memref to the given descriptor set and descriptor index.
void setResourceData(const DescriptorSetIndex setIndex, BindingIndex bindIndex,
                     float *allocated, float *aligned, int64_t offset,
                     int64_t size, int64_t stride, float value) {
  std::fill_n(allocated, size, value);
  MetalHostMemoryBuffer memBuffer{allocated,
                                   static_cast<uint32_t>(size * sizeof(float))};
  vkRuntimeManager->setResourceData(setIndex, bindIndex, memBuffer);
}

void setEntryPoint(const char *entryPoint) {
  vkRuntimeManager->setEntryPoint(entryPoint);
}

void setNumWorkGroups(uint32_t x, uint32_t y, uint32_t z) {
  vkRuntimeManager->setNumWorkGroups({x, y, z});
}

void setBinaryShader(uint8_t *shader, uint32_t size) {
  vkRuntimeManager->setShaderModule(shader, size);
}

void runOnMetal() { vkRuntimeManager->runOnMetal(); }
}
