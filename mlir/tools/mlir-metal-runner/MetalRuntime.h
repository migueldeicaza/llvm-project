//===- MetalRuntime.cpp - MLIR Metal runtime ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Metal runtime API.
//
//===----------------------------------------------------------------------===//

#ifndef METAL_RUNTIME_H
#define METAL_RUNTIME_H

#include "mlir/Analysis/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ToolOutputFile.h"

#include <metal/metal.h>

using namespace mlir;

using DescriptorSetIndex = uint32_t;
using BindingIndex = uint32_t;

/// Struct containing information regarding to a device memory buffer.
struct MetalDeviceMemoryBuffer {
  BindingIndex bindingIndex{0};
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
  VkDescriptorBufferInfo bufferInfo{};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory deviceMemory{VK_NULL_HANDLE};
};

/// Struct containing information regarding to a host memory buffer.
struct MetalHostMemoryBuffer {
  /// Pointer to a host memory.
  void *ptr{nullptr};
  /// Size of a host memory in bytes.
  uint32_t size{0};
};

/// Struct containing the number of local workgroups to dispatch for each
/// dimension.
struct NumWorkGroups {
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
};

/// Struct containing information regarding a descriptor set.
struct DescriptorSetInfo {
  /// Index of a descriptor set in descriptor sets.
  DescriptorSetIndex descriptorSet{0};
  /// Number of desriptors in a set.
  uint32_t descriptorSize{0};
  /// Type of a descriptor set.
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
};

/// MetalHostMemoryBuffer mapped into a descriptor set and a binding.
using ResourceData =
    llvm::DenseMap<DescriptorSetIndex,
                   llvm::DenseMap<BindingIndex, MetalHostMemoryBuffer>>;

/// StorageClass mapped into a descriptor set and a binding.
using ResourceStorageClassBindingMap =
    llvm::DenseMap<DescriptorSetIndex,
                   llvm::DenseMap<BindingIndex, mlir::spirv::StorageClass>>;

inline void emitMetalError(const llvm::Twine &message, VkResult error) {
  llvm::errs()
      << message.concat(" failed with error code ").concat(llvm::Twine{error});
}

#define RETURN_ON_METAL_ERROR(result, msg)                                    \
  if ((result) != VK_SUCCESS) {                                                \
    emitMetalError(msg, (result));                                            \
    return failure();                                                          \
  }

/// Metal runtime.
/// The purpose of this class is to run SPIR-V compute shader on Metal
/// device.
/// Before the run, user must provide and set resource data with descriptors,
/// SPIR-V shader, number of work groups and entry point. After the creation of
/// MetalRuntime, special methods must be called in the following
/// sequence: initRuntime(), run(), updateHostMemoryBuffers(), destroy();
/// each method in the sequence returns succes or failure depends on the Metal
/// result code.
class MetalRuntime {
public:
  explicit MetalRuntime() = default;
  MetalRuntime(const MetalRuntime &) = delete;
  MetalRuntime &operator=(const MetalRuntime &) = delete;

  /// Sets needed data for Metal runtime.
  void setResourceData(const ResourceData &resData);
  void setResourceData(const DescriptorSetIndex desIndex,
                       const BindingIndex bindIndex,
                       const MetalHostMemoryBuffer &hostMemBuffer);
  void setShaderModule(uint8_t *shader, uint32_t size);
  void setNumWorkGroups(const NumWorkGroups &numberWorkGroups);
  void setResourceStorageClassBindingMap(
      const ResourceStorageClassBindingMap &stClassData);
  void setEntryPoint(const char *entryPointName);

  /// Runtime initialization.
  LogicalResult initRuntime();

  /// Runs runtime.
  LogicalResult run();

  /// Updates host memory buffers.
  LogicalResult updateHostMemoryBuffers();

  /// Destroys all created metal objects and resources.
  LogicalResult destroy();

private:
  //===--------------------------------------------------------------------===//
  // Pipeline creation methods.
  //===--------------------------------------------------------------------===//

  LogicalResult createInstance();
  LogicalResult createDevice();
  LogicalResult getBestComputeQueue(const VkPhysicalDevice &physicalDevice);
  LogicalResult createMemoryBuffers();
  LogicalResult createShaderModule();
  void initDescriptorSetLayoutBindingMap();
  LogicalResult createDescriptorSetLayout();
  LogicalResult createPipelineLayout();
  LogicalResult createComputePipeline();
  LogicalResult createDescriptorPool();
  LogicalResult allocateDescriptorSets();
  LogicalResult setWriteDescriptors();
  LogicalResult createCommandPool();
  LogicalResult createComputeCommandBuffer();
  LogicalResult submitCommandBuffersToQueue();

  //===--------------------------------------------------------------------===//
  // Helper methods.
  //===--------------------------------------------------------------------===//

  /// Maps storage class to a descriptor type.
  LogicalResult
  mapStorageClassToDescriptorType(spirv::StorageClass storageClass,
                                  VkDescriptorType &descriptorType);

  /// Maps storage class to buffer usage flags.
  LogicalResult
  mapStorageClassToBufferUsageFlag(spirv::StorageClass storageClass,
                                   VkBufferUsageFlagBits &bufferUsage);

  LogicalResult countDeviceMemorySize();

  //===--------------------------------------------------------------------===//
  // Metal objects.
  //===--------------------------------------------------------------------===//

  VkInstance instance;
  VkDevice device;
  VkQueue queue;

  /// Specifies MetalDeviceMemoryBuffers divided into sets.
  llvm::DenseMap<DescriptorSetIndex,
                 llvm::SmallVector<MetalDeviceMemoryBuffer, 1>>
      deviceMemoryBufferMap;

  /// Specifies shader module.
  VkShaderModule shaderModule;

  /// Specifies layout bindings.
  llvm::DenseMap<DescriptorSetIndex,
                 llvm::SmallVector<VkDescriptorSetLayoutBinding, 1>>
      descriptorSetLayoutBindingMap;

  /// Specifies layouts of descriptor sets.
  llvm::SmallVector<VkDescriptorSetLayout, 1> descriptorSetLayouts;
  VkPipelineLayout pipelineLayout;

  /// Specifies descriptor sets.
  llvm::SmallVector<VkDescriptorSet, 1> descriptorSets;

  /// Specifies a pool of descriptor set info, each descriptor set must have
  /// information such as type, index and amount of bindings.
  llvm::SmallVector<DescriptorSetInfo, 1> descriptorSetInfoPool;
  VkDescriptorPool descriptorPool;

  /// Computation pipeline.
  VkPipeline pipeline;
  VkCommandPool commandPool;
  llvm::SmallVector<VkCommandBuffer, 1> commandBuffers;

  //===--------------------------------------------------------------------===//
  // Metal memory context.
  //===--------------------------------------------------------------------===//

  uint32_t queueFamilyIndex{0};
  uint32_t memoryTypeIndex{VK_MAX_MEMORY_TYPES};
  VkDeviceSize memorySize{0};

  //===--------------------------------------------------------------------===//
  // Metal execution context.
  //===--------------------------------------------------------------------===//

  NumWorkGroups numWorkGroups;
  const char *entryPoint{nullptr};
  uint8_t *binary{nullptr};
  uint32_t binarySize{0};

  //===--------------------------------------------------------------------===//
  // Metal resource data and storage classes.
  //===--------------------------------------------------------------------===//

  ResourceData resourceData;
  ResourceStorageClassBindingMap resourceStorageClassData;
};
#endif
