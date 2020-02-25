add_llvm_library(MLIRGPUtoMetalTransforms
  ConvertLaunchFuncToMetalCalls.cpp
  )

target_link_libraries(MLIRGPUtoMetalTransforms
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  MLIRTranslation
  )
