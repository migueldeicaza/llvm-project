set(LLVM_OPTIONAL_SOURCES
  mlir-metal-runner.cpp
  metal-runtime-wrappers.cpp
  MetalRuntime.cpp
  MetalRuntime.h
  )

if (MLIR_METAL_RUNNER_ENABLED)
  message(STATUS "Building the Metal runner")

  add_llvm_library(metal-runtime-wrappers SHARED
    metal-runtime-wrappers.mm
    MetalRuntime.mm
  )

  target_include_directories(metal-runtime-wrappers
    PRIVATE ${Metal_INCLUDE_DIR}
    LLVMSupport
  )

  target_link_libraries(metal-runtime-wrappers
    LLVMSupport
    MLIRSPIRVSerialization
    LLVMCore
    LLVMSupport
    "-framework Metal"
  )

  set(LIBS
    LLVMCore
    LLVMSupport
    MLIRJitRunner
    MLIRAffineOps
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRFxpMathOps
    MLIRGPU
    MLIRGPUtoCUDATransforms
    MLIRGPUtoNVVMTransforms
    MLIRGPUtoSPIRVTransforms
    MLIRGPUtoMetalTransforms
    MLIRIR
    MLIRLLVMIR
    MLIRLinalgOps
    MLIRLoopToStandard
    MLIROpenMP
    MLIRParser
    MLIRQuantOps
    MLIRROCDLIR
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRStandardOps
    MLIRStandardToLLVM
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRTranslation
    ${Metal_LIBRARY}
  )

  # Manually expand the target library, since our MLIR libraries
  # aren't plugged into the LLVM dependency tracking. If we don't
  # do this then we can't insert the CodeGen library after ourselves
  llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
  # Prepend LLVM in front of every target, this is how the library
  # are named with CMake
  SET(targets_to_link)
  FOREACH(t ${TARGET_LIBS})
    LIST(APPEND targets_to_link "LLVM${t}")
  ENDFOREACH(t)

  add_llvm_tool(mlir-metal-runner
    mlir-metal-runner.mm
  )
  add_dependencies(mlir-metal-runner metal-runtime-wrappers)
  llvm_update_compile_flags(mlir-metal-runner)
  target_link_libraries(mlir-metal-runner PRIVATE ${FULL_LINK_LIBS} ${LIBS})

endif()
