set(LLVM_OPTIONAL_SOURCES
  mlir-metal-runner.cpp
  metal-runtime-wrappers.cpp
  MetalRuntime.cpp
  MetalRuntime.h
  )

if (MLIR_METAL_RUNNER_ENABLED)
  message(STATUS "Building the Metal runner")

  # At first try "FindMetal" from:
  # https://cmake.org/cmake/help/v3.7/module/FindMetal.html
  if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
    find_package(Metal)
  endif()

  # If Metal is not found try a path specified by METAL_SDK.
  if (NOT Metal_FOUND)
    if ("$ENV{METAL_SDK}" STREQUAL "")
      message(FATAL_ERROR "Please use at least CMAKE 3.7.0 or provide "
                          "METAL_SDK path as an environment variable")
    endif()

    find_library(Metal_LIBRARY metal HINTS "$ENV{METAL_SDK}/lib" REQUIRED)
    if (Metal_LIBRARY)
      set(Metal_FOUND ON)
      set(Metal_INCLUDE_DIR "$ENV{METAL_SDK}/include")
      message(STATUS "Found Metal: " ${Metal_LIBRARY})
    endif()
  endif()

  if (NOT Metal_FOUND)
    message(FATAL_ERROR "Cannot find Metal library")
  endif()

  add_llvm_library(metal-runtime-wrappers SHARED
    metal-runtime-wrappers.cpp
    MetalRuntime.cpp
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
    ${Metal_LIBRARY}
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
    mlir-metal-runner.cpp
  )
  add_dependencies(mlir-metal-runner metal-runtime-wrappers)
  llvm_update_compile_flags(mlir-metal-runner)
  target_link_libraries(mlir-metal-runner PRIVATE ${FULL_LINK_LIBS} ${LIBS})

endif()
