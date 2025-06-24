cmake_minimum_required(VERSION 3.26)

project(split-rendering LANGUAGES C CXX CUDA)

set(CC "native" CACHE STRING "Which compute capability to build for? e.g. 75, 86, 89, ...")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(DEPENDENCIES_DIR "${CMAKE_SOURCE_DIR}/dependencies")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/stb_image")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/tinyexr")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/json")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/fpng/src")

add_subdirectory("${DEPENDENCIES_DIR}/glm")

set(SOURCES 
    "src/main.cu" 
    "src/file_reader.cu" 
    "src/renderer.cu"
    "src/util/image_buffer.cu")
set(DEPENDENCIES_SOURCES "${DEPENDENCIES_DIR}/fpng/src/fpng.cpp")

add_executable(splinter ${SOURCES} ${DEPENDENCIES_SOURCES})
set_target_properties(splinter PROPERTIES CUDA_ARCHITECTURES ${CC})

include_directories("include")
target_include_directories(splinter PUBLIC ${DEPENDENCIES_INCLUDE_DIRS})

target_compile_definitions(splinter PRIVATE SOURCE_ROOT=${CMAKE_SOURCE_DIR} FPNG_NO_SSE=0)

if (UNIX)
    target_compile_options(splinter PRIVATE $<$<COMPILE_LANGUAGE:CXX>: $<$<CONFIG:RelWithDebInfo>:-pg> -msse4.1 -mpclmul -fno-strict-aliasing -march=native>)
    target_compile_options(splinter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v>)
    target_compile_options(splinter PRIVATE $<$<CONFIG:Debug>: $<$<COMPILE_LANGUAGE:CXX>: -g>>)
else()
    set_property(TARGET splinter PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(splinter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
    target_compile_options(splinter PRIVATE $<$<CONFIG:RelWithDebInfo>: -dopt=on>)
    target_compile_options(splinter PRIVATE $<$<CONFIG:Debug>: $<$<COMPILE_LANGUAGE:CXX>: /Zi>>)
endif()

target_compile_options(splinter PRIVATE $<$<CONFIG:Debug>: $<$<COMPILE_LANGUAGE:CUDA>: -G -Xptxas -O0>>)

target_link_libraries(splinter PUBLIC glm::glm)