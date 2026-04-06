cmake_minimum_required(VERSION 3.18)
project(curfp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# C library
# ---------------------------------------------------------------------------
add_library(curfp SHARED
    src/curfp_handle.cpp
    src/curfp_ssfrk.cpp
    src/curfp_spftrf.cpp
)

target_include_directories(curfp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(curfp
    PUBLIC  CUDA::cudart
    PRIVATE CUDA::cublas CUDA::cusolver
)

# ---------------------------------------------------------------------------
# Python extension (built by scikit-build-core or via -DCURFP_BUILD_PYTHON=ON)
# ---------------------------------------------------------------------------
option(CURFP_BUILD_PYTHON "Build Python extension" OFF)

if(CURFP_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_curfp_C curfp/_ext.cpp)

    target_include_directories(_curfp_C PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # curfp is PUBLIC CUDA::cudart, so CUDA include dirs propagate here too
    target_link_libraries(_curfp_C PRIVATE curfp)

    # $ORIGIN so _curfp_C.so finds libcurfp.so in the same directory
    set_target_properties(_curfp_C PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    install(TARGETS curfp    DESTINATION curfp)
    install(TARGETS _curfp_C DESTINATION curfp)
endif()

# ---------------------------------------------------------------------------
# C++ tests
# ---------------------------------------------------------------------------
option(CURFP_BUILD_TESTS "Build C++ tests" ON)
if(CURFP_BUILD_TESTS AND NOT SKBUILD)
    enable_testing()
    add_subdirectory(tests)
endif()
