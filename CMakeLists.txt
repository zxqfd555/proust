cmake_minimum_required(VERSION 3.15)
project(proust LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /arch:AVX2 /fp:fast /favor:INTEL64 /MD")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-fno-rtti>
    )
    add_link_options($<$<CONFIG:Release>:-fuse-ld=lld>)
endif()

find_package(spdlog CONFIG REQUIRED)

add_library(project_dependencies STATIC
    ${CMAKE_SOURCE_DIR}/include/common/dataset.cpp
)
target_link_libraries(project_dependencies PUBLIC spdlog::spdlog)
target_compile_features(project_dependencies INTERFACE cxx_std_23)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(benchmarks)
add_subdirectory(tests)
