cmake_minimum_required(VERSION 3.15)
project(proust LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /arch:AVX2 /fp:fast /favor:INTEL64 /MD")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    add_compile_options(
        -O3
        -march=native
        -ffast-math
        -flto
        -fno-exceptions
        -fno-rtti
    )
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(tests)
