# proust
Durable range query engine written in C++

## Building

The whole thing is supposed to be buildable with CMake. conan 2.22.2 was used for the package management.

```
conan install . --output-folder=build --build=missing
cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
cmake --build build
```
