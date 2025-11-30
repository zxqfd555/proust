# proust
Durable range query engine written in C++

## The Task

### Generalization

### Assumptions

### Basic Engine

### Profiling and Design Decisions

### Replication and ACID

#### Synchronous replication. Bounds

#### Asynchronous replication

## Building and Dependencies

The whole thing is supposed to be buildable with CMake. Conan v2.22.2 was used for the package management. Use your best judgement, if you're familiar with the package manager. Otherwise, here are the key points you need to know in order to make everything running.

First, you will need to configure a Conan build profile. You can do that with `conan profile detect`. If nonexistent before, the profile will be created. You can view it with `conan profile show`. You may equally want to set edit certain settings. For instance, you'll need C++23 (specify that in `compiler.cppstd`). You may also want to change the type of compiler.

Note that with the default configuration, one Conan profile means one build mode: either debug or release, but not both. Due to that, if you want to develop the engine or run the tests properly, you will need put in place, another profile, for debugging purposes. You can do that by first, duplicating the existing one with `conan profile detect --name=debug`, and then editing it - in particular you'll need to set `compiler.runtime_type=Debug`.

One you're done with profiles, you can first checkout the dependencies from the C++ package repository. You can do that with `conan install . --output-folder=build --build=missing --profile=debug` (note that you'll need to omit this `profile=...` part for release). Conan will checkout only one version of a package, either debug or release. Then, prepare CMake with `cmake -B build -DCMAKE_TOOLCHAIN_FILE="build/generators/conan_toolchain.cmake"`. And then, you're ready to build with `cmake --build build` for debug or `cmake --build build --config Release` for release. Make sure that you align your Conan profile with the mode that will be used for the build. Of course, once the preparation is done, once you edit the code, you can run only that last `cmake --build build` command.

## Ensuring code quality

### Unit tests

### Stress tests

### Linters and static analysis tools

### Sanitizers

### Spellchecker

## Benchmarks