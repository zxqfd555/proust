# Build orchestration for the project.
#
# Dependencies (gtest, spdlog) are resolved with Conan; configuration and
# compilation go through CMake presets that Conan generates. The toolchain
# (cmake, conan, ninja) is expected in ~/.local/bin (pip --user install).
#
#   make debug     build the Debug configuration (all targets)
#   make release   build the Release configuration (all targets)
#   make test      build the tests under Debug + sanitizers and run them
#   make clean     remove the build/ tree

SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)

JOBS ?= $(shell nproc)

SRC_DIR := $(CURDIR)
BUILD_DEBUG := $(SRC_DIR)/build/Debug
BUILD_RELEASE := $(SRC_DIR)/build/Release

# C++23 is enforced for the dependencies too, so their ABI matches the project.
CONAN_FLAGS := -s compiler.cppstd=23 --build=missing

DEBUG_TOOLCHAIN := $(BUILD_DEBUG)/generators/conan_toolchain.cmake
RELEASE_TOOLCHAIN := $(BUILD_RELEASE)/generators/conan_toolchain.cmake

.DEFAULT_GOAL := debug
.PHONY: debug release test clean

debug: $(DEBUG_TOOLCHAIN)
	cmake --preset conan-debug
	cmake --build $(BUILD_DEBUG) -j$(JOBS)

release: $(RELEASE_TOOLCHAIN)
	cmake --preset conan-release
	cmake --build $(BUILD_RELEASE) -j$(JOBS)

# The tests target carries the sanitizer flags unconditionally (see
# tests/CMakeLists.txt); building it under the Debug config keeps asserts live.
test: $(DEBUG_TOOLCHAIN)
	cmake --preset conan-debug
	cmake --build $(BUILD_DEBUG) --target tests -j$(JOBS)
	ctest --test-dir $(BUILD_DEBUG)/tests --output-on-failure -j$(JOBS)

clean:
	rm -rf $(SRC_DIR)/build

# Conan install is re-run only when the recipe changes; it writes the toolchain
# file and the CMake preset consumed above.
$(DEBUG_TOOLCHAIN): conanfile.txt
	conan install $(SRC_DIR) -s build_type=Debug $(CONAN_FLAGS)

$(RELEASE_TOOLCHAIN): conanfile.txt
	conan install $(SRC_DIR) -s build_type=Release $(CONAN_FLAGS)
