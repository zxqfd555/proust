enable_testing()
find_package(GTest REQUIRED)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(SANITIZER_FLAGS
        -fsanitize=address,undefined,leak
        -fno-omit-frame-pointer
        -fsanitize-address-use-after-scope
    )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(SANITIZER_FLAGS
        /RTC1
        /analyze
    )
endif()

add_executable(tests
    test_dataset_serialization.cpp
    test_fat_engine.cpp
    test_point_update_engine.cpp
    test_range_update_engine.cpp
    test_simple.cpp
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tests PRIVATE ${SANITIZER_FLAGS})
    target_link_options(tests PRIVATE ${SANITIZER_FLAGS})
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(tests PRIVATE
        $<$<CONFIG:Debug>:/RTC1 /analyze>
    )
endif()

target_link_libraries(tests PRIVATE project_dependencies)
target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(tests)
