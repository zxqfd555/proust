enable_testing()
find_package(GTest REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(tests
    test_dataset_serialization.cpp
    test_in_memory_simple.cpp
    test_in_memory_stress.cpp
)

target_link_libraries(tests PRIVATE project_dependencies)
target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(tests)
