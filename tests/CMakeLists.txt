enable_testing()
find_package(GTest REQUIRED)

add_executable(tests test_in_memory_simple.cpp)

target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(tests)
