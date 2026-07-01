add_executable(generate_dataset generate_dataset.cpp)
add_executable(runner runner.cpp)
add_executable(external_microbench external_microbench.cpp)
add_executable(fence_experiment fence_experiment.cpp)
add_executable(branching_microbench branching_microbench.cpp)
add_executable(query_microbench query_microbench.cpp)

target_compile_options(runner PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

target_link_libraries(generate_dataset PRIVATE project_dependencies)
target_link_libraries(runner PRIVATE project_dependencies)
target_link_libraries(external_microbench PRIVATE project_dependencies)
target_link_libraries(fence_experiment PRIVATE project_dependencies)
target_link_libraries(branching_microbench PRIVATE project_dependencies)
target_compile_options(branching_microbench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-funroll-loops>
)
target_link_libraries(query_microbench PRIVATE project_dependencies)
target_compile_options(query_microbench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-funroll-loops>
)
# The Algorithmica wide tree's hot loops are tiny and fixed-trip (H layers); the upstream
# benchmark compiles with -funroll-loops, and the structure leans on it to turn per-layer
# offsets and shifts into immediates. Match that here.
target_compile_options(external_microbench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-funroll-loops>
)
add_executable(maxsum_microbench maxsum_microbench.cpp)
target_link_libraries(maxsum_microbench PRIVATE project_dependencies)
