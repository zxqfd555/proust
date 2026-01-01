add_executable(generate_dataset generate_dataset.cpp)
add_executable(runner runner.cpp)

target_compile_options(runner PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

target_link_libraries(generate_dataset PRIVATE project_dependencies)
target_link_libraries(runner PRIVATE project_dependencies)
