FetchContent_Declare(benchmark
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_ASSEMBLY_TESTS OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_WERROR OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(benchmark)

file(GLOB_RECURSE benchmark_files ${benchmark_SOURCE_DIR} *.cc *.c *.h)
set_source_files_properties(
    ${benchmark_files}
    TARGET_DIRECTORY benchmark
    PROPERTIES SKIP_LINTING ON
)
