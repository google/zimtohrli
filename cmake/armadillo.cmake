FetchContent_Declare(armadillo
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG 15.2.6
)
set(BUILD_SMOKE_TEST OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(armadillo)

file(GLOB_RECURSE armadillo_files ${armadillo_SOURCE_DIR} *.cc *.c *.h)
set_source_files_properties(
    ${armadillo_files}
    TARGET_DIRECTORY armadillo
    PROPERTIES SKIP_LINTING ON
)
