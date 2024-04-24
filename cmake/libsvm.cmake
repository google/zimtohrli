FetchContent_Declare(libsvm
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/cjlin1/libsvm.git
    GIT_TAG v332
)
FetchContent_MakeAvailable(libsvm)

set(libsvm_files
    ${libsvm_SOURCE_DIR}/svm.cpp
    ${libsvm_SOURCE_DIR}/svm.h
)
add_library(libsvm STATIC ${libsvm_files})
target_include_directories(libsvm PUBLIC ${libsvm_SOURCE_DIR})

set_source_files_properties(
    ${libsvm_files}
    TARGET_DIRECTORY libsvm
    PROPERTIES SKIP_LINTING ON
)
