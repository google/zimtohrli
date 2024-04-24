FetchContent_Declare(protobuf
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v26.1
)
set(protobuf_BUILD_TESTS OFF)
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(protobuf)

target_compile_options(libprotobuf PRIVATE -Wno-attributes)

file(GLOB_RECURSE protobuf_files ${protobuf_SOURCE_DIR} *.cc *.c *.h)
set_source_files_properties(
    ${protobuf_files}
    TARGET_DIRECTORY
        libprotobuf
        absl::base
        absl::strings
        absl::debugging
        absl::hash
        absl::flags
        absl::log
        absl::sample_recorder
        absl::any
        absl::time
        absl::container_common
        absl::crc_internal
        absl::numeric
        absl::synchronization
        absl::status
    PROPERTIES SKIP_LINTING ON
)
