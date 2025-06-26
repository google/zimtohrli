find_package(PkgConfig REQUIRED)
pkg_check_modules(flac REQUIRED flac)
pkg_check_modules(ogg REQUIRED ogg)
pkg_check_modules(vorbis REQUIRED vorbis)
pkg_check_modules(vorbisenc REQUIRED vorbisenc)

pkg_check_modules(soxr REQUIRED IMPORTED_TARGET soxr)

include(FetchContent)

include(cmake/protobuf.cmake)

FetchContent_Declare(libsndfile
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
    GIT_TAG 1.2.2
)
FetchContent_MakeAvailable(libsndfile)

FetchContent_Declare(googletest
     EXCLUDE_FROM_ALL
     GIT_REPOSITORY https://github.com/google/googletest.git
     GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

include(cmake/benchmark.cmake)
include(cmake/pffft.cmake)
include(cmake/libsvm.cmake)
include(cmake/armadillo.cmake)
include(cmake/visqol.cmake)
