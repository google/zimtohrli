find_package(ALSA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(flac REQUIRED flac)
pkg_check_modules(ogg REQUIRED ogg)
pkg_check_modules(vorbis REQUIRED vorbis)
pkg_check_modules(vorbisenc REQUIRED vorbisenc)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
pkg_check_modules(gles REQUIRED glesv2)

include(FetchContent)

FetchContent_Declare(highway
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.1.0
)
set(HWY_ENABLE_TESTS OFF CACHE INTERNAL "")
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(highway)

include(cmake/protobuf.cmake)

FetchContent_Declare(libsndfile
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
    GIT_TAG 1.2.2
)
FetchContent_MakeAvailable(libsndfile)

FetchContent_Declare(portaudio
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/PortAudio/portaudio.git
    GIT_TAG v19.7.0
)
target_compile_options(libprotobuf PRIVATE -Wno-deprecated-declarations)
FetchContent_MakeAvailable(portaudio)

FetchContent_Declare(googletest
     EXCLUDE_FROM_ALL
     GIT_REPOSITORY https://github.com/google/googletest.git
     GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

include(cmake/imgui.cmake)
include(cmake/benchmark.cmake)
include(cmake/pffft.cmake)
include(cmake/libsvm.cmake)
include(cmake/armadillo.cmake)
include(cmake/visqol.cmake)

FetchContent_Declare(samplerate
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/libsndfile/libsamplerate.git
    GIT_TAG 0.2.2
)
FetchContent_MakeAvailable(samplerate)