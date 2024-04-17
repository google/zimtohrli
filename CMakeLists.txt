cmake_minimum_required(VERSION 3.27)

project(Zimtohrli)

find_package(ALSA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(flac REQUIRED flac)
pkg_check_modules(ogg REQUIRED ogg)
pkg_check_modules(vorbis REQUIRED vorbis)
pkg_check_modules(vorbisenc REQUIRED vorbisenc)


include(FetchContent)

FetchContent_Declare(highway
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.1.0
)
set(HWY_ENABLE_TESTS OFF CACHE INTERNAL "")
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(highway)

if (NOT TARGET absl::base)
FetchContent_Declare(abseil-cpp
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20240116.1
)
set(ABSL_PROPAGATE_CXX_STD ON)
FetchContent_MakeAvailable(abseil-cpp)
endif()

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
FetchContent_MakeAvailable(portaudio)

FetchContent_Declare(googletest
     EXCLUDE_FROM_ALL
     GIT_REPOSITORY https://github.com/google/googletest.git
     GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(benchmark
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_ASSEMBLY_TESTS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(imgui
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.4
)
FetchContent_MakeAvailable(imgui)

enable_testing()
include(GoogleTest)

add_library(zimtohrli_base STATIC
    cpp/zimt/audio.cc
    cpp/zimt/audio.h
    cpp/zimt/cam.cc
    cpp/zimt/cam.h
    cpp/zimt/dtw.cc
    cpp/zimt/dtw.h
    cpp/zimt/elliptic.cc
    cpp/zimt/elliptic.h
    cpp/zimt/filterbank.cc
    cpp/zimt/filterbank.h
    cpp/zimt/loudness.cc
    cpp/zimt/loudness.h
    cpp/zimt/masking.cc
    cpp/zimt/masking.h
    cpp/zimt/zimtohrli.cc
    cpp/zimt/zimtohrli.h
)
target_include_directories(zimtohrli_base PUBLIC cpp)
target_link_libraries(zimtohrli_base PRIVATE absl::check)
target_link_libraries(zimtohrli_base PUBLIC hwy portaudio absl::statusor sndfile)

find_package(Python3 COMPONENTS Interpreter Development)
add_library(zimtohrli_pyohrli SHARED
    cpp/zimt/pyohrli.cc
)
set_target_properties(zimtohrli_pyohrli PROPERTIES
    PREFIX ""
    OUTPUT_NAME _pyohrli.so
    SUFFIX ""
)
target_link_libraries(zimtohrli_pyohrli zimtohrli_base Python3::Python absl::check)

add_library(zimtohrli_goohrli_glue STATIC
    cpp/zimt/goohrli.cc
    go/goohrli/goohrli.h
)
target_include_directories(zimtohrli_goohrli_glue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli)
target_link_libraries(zimtohrli_goohrli_glue zimtohrli_base)

set(goohrli_object ${CMAKE_CURRENT_BINARY_DIR}/goohrli.o)
set(goohrli_archive ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli/goohrli.a)
add_custom_command(
    OUTPUT ${goohrli_archive}
    COMMAND ${CMAKE_LINKER} -r
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_goohrli_glue.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_base.dir/ -name \"*.o\"\)
        $$\(find ${portaudio_BINARY_DIR} -name \"*.o\"\)
        $$\(find ${absl_BINARY_DIR} -name \"*.o\"\)
        $$\(find ${hwy_BINARY_DIR} -name \"*.o\"\)
        $$\(find ${libsndfile_BINARY_DIR}/CMakeFiles/sndfile.dir/ -name \"*.o\"\)
        -o ${goohrli_object}
    COMMAND ${CMAKE_AR} rcs ${goohrli_archive} ${goohrli_object}
    DEPENDS zimtohrli_goohrli_glue
)
add_custom_target(zimtohrli_goohrli ALL DEPENDS ${goohrli_archive})

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
pkg_check_modules(gles QUIET glesv2)
FetchContent_GetProperties(imgui)
add_library(zimtohrli_ux STATIC
    cpp/zimt/ux.cc
    cpp/zimt/ux.h
    ${imgui_SOURCE_DIR}/imconfig.h
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui.h
    ${imgui_SOURCE_DIR}/imgui_internal.h
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imstb_rectpack.h
    ${imgui_SOURCE_DIR}/imstb_textedit.h
    ${imgui_SOURCE_DIR}/imstb_truetype.h
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3_loader.h
)
target_include_directories(zimtohrli_ux PRIVATE ${imgui_SOURCE_DIR}/backends ${gles_INCLUDE_DIRS})
target_include_directories(zimtohrli_ux PUBLIC ${imgui_SOURCE_DIR})
target_link_libraries(zimtohrli_ux zimtohrli_base glfw OpenGL::GL)

add_executable(zimtohrli_compare
    cpp/zimt/compare.cc
)
target_link_libraries(zimtohrli_compare zimtohrli_ux zimtohrli_base absl::flags_parse)
set_target_properties(zimtohrli_compare PROPERTIES
    PREFIX ""
    OUTPUT_NAME compare
    SUFFIX ""
)

option(BUILD_ZIMTOHRLI_TESTS "Build Zimtohrli test binaries." ON)
if (BUILD_ZIMTOHRLI_TESTS)
add_executable(zimtohrli_test
    cpp/zimt/audio_test.cc
    cpp/zimt/cam_test.cc
    cpp/zimt/dtw_test.cc
    cpp/zimt/elliptic_test.cc
    cpp/zimt/filterbank_test.cc
    cpp/zimt/loudness_test.cc
    cpp/zimt/masking_test.cc
    cpp/zimt/zimtohrli_test.cc
    cpp/zimt/test_file_paths.cc
)
target_link_libraries(zimtohrli_test zimtohrli_base gtest gmock_main benchmark)
target_compile_definitions(zimtohrli_test PRIVATE CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(zimtohrli_test)

set(python3_VENV_DIR ${CMAKE_CURRENT_BINARY_DIR}/venv)
set(python3_VENV ${python3_VENV_DIR}/bin/python3)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp/zimt/pyohrli.py ${CMAKE_CURRENT_BINARY_DIR}/pyohrli.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp/zimt/pyohrli_test.py ${CMAKE_CURRENT_BINARY_DIR}/pyohrli_test.py COPYONLY)
add_test(NAME zimtohrli_pyohrli_test
    COMMAND sh -c "${Python3_EXECUTABLE} -m venv ${python3_VENV_DIR} &&
        ${python3_VENV} -m pip install jax jaxlib numpy scipy &&
        ${python3_VENV} pyohrli_test.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_test(NAME zimtohrli_python_test
    COMMAND sh -c "${Python3_EXECUTABLE} -m venv ${python3_VENV_DIR} &&
        ${python3_VENV} -m pip install jax jaxlib numpy scipy &&
        ${python3_VENV} -m unittest *_test.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)
add_test(NAME zimtohrli_ipynb_test
    COMMAND sh -c "${Python3_EXECUTABLE} -m venv ${python3_VENV_DIR} &&
        ${python3_VENV} -m pip install jax jaxlib numpy scipy jupyter nbconvert matplotlib &&
        ${python3_VENV_DIR}/bin/jupyter nbconvert --to notebook --execute ${CMAKE_CURRENT_SOURCE_DIR}/zimtohrli.ipynb --stdout"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME zimtohrli_loudness_computation_ipynb_test
    COMMAND sh -c "${Python3_EXECUTABLE} -m venv ${python3_VENV_DIR} &&
        ${python3_VENV} -m pip install jax jaxlib numpy scipy jupyter nbconvert pandas matplotlib &&
        ${python3_VENV_DIR}/bin/jupyter nbconvert --to notebook --execute ${CMAKE_CURRENT_SOURCE_DIR}/python/loudness_parameter_computation.ipynb --stdout"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


add_test(NAME zimtohrli_go_test
    COMMAND go test ./...
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(zimtohrli_benchmark
    cpp/zimt/dtw_test.cc
    cpp/zimt/elliptic_test.cc
    cpp/zimt/filterbank_test.cc
    cpp/zimt/loudness_test.cc
    cpp/zimt/masking_test.cc
    cpp/zimt/zimtohrli_test.cc
)
target_link_libraries(zimtohrli_benchmark zimtohrli_base gtest gmock benchmark_main)
endif()

