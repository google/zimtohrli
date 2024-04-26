cmake_minimum_required(VERSION 3.27)

project(Zimtohrli)

set(CMAKE_CXX_CLANG_TIDY clang-tidy)

include(cmake/deps.cmake)

enable_testing()
include(GoogleTest)

option(ZIMTOHRLI_ASAN "Build Zimtohrli with address sanitizer." OFF)
macro(asan_target TARGET)
    if (ZIMTOHRLI_ASAN)
    target_compile_options(${TARGET} PRIVATE -fsanitize=address)
    target_link_options(${TARGET} PRIVATE -fsanitize=address)
    endif()
endmacro()

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
    cpp/zimt/mos.cc
    cpp/zimt/mos.h
    cpp/zimt/zimtohrli.cc
    cpp/zimt/zimtohrli.h
)
asan_target(zimtohrli_base)
target_include_directories(zimtohrli_base PUBLIC cpp)
target_link_libraries(zimtohrli_base PRIVATE absl::check)
target_link_libraries(zimtohrli_base PUBLIC hwy portaudio absl::statusor absl::span sndfile)

add_library(zimtohrli_visqol_adapter STATIC
    cpp/zimt/visqol_model.h
    cpp/zimt/visqol_model.cc
    cpp/zimt/visqol.h
    cpp/zimt/visqol.cc
    cpp/zimt/resample.h
)
asan_target(zimtohrli_visqol_adapter)
target_include_directories(zimtohrli_visqol_adapter PUBLIC cpp)
target_link_libraries(zimtohrli_visqol_adapter PRIVATE visqol samplerate hwy)
target_link_libraries(zimtohrli_visqol_adapter PUBLIC absl::span)

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
asan_target(zimtohrli_goohrli_glue)
target_include_directories(zimtohrli_goohrli_glue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli ${CMAKE_CURRENT_SOURCE_DIR}/go/gosqol)
target_link_libraries(zimtohrli_goohrli_glue zimtohrli_base zimtohrli_visqol_adapter)

set(zimtohrli_goohrli_object ${CMAKE_CURRENT_BINARY_DIR}/goohrli.o)
set(zimtohrli_goohrli_archive ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli/goohrli.a)
add_custom_command(
    OUTPUT ${zimtohrli_goohrli_archive}
    COMMAND ${CMAKE_LINKER} -r
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_goohrli_glue.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_base.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_visqol_adapter.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/visqol.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/visqol_proto.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/libsvm.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/pffft.dir/ -name \"*.o\"\)
        $$\(find ${portaudio_BINARY_DIR} -name \"*.o\"\)
        $$\(find ${hwy_BINARY_DIR} -name \"*.o\"\)
        $$\(find ${libsndfile_BINARY_DIR}/CMakeFiles/sndfile.dir/ -name \"*.o\"\)
        $$\(find ${samplerate_BINARY_DIR} -name \"*.o\"\)
        $$\(find ${protobuf_BINARY_DIR} -name \"*.o\" ! -ipath \"*/google/protobuf/compiler/main.cc.o\"\)
        -o ${zimtohrli_goohrli_object}
    COMMAND ${CMAKE_AR} rcs ${zimtohrli_goohrli_archive} ${zimtohrli_goohrli_object}
    DEPENDS zimtohrli_goohrli_glue
)
add_custom_target(zimtohrli_goohrli ALL DEPENDS ${zimtohrli_goohrli_archive})

add_library(zimtohrli_ux STATIC
    cpp/zimt/ux.cc
    cpp/zimt/ux.h
)
asan_target(zimtohrli_ux)
target_include_directories(zimtohrli_ux PRIVATE ${imgui_SOURCE_DIR}/backends ${gles_INCLUDE_DIRS})
target_include_directories(zimtohrli_ux PUBLIC ${imgui_SOURCE_DIR})
target_link_libraries(zimtohrli_ux zimtohrli_base glfw OpenGL::GL imgui)

add_executable(zimtohrli_compare
    cpp/zimt/compare.cc
)
asan_target(zimtohrli_compare)
target_link_libraries(zimtohrli_compare zimtohrli_ux zimtohrli_base absl::flags_parse)
set_target_properties(zimtohrli_compare PROPERTIES
    PREFIX ""
    OUTPUT_NAME compare
    SUFFIX ""
)

option(BUILD_ZIMTOHRLI_TESTS "Build Zimtohrli test binaries." ON)
if (BUILD_ZIMTOHRLI_TESTS)
include(cmake/tests.cmake)
endif()