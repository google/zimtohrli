cmake_minimum_required(VERSION 3.27)

project(Zimtohrli)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_CLANG_TIDY clang-tidy)
add_compile_options(-fPIC)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3 -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -ffast-math")

include(cmake/deps.cmake)

enable_testing()
include(GoogleTest)

add_library(zimtohrli_visqol_adapter STATIC
    cpp/zimt/visqol_model.h
    cpp/zimt/visqol_model.cc
    cpp/zimt/visqol.h
    cpp/zimt/visqol.cc
    cpp/zimt/resample.h
    cpp/zimt/zimtohrli.h
)
target_include_directories(zimtohrli_visqol_adapter PUBLIC cpp)
target_link_libraries(zimtohrli_visqol_adapter PRIVATE visqol samplerate)
target_link_libraries(zimtohrli_visqol_adapter PUBLIC absl::span)

find_package(Python3 COMPONENTS Interpreter Development)
add_library(zimtohrli_pyohrli SHARED
   cpp/zimt/zimtohrli.h
   cpp/zimt/pyohrli.cc
)
target_include_directories(zimtohrli_pyohrli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set_target_properties(zimtohrli_pyohrli PROPERTIES
   PREFIX ""
   OUTPUT_NAME _pyohrli.so
   SUFFIX ""
)
target_link_libraries(zimtohrli_pyohrli Python3::Python absl::check)

add_library(zimtohrli_goohrli_glue STATIC
    cpp/zimt/goohrli.cc
    cpp/zimt/zimtohrli.h
    go/goohrli/goohrli.h
)
target_include_directories(zimtohrli_goohrli_glue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli ${CMAKE_CURRENT_SOURCE_DIR}/go/gosqol)
target_link_libraries(zimtohrli_goohrli_glue zimtohrli_visqol_adapter)

set(zimtohrli_goohrli_object ${CMAKE_CURRENT_BINARY_DIR}/goohrli.o)
set(zimtohrli_goohrli_archive ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli/goohrli.a)
add_custom_command(
    OUTPUT ${zimtohrli_goohrli_archive}
    COMMAND ${CMAKE_LINKER} -r
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_goohrli_glue.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_visqol_adapter.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/visqol.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/visqol_proto.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/libsvm.dir/ -name \"*.o\"\)
        $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/pffft.dir/ -name \"*.o\"\)
        $$\(find ${libsndfile_BINARY_DIR}/CMakeFiles/sndfile.dir/ -name \"*.o\"\)
        $$\(find ${samplerate_BINARY_DIR} -name \"*.o\"\)
        $$\(find ${protobuf_BINARY_DIR} -name \"*.o\" ! -ipath \"*/google/protobuf/compiler/main.cc.o\"\)
        -o ${zimtohrli_goohrli_object}
    COMMAND ${CMAKE_AR} rcs ${zimtohrli_goohrli_archive} ${zimtohrli_goohrli_object}
    DEPENDS zimtohrli_goohrli_glue zimtohrli_visqol_adapter
)
add_custom_target(zimtohrli_goohrli ALL DEPENDS ${zimtohrli_goohrli_archive})

add_executable(zimtohrli_compare
    cpp/zimt/audio.cc
    cpp/zimt/compare.cc
)
target_include_directories(zimtohrli_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(zimtohrli_compare sndfile absl::statusor absl::check absl::flags_parse samplerate)
set_target_properties(zimtohrli_compare PROPERTIES
    PREFIX ""
    OUTPUT_NAME compare
    SUFFIX ""
)

option(BUILD_ZIMTOHRLI_TESTS "Build Zimtohrli test binaries." ON)
if (BUILD_ZIMTOHRLI_TESTS)
include(cmake/tests.cmake)
endif()
