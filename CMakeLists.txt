cmake_minimum_required(VERSION 3.27)

project(Zimtohrli)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_CLANG_TIDY clang-tidy)
add_compile_options(-fPIC)

include(cmake/deps.cmake)

enable_testing()
include(GoogleTest)

add_library(zimtohrli_lib STATIC
    cpp/zimt/zimtohrli.h
    cpp/zimt/zimtohrli.cc
)
target_include_directories(zimtohrli_lib PUBLIC cpp)
target_link_libraries(zimtohrli_lib PRIVATE hwy)
# Use selective fast-math flags only for zimtohrli code
target_compile_options(zimtohrli_lib PRIVATE
    -fassociative-math -freciprocal-math -fno-signed-zeros -fno-math-errno)

add_library(zimtohrli_visqol_adapter STATIC
    cpp/zimt/visqol_model.h
    cpp/zimt/visqol_model.cc
    cpp/zimt/visqol.h
    cpp/zimt/visqol.cc
    cpp/zimt/resample.h
)
target_link_libraries(zimtohrli_visqol_adapter PRIVATE zimtohrli_lib visqol PkgConfig::soxr)
target_link_libraries(zimtohrli_visqol_adapter PUBLIC absl::span)
# Use selective fast-math flags only for zimtohrli code to avoid protobuf warnings
target_compile_options(zimtohrli_visqol_adapter PRIVATE 
    -freciprocal-math -fno-signed-zeros -fno-math-errno)

find_package(Python3 COMPONENTS Interpreter Development)
if (NOT Python3_FOUND)
  message(WARNING "Python 3 not found; the Python binding will not be tested.")
else ()
  add_library(zimtohrli_pyohrli SHARED
     cpp/zimt/pyohrli.cc
  )
  set_target_properties(zimtohrli_pyohrli PROPERTIES
     PREFIX ""
     OUTPUT_NAME _pyohrli.so
     SUFFIX ""
  )
  target_link_libraries(zimtohrli_pyohrli PRIVATE zimtohrli_lib Python3::Python)
endif ()

add_library(zimtohrli_goohrli_glue STATIC
    cpp/zimt/goohrli.cc
)
target_include_directories(zimtohrli_goohrli_glue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli ${CMAKE_CURRENT_SOURCE_DIR}/go/gosqol)
target_link_libraries(zimtohrli_goohrli_glue zimtohrli_visqol_adapter PkgConfig::soxr)
# Use selective fast-math flags only for zimtohrli code
target_compile_options(zimtohrli_goohrli_glue PRIVATE 
    -fassociative-math -freciprocal-math -fno-signed-zeros -fno-math-errno)

set(zimtohrli_goohrli_object ${CMAKE_CURRENT_BINARY_DIR}/goohrli.o)
set(zimtohrli_goohrli_archive_build ${CMAKE_CURRENT_BINARY_DIR}/goohrli.a)
set(zimtohrli_goohrli_archive ${CMAKE_CURRENT_SOURCE_DIR}/go/goohrli/goohrli.a)

if (NOT WIN32)
    add_custom_command(
        OUTPUT ${zimtohrli_goohrli_archive_build}
        COMMAND ${CMAKE_LINKER} -r
            $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_lib.dir/ -name \"*.o\"\)
            $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_goohrli_glue.dir/ -name \"*.o\"\)
            $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zimtohrli_visqol_adapter.dir/ -name \"*.o\"\)
            $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/visqol.dir/ -name \"*.o\"\)
            $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/visqol_proto.dir/ -name \"*.o\"\)
            $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/libsvm.dir/ -name \"*.o\"\)
            $$\(find ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/pffft.dir/ -name \"*.o\"\)
            $$\(find ${highway_BINARY_DIR}/CMakeFiles/hwy.dir/ -name \"*.o\"\)
            $$\(find ${protobuf_BINARY_DIR} -name \"*.o\" ! -ipath \"*/google/protobuf/compiler/main.cc.o\"\)
            -o ${zimtohrli_goohrli_object}
        COMMAND ${CMAKE_AR} rcs ${zimtohrli_goohrli_archive_build} ${zimtohrli_goohrli_object}
        COMMAND ${CMAKE_COMMAND} -E copy ${zimtohrli_goohrli_archive_build} ${zimtohrli_goohrli_archive}
        DEPENDS zimtohrli_goohrli_glue zimtohrli_visqol_adapter zimtohrli_lib hwy
    )
    add_custom_target(zimtohrli_goohrli ALL DEPENDS ${zimtohrli_goohrli_archive_build})
endif()

add_executable(zimtohrli_compare
    cpp/zimt/audio.cc
    cpp/zimt/compare.cc
)
target_include_directories(zimtohrli_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(zimtohrli_compare zimtohrli_lib sndfile absl::statusor absl::check absl::flags_parse PkgConfig::soxr)
set_target_properties(zimtohrli_compare PROPERTIES
    PREFIX ""
    OUTPUT_NAME compare
    SUFFIX ""
)
# Use selective fast-math flags for compare tool
target_compile_options(zimtohrli_compare PRIVATE 
    -fassociative-math -freciprocal-math -fno-signed-zeros -fno-math-errno)

option(BUILD_ZIMTOHRLI_TESTS "Build Zimtohrli test binaries." ON)
if (BUILD_ZIMTOHRLI_TESTS)
include(cmake/tests.cmake)
endif()
