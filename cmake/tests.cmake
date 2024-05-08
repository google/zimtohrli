add_executable(zimtohrli_test
    cpp/zimt/audio_test.cc
    cpp/zimt/cam_test.cc
    cpp/zimt/dtw_test.cc
    cpp/zimt/elliptic_test.cc
    cpp/zimt/filterbank_test.cc
    cpp/zimt/loudness_test.cc
    cpp/zimt/masking_test.cc
    cpp/zimt/mos_test.cc
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
        ${python3_VENV} -m pip install jax jaxlib numpy scipy unittest-parallel &&
        ${python3_VENV_DIR}/bin/unittest-parallel -s . -p '*_test.py'"
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
