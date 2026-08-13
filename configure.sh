#!/bin/bash

git submodule update --init

CLANG_OPTS=()
if [ -n "${CLANG_PREFIX}" ]; then
  CLANG_OPTS=("-DCMAKE_CXX_COMPILER=${CLANG_PREFIX}/clang++" "-DCMAKE_C_COMPILER=${CLANG_PREFIX}/clang")
fi

if [ "${1}" == "debug" ]; then
    mkdir -p debug_build
    (cd debug_build && cmake -G Ninja "${CLANG_OPTS[@]}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=RelWithDebInfo ..)
elif [ "${1}" == "asan" ]; then
    mkdir -p asan_build
    (cd asan_build && cmake -G Ninja "${CLANG_OPTS[@]}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_C{,XX}_FLAGS='-fsanitize=address' -DCMAKE_LINKER_FLAGS_DEBUG='-fsanitize=address' -DCMAKE_BUILD_TYPE=RelWithDebInfo ..)
else
    mkdir -p build
    (cd build && cmake -G Ninja "${CLANG_OPTS[@]}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release ..)
fi
