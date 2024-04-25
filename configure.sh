#!/bin/sh

if [ "${1}" == "debug" ]; then
    mkdir -p debug_build
    (cd debug_build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC' -DCMAKE_CXX_FLAGS='-fPIC' -DCMAKE_BUILD_TYPE=RelWithDebInfo ..)
elif [ "${1}" == "asan" ]; then
    mkdir -p asan_build
    (cd asan_build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC' -DZIMTOHRLI_ASAN=1 -DCMAKE_CXX_FLAGS='-fPIC' -DCMAKE_BUILD_TYPE=RelWithDebInfo ..)
else
    mkdir -p build
    (cd build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC' -DCMAKE_CXX_FLAGS='-fPIC' -DCMAKE_BUILD_TYPE=Release ..)
fi
