#!/bin/sh

if [ "${1}" == "debug" ]; then
    mkdir -p debug_build
    (cd debug_build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC' -DCMAKE_CXX_FLAGS='-fPIC' -DCMAKE_BUILD_TYPE=RelWithDebInfo ..)
elif [ "${1}" == "asan" ]; then
    mkdir -p asan_build
    (cd asan_build && cmake -G Ninja -DCMAKE_C_FLAGS='-fsanitize=address -fPIC' -DCMAKE_CXX_FLAGS='-fsanitize=address -fPIC' -DCMAKE_LINKER_FLAGS_DEBUG='-fsanitize=address' -DCMAKE_BUILD_TYPE=RelWithDebInfo ..)
else
    mkdir -p build
    (cd build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC -mavx2' -DCMAKE_CXX_FLAGS='-fPIC -mavx2' -DCMAKE_BUILD_TYPE=Release ..)
fi
