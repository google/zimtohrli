#!/bin/sh

if [ "${1}" == "debug" ]; then
    mkdir -p debug_build
    (cd debug_build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC' -DCMAKE_CXX_FLAGS='-fPIC' -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..)
elif [ "${1}" == "asan" ]; then
    mkdir -p asan_build
    (cd asan_build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC -fsanitize=address' -DCMAKE_CXX_FLAGS='-fPIC -fsanitize=address' -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..)
else
    mkdir -p build
    (cd build && cmake -G Ninja -DCMAKE_C_FLAGS='-fPIC' -DCMAKE_CXX_FLAGS='-fPIC -stdlib=libc++' -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..)
fi
