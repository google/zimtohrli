FetchContent_Declare(pffft
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://bitbucket.org/jpommier/pffft.git
    GIT_TAG 7c3b5a7dc510a0f513b9c5b6dc5b56f7aeeda422
)

FetchContent_MakeAvailable(pffft)

add_library(pffft STATIC
    ${pffft_SOURCE_DIR}/pffft.c
    ${pffft_SOURCE_DIR}/pffft.h
)
target_include_directories(pffft PUBLIC ${pffft_SOURCE_DIR})
