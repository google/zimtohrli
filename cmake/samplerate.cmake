FetchContent_Declare(samplerate
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/libsndfile/libsamplerate.git
    GIT_TAG 0.2.2
)
FetchContent_MakeAvailable(samplerate)