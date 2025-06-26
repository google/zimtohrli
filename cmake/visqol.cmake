FetchContent_Declare(visqol
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/google/visqol.git
    GIT_TAG v3.3.3
    PATCH_COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/cmake/visqol_manager.cc src/visqol_manager.cc
)
FetchContent_MakeAvailable(visqol)

add_library(visqol_proto STATIC ${visqol_SOURCE_DIR}/src/proto/similarity_result.proto ${visqol_SOURCE_DIR}/src/proto/visqol_config.proto)

include(${protobuf_BINARY_DIR}/cmake/protobuf/protobuf-generate.cmake)

set(visqol_PROTO_DIR ${visqol_BINARY_DIR}/generated)
protobuf_generate(
    TARGET visqol_proto
    IMPORT_DIRS ${visqol_SOURCE_DIR}
    PROTOC_OUT_DIR ${visqol_PROTO_DIR}
    LANGUAGE cpp
)
target_link_libraries(visqol_proto protobuf::libprotobuf)
target_include_directories(visqol_proto PUBLIC ${visqol_PROTO_DIR})

set(visqol_MODEL_H ${visqol_SOURCE_DIR}/src/include/libsvm_nu_svr_model.h)
add_custom_command(
    OUTPUT ${visqol_MODEL_H}
	COMMAND sh -c "xxd -i ${visqol_SOURCE_DIR}/model/libsvm_nu_svr_model.txt | sed 's/[A-Za-z_]*libsvm_nu_svr_model_txt/visqol_model_bytes/' > ${visqol_MODEL_H}"
    DEPENDS ${visqol_SOURCE_DIR}/model/libsvm_nu_svr_model.txt
	VERBATIM
) 
add_custom_target(visqol_model DEPENDS ${visqol_MODEL_H})

add_library(visqol STATIC
    ${visqol_SOURCE_DIR}/src/alignment.cc
    ${visqol_SOURCE_DIR}/src/amatrix.cc
    ${visqol_SOURCE_DIR}/src/analysis_window.cc
    ${visqol_SOURCE_DIR}/src/commandline_parser.cc
    ${visqol_SOURCE_DIR}/src/comparison_patches_selector.cc
    ${visqol_SOURCE_DIR}/src/complex_valarray.cc
    ${visqol_SOURCE_DIR}/src/convolution_2d.cc
    ${visqol_SOURCE_DIR}/src/envelope.cc
    ${visqol_SOURCE_DIR}/src/equivalent_rectangular_bandwidth.cc
    ${visqol_SOURCE_DIR}/src/fast_fourier_transform.cc
    ${visqol_SOURCE_DIR}/src/fft_manager.cc
    ${visqol_SOURCE_DIR}/src/gammatone_filterbank.cc
    ${visqol_SOURCE_DIR}/src/gammatone_spectrogram_builder.cc
    ${visqol_SOURCE_DIR}/src/image_patch_creator.cc
    ${visqol_SOURCE_DIR}/src/include/alignment.h
    ${visqol_SOURCE_DIR}/src/include/amatrix.h
    ${visqol_SOURCE_DIR}/src/include/analysis_window.h
    ${visqol_SOURCE_DIR}/src/include/audio_channel.h
    ${visqol_SOURCE_DIR}/src/include/audio_signal.h
    ${visqol_SOURCE_DIR}/src/include/commandline_parser.h
    ${visqol_SOURCE_DIR}/src/include/comparison_patches_selector.h
    ${visqol_SOURCE_DIR}/src/include/complex_valarray.h
    ${visqol_SOURCE_DIR}/src/include/conformance.h
    ${visqol_SOURCE_DIR}/src/include/convolution_2d.h
    ${visqol_SOURCE_DIR}/src/include/envelope.h
    ${visqol_SOURCE_DIR}/src/include/equivalent_rectangular_bandwidth.h
    ${visqol_SOURCE_DIR}/src/include/fast_fourier_transform.h
    ${visqol_SOURCE_DIR}/src/include/fft_manager.h
    ${visqol_SOURCE_DIR}/src/include/file_path.h
    ${visqol_SOURCE_DIR}/src/include/gammatone_filterbank.h
    ${visqol_SOURCE_DIR}/src/include/gammatone_spectrogram_builder.h
    ${visqol_SOURCE_DIR}/src/include/image_patch_creator.h
    ${visqol_SOURCE_DIR}/src/include/libsvm_target_observation_convertor.h
    ${visqol_SOURCE_DIR}/src/include/machine_learning.h
    ${visqol_SOURCE_DIR}/src/include/misc_audio.h
    ${visqol_SOURCE_DIR}/src/include/misc_math.h
    ${visqol_SOURCE_DIR}/src/include/misc_vector.h
    ${visqol_SOURCE_DIR}/src/include/neurogram_similiarity_index_measure.h
    ${visqol_SOURCE_DIR}/src/include/patch_similarity_comparator.h
    ${visqol_SOURCE_DIR}/src/include/rms_vad.h
    ${visqol_SOURCE_DIR}/src/include/signal_filter.h
    ${visqol_SOURCE_DIR}/src/include/similarity_result.h
    ${visqol_SOURCE_DIR}/src/include/similarity_to_quality_mapper.h
    ${visqol_SOURCE_DIR}/src/include/sim_results_writer.h
    ${visqol_SOURCE_DIR}/src/include/spectrogram_builder.h
    ${visqol_SOURCE_DIR}/src/include/spectrogram.h
    ${visqol_SOURCE_DIR}/src/include/speech_similarity_to_quality_mapper.h
    ${visqol_SOURCE_DIR}/src/include/status_macros.h
    ${visqol_SOURCE_DIR}/src/include/support_vector_regression_model.h
    ${visqol_SOURCE_DIR}/src/include/svr_similarity_to_quality_mapper.h
    ${visqol_SOURCE_DIR}/src/include/vad_patch_creator.h
    ${visqol_SOURCE_DIR}/src/include/visqol_api.h
    ${visqol_SOURCE_DIR}/src/include/visqol.h
    ${visqol_SOURCE_DIR}/src/include/wav_reader.h
    ${visqol_SOURCE_DIR}/src/include/xcorr.h
    ${visqol_SOURCE_DIR}/src/libsvm_target_observation_convertor.cc
    ${visqol_SOURCE_DIR}/src/misc_audio.cc
    ${visqol_SOURCE_DIR}/src/misc_math.cc
    ${visqol_SOURCE_DIR}/src/misc_vector.cc
    ${visqol_SOURCE_DIR}/src/neurogram_similiarity_index_measure.cc
    ${visqol_SOURCE_DIR}/src/rms_vad.cc
    ${visqol_SOURCE_DIR}/src/signal_filter.cc
    ${visqol_SOURCE_DIR}/src/spectrogram.cc
    ${visqol_SOURCE_DIR}/src/speech_similarity_to_quality_mapper.cc
    ${visqol_SOURCE_DIR}/src/support_vector_regression_model.cc
    ${visqol_SOURCE_DIR}/src/svr_similarity_to_quality_mapper.cc
    ${visqol_SOURCE_DIR}/src/svr_training/training_data_file_reader.cc
    ${visqol_SOURCE_DIR}/src/svr_training/training_data_file_reader.h
    ${visqol_SOURCE_DIR}/src/vad_patch_creator.cc
    ${visqol_SOURCE_DIR}/src/visqol_api.cc
    ${visqol_SOURCE_DIR}/src/visqol.cc
    ${visqol_SOURCE_DIR}/src/visqol_manager.cc
    ${visqol_SOURCE_DIR}/src/wav_reader.cc
    ${visqol_SOURCE_DIR}/src/xcorr.cc
)
target_include_directories(visqol PUBLIC ${visqol_SOURCE_DIR} ${visqol_SOURCE_DIR}/src/include)
target_link_libraries(visqol visqol_proto libsvm armadillo absl::span pffft)
add_dependencies(visqol visqol_model)

file(GLOB_RECURSE visqol_files ${visqol_SOURCE_DIR} *.cc *.c *.h)
set_source_files_properties(
    ${visqol_files}
    TARGET_DIRECTORY visqol
    PROPERTIES SKIP_LINTING ON
)
