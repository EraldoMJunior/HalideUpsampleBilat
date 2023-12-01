cmake_minimum_required(VERSION 3.9)

project( stereo_estimation )
set(TARGET "bilateral")
include(ExternalProject)

###############
# packages #
###############

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#c++17
set(CMAKE_CXX_STANDARD 17)

#export HALIDE_ROOT="/home/${USER}/Halide"
set(HALIDE_ROOT "/home/eraldomr/Halide")

find_package(OpenMP)


## Halide setup ##
if(ANDROID)
    if(CMAKE_ANDROID_ARCH_ABI MATCHES "arm64-v8a")
        set(HALIDE_TARGET "arm-64-android")
    else()
        set(HALIDE_TARGET "arm-32-android")
    endif()
else()
    set(HALIDE_TARGET "host")
endif()


macro(add_plugin)
    cmake_parse_arguments(PLUGIN_ARGS "" "PATH" "" ${ARGN} )
    if(NOT PLUGIN_ARGS_PATH)
        message(FATAL_ERROR "You must provide a plugin path")
    endif()

    #check if using MSYS shell but not the MSYS version of cmake, i.e: android studio cmake
    if(uname MATCHES "^MSYS")
	    execute_process(COMMAND cygpath -w ${PLUGIN_ARGS_PATH}
            OUTPUT_VARIABLE WPLUGIN OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        set(WPLUGIN ${PLUGIN_ARGS_PATH})
    endif()

    set(HALIDE_PLUGINS "${HALIDE_PLUGINS}${SEP}${WPLUGIN}")

    set(SEP ",")
endmacro(add_plugin)

if(UNIX)
    set(LIBRARY_EXTENSION so)
else()
    set(LIBRARY_EXTENSION dll)
endif()

add_plugin(PATH "libautoschedule_adams2019.${LIBRARY_EXTENSION}")
add_plugin(PATH "libautoschedule_mullapudi2016.${LIBRARY_EXTENSION}")
add_plugin(PATH "libautoschedule_li2018.${LIBRARY_EXTENSION}")

set(GENERATOR "${CMAKE_BINARY_DIR}/bin/Generator")

macro(add_generator)
    cmake_parse_arguments(GEN_ARGS "INCLUDED" "NAME;SOURCE;FUNC" "PARAMS" ${ARGN} )

    message(STATUS  "add_generator: ${GEN_ARGS_NAME} ${GEN_ARGS_SOURCE} ${GEN_ARGS_FUNC} ${GEN_ARGS_PARAMS}")

    if(NOT GEN_ARGS_NAME)
        message(FATAL_ERROR "You must provide a generator name")
    endif()

    if(NOT GEN_ARGS_SOURCE)
        message(FATAL_ERROR "You must provide a generator source file")
    endif()

    if(NOT GEN_ARGS_FUNC)
        set(GEN_ARGS_FUNC ${GEN_ARGS_NAME})
    endif()

    set(HALIDE_DIR "${CMAKE_BINARY_DIR}/halide")
    set(GEN_TARGET "${HALIDE_DIR}/${GEN_ARGS_FUNC}")

    add_custom_command(
        COMMAND mkdir -p ${HALIDE_DIR} && ${CMAKE_COMMAND} -E env "PATH=\"$ENV{PATH};$ENV{HALIDE_ROOT}\"" ${GENERATOR}
            -g ${GEN_ARGS_NAME} -f ${GEN_ARGS_FUNC} -e h,o,schedule,stmt -o ${HALIDE_DIR} -p ${HALIDE_PLUGINS}
            target=${HALIDE_TARGET}${HALIDE_TARGET_RUNTIME} ${GEN_ARGS_PARAMS} ${GENERATOR_PARAMS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        OUTPUT ${GEN_TARGET}${CMAKE_C_OUTPUT_EXTENSION}
        OUTPUT ${GEN_TARGET}.h
        COMMENT "generate ${GEN_ARGS_NAME} for target=${HALIDE_TARGET}${HALIDE_TARGET_RUNTIME} ${GEN_ARGS_PARAMS} ${GENERATOR_PARAMS}"
        DEPENDS ${GEN_ARGS_SOURCE}
    )

    message(STATUS "add_custom_command: ${GEN_TARGET}${CMAKE_C_OUTPUT_EXTENSION} ${GEN_TARGET}.h")

    if(NOT GEN_ARGS_INCLUDED)
        set(GEN_SOURCE "${GEN_SOURCE} ${GEN_ARGS_SOURCE}")
        message(STATUS "add ${GEN_ARGS_SOURCE} to GEN_SOURCE")
    endif()

    list(APPEND GEN_OBJS "${GEN_TARGET}${CMAKE_C_OUTPUT_EXTENSION}")
    list(APPEND GEN_HEADERS "${GEN_TARGET}.h")
    message(STATUS "add ${GEN_TARGET}${CMAKE_C_OUTPUT_EXTENSION} to GEN_OBJS")

    set_source_files_properties("${GEN_TARGET}${CMAKE_C_OUTPUT_EXTENSION}" PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
    )

    set(HALIDE_TARGET_RUNTIME "-no_runtime")
endmacro(add_generator)

list(APPEND GENERATOR_PARAMS_UPSAMPLE "")

add_generator(
  NAME bilateral_upsample
  FUNC bilateral_upsample
  SOURCE "${PROJECT_SOURCE_DIR}/halide/src/halide_bilateral_upsample.cpp"
  PARAMS "${GENERATOR_PARAMS_UPSAMPLE}"
)


externalproject_add(Generator
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/halide
    BUILD_ALWAYS true
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE};-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>;-DGEN_SOURCE=${GEN_SOURCE}
)



###############
# executables #
###############

add_executable( ${TARGET}  src/main.cpp ${GEN_OBJS} )

add_dependencies( ${TARGET}  Generator)

set_target_properties( ${TARGET} PROPERTIES PUBLIC_HEADER "${GEN_HEADERS}")


target_link_directories(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    ${HALIDE_ROOT}/lib

)



target_include_directories(${TARGET} PUBLIC
    ${HALIDE_ROOT}/tools
    ${HALIDE_ROOT}/include
    ${CMAKE_BINARY_DIR}/halide

)

FIND_PACKAGE(JPEG REQUIRED)
IF(JPEG_FOUND)
  INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(${TARGET} ${JPEG_LIBRARIES})
ENDIF()

FIND_PACKAGE(PNG REQUIRED)
IF(PNG_FOUND)
  INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(${TARGET} ${PNG_LIBRARIES})
ENDIF()


if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET}  OpenMP::OpenMP_CXX  Halide  ${OpenCV_LIBS} ${PTHREAD_LIB} dl)
endif()