project(HALIDE_GENERATOR)
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories($ENV{HALIDE_ROOT}/lib)


string(REPLACE " " ";" GEN_SOURCE ${GEN_SOURCE})

add_executable(Generator
    $ENV{HALIDE_ROOT}/tools/GenGen.cpp
    ${GEN_SOURCE}
)

target_include_directories(Generator PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    $ENV{HALIDE_ROOT}/include
)

target_link_libraries(Generator PUBLIC Halide)

install(TARGETS Generator
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PRIVATE_HEADER DESTINATION include
)
