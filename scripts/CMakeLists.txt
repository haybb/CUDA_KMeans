cmake_minimum_required(VERSION 3.10)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(generate_expected generate_expected.c ${PROJECT_SOURCE_DIR}/src/kmeans.c ${PROJECT_SOURCE_DIR}/src/common.c)

target_link_libraries(generate_expected m)