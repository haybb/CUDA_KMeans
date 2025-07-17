enable_testing()
include_directories(${GTEST_HOME}/googletest/include)


#############
# Build tests
#############

set (SOURCE_FOLDER ${PROJECT_SOURCE_DIR}/src)


########## Seq version

add_executable(kmeans_test ${SOURCE_FOLDER}/kmeans.c ${SOURCE_FOLDER}/common.c kmeans_test.cpp)
target_link_libraries(kmeans_test ${GTEST_BOTH_LIBRARIES})


########## CUDA centroids version

add_executable(kmeans_centroids_test ${SOURCE_FOLDER}/kmeans_centroids.cu ${SOURCE_FOLDER}/common.c kmeans_test.cpp)
target_link_libraries(kmeans_centroids_test ${GTEST_BOTH_LIBRARIES})


########## CUDA distances version  

add_executable(kmeans_distances_test ${SOURCE_FOLDER}/kmeans_distances.cu ${SOURCE_FOLDER}/common.c kmeans_test.cpp)
target_link_libraries(kmeans_distances_test ${GTEST_BOTH_LIBRARIES})


########## CUDA full version

add_executable(kmeans_full_test ${SOURCE_FOLDER}/kmeans_full.cu ${SOURCE_FOLDER}/common.c kmeans_test.cpp)
target_link_libraries(kmeans_full_test ${GTEST_BOTH_LIBRARIES})


########## Copy test files

message ("-- Copying test files")
file(GLOB FILES ../test_files/*)
file(COPY ${FILES} DESTINATION ${CMAKE_BINARY_DIR}/test)

file(GLOB FILES expected*.txt)
file(COPY ${FILES} DESTINATION ${CMAKE_BINARY_DIR}/test)