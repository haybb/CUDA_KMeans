add_executable(kmeans_seq main.c common.c kmeans.c)
target_link_libraries(kmeans_seq m)

add_executable(kmeans_centroids main.c common.c kmeans_centroids.cu)
target_link_libraries(kmeans_centroids m)

add_executable(kmeans_distances main.c common.c kmeans_distances.cu)
target_link_libraries(kmeans_distances m)

add_executable(kmeans_full main.c common.c kmeans_full.cu)
target_link_libraries(kmeans_full m)