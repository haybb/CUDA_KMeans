//
// Created by Hervé Paulino on 03/04/2025.
//

#ifndef PROJ_CUDA_KMEANS_H
#define PROJ_CUDA_KMEANS_H

#ifdef __cplusplus
extern "C" {
#endif
void kmeans(const char* input_filename, int K, int maxIterations,
            float minChangesPer, float maxThreshold, const char* output_filename);
#ifdef __cplusplus
}
#endif

/*
Function showFileError: It displays the corresponding error during file reading.
*/
#ifdef __cplusplus
extern "C" {
#endif
void showFileError(int error, const char* filename);
#ifdef __cplusplus
}
#endif

/*
Function readInput: It reads the file to determine the number of rows and columns.
*/
#ifdef __cplusplus
    extern "C" {
#endif
int readInput(const char* filename, int *lines, int *samples);
#ifdef __cplusplus
}
#endif

/*
Function readInput2: It loads data from file.
*/
#ifdef __cplusplus
        extern "C" {
#endif
int readInput2(const char* filename, float* data);
#ifdef __cplusplus
}
#endif

/*
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
#ifdef __cplusplus
            extern "C" {
#endif
int writeResult(int *classMap, int lines, const char* filename);
#ifdef __cplusplus
}
#endif

#endif //PROJ_CUDA_KMEANS_H
