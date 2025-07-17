/*
 * k-Means clustering algorithm
 *
 * CUDA version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <ctype.h>
 #include <math.h>
 #include <time.h>
 #include <string.h>
 #include <float.h>
 #include <kmeans.h>
 //#include <cuda.h>
 
 #define MAXCAD 200
 
 //Macros
 #define MIN(a,b) ((a) < (b) ? (a) : (b))
 #define MAX(a,b) ((a) > (b) ? (a) : (b))
 
 /*
  * Macros to show errors when calling a CUDA library function,
  * or after launching a kernel
  */
 #define CHECK_CUDA_CALL( a )	{ \
     cudaError_t ok = a; \
     if ( ok != cudaSuccess ) \
         fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
     }
 #define CHECK_CUDA_LAST()	{ \
     cudaError_t ok = cudaGetLastError(); \
     if ( ok != cudaSuccess ) \
         fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
     }
 
 
 /*
 Function initCentroids: This function copies the values of the initial centroids, using their
 position in the input data structure as a reference map.
 */
 void initCentroids(const float *data, float* centroids, const int* centroidPos, int samples, int K)
 {
     int i;
     int idx;
     for(i=0; i<K; i++)
     {
         idx = centroidPos[i];
         memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
     }
 }
 
 /*
 Function euclideanDistance: Euclidean distance
 This function could be modified
 */
 float euclideanDistance(const float *point, const float *center, int samples)
 {
     float dist=0.0;
     for(int i=0; i<samples; i++)
     {
         dist+= (point[i]-center[i])*(point[i]-center[i]);
     }
     dist = sqrt(dist);
     return(dist);
 }
 
 /*
 Function zeroFloatMatriz: Set matrix elements to 0
 This function could be modified
 */
 void zeroFloatMatriz(float *matrix, int rows, int columns)
 {
     int i,j;
     for (i=0; i<rows; i++)
         for (j=0; j<columns; j++)
             matrix[i*columns+j] = 0.0;
 }
 
 /*
 Function zeroIntArray: Set array elements to 0
 This function could be modified
 */
 void zeroIntArray(int *array, int size)
 {
     int i;
     for (i=0; i<size; i++)
         array[i] = 0;
 }
 
 __global__ void computeCentroids(const float* data, float* centroids, const int* classMap, int* pointsPerClass, int lines, int samples, int K) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < lines)
     {
         int cluster = classMap[idx]-1;
         if (cluster >= 0 && cluster < K) {
             atomicAdd(&pointsPerClass[cluster], 1);
             for (int d = 0; d < samples; ++d) {
                 atomicAdd(&centroids[cluster * samples + d], data[idx * samples + d]);
             }
         }
     }
 }
 
 
 
 void kmeans(const char* input_filename, int K, int maxIterations,
             float minChangesPer, float maxThreshold, const char* output_filename) {
 
     //START CLOCK***************************************
     clock_t start, end;
     start = clock();
 
     // Reading the input data
     // lines = number of points; samples = number of dimensions per point
     int lines = 0, samples= 0;
 
     int error = readInput(input_filename, &lines, &samples);
     if(error != 0)
     {
         showFileError(error,input_filename);
         exit(error);
     }
 
     float *data = (float*)calloc(lines*samples,sizeof(float));
     if (data == NULL)
     {
         fprintf(stderr,"Memory allocation error.\n");
         exit(-4);
     }
     error = readInput2(input_filename, data);
     if(error != 0)
     {
         showFileError(error,input_filename);
         exit(error);
     }
 
     int minChanges= (int)(lines*minChangesPer/100.0);
     int *centroidPos = (int*)calloc(K,sizeof(int));
     float *centroids = (float*)calloc(K*samples,sizeof(float));
     int *classMap = (int*)calloc(lines,sizeof(int));
 
     if (centroidPos == NULL || centroids == NULL || classMap == NULL)
     {
         fprintf(stderr,"Memory allocation error.\n");
         exit(-4);
     }
 
     // Initial centroids
     srand(0);
     int i;
     for(i=0; i<K; i++)
         centroidPos[i]=rand()%lines;
 
     // Loading the array of initial centroids with the data from the array data
     // The centroids are points stored in the data array.
     initCentroids(data, centroids, centroidPos, samples, K);
 
 
     printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", input_filename, lines, samples);
     printf("\tNumber of clusters: %d\n", K);
     printf("\tMaximum number of iterations: %d\n", maxIterations);
     printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, minChangesPer, lines);
     printf("\tMaximum centroid precision: %f\n", maxThreshold);
 
     //END CLOCK*****************************************
     end = clock();
     printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
     fflush(stdout);
 
     CHECK_CUDA_CALL( cudaSetDevice(0) );
     CHECK_CUDA_CALL( cudaDeviceSynchronize() );
     //**************************************************
     //START CLOCK***************************************
     start = clock();
     //**************************************************
     char *outputMsg = (char *)calloc(10000,sizeof(char));
     char line[100];
 
     int j;
     int clazz;
     float dist, minDist;
     int it=0;
     int changes = 0;
     float maxDist;
 
     //pointPerClass: number of points classified in each class
     //auxCentroids: mean of the points in each class
     int *pointsPerClass = (int *)malloc(K*sizeof(int));
     float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
     float *distCentroids = (float*)malloc(K*sizeof(float));
     if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
     {
         fprintf(stderr,"Memory allocation error.\n");
         exit(-4);
     }
 
 /*
  *
  * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
  *
  */
     float *d_data, *d_centroidSums;
     int *d_classMap, *d_pointsPerClass;
 
     CHECK_CUDA_CALL(cudaMalloc(&d_data, lines * samples * sizeof(float)));
     CHECK_CUDA_CALL(cudaMalloc(&d_centroidSums, K * samples * sizeof(float)));
     CHECK_CUDA_CALL(cudaMalloc(&d_classMap, lines * sizeof(int)));
     CHECK_CUDA_CALL(cudaMalloc(&d_pointsPerClass, K * sizeof(int)));
 
     CHECK_CUDA_CALL(cudaMemcpy(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice));
 
     dim3 blockSize(256);
     dim3 gridSize((lines + blockSize.x - 1) / blockSize.x);
 
     do{
         it++;
 
         //1. Calculate the distance from each point to the centroid
         //Assign each point to the nearest centroid.
         changes = 0;
         for(i=0; i<lines; i++)
         {
             clazz=1;
             minDist=FLT_MAX;
             for(j=0; j<K; j++)
             {
                 dist=euclideanDistance(&data[i*samples], &centroids[j*samples], samples);
 
                 if(dist < minDist)
                 {
                     minDist=dist;
                     clazz=j+1;
                 }
             }
             if(classMap[i]!=clazz)
             {
                 changes++;
             }
             classMap[i]=clazz;
         }
 
         // 2. Recalculates the centroids: calculates the mean within each cluster
         CHECK_CUDA_CALL(cudaMemcpy(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice));
         
         CHECK_CUDA_CALL(cudaMemset(d_centroidSums, 0, K * samples * sizeof(float)));
         CHECK_CUDA_CALL(cudaMemset(d_pointsPerClass, 0, K * sizeof(int)));
         
         computeCentroids<<<gridSize, blockSize>>>(d_data, d_centroidSums, d_classMap, d_pointsPerClass, lines, samples, K);
         CHECK_CUDA_LAST();
         
         CHECK_CUDA_CALL(cudaMemcpy(auxCentroids, d_centroidSums, K * samples * sizeof(float), cudaMemcpyDeviceToHost));
         CHECK_CUDA_CALL(cudaMemcpy(pointsPerClass, d_pointsPerClass, K * sizeof(int), cudaMemcpyDeviceToHost))
 
         for(i=0; i<K; i++)
         {
             if (pointsPerClass[i] > 0) {
                 for(j=0; j<samples; j++){
                     auxCentroids[i*samples+j] /= pointsPerClass[i];
                 }
             }
         }
 
         maxDist=FLT_MIN;
         for(i=0; i<K; i++){
             distCentroids[i]=euclideanDistance(&centroids[i*samples], &auxCentroids[i*samples], samples);
             if(distCentroids[i]>maxDist) {
                 maxDist=distCentroids[i];
             }
         }
         memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));
 
         sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
         outputMsg = strcat(outputMsg,line);
 
     } while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));
 
     cudaFree(d_data);
     cudaFree(d_centroidSums);
     cudaFree(d_classMap);
     cudaFree(d_pointsPerClass);
 
 /*
  *
  * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
  *
  */
     // Output and termination conditions
     printf("%s",outputMsg);
 
     CHECK_CUDA_CALL( cudaDeviceSynchronize() );
 
     //END CLOCK*****************************************
     end = clock();
     printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
     fflush(stdout);
     //**************************************************
     //START CLOCK***************************************
     start = clock();
     //**************************************************
 
 
 
     if (changes <= minChanges) {
         printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
     }
     else if (it >= maxIterations) {
         printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
     }
     else {
         printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
     }
 
     // Writing the classification of each point to the output file.
     error = writeResult(classMap, lines, output_filename);
     if(error != 0)
     {
         showFileError(error, output_filename);
         exit(error);
     }
 
     //Free memory
     free(data);
     free(classMap);
     free(centroidPos);
     free(centroids);
     free(distCentroids);
     free(pointsPerClass);
     free(auxCentroids);
 
     //END CLOCK*****************************************
     end = clock();
     printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
     fflush(stdout);
 
 
 }
 