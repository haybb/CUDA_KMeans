/*
 * k-Means clustering algorithm
 *
 * CUDA version (fully optimized)
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
 #include <cuda.h>
 
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
 GPU Kernel for Euclidean distance calculation
 Calculates the distance between each point and each centroid.
 */
 __global__ void euclideanDistanceKernel(
     const float* data, 			// [lines * samples]
     const float* centroids,    	// [K * samples]
     float* distances,          	// [lines * K]
     int lines, int K, int samples)
 {
     int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
     int centroidIdx = blockIdx.y * blockDim.y + threadIdx.y;
 
     if (pointIdx < lines && centroidIdx < K) {
         float dist = 0.0f;
         for (int s = 0; s < samples; ++s) {
             float diff = data[pointIdx * samples + s] - centroids[centroidIdx * samples + s];
             dist += diff * diff;
         }
         distances[pointIdx * K + centroidIdx] = sqrtf(dist);
     }
 }
 
 /*
 GPU Kernel for finding closest centroid for each point
 */
 __global__ void findClosestCentroid(
     const float* distances,  // [lines * K]
     int* classMap,           // [lines]
     int* changes,            // Single counter of changes
     const int* prevClassMap, // [lines]
     int lines, int K)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < lines) {
         float minDist = FLT_MAX;
         int bestClass = 0;
         
         // Find minimum distance
         for (int k = 0; k < K; k++) {
             float dist = distances[idx * K + k];
             if (dist < minDist) {
                 minDist = dist;
                 bestClass = k + 1; // Classes are 1-indexed
             }
         }
         
         // Check if classification changed
         if (prevClassMap[idx] != bestClass) {
             atomicAdd(changes, 1);
         }
         
         // Update class map
         classMap[idx] = bestClass;
     }
 }

/*
 * First cuda version that computes the new position
 * of centroids without shared memory optimisation
 */
__global__ void computeCentroidsKernel(const float* data,
    float* centroids,        // [K * samples]
    const int* classMap,     // [lines]
    int* pointsPerClass,      // [K]
    int lines, int samples, int K) {
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
 
 /*
 GPU Kernel for centroid calculation with shared memory optimisation
 */
 __global__ void computeCentroidsKernelShared(
     const float* data,        // [lines * samples]
     float* centroidSums,      // [K * samples]
     const int* classMap,      // [lines]
     int* pointsPerClass,      // [K]
     int lines, int samples, int K)
 {
     extern __shared__ float shared[];
     float* sharedCentroidSums = shared;
     int* sharedPointsPerClass = (int*)&sharedCentroidSums[K*samples];
     int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory to zero
     for (int i=threadIdx.x;i<K*samples;i+=blockDim.x)
     {
         sharedCentroidSums[i] = 0.0f;
     }
     for (int i=threadIdx.x;i<K;i+= blockDim.x)
     {
         sharedPointsPerClass[i] = 0;
     }
     __syncthreads();

     if (idx < lines)
     {
         int cluster = classMap[idx] - 1; // Convert to 0-indexed
         if (cluster >= 0 && cluster < K) {
             atomicAdd(&sharedPointsPerClass[cluster], 1);
             for (int d = 0; d < samples; ++d) {
                 atomicAdd(&sharedCentroidSums[cluster * samples + d], data[idx * samples + d]);
             }
         }
     }

     __syncthreads();

     for (int i=threadIdx.x;i<K;i+=blockDim.x)
     {
         atomicAdd(&pointsPerClass[i], sharedPointsPerClass[i]);
     }
     for (int i=threadIdx.x;i<K*samples;i+=blockDim.x)
     {
         atomicAdd(&centroidSums[i], sharedCentroidSums[i]);
     }
 }
 
 /*
 CPU function for euclidean distance calculation
 */
 float euclideanDistanceCPU(const float* a, const float* b, int samples)
 {
     float sum = 0.0f;
     for (int i = 0; i < samples; ++i) {
         float diff = a[i] - b[i];
         sum += diff * diff;
     }
     return sqrtf(sum);
 }
 
 /*
 Function zeroFloatMatrix: Set matrix elements to 0
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
 */
 void zeroIntArray(int *array, int size)
 {
     int i;
     for (i=0; i<size; i++)
         array[i] = 0;
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
     int *prevClassMap = (int*)calloc(lines,sizeof(int)); // Store previous classification
 
     if (centroidPos == NULL || centroids == NULL || classMap == NULL || prevClassMap == NULL)
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

     int sharedMemPerBlock;
     cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
     //printf("Max shared memory per block: %d bytes\n", sharedMemPerBlock);

     float *d_data, *d_centroids, *d_distances, *d_centroidSums;
     int *d_classMap, *d_prevClassMap, *d_pointsPerClass, *d_changes;
     
     CHECK_CUDA_CALL(cudaMalloc(&d_data, lines * samples * sizeof(float)));
     CHECK_CUDA_CALL(cudaMalloc(&d_centroids, K * samples * sizeof(float)));
     CHECK_CUDA_CALL(cudaMalloc(&d_distances, lines * K * sizeof(float)));
     CHECK_CUDA_CALL(cudaMalloc(&d_centroidSums, K * samples * sizeof(float)));
     CHECK_CUDA_CALL(cudaMalloc(&d_classMap, lines * sizeof(int)));
     CHECK_CUDA_CALL(cudaMalloc(&d_prevClassMap, lines * sizeof(int)));
     CHECK_CUDA_CALL(cudaMalloc(&d_pointsPerClass, K * sizeof(int)));
     CHECK_CUDA_CALL(cudaMalloc(&d_changes, sizeof(int)));
 
     CHECK_CUDA_CALL(cudaMemcpy(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice));
     CHECK_CUDA_CALL(cudaMemset(d_classMap, 0, lines * sizeof(int)));
     CHECK_CUDA_CALL(cudaMemset(d_prevClassMap, 0, lines * sizeof(int)));
 
     dim3 distBlockSize(16, 16);
     dim3 distGridSize(
         (lines + distBlockSize.x - 1) / distBlockSize.x,
         (K + distBlockSize.y - 1) / distBlockSize.y
     );
     
     dim3 pointBlockSize(256);
     dim3 pointGridSize((lines + pointBlockSize.x - 1) / pointBlockSize.x);
     
     do {
         it++;
 
         CHECK_CUDA_CALL(cudaMemcpy(d_prevClassMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToDevice));
         
         CHECK_CUDA_CALL(cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice));
         
         changes = 0;
         CHECK_CUDA_CALL(cudaMemcpy(d_changes, &changes, sizeof(int), cudaMemcpyHostToDevice));
         
         // 1. Calculate distances between all points and centroids
         euclideanDistanceKernel<<<distGridSize, distBlockSize>>>(d_data, d_centroids, d_distances, lines, K, samples);
         CHECK_CUDA_LAST();
         
         // 2. Find closest centroid for each point
         findClosestCentroid<<<pointGridSize, pointBlockSize>>>(d_distances, d_classMap, d_changes, d_prevClassMap, lines, K);
         CHECK_CUDA_LAST();
         
         CHECK_CUDA_CALL(cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));
         
         // 3. Calculate new centroids
         CHECK_CUDA_CALL(cudaMemset(d_centroidSums, 0, K * samples * sizeof(float)));
         CHECK_CUDA_CALL(cudaMemset(d_pointsPerClass, 0, K * sizeof(int)));
         size_t requiredSharedMem = K * samples * sizeof(float) + K * sizeof(int);
         if (requiredSharedMem <= sharedMemPerBlock)
         {
            //  printf("Computed centroids with shared memory optimisations");
             computeCentroidsKernelShared<<<pointGridSize, pointBlockSize, requiredSharedMem>>>(d_data, d_centroidSums, d_classMap, d_pointsPerClass, lines, samples, K);
            //  computeCentroidsKernel<<<pointGridSize, pointBlockSize>>>(d_data, d_centroidSums, d_classMap, d_pointsPerClass, lines, samples, K);

         }
         else
         {
            //  printf("Computed centroids without shared memory optimisations");
             computeCentroidsKernel<<<pointGridSize, pointBlockSize>>>(d_data, d_centroidSums, d_classMap, d_pointsPerClass, lines, samples, K);

         }
         CHECK_CUDA_LAST();
         
         CHECK_CUDA_CALL(cudaMemcpy(auxCentroids, d_centroidSums, K * samples * sizeof(float), cudaMemcpyDeviceToHost));
         CHECK_CUDA_CALL(cudaMemcpy(pointsPerClass, d_pointsPerClass, K * sizeof(int), cudaMemcpyDeviceToHost));
         
         for(i=0; i<K; i++) {
             if (pointsPerClass[i] > 0) {
                 for(j=0; j<samples; j++){
                     auxCentroids[i*samples+j] /= pointsPerClass[i];
                 }
             }
         }
         
         maxDist = FLT_MIN;
         for(i=0; i<K; i++){
             float dist = euclideanDistanceCPU(&centroids[i*samples], &auxCentroids[i*samples], samples);
             if(dist > maxDist) {
                 maxDist = dist;
             }
         }
         
         memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));
         
         sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
         outputMsg = strcat(outputMsg, line);
 
     } while((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));
 
     CHECK_CUDA_CALL(cudaMemcpy(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost));
 
     cudaFree(d_data);
     cudaFree(d_centroids);
     cudaFree(d_distances);
     cudaFree(d_centroidSums);
     cudaFree(d_classMap);
     cudaFree(d_prevClassMap);
     cudaFree(d_pointsPerClass);
     cudaFree(d_changes);
     free(prevClassMap);
 
 /*
  *
  * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
  *
  */
     // Output and termination conditions
     printf("%s", outputMsg);    
 
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