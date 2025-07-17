[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pUutzZJM)
[![Work in MakeCode](https://classroom.github.com/assets/work-in-make-code-8824cc13a1a3f34ffcd245c82f0ae96fdae6b7d554b6539aec3a03a70825519c.svg)](https://classroom.github.com/online_ide?assignment_repo_id=19064847&assignment_repo_type=AssignmentRepo)
# Computação de Alto Desempenho 2024/25 

## CUDA Project Assignment - K-Means Clustering Algorithm

This assignment is strongly based
in the _EduHPC’23 Peachy Assignment_, developed by the Trasgo Research Group at the Universidad de Valladolid. To know more about Peachy Assignments, 
visit https://tcpp.cs.gsu.edu/curriculum/?q=peachy.

## Structure

- **include**: project header files.
- **report**: template for the project's report. You must place your final report in this folder.
- **src**: project source files.
- **test**: project test files that make use of the Google Test framework.
- **test_files**: set of files for you to test. Your performance analysis may include more files
  than the ones included here.
- **third_pary**: external libraries needed for the project.

## Installation requirements

C++ compiler and a profiler.

cmake is advised

## Compilation and Execution
To compile import in a IDE that supports cmake or run the following sequence of command in a terminal:


```
mkdir build
cd build
cmake ..
make
```

To run the executable in the IDE, simply 
select the target from the target list and edit the 
configuration to provide the arguments.

From the command line, go to the build directory (such as cmake-build-release) and type:

```
./src/kmeans [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]
```

proving the values for each input parameter.

To run the tests in the IDE,  simply
select the target from the target list.

From the command line, type:

```
test/kmeans_test
```
