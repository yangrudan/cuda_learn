# cublas build
1. ```bash
   $ nvcc main.cu -o cublas_learn -lcublas
   $ ./cublas_learn
2. 修改CMakeLists.txt
   * target_link_libraries(untitled1  cublas)