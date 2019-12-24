# TC_CUDA
CCF BDCI 2019 三角形计数 赛题第2名解决方案

## 程序代码说明
+ 源程序共有四个文件：CalcTriangleMain.cpp, CalcTriangleCuda.cu, mapfile.hpp
- CalcTriangleMain.cpp：内容包括主程序、由CPU计算实现的子函数、与CUDA的通信接口。
- CalcTriangleCuda.cu：为GPU计算模块，包括GPU参数设置、由GPU计算实现的子函数等。
- mapfile.hpp：mmap函数调用函数

## 程序代码编译说明
+ 编译过程由makefile进行批处理，请在源文件目录下，运行命令make完成程序编译，将生成可执行文件CalcTriangle。
+ 特别说明，本程序仅在Tesla V100及CUDA 10.1环境下完成开发和测试，不保证在其它版本和其它环境中的运行结果。如果需要尝试，请先修改makefile文件中GPU型号所对应的gpu-architecture和gpu-code参数，才可正确编译。

## 程序代码运行使用说明
+ 运行程序请用命令格式： 
···
./CalcTriangle -f [数据所在目录/图数据文件]
···
例如：
···
./CalcTriangle -f ../datasets/s29.e10.kron.edgelist.bin
···
