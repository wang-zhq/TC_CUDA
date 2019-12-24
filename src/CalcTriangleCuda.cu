#include <cstdint>
#include <unistd.h>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <omp.h>
#include "cuda_runtime.h"
#include <iostream>
#include "mapfile.hpp"

void gpuUllSort(uint64_t *cpuBlockStart, uint32_t blockSize);

using namespace std;
void DebugArray2(const uint32_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx);
void DebugArray4(const uint64_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx);
void DebugArrayMatrix2(const char* title, const uint32_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx, uint64_t RowLen);
void DebugArrayMatrix4(const char* title, const uint64_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx, uint64_t RowLen);

const uint32_t WARPSIZE  = 32;
const uint32_t CALC_SIZE = 128;	           //计算三角形时使用的线程块的线程数
const uint32_t BLOCKSIZE = 512;	           //预处理时使用的线程块的线程数
const uint64_t MIN_EDGENUMB = 0x0000000030000000;  //按顶点分桶操作最少输出边数, 这里的值是768M个边
const uint64_t MAP_BOUNDARY = 0xFFFFFFFFFFFF0000;  //文件影像的边界选择, 必须是系统页的倍数, 我们选择64K的倍数

extern unsigned int N;               //保存图的顶点数
extern uint64_t TotalEdgeCount;      //保存图的总边数
uint64_t OrderTabSize;               //各顶点新旧序号对照表占用的内存大小(字节数)
extern uint64_t EdgeNumbSize;        //各顶点频度占用内存的大小(字节数)
extern uint64_t EdgeAddrSize;        //各顶点首边偏移位置占用的内存大小(字节数)
extern uint64_t EdgeListSize;        //保存所有的边的数组占用的内存大小(字节数)

cudaDeviceProp deviceProp;		     //GPU的参数

uint64_t  TOTALMEMOSIZE = 0x400000000;	//GPU上可用的总内存数量: 16G
uint64_t  TOTALTHDCOUNT = 65536;	 //GPU允许的最大线程数
uint64_t *gpuEdgeAddr = NULL;        //在GPU上分配的,保存每个顶点的首边位置的数组地址;
uint32_t* gpuOrderTab = NULL;        //在GPU上分配的,保存每个顶点的新索引对照表地址;

uint32_t* tmpEdgeAdr;


//定义捕获显示错误的宏
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
//功能: 初始化GPU, 获取GPU相关参数
//输入: ReserveMemo＝预留给系统的存储器数(兆字节)
void initGPU(const uint64_t ReserveMemo)
{
    gpuErrchk( cudaGetDeviceProperties(&deviceProp, 0) );
#ifdef DEBUG_DF
    /*
    cout << "使用GPU 设备 0: " << deviceProp.name << endl;
    cout << "设备常量内存： " << deviceProp.totalConstMem << " bytes." << endl;
    cout << "纹理内存对齐： " << deviceProp.textureAlignment << " bytes." << endl;
    cout << "线程warp大小： " << deviceProp.warpSize << endl;
    cout << "SM的数量：" << deviceProp.multiProcessorCount << std::endl;
    cout << "每个线程块的共享内存大小：" << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
    cout << "每个线程块的最大线程数：" << deviceProp.maxThreadsPerBlock << endl;
    cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << deviceProp.regsPerBlock << endl;
    cout << "每个EM的最大线程数：" << deviceProp.maxThreadsPerMultiProcessor << endl;
    cout << "每个EM的最大线程束数：" << deviceProp.maxThreadsPerMultiProcessor / 32 << endl;
    cout << "设备上多处理器的数量： " << deviceProp.multiProcessorCount << endl;
    cout << "线程块每维最大线程数： " << deviceProp.maxThreadsDim[0]<< ", " << deviceProp.maxThreadsDim[1] << endl;
    cout << "网格每维最大线程数： " << deviceProp.maxGridSize[0]<< ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << endl;
    */
#endif
    
    //我们最少需要使用的存储器数量
    uint64_t usageMem = 1024*1024*1024;       //1G       edge_num / 2 * 8 + (N + 1) * 4 * 3 + 1024 * 1024 * 64;
    if(usageMem > deviceProp.totalGlobalMem){
        cerr << "Global memory(" 
                << deviceProp.totalGlobalMem / 1024 / 1024 
                << "MB) is not enough. Require " 
                << usageMem / 1024 / 1024 << "MB" << endl;
        exit(2);
    }
    TOTALMEMOSIZE = deviceProp.totalGlobalMem;	   //保存显示器总的现存数量
    cout << "GPU特性表返回内存总量：" << TOTALMEMOSIZE << " Bytes." << endl;
    TOTALMEMOSIZE -= ReserveMemo * 1024 * 1024;   //保留内存
    cout << "给系统预留后可用内存总量：" << TOTALMEMOSIZE << " Bytes." << endl;
}

//功能: 用二分法查找值在数组cpuEdgeAddr中位置
//输入: Value＝要查找的值, MinPos＝开始查找位置, MaxPos＝结束查找位置
//输出: 返回值相等位置,无相等值时返回小于值的最大的位置(就是插入位置)
uint32_t FindValuePos(uint64_t Value, uint32_t MinPos, uint32_t MaxPos, uint64_t* cpuEdgeAddr)
{
     uint32_t   MiddlePos;
     
     if (cpuEdgeAddr[MaxPos] <= Value) return MaxPos;
     while (MaxPos - MinPos > 1)
     {
         MiddlePos = (MinPos + MaxPos) / 2;
         if (cpuEdgeAddr[MiddlePos] == Value) return MiddlePos;
         if (cpuEdgeAddr[MiddlePos] > Value)
             MaxPos = MiddlePos;
         else MinPos = MiddlePos;
     }
     return MinPos;
}
//******************************************************************************************************
//功能: 并行运算查找最大的顶点序号, 计算各顶点边的总频度数
//输入: gpuEdgeList ＝原始边数据缓冲区地址, EdgeCount＝原始数据的边数目
//输出: gpuFrequence＝返回各顶点的边总频度数
//      最大的顶点序号在全局变量 devNodeCount 中
__device__ unsigned int devNodeCount = 0;
__global__ void __CalcNodeDegree(const uint64_t* __restrict__ gpuEdgeList, const uint64_t __restrict__ EdgeCount, uint32_t* __restrict__ gpuFrequence)
{
    uint64_t  tmpNodeCount = 0;     //保存当前线程找到的最大顶点序号
    uint64_t  EdgeData, FrontNode, BehindNode;
    unsigned int* tempFrequence = (unsigned int*)gpuFrequence;
    int threadCount = blockDim.x * gridDim.x;        //线程总数
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
        
    for (uint64_t i = threadId; i < EdgeCount; i += threadCount)
    {
        EdgeData  = gpuEdgeList[i];
        FrontNode = EdgeData & 0xFFFFFFFF;
        BehindNode = (EdgeData >> 32);
        atomicAdd(tempFrequence + FrontNode, 1);   //对应的顶点的边数增加
        atomicAdd(tempFrequence + BehindNode, 1);  //对应的顶点的边数增加

        if (FrontNode > BehindNode)
            BehindNode = FrontNode;
        if (tmpNodeCount < BehindNode)
            tmpNodeCount = BehindNode;
    }
    atomicMax(&devNodeCount, (unsigned int)tmpNodeCount);
}
//功能: 生成各顶点频度序号表(高32位为频度,低32位为顶点序号) 为排序准备
//输入: gpuFrequence＝各顶点边的总频度数, NodeCount＝顶点数目
//输出: gpuOrderDeg ＝保存结果的缓冲区
__global__ void __GenerateOrderDegreeTab(const uint32_t* __restrict__ gpuFrequence, const uint64_t __restrict__ NodeCount, uint64_t* __restrict__ gpuOrderDeg)
{
    int threadCount = blockDim.x * gridDim.x;        //线程总数
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t tmpFrequence;
    
    for (uint64_t i = threadId; i < NodeCount; i += threadCount)
    {
        tmpFrequence  = (uint64_t)gpuFrequence[i];
        gpuOrderDeg[i] = (tmpFrequence << 32) | i;
    }
}
//功能: 查找顶点的数目, 计算各顶点边的总频度数
//输入: cpuEdgeList＝原始数据的边数据地址, EdgeCount＝原始数据的边数目
//返回: 保存各顶点的边总频度数(高32bit)和顶点序号(低32bit)缓冲区的地址
//      查找到的顶点数返回在全局变量 N 中
uint64_t* gpuCalcDegreeFirst(uint64_t* cpuEdgeList, uint64_t EdgeCount)
{
    uint32_t  ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;   //线程块数
    uint64_t  MaxBlockSize;
    uint64_t* gpuEdgeList;         //GPU上分配的源数据边缓冲区
    uint32_t* gpuFrequence;        //GPU上计算各顶点总频度的缓冲区
    uint64_t* gpuOrderDeg;         //GPU上生成各顶点总频度(高32bit)和顶点序号(低32bit)的缓冲区
    uint64_t* cpuOrderDeg;         //CPU上分配的各顶点总频度(高32bit)和顶点序号(低32bit)的缓冲区
    
    cout << "使用GPU计算顶点度: gpuCalcDegreeFirst" << endl;
    //首先分配各顶点度计数占用的内存, 每顶点32bit 
    EdgeNumbSize = 0x40000000 * sizeof(uint32_t);                   //最多可以容纳1G个边
    gpuErrchk( cudaMalloc((void**)&gpuFrequence,  EdgeNumbSize));
    gpuErrchk( cudaMemset(gpuFrequence, 0, EdgeNumbSize));          //初始化为零
    
    //剩余内存用于拷贝一段边数据, 每次拷贝的块尽可能大．
    MaxBlockSize = (TOTALMEMOSIZE - EdgeNumbSize) / sizeof(uint64_t);    //计算每次拷贝块尺寸(边数)
    gpuErrchk( cudaMalloc((void**)&gpuEdgeList,  MaxBlockSize * sizeof(uint64_t)));
    //循坏调用每个块直至全部的边
    for (uint64_t i = 0; i < EdgeCount; i += MaxBlockSize)
    {
        uint64_t copySize = min(EdgeCount - i, MaxBlockSize);  
        cout << "GPU: Start mapfile MapData, i = " << i << ", BlockSize = " << copySize << endl;
        gpuErrchk( cudaMemcpy(gpuEdgeList, cpuEdgeList + i, copySize * sizeof(uint64_t), cudaMemcpyHostToDevice) );
        __CalcNodeDegree<<<ThreadBlockCount, BLOCKSIZE>>>(gpuEdgeList, copySize, gpuFrequence);
        gpuErrchk( cudaDeviceSynchronize() );
    }
    gpuErrchk( cudaMemcpyFromSymbol(&N, devNodeCount, sizeof(unsigned int)) );  //最大顶点序号
    N++;         //最大序号加1为顶点数
    
    //生成各顶点总频度(高32bit)和顶点序号, 即将gpuEdgeNumb中的频度移至高32bit, 低32bit填入顶点序号
    gpuOrderDeg = gpuEdgeList;      //直接使用边的缓冲区
    __GenerateOrderDegreeTab<<<ThreadBlockCount, BLOCKSIZE>>>(gpuFrequence, N, gpuOrderDeg);
    
    EdgeNumbSize = N * sizeof(uint64_t); 
    cpuOrderDeg = (uint64_t *)malloc(EdgeNumbSize);
    gpuErrchk( cudaMemcpy(cpuOrderDeg, gpuOrderDeg, EdgeNumbSize, cudaMemcpyDeviceToHost));   //保存每个顶点的频度和序号到CPU内存
   
    cudaFree(gpuEdgeList);
    cudaFree(gpuFrequence);
    return cpuOrderDeg;
}
// ********************************根据排序好的各顶点的边频度, 生成新旧序号对照表********************************
//功能: 并行生成新旧序号对照表
//参数: gpuEdgeNumb＝已排序的各顶点边频度表, 低32bit为原顶点序号, 高32bit为边频度(在这里没有实际用途)
//输出：生成的频度对照表放在GPU内存中, 其地址放入全局变量 gpuOrderTab 中.
__global__ void __GenerateNodeOrderTab(uint64_t* __restrict__ gpuEdgeNumb, uint32_t* __restrict__ gpuOrderTab, uint32_t __restrict__ NodeCount)
{
    int threadCount = blockDim.x * gridDim.x;        //线程总数
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (uint64_t i = threadId; i < NodeCount; i += threadCount)
    {
        uint32_t SaveIndex = (uint32_t)gpuEdgeNumb[i];        //取频度对应的顶点序号
        gpuOrderTab[SaveIndex] = i;
    }
}
//功能: 根据排序好的各顶点的边频度, 生成新旧序号对照表
//参数: cpuEdgeNumb＝已排序的各顶点边频度表, 低32bit为原顶点序号, 高32bit为边频度(在这里没有用途)
//输出：生成的频度对照表放在GPU内存中, 其地址放入全局变量 gpuOrderTab 中.
void gpuGenerateNodeOrderTab(uint64_t* cpuEdgeNumb)
{ 
    uint32_t  ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;   //线程块数
    uint64_t* gpuEdgeNumb;      //GPU上分配的各顶点边频度表
    uint64_t  MaxSortNumb;
    
    MaxSortNumb  = TOTALMEMOSIZE / sizeof(uint64_t) / 2;
    MaxSortNumb &= 0xFFFFFFFFFFFFFF00;
    if (N <= MaxSortNumb)                    //排序cpuEdgeNumb, 因高32位是频度值，相当于按频度排序
        gpuUllSort(cpuEdgeNumb, N);       
    else
        sort(cpuEdgeNumb, cpuEdgeNumb + N);  //排序cpuEdgeNumb, 因高32位是频度值，相当于按频度排序
    //分配各顶点边频度占用的内存
    EdgeAddrSize = (N + 1)* sizeof(uint64_t); 
    gpuErrchk( cudaMalloc((void**)&gpuEdgeNumb,  EdgeAddrSize));
    gpuErrchk( cudaMemcpy(gpuEdgeNumb, cpuEdgeNumb, N * sizeof(uint64_t), cudaMemcpyHostToDevice) );
    //分配各顶点的由旧序号转新序号的对照表占用的内存
    OrderTabSize =  N * sizeof(uint32_t); 
    gpuErrchk( cudaMalloc((void**)&gpuOrderTab,  OrderTabSize));
    //各顶点的由旧序号转新序号的对照表
    __GenerateNodeOrderTab<<<ThreadBlockCount, BLOCKSIZE>>>(gpuEdgeNumb, gpuOrderTab, N);
    
    gpuEdgeAddr = gpuEdgeNumb;    //gpuEdgeNumb占用的内存不释放, 以后由各顶点首边偏移位置gpuEdgeAddr使用
    //cudaFree(gpuOrderTab);      //gpuOrderTab占用的内存不释放, 在下面处理中直接使用
}
// ********************************按新顶点序,计算各顶点首边位置***********************************
//功能: 并行计算按顶点重排时, 各顶点的边频度数
//输入: cpuEdgeList＝原始边数据首地址,  EdgeCount＝边数据总数, gpuOrderTab＝将顶点旧序号映射成新序号的对照表地址
//输出: gpuEdgeNumb＝返回各顶点边频度 
//注释：按照根据边总频度大小重排后的新顶点序计算, 每个线程处理一个边的数据
__global__ void __CalcNodeDegreeAgain(const uint64_t* __restrict__ gpuEdgeList, const uint64_t __restrict__ EdgeCount, 
    uint64_t* __restrict__ gpuEdgeNumb, uint32_t* __restrict__ gpuOrderTab)
{
    int threadCount = blockDim.x * gridDim.x;        //线程总数
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long* tmpEdgeNumb = (unsigned long long*)gpuEdgeNumb;
    uint64_t tempData, FrontNode, BehindNode;
    
    for (uint64_t i = threadId; i < EdgeCount; i += threadCount)
    {
        tempData   = gpuEdgeList[i];
        FrontNode  = tempData & 0xFFFFFFFF;
        BehindNode = (tempData >> 32);
        if (FrontNode != BehindNode) 
        {   //映射为新序号
            FrontNode  = gpuOrderTab[FrontNode];
            BehindNode = gpuOrderTab[BehindNode];
            //计算频度
            if (FrontNode > BehindNode)
                FrontNode = BehindNode;
            atomicAdd(tmpEdgeNumb + FrontNode, 1);  //对应的顶点的边数增加
        }
    }
}
//功能: 计算按顶点重排时各顶点的邻接边新的存储偏移位置
//输入: cpuEdgeList＝原始边数据首地址,  EdgeCount＝边数据总数, 
//输出: cpuEdgeAddr＝返回各顶点首边偏移位置
//注释: 1.按照根据边总频度大小重排后的新顶点序计算
//      2.使用全局变量gpuOrderTab＝将顶点旧序号映射成新序号的对照表, 已与存入GPU内存
void gpuCalcDegreeAgain(uint64_t* cpuEdgeList, uint64_t EdgeCount, uint64_t* cpuEdgeAddr)
{
    uint32_t  ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;   //线程块数
    uint64_t  BlockSize;
    uint64_t* gpuEdgeList;      //GPU上分配的边列表
    uint64_t* gpuEdgeNumb;      //GPU上分配的各顶点边频度表
    
    cout << "使用GPU计算顶点度: gpuCalcDegreeAgain" << endl;
    //各顶点边频度占用的内存, 直接使用gpuEdgeAddr的内存
    gpuEdgeNumb = gpuEdgeAddr;
    gpuErrchk( cudaMemset(gpuEdgeNumb, 0, EdgeAddrSize));          //初始化为零
    
    //各顶点对应的新索引表已经在GPU内存中, 地址为 gpuOrderTab
    //BlockSize =  N * sizeof(uint32_t); 
    //gpuErrchk( cudaMalloc((void**)&gpuOrderTab,  BlockSize));
    //gpuErrchk( cudaMemcpy(gpuOrderTab, cpuOrderTab, BlockSize, cudaMemcpyHostToDevice) );
    
    //剩余内存用于拷贝原始边数据, 一次不能全部放入时分段拷贝
    BlockSize = (TOTALMEMOSIZE - EdgeAddrSize - OrderTabSize) / sizeof(uint64_t);    //计算每次拷贝块尺寸
    gpuErrchk( cudaMalloc((void**)&gpuEdgeList,  BlockSize * sizeof(uint64_t)));
    
    //循坏调用每个块直至全部的边
    for (uint64_t i = 0; i < EdgeCount; i += BlockSize)
    {
        uint64_t copySize = min(EdgeCount - i, BlockSize);   
        gpuErrchk( cudaMemcpy(gpuEdgeList, cpuEdgeList + i, copySize * sizeof(uint64_t), cudaMemcpyHostToDevice) );
        __CalcNodeDegreeAgain<<<ThreadBlockCount, BLOCKSIZE>>>(gpuEdgeList, copySize, gpuEdgeNumb, gpuOrderTab);
        gpuErrchk( cudaDeviceSynchronize() );
    }
    //保存各顶点边频度到CPU内存
    gpuErrchk( cudaMemcpy(cpuEdgeAddr + 1, gpuEdgeNumb, EdgeAddrSize - sizeof(uint64_t) , cudaMemcpyDeviceToHost)); 
    //计算各顶点首边偏移位置
    cpuEdgeAddr[0] = 0;
    for (uint64_t i = 0; i < N; i++) 
        cpuEdgeAddr[i+1] += cpuEdgeAddr[i];
    //释放资源    
    cudaFree(gpuEdgeList);
    //gpuEdgeNumb;                 //gpuEdgeAddr占用的内存不释放, 其内存在分桶操作时还要使用
    //cudaFree(gpuOrderTab);       //gpuOrderTab占用的内存不释放, 其数据在分桶操作时还要使用
}

// ********************************按新顶点序, 将原始边数据按顶点分桶***********************************
const uint32_t ORDER_SHIFT_BIT = 34;
const uint64_t OFFSET_MASK_VAL = 0x3FFFFFFFF;
__global__ void __MergeOrderAndOffset(const uint32_t* __restrict__ gpuOrderTab, const uint32_t __restrict__ NodeCount,
    uint64_t* __restrict__ gpuEdgeOffset)
{
    uint64_t tempData;
    int threadCount = blockDim.x * gridDim.x;        //线程总数
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    for (uint64_t i = threadId; i < NodeCount; i += threadCount)
    {   
        tempData  = (uint64_t)gpuOrderTab[i];
        gpuEdgeOffset[i] |= (tempData << ORDER_SHIFT_BIT);
    }
}
void gpuMergeOrderAndOffset(uint64_t* cpuEdgeAddr)
{    
    uint32_t  ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;   //线程块数
    uint64_t  tempDataVal;
    if (gpuEdgeAddr == NULL)
	    gpuErrchk( cudaMalloc((void**)&gpuEdgeAddr, EdgeAddrSize));
    gpuErrchk( cudaMemcpy(gpuEdgeAddr, cpuEdgeAddr, EdgeAddrSize, cudaMemcpyHostToDevice) );
    __MergeOrderAndOffset<<<ThreadBlockCount, BLOCKSIZE>>>(gpuOrderTab, N, gpuEdgeAddr);
    cudaFree(gpuOrderTab);
    /*
    tmpEdgeAdr = (uint32_t *)malloc(EdgeAddrSize);
    uint8_t* tmpHighAdr = (uint8_t*)(tmpEdgeAdr + N + 1);
    for (uint32_t i = 0; i <= N; i++)
    {
        tempDataVal = cpuEdgeAddr[i];
        tmpEdgeAdr[i] = (uint32_t)tempDataVal;
        tmpHighAdr[i] = (uint8_t)(tempDataVal >> 32);
    }
    */
}
//*********************************************


//功能: 并行将原始边数据按顶点序号重新存储(按顶点分桶)
//输入: cpuEdgeList＝原始边数据首地址, cpuEdgeAddr＝各顶点首边偏移位置, EdgeCount＝边数据总数, 
//      minNode＝由于内存限制,本次能处理的最小顶点序号(包含), maxNode＝由于内存限制,本次能处理的最大顶点序号(不包含)
//      StartPos＝由于内存限制, 本次能处理输出缓冲区相对于总缓冲区的偏移
//      gpuOrderTab＝将顶点旧序号映射成新序号的对照表
//输出: gpuOutEdge＝当前输出边缓冲区 
__global__ void __RearrangeEdge(const uint64_t* __restrict__ gpuEdgeList, const uint64_t* __restrict__ gpuEdgeAddr, 
    const uint32_t __restrict__ minNode, const uint32_t __restrict__ maxNode, const uint64_t __restrict__ StartPos, 
	const uint64_t __restrict__ EdgeCount, uint32_t* __restrict__ gpuOutEdge)
{
    uint64_t tempData, FrontNode, BehindNode;
    unsigned long long* tmpEdgePos = (unsigned long long*)gpuEdgeAddr;
    
    int threadCount = blockDim.x * gridDim.x;        //线程总数
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    for (uint64_t i = threadId; i < EdgeCount; i += threadCount)
    {   
        tempData   = gpuEdgeList[i];
        FrontNode  = tempData & 0xFFFFFFFF;
        BehindNode = (tempData >> 32);
        if (FrontNode != BehindNode) 
        {   //映射为新序号
            FrontNode  = tmpEdgePos[FrontNode] >> ORDER_SHIFT_BIT;
            BehindNode = tmpEdgePos[BehindNode] >> ORDER_SHIFT_BIT;
            if (FrontNode > BehindNode)
            {
                tempData   = FrontNode;
                FrontNode  = BehindNode;
                BehindNode = tempData;
            }
            if (FrontNode >= minNode && FrontNode < maxNode)
            {
                uint64_t SavePos = atomicAdd(tmpEdgePos + FrontNode, (unsigned long long)1);
                SavePos &= OFFSET_MASK_VAL;
                gpuOutEdge[SavePos - StartPos] = BehindNode;
            }
        }
    }
}

//功能: 将原始边数据按顶点序号重新存储(按顶点分桶)
//参数: mapfile＝原始数据影像类, cpuEdgeAddr＝已计算好各顶点首边偏移位置
//输出：gpuOutEdge＝保存重排后的边缓冲区地址
//注释: 1.按照根据边总频度大小重排后的新顶点序计算
//      2.使用全局变量gpuOrderTab＝将顶点旧序号映射成新序号的对照表, 已与存入GPU内存
//      3.计算结束时，gpuEdgeAddr中的值被破坏，需要重新拷贝


void gpuRearrangeEdge(uint64_t* cpuEdgeList, uint64_t* cpuEdgeAddr, uint32_t* cpuOutEdge)
{    
    uint32_t  ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;   //线程块数
    uint64_t  SrcSize;       //每次处理的边数
    uint64_t  DstSize;       //每次输出的边数
    uint32_t  minNode, maxNode;
    uint64_t  StartPos;                      //, OrderTabSize;
    uint64_t  SrcNumb, DstNumb;
    //uint64_t* endpoints;
    
    uint64_t* gpuEdgeList;    //要处理的边数组
    uint32_t* gpuOutEdge;    //要处理的边数组
    
    cout << "使用GPU生成邻接表: gpuRearrangeEdge" << endl;
    //这个操作比较麻烦，由于GPU的内存限制，对于大数据集, 不管是重排后的结果，还是重排前的原数据，都不一定能够被1次放入GPU内存中，都可能需要分段处理
    //合理分配GPU内存  
    DstSize  = MIN_EDGENUMB * sizeof(uint32_t);         //GPU内存分配: 保留给输出边的内存数, 至少能保存MINOUTSIZE个边
    SrcSize  = TOTALMEMOSIZE - EdgeAddrSize - DstSize;  //GPU内存分配: 剩余内存用于拷贝原始文件边数据 
    SrcSize &= MAP_BOUNDARY;                            //原始边数据映像必须是PAGE_SIZE的倍数, 这里取64K边界
    DstSize  = TOTALMEMOSIZE - EdgeAddrSize - SrcSize;  //重新计算留给输出边的内存数, 把原始边数据不能映像用的内存加进来
 
    SrcSize /= sizeof(uint64_t);     //原始边数据内存折算成边数
    DstSize /= sizeof(uint32_t);     //输出边数据内存折算成边数
       
    //DstSize = 0x4000000;        //只调试用: 设置边输出限制边数, 强迫分块
    //SrcSize = 0x4000000;        //只调试用: 设置源数据限制边数, 强迫分块
    gpuErrchk( cudaMalloc((void**)&gpuEdgeList, SrcSize * sizeof(uint64_t)));
    gpuErrchk( cudaMalloc((void**)&gpuOutEdge,  DstSize * sizeof(uint32_t)));

    cout << "内存分配: 源数据限制边数 = " << SrcSize  << ", 边输出限制边数 = " <<  DstSize  << endl;
    
    //外层循环对源数循环, 以最大限制数分段映像, 并拷贝到GPU内存中
    for (uint64_t i = 0; i < TotalEdgeCount; i += SrcSize)
    {   
        SrcNumb = min(TotalEdgeCount - i, SrcSize);        //源数据拷贝边数
        cout << "GPU: Start Position, i = " << i << ", SrcNumb = " << SrcNumb << endl;
        gpuErrchk( cudaMemcpy(gpuEdgeList,  cpuEdgeList + i,  SrcNumb * sizeof(uint64_t), cudaMemcpyHostToDevice))
        //对输出边分段循环, 以顶点为界限, 根据各顶点度数, 计算分段位置(每次处理尽可能多的顶点数据)
        minNode = 0;
        while (minNode < N)
        {
            StartPos = cpuEdgeAddr[minNode];
            maxNode = FindValuePos(StartPos +  DstSize, minNode, N, cpuEdgeAddr);
            DstNumb = cpuEdgeAddr[maxNode] - StartPos;                            //本次分段边数
            //cout << "GPU: minNode = " << minNode << ", maxNode = " << maxNode << ", DstNumb = " << DstNumb << endl;
            if (i > 0) gpuErrchk( cudaMemcpy(gpuOutEdge,cpuOutEdge + StartPos,  DstNumb * sizeof(uint32_t), cudaMemcpyHostToDevice));
            __RearrangeEdge<<<ThreadBlockCount, BLOCKSIZE>>>(gpuEdgeList, gpuEdgeAddr, minNode, maxNode, StartPos, SrcNumb, gpuOutEdge);
            gpuErrchk( cudaMemcpy(cpuOutEdge + StartPos, gpuOutEdge, DstNumb * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            minNode = maxNode;
        }
    }
    cudaFree(gpuOutEdge);
    cudaFree(gpuEdgeList);	
}









/*
void gpuRearrangeEdge(MapFile mapfile, uint64_t* cpuEdgeAddr, uint32_t* cpuOutEdge)
{    
    uint32_t  ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;   //线程块数
    uint64_t  SrcSize;       //每次处理的边数
    uint64_t  DstSize;       //每次输出的边数
    uint32_t  minNode, maxNode;
    uint64_t  StartPos;                      //, OrderTabSize;
    uint64_t  SrcNumb, DstNumb;
    uint64_t* endpoints;
    
    uint64_t* gpuEdgeList;    //要处理的边数组
    uint32_t* gpuOutEdge;    //要处理的边数组
    
    cout << "使用GPU生成邻接表: gpuRearrangeEdge" << endl;
    //这个操作比较麻烦，由于GPU的内存限制，对于大数据集, 不管是重排后的结果，还是重排前的原数据，都不一定能够被1次放入GPU内存中，都可能需要分段处理
    //合理分配GPU内存  
    DstSize  = MIN_EDGENUMB * sizeof(uint32_t);         //GPU内存分配: 保留给输出边的内存数, 至少能保存MINOUTSIZE个边
    SrcSize  = TOTALMEMOSIZE - EdgeAddrSize - DstSize;  //GPU内存分配: 剩余内存用于拷贝原始文件边数据 
    SrcSize &= MAP_BOUNDARY;                            //原始边数据映像必须是PAGE_SIZE的倍数, 这里取64K边界
    DstSize  = TOTALMEMOSIZE - EdgeAddrSize - SrcSize;  //重新计算留给输出边的内存数, 把原始边数据不能映像用的内存加进来
 
    SrcSize /= sizeof(uint64_t);     //原始边数据内存折算成边数
    DstSize /= sizeof(uint32_t);     //输出边数据内存折算成边数
       
    //DstSize = 0x4000000;        //只调试用: 设置边输出限制边数, 强迫分块
    //SrcSize = 0x4000000;        //只调试用: 设置源数据限制边数, 强迫分块
    gpuErrchk( cudaMalloc((void**)&gpuEdgeList, SrcSize * sizeof(uint64_t)));
    gpuErrchk( cudaMalloc((void**)&gpuOutEdge,  DstSize * sizeof(uint32_t)));

    cout << "内存分配: 源数据限制边数 = " << SrcSize  << ", 边输出限制边数 = " <<  DstSize  << endl;
    
    //外层循环对源数循环, 以最大限制数分段映像, 并拷贝到GPU内存中
    for (uint64_t i = 0; i < TotalEdgeCount; i += SrcSize)
    {   
        SrcNumb = min(TotalEdgeCount - i, SrcSize);        //源数据拷贝边数
        cout << "GPU: Start mapfile MapData, i = " << i << ", SrcNumb = " << SrcNumb << endl;
        endpoints = (uint64_t*)mapfile.MapData(i * sizeof(uint64_t), SrcNumb * sizeof(uint64_t));
        gpuErrchk( cudaMemcpy(gpuEdgeList,  endpoints,  SrcNumb * sizeof(uint64_t), cudaMemcpyHostToDevice))
        mapfile.UnMapData(endpoints, SrcNumb * sizeof(uint64_t));
        //对输出边分段循环, 以顶点为界限, 根据各顶点度数, 计算分段位置(每次处理尽可能多的顶点数据)
        minNode = 0;
        while (minNode < N)
        {
            StartPos = cpuEdgeAddr[minNode];
            maxNode = FindValuePos(StartPos +  DstSize, minNode, N, cpuEdgeAddr);
            DstNumb = cpuEdgeAddr[maxNode] - StartPos;                            //本次分段边数
            //cout << "GPU: minNode = " << minNode << ", maxNode = " << maxNode << ", DstNumb = " << DstNumb << endl;
            if (i > 0) gpuErrchk( cudaMemcpy(gpuOutEdge,cpuOutEdge + StartPos,  DstNumb * sizeof(uint32_t), cudaMemcpyHostToDevice));
            __RearrangeEdge<<<ThreadBlockCount, BLOCKSIZE>>>(gpuEdgeList, gpuEdgeAddr, minNode, maxNode, StartPos, SrcNumb, gpuOutEdge);
            gpuErrchk( cudaMemcpy(cpuOutEdge + StartPos, gpuOutEdge, DstNumb * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            minNode = maxNode;
        }
    }
    cudaFree(gpuOutEdge);
    cudaFree(gpuEdgeList);	
}
*/
//----------------------------------------------------MultWarp-------------------------------------------------------------------
__device__ unsigned int devNodeIdx = 0;
__device__ unsigned int devBlock1End;            //第一个边数据块结束节点索引
__device__ unsigned int devBlock2Start;          //第二个边数据块开始节点索引
__device__ unsigned int devBlock2End;            //第二个边数据块结束节点索引
__device__ unsigned long long devBlock1Pos;            //第一个边数据块边开始位置
__device__ unsigned long long devBlock2Pos;            //第二个边数据块边开始位置
__device__ unsigned long long devTriangleSum;

//功能：计算两组边的交集元素个数(即三角形数)
//输入：EdgeList1＝第一组邻接边地址, EdgeNumb1＝第一组邻接边数目
//      EdgeList2＝第二组邻接边地址, EdgeNumb2＝第二组邻接边数目; 
//输出：计算结果直接累加到变量 devTriangleCount中
__device__ void intersectMultWarp(const uint32_t* __restrict__ EdgeList1, const uint32_t* __restrict__ EdgeList2, 
    const uint32_t __restrict__ EdgeNumb1, const uint32_t __restrict__ EdgeNumb2)
{
    __shared__ uint32_t Edgeblock1[CALC_SIZE];
    __shared__ uint32_t Edgeblock2[CALC_SIZE];

    const int WarpBegin = threadIdx.x & (~(WARPSIZE - 1));   //一个Warp开始读数据的位置
    const int thd_Index = threadIdx.x & (WARPSIZE - 1);      //在Warp内线程的索引

    uint32_t i = 0, j = 0, sum = 0;
    uint32_t Block1Size = WARPSIZE;
    uint32_t Block2Size = WARPSIZE;

    while (i < EdgeNumb1 && j < EdgeNumb2) 
    {
        Block1Size = min(EdgeNumb1 - i, WARPSIZE);
        Block2Size = min(EdgeNumb2 - j, WARPSIZE);

        if(i + thd_Index < EdgeNumb1) Edgeblock1[threadIdx.x] = EdgeList1[i + thd_Index];
        if(j + thd_Index < EdgeNumb2) Edgeblock2[threadIdx.x] = EdgeList2[j + thd_Index];

        __threadfence_block();

        for(int k = 0; k < Block2Size; ++k)
            sum += (thd_Index < Block1Size) & (Edgeblock1[threadIdx.x] == Edgeblock2[WarpBegin + k]);

        uint32_t LastEdge1 = Edgeblock1[WarpBegin + Block1Size - 1];
        uint32_t LastEdge2 = Edgeblock2[WarpBegin + Block2Size - 1];

        if(LastEdge1 >= LastEdge2) j += Block2Size;
        if(LastEdge1 <= LastEdge2) i += Block1Size;
    }
    
    atomicAdd(&devTriangleSum, sum);
}
//功能：并行计算两个分块的三角形数目
//输入：gpuEdgeAddr＝各顶点首边的偏移位置, gpuEdgeList1＝第一组分块边地址, gpuEdgeList2＝第二组分块边地址
//输出：计算结果直接累加到变量 devTriangleCount中
__global__ void __CalcTriangleMultWarp(const uint64_t* gpuEdgeAddr, uint32_t* gpuEdgeList1, uint32_t* gpuEdgeList2)
{
    uint32_t  *FirstList, FirstNumb;
    uint32_t  *SecondList, SecondNumb;
    uint32_t  SecondNode;
    uint32_t  NodeIdx, EdgeIdx; 
    
    const int Warp_Idx = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    const int WarpNumb = blockDim.x  / WARPSIZE;    //一个线程块的Warp数
  
    __shared__ unsigned int BlockNodeIdx ;

    while (true)
    {   
        if(threadIdx.x == 0)
        {
             BlockNodeIdx = atomicAdd(&devNodeIdx, 1);
        } 
        __syncthreads();
        NodeIdx = BlockNodeIdx;
        if(NodeIdx >= devBlock1End) break;
        FirstList = gpuEdgeList1 + (gpuEdgeAddr[NodeIdx] - devBlock1Pos);
        FirstNumb = gpuEdgeAddr[NodeIdx + 1] - gpuEdgeAddr[NodeIdx];
        if (FirstNumb > 0)  
        for (EdgeIdx = Warp_Idx; EdgeIdx < FirstNumb; EdgeIdx += WarpNumb) 
        {
            SecondNode = FirstList[EdgeIdx];
            if (SecondNode >= devBlock2End) break;
            if (SecondNode >= devBlock2Start)
            {
                SecondList = gpuEdgeList2 + (gpuEdgeAddr[SecondNode] - devBlock2Pos);
                SecondNumb = gpuEdgeAddr[SecondNode + 1] - gpuEdgeAddr[SecondNode];
                if (SecondNumb > 0)
                    intersectMultWarp(FirstList + EdgeIdx + 1, SecondList, FirstNumb - EdgeIdx - 1, SecondNumb);
            }
        }
    }
}


//功能：计算数据集的三角形数目, 当数据集大到GPU内存不足时,自动分块计算
//输入：cpuEdgeList＝数据集的邻接边地址, cpuEdgeAddr＝各顶点首边的偏移位置
//输出：计算结果直接累加到变量 devTriangleCount中
//注释, 使用全局变量gpuEdgeAddr, 其在GPU上的内存已经分配
uint64_t gpuCalcTriangle(uint32_t* cpuEdgeList, uint64_t* cpuEdgeAddr)
{
    int numBlocks = TOTALTHDCOUNT/CALC_SIZE;
    uint32_t* gpuEdgeList;
    unsigned int Block1Start, Block1End;         //第一个分块的起始顶点序号和结束顶点序号
    unsigned int Block2Start, Block2End;         //第二个分块的起始顶点序号和结束顶点序号
    unsigned long long Block1Pos, Block2Pos;     //分别为第一二个分块的邻接边的起始位置偏移
    uint64_t  TotleMemoSize, CopySize;
    uint64_t  TempBlockSize;
    uint32_t* gpuTempList;                        //分块计算时保存第二块在GPU中的起始地址
    
    cout << "使用GPU计算三角形:  gpuCalcTriangle." << endl; 
    
    unsigned long long tmpSum = 0;
    gpuErrchk( cudaMemcpyToSymbol(devTriangleSum, &tmpSum, sizeof(unsigned long long)) );
    
    //首先分配cpuEdgeAddr占用的内存
    if (gpuEdgeAddr == NULL)
        gpuErrchk( cudaMalloc((void**)&gpuEdgeAddr, EdgeAddrSize) );
    gpuErrchk( cudaMemcpy(gpuEdgeAddr, cpuEdgeAddr, EdgeAddrSize, cudaMemcpyHostToDevice) );
    //剩余内存将保存边的数据  
    TotleMemoSize = (TOTALMEMOSIZE - EdgeAddrSize) / sizeof(uint32_t);   //计算剩余内存能容纳的边数 
    //TotleMemoSize = 100000000;      //只调试使用, 限制内存使强迫分块计算
       
    Block1Start = 0;    Block1Pos = 0;
    while (Block1Start < N)
    {   
        gpuErrchk( cudaMemcpyToSymbol(devBlock1Pos,  &Block1Pos,  sizeof(unsigned long long)));
        if (cpuEdgeAddr[N] - Block1Pos < TotleMemoSize)
        {
            //(剩余的)计算能够一次性拷贝完成
            CopySize = TotalEdgeCount - Block1Pos;
            cout << "(剩余的)计算一次完成: CopySize = " << CopySize << endl; 
            cout << "Block1Start=" << Block1Start << ", Block1End=" << N << ", Block1Pos=" << Block1Pos << endl; 
            gpuErrchk( cudaMalloc((void**)&gpuEdgeList, TotleMemoSize * sizeof(uint32_t)) );
            gpuErrchk( cudaMemcpy(gpuEdgeList, cpuEdgeList + Block1Pos, CopySize * sizeof(uint32_t), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpyToSymbol(devNodeIdx,     &Block1Start, sizeof(unsigned int)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock1End,   &N,           sizeof(unsigned int)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock2Pos,   &Block1Pos,   sizeof(unsigned long long)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock2Start, &Block1Start, sizeof(unsigned int)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock2End,   &N,           sizeof(unsigned int)) );
            __CalcTriangleMultWarp<<<numBlocks, CALC_SIZE>>>(gpuEdgeAddr, gpuEdgeList, gpuEdgeList);
      
            //gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            cudaFree(gpuEdgeList);
            break;
        } else
        {
            cout << "计算不能一次完成, 只能分块进行" << endl; 
            //首次计算, 两块实际上是连续的一大块. 第一块作为要完成所有计算的一组边, 第一块以及第二块整体是参与计算的第二组边.
            TempBlockSize = TotleMemoSize / 2;                           //第一块的最大尺寸(边数) 
            Block2End = FindValuePos(Block1Pos +  TotleMemoSize, Block1Start, N, cpuEdgeAddr);
            Block1End = FindValuePos(Block1Pos +  TempBlockSize, Block1Start, Block2End, cpuEdgeAddr);
                                 
            cout << "分块首次计算: Block1Start=" << Block1Start << ", Block1End=" << Block1End << ", Block2End=" << Block2End << endl; 
            CopySize = cpuEdgeAddr[Block2End] - Block1Pos;
            gpuErrchk( cudaMalloc((void**)&gpuEdgeList, TotleMemoSize * sizeof(uint32_t)) );
            gpuErrchk( cudaMemcpy(gpuEdgeList, cpuEdgeList + Block1Pos, CopySize * sizeof(uint32_t), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpyToSymbol(devNodeIdx,     &Block1Start, sizeof(unsigned int)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock1End,   &Block1End,   sizeof(unsigned int)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock2Pos,   &Block1Pos,   sizeof(unsigned long long)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock2Start, &Block1Start, sizeof(unsigned int)) );
            gpuErrchk( cudaMemcpyToSymbol(devBlock2End,   &Block2End,   sizeof(unsigned int)) );
            cout << "Block1Start=" << Block1Start << ", Block1End=" << Block1End << ", Block1Pos=" << Block1Pos << endl; 
            cout << "Block2Start=" << Block1Start << ", Block2End=" << Block2End << ", Block2Pos=" << Block1Pos << endl; 
            __CalcTriangleMultWarp<<<numBlocks, CALC_SIZE>>>(gpuEdgeAddr, gpuEdgeList, gpuEdgeList);
            gpuErrchk( cudaDeviceSynchronize() );
            cudaFree(gpuEdgeList);
            gpuErrchk( cudaMemcpyFromSymbol(&tmpSum, devTriangleSum, sizeof(tmpSum)) );
            cout << "第一次计算结束: 当前三角形数 tmpSum=" << tmpSum << endl; 
            
            //后续的计算, 需要拷贝不连续的两个块. 先拷贝第一块 
            CopySize = cpuEdgeAddr[Block1End] - Block1Pos;
            gpuErrchk( cudaMalloc((void**)&gpuEdgeList, CopySize * sizeof(uint32_t)) );
            gpuErrchk( cudaMemcpy(gpuEdgeList, cpuEdgeList + Block1Pos, CopySize * sizeof(uint32_t), cudaMemcpyHostToDevice) );
            //分配第二块内存          
            TempBlockSize = TotleMemoSize - CopySize;      //第二块容许的最大尺寸
            gpuErrchk( cudaMalloc((void**)&gpuTempList, TempBlockSize * sizeof(uint32_t)) );
            //循环直至所有的块
            while (Block2End < N)
            {   
                Block2Start = Block2End;
                Block2Pos   = cpuEdgeAddr[Block2Start];
                Block2End = FindValuePos(Block2Pos +  TempBlockSize, Block2Start, N, cpuEdgeAddr);
              
                cout << "分块后续计算: Block2Start=" << Block2Start << ", Block2End=" << Block2End << ", Block2Pos=" << Block2Pos << endl; 
                CopySize = cpuEdgeAddr[Block2End] - Block2Pos;
                gpuErrchk( cudaMemcpy(gpuTempList, cpuEdgeList + Block2Pos, CopySize * sizeof(uint32_t), cudaMemcpyHostToDevice) );
                gpuErrchk( cudaMemcpyToSymbol(devNodeIdx,     &Block1Start, sizeof(unsigned int)) );
                gpuErrchk( cudaMemcpyToSymbol(devBlock2Pos,   &Block2Pos,   sizeof(unsigned long long)) );
                gpuErrchk( cudaMemcpyToSymbol(devBlock2Start, &Block2Start, sizeof(unsigned int)) );
                gpuErrchk( cudaMemcpyToSymbol(devBlock2End,   &Block2End,   sizeof(unsigned int)) );
               
                __CalcTriangleMultWarp<<<numBlocks, CALC_SIZE>>>(gpuEdgeAddr, gpuEdgeList, gpuTempList);
                gpuErrchk( cudaDeviceSynchronize() );
                
                gpuErrchk( cudaMemcpyFromSymbol(&tmpSum, devTriangleSum, sizeof(tmpSum)) );
                cout << "完成一次后续计算: 当前三角形数 tmpSum=" << tmpSum << endl; 
            }
            cudaFree(gpuEdgeList);
            cudaFree(gpuTempList);
        }
        Block1Start = Block1End;
        Block1Pos = cpuEdgeAddr[Block1Start];
    }
    gpuErrchk( cudaMemcpyFromSymbol(&tmpSum, devTriangleSum, sizeof(tmpSum)) );
    cudaFree(gpuEdgeAddr);
  
    return tmpSum;
}
// 使用 Thrust 库中的排序算法 
void gpuUllSort(uint64_t *cpuBlockStart, uint32_t blockSize)
{
    // cout << "get int gpu" << endl;
    thrust::device_vector<unsigned long long> gpuBlockData(blockSize);
    // cout << "start put value " << cpuBlockStart[0] << " " << blockSize << endl;
    thrust::copy(cpuBlockStart, cpuBlockStart+blockSize, gpuBlockData.begin()); 

    thrust::sort(gpuBlockData.begin(), gpuBlockData.end());
    thrust::copy(gpuBlockData.begin(),gpuBlockData.end(), cpuBlockStart);

}
