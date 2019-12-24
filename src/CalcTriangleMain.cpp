#include <iostream>
#include <fstream>
#include <cstring>
#include <omp.h>

#include <algorithm>
#include <random>
#include <chrono>

#include "mapfile.hpp"

void DebugArray2(const uint32_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx);
void DebugArray4(const uint64_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx);
void DebugArrayMatrix2(const char* title, const uint32_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx, uint64_t RowLen);
void DebugArrayMatrix4(const char* title, const uint64_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx, uint64_t RowLen);

void initGPU(const uint64_t ReserveMemo);
void gpuUllSort(uint64_t *cpuBlockStart, uint32_t blockSize);

uint64_t* gpuCalcDegreeFirst(uint64_t* cpuEdgeList, uint64_t EdgeCount);
void gpuGenerateNodeOrderTab(uint64_t* cpuEdgeNumb);
void gpuCalcDegreeAgain(uint64_t* cpuEdgeList, uint64_t EdgeCount, uint64_t* cpuEdgeAddr);
void gpuMergeOrderAndOffset(uint64_t* cpuEdgeAddr);
//void gpuRearrangeEdge(MapFile mapfile, uint64_t* cpuEdgeAddr, uint32_t* cpuOutEdge);
void gpuRearrangeEdge(uint64_t* cpuEdgeList, uint64_t* cpuEdgeAddr, uint32_t* cpuOutEdge);
void cpuSortEdgeEachNode(uint32_t* cpuEdgeList, uint64_t* cpuEdgeAddr);
uint64_t gpuCalcTriangle(uint32_t* cpuEdgeList, uint64_t* cpuEdgeAddr);


using namespace std;

#ifdef DEBUG_DF
    TimeInterval allTime;
    TimeInterval preProcessTime;
    TimeInterval tmpTime;
#endif

//***********************************************
unsigned int N = 0;		    //保存图的顶点数
uint64_t TotalEdgeCount;	//保存图的总边数
uint64_t TotalTriCount;	    //保存计算结果: 三角形总数
uint64_t EdgeNumbSize;      //各顶点频度占用内存的大小(字节数)
uint64_t EdgeAddrSize;      //保存每个顶点首个边位置的数组占用的空间大小(字节数)
uint64_t EdgeListSize;      //保存所有的边的数组占用的空间大小(字节数)

int main(int argc, char *argv[]) 
{
    char const *dt_name;
    uint64_t  i;
    // 接收命令并读取文件
    if (argc < 1)
    {
        printf("Usage: %s filename.\n", argv[0]);
        exit(1);
    }
    else if (argc == 3)
        dt_name = argv[2];
    else
        dt_name = argv[1];
    /*
    cout << "uint32_t " << sizeof(uint32_t) << " bytes." << endl;
    cout << "uint64_t " << sizeof(uint64_t) << " bytes." << endl;
    cout << "unsigned int  " << sizeof(unsigned int)  << " bytes." << endl;
    cout << "unsigned long " << sizeof(unsigned long) << " bytes." << endl;
    cout << "unsigned long long  " << sizeof(unsigned long long)  << " bytes." << endl;
    */
     //初始化GPU, 获取参数
    initGPU(512);
    // 通过内存映射实现快速读取
    MapFile mapfile(dt_name);
    uint64_t dt_size = mapfile.getLen();
    cout << "The size of data file " << dt_name << " is " << dt_size << " Bytes."  << endl;
   
#ifdef DEBUG_DF
    cout << "Start reading file" << endl;
    tmpTime.check();
#endif
    
    uint64_t *endpoints = (uint64_t*)mapfile.MapData(0, dt_size);
    //计算图的边数
    TotalEdgeCount = dt_size / sizeof(uint64_t);   //计算边数
    cout << "Data is loaded successfully. " << endl;
    cout << "There are " << TotalEdgeCount << "  edges." << endl;
  
#ifdef DEBUG_DF
    tmpTime.print("Mapfile Time Cost");
    preProcessTime.check();
    tmpTime.check();
#endif
    //查找最大的顶点号，同时计算各顶点的度
    uint64_t *TempEdgeNumb = gpuCalcDegreeFirst(endpoints, TotalEdgeCount);  //返回各顶点的边总频度数(高32bit)和顶点序号(低32bit)缓冲区的地址
    cout << "There are " << N << "  Nodes." << endl;
    cout << "There are " << TotalEdgeCount << "  edges." << endl;
#ifdef DEBUG_DF
    tmpTime.print("查找顶点数和计算顶点的度时间消耗");
    tmpTime.check();
#endif 
    //按顶点边的频度, 重新对顶点编索引
    gpuGenerateNodeOrderTab(TempEdgeNumb);
    free(TempEdgeNumb);
#ifdef DEBUG_DF
    tmpTime.print("生成顶点新旧序号对照表时间消耗");
    tmpTime.check();
#endif     
    //按新序号再次计算各顶点的边频度, 计算各顶点首边的偏移位置
    EdgeAddrSize = sizeof(uint64_t) * (N + 1);
    uint64_t *RowEdgeAddr = (uint64_t *)malloc(EdgeAddrSize);                       //按行存储: 保存每个顶点的首边位置  
    gpuCalcDegreeAgain(endpoints, TotalEdgeCount, RowEdgeAddr);
    //mapfile.UnMapData(endpoints, dt_size);
    cout << "After Remove Self-Cycle Edges, There are " << RowEdgeAddr[N] << "  edges." << endl;
#ifdef DEBUG_DF
    tmpTime.print("计算各顶点首边的偏移位置时间消耗");
    tmpTime.check();
#endif 
    gpuMergeOrderAndOffset(RowEdgeAddr);
#ifdef DEBUG_DF
    tmpTime.print("合并顶点新旧序号和偏移表时间消耗");
    tmpTime.check();
#endif  
    //按顶点对边重新分桶
    EdgeListSize = sizeof(uint32_t) * RowEdgeAddr[N];
    uint32_t *RowEdgeList = (uint32_t *)malloc(EdgeListSize);                       //按行存储: 保存每个顶点的边列表
    gpuRearrangeEdge(endpoints, RowEdgeAddr, RowEdgeList);
    //调用函数gpuRearrangeEdge后，已经去掉了自环的边, 总边数在RowEdgeAddr[N]中
    TotalEdgeCount = RowEdgeAddr[N];
    // 关闭镜像通道
    mapfile.UnMapData(endpoints, dt_size);
    mapfile.release();  
#ifdef DEBUG_DF
    tmpTime.print("生成邻接表结构时间消耗");
    tmpTime.check();
#endif

    //对每个顶点的边排序, 并去掉重复的边 
    cpuSortEdgeEachNode(RowEdgeList, RowEdgeAddr);
    //cout << "Sort End!! " << endl;
    //for (i = 0; i < N; i++)
    //    RowEdgeNumb = RowEdgeAddr[i+1] - RowEdgeAddr[i];
    EdgeListSize = sizeof(uint32_t) * TotalEdgeCount;
    cout << "After Remove Repetitive Edges, There are " << TotalEdgeCount << "  edges." << endl;
    
#ifdef DEBUG_DF
    tmpTime.print("去重边时间消耗");
    preProcessTime.print("预处理时间消耗");
    tmpTime.check();
#endif
    //计算三角形个数
    TotalTriCount = gpuCalcTriangle(RowEdgeList, RowEdgeAddr);
   //输出结果
    cout << "There are " << TotalTriCount << " triangles in the input graph." << endl;
#ifdef DEBUG_DF
    tmpTime.print("Counting Time Cost");
    allTime.print("All Time Cost");
#endif
    return 0;
}

//******************************************************************
//功能: 最大顶点序号
//#pragma omp parallel for reduction(max: dim)
//    for (i = 0; i < num; i++)
//        dim = max(edges[i].first, dim);
unsigned int cpuCalcDegreeAndMaxNode(uint64_t* cpuEdgeList, uint64_t EdgeCount, uint32_t* cpuEdgeNumb)
{
    uint64_t EdgeData, FrontNode, BehindNode;
    uint64_t maxNode = 0;
    cout << "使用cPU计算顶点度: cpuCalcDegreeAndMaxNode" << endl;
    //#pragma omp parallel for reduction(max : maxNode)
    for (uint64_t i = 0; i < EdgeCount; i++)
    {
        EdgeData  = cpuEdgeList[i];
        FrontNode = EdgeData & 0xFFFFFFFF;
        BehindNode = (EdgeData >> 32);
        //cout << " i = " << i << " FrontNode = " << FrontNode << " BehindNode = " << BehindNode << endl;
        if (FrontNode != BehindNode) 
        {
            if (FrontNode > BehindNode)
            {
                FrontNode  = BehindNode;
                BehindNode = EdgeData & 0xFFFFFFFF;
            }
            maxNode = max(BehindNode, maxNode);
            cpuEdgeNumb[FrontNode]++;
        } 
    }
    return maxNode;
}
uint32_t cpuFindMaxNodeNo(uint64_t* EdgeList, uint32_t* cpuFrontNode, uint32_t* cpuBehindNode)
{
    uint64_t EdgeSelfCount = 0;    //自环的边计数
    uint64_t EdgeData, FrontNode, BehindNode;
    uint32_t* cpuFrontNode1 = cpuFrontNode;
    uint32_t* cpuBehindNode1 = cpuBehindNode;
    unsigned int  maxNode = 0;
    #pragma omp parallel for reduction(max : maxNode)
    for (uint64_t i = 0; i < TotalEdgeCount; i++)
    {
        EdgeData = EdgeList[i];
        FrontNode = EdgeData & 0xFFFFFFFF;
        BehindNode = (EdgeData >> 32);
        if (FrontNode != BehindNode) 
        {
            if (FrontNode > BehindNode)
            {
                FrontNode  = BehindNode;
                BehindNode = EdgeData & 0xFFFFFFFF;
            }
            cpuFrontNode[i]  = (uint32_t)FrontNode;
            cpuBehindNode[i] = (uint32_t)BehindNode;
             
            maxNode = max((unsigned int)BehindNode, maxNode);
        } else
        { 
            //cout << "cpuFindMaxNodeNo: Self Cycle i = " << i << " Node = " << FrontNode << endl;
            cpuFrontNode[i]  = 0xFFFFFFFF;
            cpuBehindNode[i] = 0xFFFFFFFF;
        }
          //  EdgeSelfCount++;
    }
    return (uint32_t)maxNode + 1;
    //N = maxNode + 1;     //加1即为顶点数
    //TotalEdgeCount -= EdgeSelfCount;
    //cout << "There are " << EdgeSelfCount << " Self Cycle edges." << endl;
   
}

//功能: 计算按顶点重排时,每个顶点的邻接边数,以及新的存储位置
//参数: cpuEdgeList＝重排的顶点边数据;;
//      cpuEdgeNumb＝保存每个顶点的边数，cpuEdgeAddr＝保存各顶点首个边的位置, cpuEdgePos＝保存按顶点重排后边的新存储位置
//注释：需要使用全局变量：TotalEdgeCount边的总数，N 顶点的总数
//注释: 在大数据集情况下,不能保证所有的边数据一次性能全部拷贝到GPU内存中, 所以不能简单地调用1次核函数完成，可能需要分段完成, 进行多次调用
//void cpuCalcNodeDegree(const uint32_t* cpuFrontNode, const uint32_t* cpuBehindNode, uint64_t* cpuEdgeNumb, uint64_t* cpuEdgeAddr)
void cpuCalcNodeDegree(const uint32_t* cpuFrontNode, const uint32_t* cpuBehindNode, uint64_t* cpuEdgeAddr)
{
    uint32_t FrontNode;

    memset(cpuEdgeAddr, 0, EdgeAddrSize);              //初始化为零
    cout << "使用CPU计算顶点度: cpuCalcNodeDegree" << endl;
    for (uint64_t i = 0; i < TotalEdgeCount; i ++)
    {   
        FrontNode = *cpuFrontNode++;
        if (FrontNode != cpuBehindNode[i])
            cpuEdgeAddr[FrontNode + 1] += 1;
     }
    //计算按顶点分桶各顶点首个边的位置
    //DebugArray4(cpuEdgeAddr, 0, N + 1);
    for (uint32_t k = 1; k < N; ++k) 
        cpuEdgeAddr[k+1] += cpuEdgeAddr[k];
  
}


//功能: 按顶点序重新存储边
//参数: gpuEdgeRow、gpuEdgeCol＝待重排的边数据, gpuEdgePos＝已计算好的新存储位置, gpuEdgeAddr＝已计算好的各顶点首个边的位置
//输出：gpuOutEdge＝保存重排后的边
void cpuRearrangeEdge1(uint64_t* EdgeList, uint64_t EdgeCount, uint64_t* cpuEdgeAddr, uint32_t* cpuOutEdge)
{    
    uint64_t  EdgeData, FrontNode, BehindNode;
    uint64_t  SaveIndex;
    uint64_t *TempEdgeAddr;
    
    cout << "使用CPU生成邻接表: cpuRearrangeEdge1" << endl;
    TempEdgeAddr = (uint64_t *)malloc(EdgeAddrSize);                       //按行存储: 保存每个顶点的首边位置
    memcpy(TempEdgeAddr, cpuEdgeAddr, EdgeAddrSize);
    for (uint64_t i = 0; i < EdgeCount; i++ )
    {   
	    EdgeData  = EdgeList[i];
        FrontNode = EdgeData & 0xFFFFFFFF;
        BehindNode = (EdgeData >> 32);
        if (FrontNode != BehindNode) 
        {
            if (FrontNode > BehindNode)
            {
                FrontNode  = BehindNode;
                BehindNode = EdgeData & 0xFFFFFFFF;
            }
            SaveIndex  = cpuEdgeAddr[FrontNode];
            cpuOutEdge[SaveIndex] = BehindNode;
            cpuEdgeAddr[FrontNode]++;  
        } 
    }
    free(TempEdgeAddr);
}

//功能: 按顶点序重新存储边
//参数: gpuEdgeRow、gpuEdgeCol＝待重排的边数据, gpuEdgePos＝已计算好的新存储位置, gpuEdgeAddr＝已计算好的各顶点首个边的位置
//输出：gpuOutEdge＝保存重排后的边
void cpuRearrangeEdge(const uint32_t* cpuFrontNode, const uint32_t* cpuBehindNode, uint64_t* cpuEdgeAddr, uint32_t* cpuOutEdge)
{    
    uint32_t   FrontNode, BehindNode;
    uint64_t   SaveIndex;
    cout << "使用CPU生成邻接表: cpuRearrangeEdge" << endl;
    uint64_t  *tmpEdgeAddr = (uint64_t *)malloc(sizeof(uint64_t) * (N + 1)); 
    memcpy(tmpEdgeAddr, cpuEdgeAddr, sizeof(uint64_t) * (N + 1));
    for (uint64_t i = 0; i < TotalEdgeCount; i++ )
    {   
	    FrontNode  = *cpuFrontNode++;
        BehindNode = *cpuBehindNode++;
        if (FrontNode != BehindNode)
        {
            SaveIndex  = tmpEdgeAddr[FrontNode];
            cpuOutEdge[SaveIndex] = BehindNode;
            tmpEdgeAddr[FrontNode]++;
        }
    }
    free(tmpEdgeAddr);
}
//******************************************************************
//功能：对按顶点分桶后的边排序，排序后再去掉无效的边
//输入：cpuEdgeList＝邻接边列表; cpuEdgeNumb＝各结点的邻接边数; cpuEdgeAddr＝各结点首个邻接边索引
int comp_uint(const void *a, const void *b)
{
    return ((*(uint32_t *)a < *(uint32_t *)b) ? -1 : 1);
}

void  cpuSortEdgeEachNode(uint32_t* cpuEdgeList, uint64_t* cpuEdgeAddr)
{
    uint32_t   TempData, i;
    uint32_t*  SortBegin;
    uint64_t   SortLen;
    uint64_t   NewLstPos = 0;
    cout << "使用CPU生成排序去重: cpuSortEdgeEachNode" << endl;
    
    #pragma omp parallel for schedule(dynamic, 256)
    for (i = 0; i < N; i++)
    {   
	    SortBegin = cpuEdgeList + cpuEdgeAddr[i];
        SortLen   = cpuEdgeAddr[i + 1] - cpuEdgeAddr[i];
        if (SortLen > 1)
            //qsort(SortBegin, SortLen, sizeof(uint32_t), comp_uint);
            sort(SortBegin, SortBegin + SortLen);
    }
#ifdef DEBUG_DF
    tmpTime.print("分桶后排序时间消耗");
    tmpTime.check();
#endif   
  	//去掉重复的边
    for (uint32_t i = 0; i < N; i++)
    {
        SortBegin = cpuEdgeList + cpuEdgeAddr[i];
        SortLen   = cpuEdgeAddr[i + 1] - cpuEdgeAddr[i];
        cpuEdgeAddr[i] = NewLstPos;
        if (SortLen > 0)
        {   
            TempData = *SortBegin++;
            cpuEdgeList[NewLstPos++] = TempData;
            for (uint64_t k = 1; k < SortLen; k++, SortBegin++)
            if (*SortBegin != TempData) 
            {  
               TempData = *SortBegin;
               cpuEdgeList[NewLstPos++] = TempData;
	        }
        }
        
    }
    TotalEdgeCount = NewLstPos;
    cpuEdgeAddr[N] = NewLstPos;
 }
//******************************************************************
//功能：计算三角形总数目
//输入：cpuEdgeList＝邻接边列表; cpuEdgeNumb＝各结点的邻接边数; cpuEdgeAddr＝各结点首个邻接边索引
//返回：三角形计数结果
uint64_t cpuCalcTriangle(uint32_t* gpuEdgeList, uint64_t* gpuEdgeAddr)
{
    unsigned long long TriangleSum = 0;
    uint32_t  *FirstList, FirstNumb;
    uint32_t  *SecondList, SecondNumb;
    uint32_t  SecondNode;
    uint32_t  NodeIdx, EdgeIdx, i, j;
    cout << "使用CPU计算三角形: cpuCalcTriangle" << endl;
    for  (NodeIdx = 0; NodeIdx < N; NodeIdx++)
    {    
        FirstList = gpuEdgeList + gpuEdgeAddr[NodeIdx];
        FirstNumb = gpuEdgeAddr[NodeIdx + 1] - gpuEdgeAddr[NodeIdx];
        if (FirstNumb > 0)
        {  
            for (EdgeIdx = 0; EdgeIdx < FirstNumb; EdgeIdx ++)
            {
                SecondNode = FirstList[EdgeIdx];
                SecondList = gpuEdgeList + gpuEdgeAddr[SecondNode];
                SecondNumb = gpuEdgeAddr[SecondNode + 1] - gpuEdgeAddr[SecondNode];
                i =  EdgeIdx + 1; j = 0;
                while (i < FirstNumb && j < SecondNumb)
                {
                    if (FirstList[i] == SecondList[j]) 
                    {   
                        TriangleSum++;
                        i++; j++;
                    } else if (FirstList[i] < SecondList[j])
                        i++;
                    else
                       j++;
                }
            }
        }
    }
    return TriangleSum;
}





void DebugArray2(const uint32_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx)
{
    for  (uint64_t i = StartIdx; i < EndIdx; i++)
   {    
        cout << "cpuArray[" << i << "]=" << cpuArray[i] << ",  地址=" << cpuArray + i  <<  endl;
    }
}
void DebugArray4(const uint64_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx)
{
    for  (uint64_t i = StartIdx; i < EndIdx; i++)
   {    
        cout << "cpuArray[" << i << "]=" << cpuArray[i] << ", 地址=" << cpuArray + i <<  endl;
   }
}
void DebugArrayMatrix2(const char* title, const uint32_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx, uint64_t RowLen)
{
   cout << title << EndIdx - StartIdx <<  endl;
   for  (uint64_t i = StartIdx; i < EndIdx; i++)
   {    
       cout <<  cpuArray[i] << ", \009";
       if ( (i - StartIdx + 1) % RowLen == 0) cout  << "" <<  endl;
   } 
   cout  << "" << endl;
}
void DebugArrayMatrix4(const char* title, const uint64_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx, uint64_t RowLen)
{
   cout << title << EndIdx - StartIdx <<  endl;
   for  (uint64_t i = StartIdx; i < EndIdx; i++)
   {    
       cout <<  cpuArray[i] << ", \009";
       if ( (i - StartIdx + 1) % RowLen == 0) cout  << "" <<  endl;
   } 
   cout  << "" << endl;
}
void DebugEdgeAddr(const uint64_t* cpuArray, uint64_t StartIdx, uint64_t EndIdx)
{
    uint64_t MaxEdgeCount = 0;
    uint64_t GreaterThan500 = 0;
    cout << "EdgeAddr Output Size = " << EndIdx - StartIdx  <<  endl;
    for  (uint64_t i = StartIdx; i < EndIdx; i++)
    {    
        uint64_t NodeEdgeCount = cpuArray[i + 1] - cpuArray[i];
        if (NodeEdgeCount > MaxEdgeCount) MaxEdgeCount = NodeEdgeCount;
        if (NodeEdgeCount > 500) GreaterThan500++;
        cout << "Index = " << i << ", EdgeNumb = " << NodeEdgeCount << ", EdgeStartPos = " <<  cpuArray[i]  <<  endl;
    }
    cout << "EdgeAddr MaxEdgeCount = " << MaxEdgeCount  <<  endl;
    cout << "EdgeAddr GreaterThan500 = " << GreaterThan500  <<  endl;
}