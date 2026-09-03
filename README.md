# CUTLASS-CuTe
# CuTe GEMM Learning Practice

## 项目简介
本仓库是个人学习 NVIDIA CUTLASS CuTe 编程模型的实践项目。通过从零实现分层分块 GEMM 算子，逐步掌握 CuTe 的核心抽象、硬件原语映射与分块性能优化方法，沉淀底层算子开发的工程经验。

## 核心学习内容
项目围绕 CuTe 的分层设计思想展开，覆盖从硬件原子到分块调度的完整技术栈：
- **底层硬件原子**：MMA Atom（Tensor Core 矩阵乘加原语）、Copy Atom（分级存储间数据搬运原语）
- **分块流水线抽象**：TiledMMA（计算分块与线程映射）、TiledCopy（数据搬运分块与访存调度）
- **核心布局系统**：Layout 抽象、Stride 编排、Coord 坐标体系、多层 Tensor 表示
- **架构级映射**：基于 SM80 架构的 WMMA 指令（M16N8K16）、ldmatrix.x4 数据搬运指令的适配与调用

## 环境依赖
- CUDA Toolkit 12.4 及以上
- CUTLASS 4.6.1（包含 CuTe 头文件库）
- 支持 Compute Capability 8.0+ 的 NVIDIA GPU（Ampere 及以上架构）
- 编译环境：nvcc + C++17 标准

## 编译与运行
1. 确保 CUTLASS 头文件路径已在编译选项中正确配置
2. 编译算子：
bash
nvcc -std=c++17 -O3 practice_gemm.cu -o cute_gemm

3. 运行可执行文件：
bash
./cute_gemm


## 项目结构

.
├── include/          # 辅助头文件与公共定义
├── practice_gemm.cu  # CuTe 版 GEMM 算子主实现
└── README.md
```

## 学习进度
### 已完成
- CuTe Layout / Tensor / Coord 核心基础概念与接口使用
- MMA Atom、Copy Atom 硬件原语的参数配置与调用
- TiledMMA、TiledCopy 分块流水线的搭建与线程映射
- 基础版分层分块 GEMM 算子的完整实现与功能验证

### 进行中 / 后续计划
- 寄存器级双缓冲（Ping-Pong）优化，隐藏数据搬运延迟
- 共享内存 Bank 冲突分析与数据排布优化
- 多尺寸矩阵的通用化模板适配
- 小型算子自动调优脚本实现
- FlashAttention 算子的 CuTe 版本移植

## 说明
本项目为个人学习记录，核心目标是梳理 CuTe 编程模型的设计逻辑与算子开发方法论，性能以学习验证为主，暂未做极致工程化调优。
