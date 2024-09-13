# NVIDIA MPS 工程实践笔记

**官方文档**：https://docs.nvidia.com/deploy/mps/index.html

**概述**：
在支持 Hyper-Q 架构的 GPU 上，MPS可以对多进程的推理任务进行空分复用，提高GPU利用率及SM利用率。<br>
MPS 允许多个 stream 或者多个 CPU 进程同时对 GPU launch kernel， 并把他们结合成单一程序的上下文在 GPU 上运行。<br>
MSP Server 会通过一个 CUDA Context 管理 GPU 资源，接受多个 MPS client 的请求，聚合并进行并行计算（越过硬件时间片的限制）<br>
假如同时又三个 gpu 利用率都不高的任务被发给同一个 gpu，因为时间片的轮转机制，他们要串行的被执行。使用 MPS 进行指令的聚合（空分复用），则可以并行执行者三个任务：<br>
<img width="411" alt="CPU Processes" src="https://github.com/user-attachments/assets/1a2aef97-77b3-47b0-a210-29a55e762a7e">
<br>
**关闭 MPS， 多进程多任务通过时间分片的调度方式共享 GPU**<br>**开启 MPS， 多进程多任务通过 MPS server 管理的 CUDA context 调度共享 GPU**

**更多优化点**：
在聚合的过程中进行资源的合理配置，给每个 client context 分配合理的 threads。
- [ ] 支持不同进程的 kernel & memcopy 并行。
- [ ] 减少 GPU 侧 CUDA context 的存储空间：每个进程一个 CUDA context -> MPS Server 管理一个 CUDA context，调度子任务。
- [ ] MPS Server 管理一套 GPU 侧调度资源，减少多进程单独申请调度资源的开销。
- [ ] 提高 SM Occupancy： 假如有 threads-per-grid 较低的线程（grid 对应 SM；即占用较多 SM 块，但是每个 SM 并行的 threads 少、不足），MPS 在并行 threads 不变的情况下，自动把该任务的 blocks-per-grid 提高，把 threads-per-block 减低（即使用更少的块，每块使用更多并发）；<br>通过这个手段，自动提高 SM 的 Occupancy，并把省下来的 blocks 分配给其他并行的 client 进程
（MPS 不会影响非 GPU 操作的并发性。）

**MPS 的缺点**：
- MPS 的错误处理 （工程重点）：<br>
假如一个 MPS client 的任务 A 执行时发生 fatal，这个 MPS Server 管理的所有子任务都会 fatal，并且返回任务 A 的错误信息。<br>
这体现出 MPS 的错误处理很草率，容易跨请求带来错误的蔓延，即使这本来是一个正常的请求也会被带错，并且无法做合理的区分和重试！<br>
发生错误后，MPS Server 的状态会从“ACTIVE” 变为 “FAULT”，不再处理新请求。直到所有子任务退出，重启 CUDA Context，状态重新变成“ACTIVE”<br>
基于错误处理最佳实践：一个 client 任务 X，只发给一个 GPU 的 MPS Server 去执行。如果一个 client 的任务发给多个 GPU，即使只是其中一个 GPU 上 MPS Server 任务发生 fatal，也会会带坏该 GPU 的 MPS Server 及任务 X 所有发送任务的 GPU 的 MPS Server，可能造成服务的链式雪崩。<br>

- MPS 要求优雅退出：<br>
如果不优雅退出，某一个 client 直接 CTRL-C 或信号退出，会导致意外发生，如故障或挂起。<br>
优雅退出：某个 client 退出前<br>
代码：通过 stream 调用 `cudaDeviceSynchronize` 或 `cudaStreamSynchronize`，把客户端置为空闲状态<br>
命令：`terminate_client <服务器 PID> < 客户端 PID>` 指示 MPS 服务器终止 MPS 客户端进程的 CUDA 上下文。此时该 client 的上下文状态变成 INACTIVE，但不会立马退出（可能还有显存清理/状态重置逻辑需要运行）。<br>

**MPS 细化管理**：
- 计算资源管理（`nvidia-cuda-mps-control`）：<br>
可以通过划分每个 MPS client 可以使用的 threads 占总 thread 的百分比来划分资源(`via set_default_active_thread_percentage || set_active_thread_percentage`)<br>
更严格的话，可以通过 SM Partitioning 来细化管理每个 client 可以使用的 SM 数量，合理限制及分配资源<br>

- 显存资源管理：<br>
MPS 可以在 Server 端对 client 限制可用显存大小，如果超限 cuda 内存分配返回内存不足错误。（`via set_default_device_pinned_mem_limit || set_device_pinned_mem_limit control`）<br>
MPS 的客户端也可以自行通过 `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 环境变量约束占用的显存。<br>
开启 MPS 时支持 nsys 及 nvprof 的性能 pref<br>

- 优先级管理：<br>
优先级级别值为 0（正常）和 1（低于正常）<br>
代码层面：通过 API  `cudaStreamCreateWithPriority()` 来控制其内核的 GPU 优先级级别。<br>
MPS Server 配置层面：命令 `set_default_client_priority < 优先级级别>`  将给定客户端的 stream 优先级映射到内部 CUDA 优先级。<br>
环境变量层面：启动前设置 `CUDA_MPS_CLIENT_PRIORITY` 环境变量来配置。<br>
