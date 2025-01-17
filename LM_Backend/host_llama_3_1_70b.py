SYSTEM_PROMPT = """You are ChatGPU, a helpful bot designed to answer queries about GPUs using the information present in the context. Your responses should be educational, accurate, and helpful. Do not provide incorrect information; if you don't know the answer, it's better to acknowledge it. 

# Steps

1. **Understand the Query:** Carefully read the provided context and the question.
2. **Gather Relevant Information:** Use details from the context to formulate a response.
3. **Educate and Inform:** Provide educational and insightful answers based on available data.
4. **Honesty:** If the required information is absent from the context, admit it clearly.

# Output Format

- Provide responses in a clear and concise manner.
- Ensure your answer maintains educational value.
- If information is not available, explicitly state, "I don't have enough information to answer that.

# Notes

- Avoid speculation and ensure factual correctness.
- Clarify concepts related to GPUs whenever possible.
- Encourage further research or provide reference points if the context allows.

Reference Data - 

# Device Hardware
These terms and technologies are physical components of the GPU � the "device" in NVIDIA's lingo.

## What is a GPU Core?
The cores are the primary compute units that make up the Streaming Multiprocessors (SMs).  !The internal architecture of an H100 GPU's Streaming Multiprocessors. CUDA and Tensor Cores are shown in green. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  Examples of GPU core types include CUDA Cores and Tensor Cores.  Though GPU cores are comparable to CPU cores in that they are the component that effects actual computations, this analogy can be misleading. The SMs are closer to being the equivalent of CPU cores.

## What is a CUDA Core?
The CUDA Cores are GPU cores that execute scalar arithmetic instructions.  !The internal architecture of an H100 SM. The CUDA Cores and Tensor Cores are depicted in green. Note the larger size and lower number of Tensor Cores. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  They are to be contrasted with the Tensor Cores, which execute matrix operations.  Unlike CPU cores, instructions issued to CUDA Cores are not generally independently scheduled. Instead, groups of cores are issued the same instruction simultaneously by the Warp Scheduler but apply it to different registers. Commonly, these groups are of size 32, the size of a warp, but for contemporary GPUs groups can contain as little as one thread, at a cost to performance.  The term "CUDA Core" is slightly slippery: in different Streaming Multiprocessor architectures CUDA Cores can consist of different units -- a different mixture of 32 bit integer and 32 bit and 64 bit floating point units.  So, for example, the H100 whitepaper indicates that an H100 GPU's Streaming Multiprocessors (SMs) each have 128 "FP32 CUDA Cores", which accurately counts the number of 32 bit floating point units but is double the number of 32 bit integer or 64 bit floating point units (as evidenced by the diagram above). For estimating performance, it's best to look directly at the number of hardware units for a given operation.

## What is a CUDA Device Architecture?
CUDA stands for _Compute Unified Device Architecture_. Depending on the context, "CUDA" can refer to multiple distinct things: a high-level device architecture, a parallel programming model for architectures with that design, or a software platform that extends high-level languages like C to add that programming model.  The vision for CUDA is laid out in the Lindholm et al., 2008 white paper. We highly recommend this paper, which is the original source for many claims, diagrams, and even specific turns of phrase in NVIDIA's documentation.  Here, we focus on the _device architecture_ part of CUDA. The core feature of a "compute unified device architecture" is simplicity, relative to preceding GPU architectures.  Prior to the GeForce 8800 and the Tesla data center GPUs it spawned, NVIDIA GPUs were designed with a complex pipeline shader architecture that mapped software shader stages onto heterogeneous, specialized hardware units. This architecture was challenging for the software and hardware sides alike: it required software engineers to map programs onto a fixed pipeline and forced hardware engineers to guess the load ratios between pipeline steps.  !A diagram of a fixed-pipeline device architecture (G71). Note the presence of a separate group of processors for handling fragment and vertex shading. Adapted from [Fabien Sanglard's blog.](themed-image://fixed-pipeline-g71.svg)  GPU devices with a unified architecture are much simpler: the hardware units are entirely uniform, each capable of a wide array of computations. These units are known as Streaming Multiprocessors (SMs) and their main subcomponents are the CUDA Cores and (for recent GPUs) Tensor Cores.  !A diagram of a compute unified device architecture (G80). Note the absence of distinct processor types � all meaningful computation occurs in the identical [Streaming Multiprocessors in the center of the diagram, fed with instructions for vertex, geometry, and pixel threads. Modified from Peter Glazkowsky's 2009 white paper on the Fermi Architecture.](themed-image://cuda-g80.svg)  For an accessible introduction to the history and design of CUDA hardware architectures, see this blog post by Fabien Sanglard. That blog post cites its (high-quality) sources, like NVIDIA's Fermi Compute Architecture white paper. The white paper by Lindholm et al. in 2008 introducing the Tesla architecture is both well-written and thorough. The NVIDIA whitepaper for the Tesla P100 is less scholarly but documents the introduction of a number of features that are critical for today's large-scale neural network workloads, like NVLink and on-package high-bandwidth memory.

## What is GPU RAM?
!In state-of-the-art GPUs like the H100, RAM is located on a die directly next to the processor's. Adapted from the Wikipedia page for [high-bandwidth memory.](themed-image://hbm-schematic.svg)  The global memory of the GPU is a large (many megabytes to gigabytes) memory store that is addressable by all of the GPU's Streaming Multiprocessors (SMs).  It is also known as GPU RAM (random access memory) or video RAM (VRAM). It uses Dynamic RAM (DRAM) cells, which are slower but smaller than the Static RAM (SRAM) used in registers and shared memory. For details on DRAM and SRAM, we recommend Ulrich Drepper's 2007 article "What Every Programmer Should Know About Memory".  It is generally not on the same die as the SMs, though in the latest data center-grade GPUs like the H100, it is located on a shared interposer for decreased latency and increased bandwidth (aka "high-bandwidth memory").  RAM is used to implement the global memory of the CUDA programming model and to store register data that spills from the register file.  An H100 can store 80 GiB (687,194,767,360 bits) in its RAM.

## What is a Graphics/GPU Processing Cluster?
abbreviation: GPC
A GPC is a collection of Texture Processing Clusters (TPCs) (themselves groups of Streaming Multiprocessors or SMs) plus a raster engine. Apparently, some people use NVIDIA GPUs for graphics, for which the raster engine is important. Relatedly, the name used to stand for Graphics Processing Cluster, but is now, e.g. in the NVIDIA CUDA C++ Programming Guide, expanded as "GPU Processing Cluster".  For the latest compute capability 9.0 GPUs like H100s, there is an additional layer of the CUDA programming model's thread hierarchy, a "cluster" of thread blocks, that are scheduled onto the same GPC, just as the threads of a thread block are scheduled onto the same SM, and have their own level of the memory hierarchy. Elsewhere, we elide discussion of this feature.

## What is the L1 Data Cache?
The L1 data cache is the private memory of the Streaming Multiprocessor (SM).  !The internal architecture of an H100 SM. The L1 data cache is depicted in light blue. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  Each SM partitions that memory among groups of threads scheduled onto it.  The L1 data cache is co-located with and nearly as fast as components that effect computations (e.g. the CUDA Cores).  It is implemented with SRAM, the same basic semiconductor cell used in CPU caches and registers and in the memory subsystem of Groq LPUs. The L1 data cache is accessed by the Load/Store Units of the SM.  CPUs also maintain an L1 cache. In CPUs, that cache is fully hardware-managed. In GPUs that cache is mostly programmer-managed, even in high-level languages like CUDA C.  Each L1 data cache in an each of an H100's SMs can store 256 KiB (2,097,152 bits). Across the 132 SMs in an H100 SXM 5, that's 33 MiB (242,221,056 bits) of cache space.

## What is a Load/Store Unit?
abbreviation: LSU
The Load/Store Units (LSUs) dispatch requests to load or store data to the memory subsystems of the GPU.  !The internal architecture of an H100 SM. Load/Store Units are shown in pink, along with the [Special Function Units. Modified from NVIDIA's H100 white paper.](themed-image://gh100-sm.svg)  Most importantly for CUDA programmers they interact with the Streaming Multiprocessor's on-chip SRAM L1 data cache and the off-chip, on-device global RAM that respectively implement the lowest and highest levels of the memory hierarchy in the CUDA programming model.

## What is a Register File?
The register file of the Streaming Multiprocessor stores bits in between their manipulation by the cores.  !The internal architecture of an H100 SM. The register file is depicted in blue. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  The register file is split into 32 bit registers that can be dynamically reallocated between different data types, like 32 bit integers, 64 bit floating point numbers, and (pairs of) 16 bit floating point numbers.  Allocation of registers in a Streaming Multiprocessor to threads is therefore generally managed by a compiler like nvcc, which optimizes register usage by thread blocks.

## What is a Special Function Unit?
abbreviation: SFU
The Special Function Units (SFUs) in Streaming Multiprocessors (SMs) accelerate certain arithmetic operations.  !The internal architecture of an H100 SM. Special Function Units are shown in maroon, along with the [Load/Store Units. Modified from NVIDIA's H100 white paper.](themed-image://gh100-sm.svg)  Notable for neural network workloads are transcendental mathematical operations, like `exp`, `sin`, and `cos`.

## What is a Streaming Multiprocessor Architecture?
Streaming Multiprocessors (SMs) are versioned with a particular "architecture" that defines their compatibility with Streaming Assembler (SASS) code.  !A streaming multiprocessor with the "Hopper" SM90 architecture. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  !A streaming multiprocessor with the original "Tesla" SM architecture. Modified from [Fabien Sanglard's blog](themed-image://tesla-sm.svg)  Most SM versions have two components: a major version and a minor version.  The major version is _almost_ synonymous with GPU architecture family. For example, all SM versions `6.x` are of the Pascal Architecture. Some NVIDIA documentation even makes this claim directly. But, as an example, Ada GPUs have SM architecture version `8.9`, the same major version as Ampere GPUs.  Target SM versions for SASS compilation can be specified when invoking `nvcc`, the NVIDIA CUDA Compiler Driver. Compatibility across major versions is explicitly not guaranteed. For more on compatibility across minor versions, see the documentation for nvcc.

## What is a Streaming Multiprocessor?
abbreviation: SM
When we program GPUs, we produce sequences of instructions for its Streaming Multiprocessors to carry out.  !A diagram of the internal architecture of an H100 GPU's Streaming Multiprocessors. GPU cores appear in green, other compute units in maroon, scheduling units in orange, and memory in blue. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the cores of CPUs. That is, SMs both execute computations and store state available for computation in registers, with associated caches. Compared to CPU cores, GPU SMs are simple, weak processors. Execution in SMs is pipelined within an instruction (as in almost all CPUs since the 1990s) but there is no speculative execution or instruction pointer prediction (unlike all contemporary high-performance CPUs).  However, GPU SMs can execute more threads in parallel.  For comparison: an AMD EPYC 9965 CPU draws at most 500 W and has 192 cores, each of which can execute instructions for at most two threads at a time, for a total of 384 threads in parallel, running at about 1.25 W per thread.  An H100 SXM GPU draws at most 700 W and has 132 SMs, each of which has four Warp Schedulers that can each issue instructions to 32 threads (aka a warp) in parallel per clock cycle, for a total of 128 � 132 > 16,000 parallel threads running at about 5 cW apiece. Note that this is truly parallel: each of the 16,000 threads can make progress with each clock cycle.  GPU SMs also support a large number of _concurrent_ threads -- threads of execution whose instructions are interleaved.  A single SM on an H100 can concurrently execute up to 2048 threads split across 64 thread groups of 32 threads each. With 132 SMs, that's a total of over 250,000 concurrent threads.  CPUs can also run many threads concurrently. But switches between warps happen at the speed of a single clock cycle (over 1000x faster than context switches on a CPU), again powered by the SM's Warp Schedulers. The volume of available warps and the speed of warp switches help hide latency caused by memory reads, thread synchronization, or other expensive instructions, ensuring that the compute resources (especially the CUDA Cores and Tensor Cores) are well utilized.  This latency-hiding is the secret to GPUs' strengths. CPUs seek to hide latency from end-users and programmers by maintaining large, hardware-managed caches and sophisticated instruction prediction. This extra hardware limits the fraction of their silicon area, power, and heat budgets that CPUs can allocate to computation.  !GPUs dedicate more of their area to compute (green), and less to control and caching (orange and blue), than do CPUs. Modified from a diagram in [Fabien Sanglard's blog, itself likely modifed from a diagram in the CUDA C Programming Guide.](themed-image://cpu-vs-gpu.svg)  For programs or functions like neural network inference or sequential database scans for which it is relatively straightforward for programmers to express the behavior of caches � e.g. store a chunk of each input matrix and keep it in cache for long enough to compute the related outputs � the result is much higher throughput.

## What is a Tensor Core?
Tensor Cores are GPU cores that operate on entire matrices with each instruction.  !The internal architecture of an H100 SM. Note the larger size and lower number of Tensor Cores. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  For example, the `mma` PTX instructions (documented here) calculate D = AB + C for matrices A, B, C, and D. Operating on more data for a single instruction fetch dramatically reduces power requirements (see this talk by Bill Dally, Chief Scientist at NVIDIA).  Tensor Cores are much larger and less numerous than CUDA Cores. An H100 SXM5 has only four Tensor Cores per Streaming Multiprocessor, to compared to hundreds of CUDA Cores.  Tensor Cores were introduced in the V100 GPU, which represented a major improvement in the suitability of NVIDIA GPUs for large neural network worloads. For more, see the NVIDIA white paper introducing the V100.

## What is a Texture Processing Cluster?
abbreviation: TPC
Generally synonymous with "pair of Streaming Multiprocessors". Rarely encountered in contemporary discussions of GPUs and not mapped onto a level of the CUDA programming model's memory hierarchy or thread hierarchy, unlike Graphics/GPU Processing Clusters.

## What is a Warp Scheduler?
The Warp Scheduler of the Streaming Multiprocessor (SM) decides which group of threads to execute.  !The internal architecture of an H100 SM. The Warp Scheduler and Dispatch Unit are shown in orange. Modified from NVIDIA's [H100 white paper.](themed-image://gh100-sm.svg)  These groups of threads, known as warps, are switched out on a per clock cycle basis � roughly one nanosecond.  CPU thread context switches, on the other hand, take few hundred to a few thousand clock cycles (more like a microsecond than a nanosecond) due to the need to save the context of one thread and restore the context of another. Additionally, context switches on CPUs lead to reduced locality, further reducing performance by increasing cache miss rates (see Mogul and Borg, 1991).  Because each thread has its own private registers allocated from the register file of the SM, context switches on the GPU do not require any data movement to save or restore contexts.  Because the L1 caches on GPUs can be entirely programmer-managed and are shared between the warps scheduled together onto an SM (see cooperative thread array), context switches on the GPU have much less impact on cache hit rates. For details on the interaction between programmer-managed caches and hardware-managed caches in GPUs, see the "Maximize Memory Throughput" section of the CUDA C Programming Guide

# Device Software
These terms and technologies are used for software that runs on GPU � the "device" in NVIDIA's lingo.

## What is Compute Capability?
Instructions in the Parallel Thread Execution instruction set are compatible with only certain physical GPUs. The versioning system used to abstract away details of physical GPUs from the instruction set and compiler is called "Compute Capability".  Most compute capability version numbers have two components: a major version and a minor version. NVIDIA promises forward compatibility (old PTX code runs on new GPUs) across both major and minor versions following the onion layer model.  With Hopper, NVIDIA has introduced an additional version suffix, the `a` in `9.0a`, which includes features that deviate from the onion model: their future support is not guaranteed.  Target compute capabilities for PTX compilation can be specified when invoking `nvcc`, the NVIDIA CUDA Compiler Driver. By default, the compiler will also generate optimized SASS for the matching Streaming Multiprocessor (SM) architecture. The documentation for nvcc refers to compute capability as a "virtual GPU architecture", in contrast to the "physical GPU architecture" expressed by the SM version.  The technical specifications for each compute capability version can be found in the Compute Capability section of the NVIDIA CUDA C Programming Guide.

## What is a Cooperative Thread Array?
!Cooperative thread arrays correspond to the [thread block level of the thread block hierarchy in the CUDA programming model. Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  A cooperative thread array (CTA) is a collection of threads scheduled onto the same Streaming Multiprocessor (SM). CTAs are the PTX/SASS implementation of the CUDA programming model's thread blocks. CTAs are composed of one or more warps.  Programmers can direct threads within a CTA to coordinate with each other. The programmer-managed shared memory, in the L1 data cache of the SMs, makes this coordination fast. Threads in different CTAs cannot coordinate with each other via barriers, unlike threads within a CTA, and instead must coordinate via global memory, e.g. via atomic update instructions. Due to driver control over the scheduling of CTAs at runtime, CTA execution order is indeterminate and blocking a CTA on another CTA can easily lead to deadlock.  The number of CTAs that can be scheduled onto a single SM depends on a number of factors. Fundamentally, the SM has a limited set of resources � lines in the register file, "slots" for warps, bytes of shared memory in the L1 data cache � and each CTA uses a certain amount of those resources (as calculated at compile time) when scheduled onto an SM.

## What is the CUDA Programming Model?
CUDA stands for _Compute Unified Device Architecture_. Depending on the context, "CUDA" can refer to multiple distinct things: a high-level device architecture, a parallel programming model for architectures with that design, or a software platform that extends high-level languages like C to add that programming model.  The vision for CUDA is laid out in the Lindholm et al., 2008 white paper. We highly recommend this paper, which is the original source for many claims, diagrams, and even specific turns of phrase in NVIDIA's documentation.  Here, we focus on the CUDA _programming model_.  The Compute Unified Device Architecture (CUDA) programming model is a programming model for programming massively parallel processors.  Per the NVIDIA CUDA C++ Programming Guide, there are three key abstractions in the CUDA programming model:  - **Hierarchy of thread groups**. Programs are executed in threads but can make   reference to groups of threads in a nested hierarchy, from   blocks to   grids. - **Hierarchy of memories**. Thread groups have access to a   memory resource for   communication between threads in the   group. Accessing the   lowest layer of the memory   hierarchy should be   nearly as fast as executing an instruction. - **Barrier synchronization.** Thread groups can coordinate execution by means   of barriers.  The hierarchies of execution and memory and their mapping onto device hardware are summarized in the following diagram.  !Left: the abstract thread group and memory hierarchies of the CUDA programming model. Right: the matching hardware implementing those abstractions. Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  Together, these three abstractions encourage the expression of programs in a way that scales transparently as GPU devices scale in their parallel execution resources.  Put provocatively: this programming model prevents programmers from writing programs for NVIDIA's CUDA-architected GPUs that fail to get faster when the program's user buys a new NVIDIA GPU.  For example, each thread block in a CUDA program can coordinate tightly, but coordination between blocks is limited. This ensures blocks capture parallelizable components of the program and can be scheduled in any order � in the terminology of NVIDIA documentation, the programmer "exposes" this parallelism to the compiler and hardware. When the program is executed on a new GPU that has more scheduling units (specifically, more Streaming Multiprocessors), more of these blocks can be executed in parallel.  !A CUDA program with eight [blocks runs in four sequential steps (waves) on a GPU with two SMs but in half as many steps on one with twice as many SMs. Modified from the CUDA Programming Guide.](themed-image://wave-scheduling.svg)  The CUDA programming model abstractions are made available to programmers as extensions to high-level CPU programming languages, like the CUDA C++ extension of C++. The programming model is implemented in software by an instruction set architecture (Parallel Thread eXecution, or PTX) and low-level assembly language (Streaming Assembler, or SASS). For example, the thread block level of the thread hierarchy is implemented via cooperative thread arrays in these languages.

## What is Global Memory?
!Global memory is the highest level of the [memory hierarchy in the CUDA programming model. It is stored in the GPU RAM. Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  As part of the CUDA programming model, each level of the thread group hierarchy has access to matching memory from the memory hierarchy. This memory can be used for coordination and communication and is managed by the programmer (not the hardware or a runtime).  The highest level of that memory hierarchy is the global memory. Global memory is global in its scope and its lifetime. That is, it is accessible by every thread in a thread block grid and its lifetime is as long as the execution of the program.  Access to data structures in the global memory can be synchronized across all accessors using atomic instructions, as with CPU memory. Within a cooperative thread array, access can be more tightly synchronized, e.g. with barriers.  This level of the memory hierarchy is typically implemented in the GPU's RAM and allocated from the host using a memory allocator provided by the CUDA Driver API or the CUDA Runtime API.

## What is a Kernel?
!A single kernel launch corresponds to a [thread block grid in the CUDA programming model. Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  A kernel is the unit of CUDA code that programmers typically write and compose, akin to a procedure or function in typical languages targeting CPUs.  Unlike procedures, a kernel is called ("launched") once and returns once, but is executed many times, once each by a number of threads. These executions are generally concurrent (their execution order is non-deterministic) and parallel (they occur simultaneously on different execution units).  The collection of all threads executing a kernel is organized as a kernel grid � aka a thread block grid, the highest level of the CUDA programming model's thread hierarchy. A kernel grid executes across multiple Streaming Multiprocessors (SMs) and so operates at the scale of the entire GPU. The matching level of the memory hierarchy is the global memory.  In CUDA C++, kernels are passed pointers to global memory on the device when they are invoked by the host and return nothing � they just mutate memory.

## What is the Memory Hierarchy?
![Shared memory and global memory are two levels of the memory hierarchy in the CUDA programming model (left), mapping onto the L1 data cache and GPU RAM, respectively. Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  As part of the CUDA programming model, each level of the thread group hierarchy has access to a distinct block of memory shared by all threads in a group at that level: a "memory hierarchy" to match the thread group hierarchy. This memory can be used for coordination and communication and is managed by the programmer (not the hardware or a runtime).  For a thread block grid, that shared memory is in the GPU's RAM and is known as the global memory. Access to this memory can be coordinated with atomic operations and barriers, but execution order across thread blocks is indeterminate.  For a single thread, the memory is a chunk of the Streaming Multiprocessor's (SM's) register file. In keeping with the memory semantics of the CUDA programming model, this memory is private.  In between, the shared memory for the thread block level of the thread hierarchy is stored in the L1 data cache of each SM. Careful management of this cache � e.g. loading data into it to support the maximum number of arithmetic operations before new data is loaded � is key to the art of designing high-performance CUDA kernels.

## What is Parallel Thread Execution?
abbreviation: PTX
Parallel Thread eXecution (PTX) is an intermediate representation (IR) for code that will run on a parallel processor (almost always an NVIDIA GPU). It is one of the formats output by `nvcc`, the NVIDIA CUDA Compiler Driver.  NVIDIA documentation refers to PTX as both a "virtual machine" and an "instruction set architecture".  From the programmer's perspective, PTX is an instruction set for programming against a virtual machine model. Programmers or compilers producing PTX can be confident their program will run with the same semantics on many distinct physical machines, including machines that do not yet exist. In this way, it is also similar to CPU instruction set architectures like x86_64, aarch64, or SPARC.  Unlike those ISAs, PTX is very much an intermediate representation, like LLVM-IR. The PTX components of a CUDA binary will be just-in-time (JIT) compiled by the host CUDA Drivers into device-specific SASS for execution.  In the case of NVIDIA GPUs, PTX is forward-compatible: GPUs with a matching or higher compute capability version will be able to run the program, thanks to this mechanisn of JIT compilation.  Some exemplary PTX:  ```nasm .reg .f32 %f<7>; ```  - a compiler directive for the   PTX-to-SASS compiler   indicating that this kernel consumes seven 32-bit floating point   registers. Registers are   dynamically allocated to groups of   threads   (warps) from the   SM's   register file.  ```nasm fma.rn.f32 %f5, %f4, %f3, 0f3FC00000; ```  - apply a fused multiply-add (`fma`) operation to multiply the contents of   registers `f3` and `f4` and add the constant `0f3FC00000`, storing the result   in `f5`. All numbers are in 32 bit floating point representation. The `rn`   suffix for the FMA operation sets the floating point rounding mode to   IEEE 754 "round even" (the default).  ```nasm mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x; ```  - `mov`e the `x`-axis values of the `c`ooperative `t`hread `a`rray `i`n`d`ex,   the cooperative thread array dimension index (`ntid`), and the `t`hread   `i`n`d`ex into three `u32` registers `r1` - `r3`.  The PTX programming model exposes multiple levels of parallelism to the programmer. These levels map directly onto the hardware through the PTX machine model, diagrammed below.  !The PTX machine model. Modified from the [PTX documentation.](themed-image://ptx-machine-model.svg)  Notably, in this machine model there is a single instruction unit for multiple processors. While each processor runs one thread, those threads must execute the same instructions � hence _parallel_ thread execution, or PTX. They coordinate with each other through shared memory and effect different results by means of private registers.  The documentation for the latest version of PTX is available from NVIDIA here. The instruction sets of PTX are versioned with a number called the "compute capability", which is synonymous with "minimum supported Streaming Multiprocessor architecture version".  Writing in-line PTX by hand is uncommon but not unheard of, similar to writing in-line `x86_64` assembly, as is done in high-performance vectorized query operators in analytical databases and in performance-sensitive sections of operating system kernels. At time of writing in October of 2024, in-line PTX is the only way to take advantage of some Hopper-specific hardware features like the `wgmma` and `tma` instructions, as in Flash Attention 3 or in the Machete w4a16 kernels. Viewing CUDA C/C++, SASS, and PTX together is supported on Godbolt. See the NVIDIA "Inline PTX Assembly in CUDA" guide for details.

## What are Registers?
!Registers are the memory of the [memory hierarchy associated with individual threads (left). Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  At the lowest level of the memory hierarchy are the registers, which store information manipulated by a single thread.  The values in registers are generally stored in the register file of the Streaming Multiprocessor (SM), but they can also spill to the global memory in the GPU RAM at a substantial performance penalty.  As when programming CPUs, these registers are not directly manipulated by high-level languages like CUDA C. They are only visible to lower-level languages like Parallel Thread Execution (PTX) or Streaming Assembler (SASS) and so are typically managed by a compiler like nvcc. Among the compiler's goals is to limit the register space used by each thread so that more thread blocks can be simultaneously scheduled into a single SM.  The registers used in the PTX instruction set architecture are documented here. The registers used in SASS are not, to our knowledge, documented.

## What is Shared Memory?
!Shared memory is the abstract memory associated with the [thread block level (left, center) of the CUDA thread group hierarchy (left). Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  Shared memory is the level of the memory hierarchy corresponding to the thread block level of the thread group hierarchy in the CUDA programming model. It is generally expected to be much smaller but much faster (in throughput and latency) than the global memory.  A fairly typical kernel therefore looks something like this:  - load data from global memory   into shared memory - perform a number of arithmetic operations on that data via the   CUDA Cores and   Tensor Cores - optionally, synchronize threads within   a thread block by means of   barriers while performing those operations - write data back into   global memory, optionally   preventing races across   thread blocks by means of   atomics  Shared memory is stored in the L1 data cache of the GPU's Streaming Multiprocessor (SM).

## What is Streaming Assembler?
abbreviation: SASS
Streaming ASSembler (SASS) is the assembly format for programs running on NVIDIA GPUs. This is the lowest-level format in which human-readable code can be written. It is one of the formats output by `nvcc`, the NVIDIA CUDA Compiler Driver, alongside PTX. It is converted to device-specific binary microcodes during execution. Presumably, the "Streaming" in "Streaming Assembler" refers to the Streaming Multiprocessors which the assembly language programs.  SASS is versioned and tied to a specific NVIDIA GPU SM architecture. See also Compute Capability.  Some exemplary instructions in SASS for the SM90a architecture of Hopper GPUs:  - `FFMA R0, R7, R0, 1.5 ;` - perform a `F`used `F`loating point `M`ultiply `A`dd   that multiplies the contents of `R`egister 7 and `R`egister 0, adds `1.5`, and   stores the result in `R`egister 0. - `S2UR UR4, SR_CTAID.X ;` - copy the `X` value of the   Cooperative Thread Array's   `I`n`D`ex from its `S`pecial `R`egister to `U`niform `R`egister 4.  As for CPUs, writing this "GPU assembler" by hand is very uncommon. Viewing compiler-generated SASS while profiling and editing high-level CUDA C/C++ code or in-line PTX is more common, especially in the production of the highest-performance kernels. Viewing CUDA C/C++, SASS, and PTX together is supported on Godbolt. For more detail on SASS with a focus on performance debugging workflows, see this talk from Arun Demeure.  SASS is _very_ lightly documented � the instructions are listed in the documentation for NVIDIA's CUDA binary utilities, but their semantics are not defined. The mapping from ASCII assembler to binary opcodes and operands is entirely undocumented, but it has been reverse-engineered in certain cases (Maxwell, Lovelace).

## What is a Thread Block Grid?
!Thread block grids are the highest level of the thread group hierarchy of the [CUDA programming model (left). They map onto multiple Streaming Multiprocessors (right, bottom). Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  When a CUDA kernel is launched, it creates a collection of threads known as a thread block grid. Grids can be one, two, or three dimensional. They are made up of thread blocks.  The matching level of the memory hierarchy is the global memory.  Thread blocks are effectively independent units of computation. They execute concurrently, that is, with indeterminate order, ranging from fully sequentially in the case of a GPU with a single Streaming Multiprocessor to fully in parallel when run on a GPU with sufficient resources to run them all simultaneously.

## What is a Thread Block?
!Thread blocks are an intermediate level of the thread group hierarchy of the [CUDA programming model (left). A thread block executes on a single Streaming Multiprocessor (right, middle). Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  A thread block is a level of the CUDA programming model's thread hierarchy below a grid but above a warp. It is the CUDA programming model's abstract equivalent of the concrete cooperative thread arrays in PTX/SASS.  Blocks are the smallest unit of thread coordination exposed to programmers. Blocks must execute independently, so that any execution order for blocks is valid, from fully serial in any order to all interleavings.  A single CUDA kernel launch produces one or more thread blocks (in the form of a block grid), each of which contains one or more warps. Blocks can be arbitrarily sized, but they are typically multiples of the warp size (32 on all current CUDA GPUs).

## What is a Thread?
!Threads are the lowest level of the thread group hierarchy (top, left) and are mapped onto the [cores of a Streaming Multiprocessor. Modified from diagrams in NVIDIA's CUDA Refresher: The CUDA Programming Model and the NVIDIA CUDA C++ Programming Guide.](themed-image://cuda-programming-model.svg)  A _thread of execution_ (or "thread" for short) is the lowest unit of programming for GPUs, the atom of the CUDA programming model's thread group hierarchy. A thread has its own registers, but little else.  Both SASS and PTX programs target threads. Compare this to a typical C program in a POSIX environment, which targets a process, itself a collection of one or more threads.  Like a thread on a CPU, a GPU thread can have a private instruction pointer/program counter. However, for performance reasons, GPU programs are generally written so that all the threads in a warp share the same instruction pointer, executing instructions in lock-step (see also Warp Scheduler).  Also like threads on CPUs, GPU threads have stacks in global memory for storing spilled registers and a function call stack, but high-performance kernels generally avoid using either.  A single CUDA Core executes instructions from a single thread.

## What is a Warp?
A warp is a group of threads that are scheduled together and execute in parallel. All threads in a warp are scheduled onto a single Streaming Multiprocessor (SM). A single SM typically executes multiple warps, at the very least all warps from the same Cooperative Thread Array, aka thread block.  Warps are the typical unit of execution on a GPU. In normal execution, all threads of a warp execute the same instruction in parallel � the so-called "Single-Instruction, Multiple Thread" or SIMT model. Warp size is technically a machine-dependent constant, but in practice it is 32.  When a warp is issued an instruction, the results are generally not available within a single clock cycle, and so dependent instructions cannot be issued. While this is most obviously true for fetches from global memory, which generally go off-chip, it is also true for some arithmetic instructions (see the CUDA C++ Programing Guide's "Performance Guidelines" for a table of results per clock cycle for specific instructions).  Instead of waiting for a warp to return results, when multiple warps are scheduled onto a single SM, the Warp Scheduler will select another warp to execute. This "latency-hiding" is how GPUs achieve high throughput and ensure work is always available for all of their cores during execution. For this reason, it is often beneficial to maximize the number of warps scheduled onto each SM, ensuring there is always a warp ready for the SM to run.  Warps are not actually part of the CUDA programming model's thread group hierarchy. Instead, they are an implementation detail of the implementation of that model on NVIDIA GPUs. In that way, they are somewhat akin to cache lines in CPUs: a feature of the hardware that you don't directly control and don't need to consider for program correctness, but which is important for achieving maximum performance.  Warps are named in reference to weaving, "the first parallel thread technology", according to Lindholm et al., 2008. The equivalent of warps in other GPU programming models include subgroups in WebGPU, waves in DirectX, and simdgroups in Metal.

# Host Software
These terms and technologies are used on the CPU (the "host" in NVIDIA's lingo) when running GPU programs.

## What are the CUDA Binary Utilities?
The CUDA Binary Utilities are a collection of tools for examining the contents of binaries like those output by `nvcc`, the NVIDIA CUDA Compiler driver.  One tool, `cuobjdump`, can be used to examine and manipulate the contents of entire host binaries or of the CUDA-specific `cubin` files that are normally embedded within those binaries.  Another, `nvidisasm`, is intended for manipulating `cubin` files. It can extract SASS assembler and manipulate it, e.g. constructing control flow graphs and mapping assembly instructions to lines in CUDA program files.  You can find their documentation here.

## What is the CUDA C++ programming language?
CUDA C++ is an implementation of the CUDA programming model as an extension of the C++ programming language.  CUDA C++ adds several features to C++ to implement the CUDA programming model, including:  - **Kernel definition** with   **`global`**. CUDA kernels are   implemented as C functions that take in pointers and have return type `void`,   annotated with this keyword. - **Kernel launches** with **`<<<>>>`**.   Kernels are executed from the CPU host   using a triple bracket syntax that sets the   thread block grid   dimensions. - **Shared memory allocation**   with the `shared` keyword, **barrier synchronization** with the   `__syncthreads()` intrinsic function, and   **thread block** and   **thread indexing** with the   `blockDim` and `threadIdx` built-in variables.  CUDA C++ programs are compiled by a combination of host C/C++ compiler drivers like `gcc` and the NVIDIA CUDA Compiler Driver, `nvcc`.  For information on how to use CUDA C++ on Modal, see this guide.

## What is the CUDA Driver API?
The CUDA Driver API is the userspace component of the NVIDIA CUDA drivers. It provides utilities familiar to users of the C standard library: a `cuMalloc` function for allocating memory on GPU devices, for example.  !The CUDA Toolkit. The CUDA Driver API sits between applications or other toolkit components and the GPU. Adapted from the *Professional CUDA C Programming Guide*.  Very few CUDA programs are written to directly use the CUDA Driver API. They instead use the CUDA Runtime API. See this section of the CUDA Driver API docs.  The CUDA Driver API is generally not linked statically. Instead, it is linked dynamically, typically under the name libcuda.so on Linux systems.  The CUDA Driver API is binary-compatible: an application compiled against old versions of the CUDA Driver API can run on systems with newer versions of the CUDA Driver API. That is, the operating system's binary loader may load a newer version of the CUDA Driver API and the program will function the same.  For details on distributing CUDA C applications, see the CUDA C/C++ Best Practices Guide from NVIDIA.  The CUDA Driver API is closed source. You can find its documentation here.

## What is the CUDA Runtime API?
The CUDA Runtime API wraps the CUDA Driver API and provides a higher-level API for the same functions.  !The CUDA Toolkit. The CUDA Runtime API wraps the CUDA Driver API to make it more amenable to application programming. Adapted from the *Professional CUDA C Programming Guide*.  It is generally preferred over the Driver API for better ergonomics, but there are some small caveats around control of kernel launches and context management. See this section of the CUDA Runtime API docs for more.  While the Runtime API may be statically linked, per Attachment A of the NVIDIA CUDA Toolkit EULA, it does not have to be. The shared object file for dynamic linking is usually named libcudart.so on Linux systems.  The CUDA Runtime API is closed source. You can find its documentation here.

## What is the CUDA Software Platform?
CUDA stands for _Compute Unified Device Architecture_. Depending on the context, "CUDA" can refer to multiple distinct things: a high-level device architecture, a parallel programming model for architectures with that design, or a software platform that extends high-level languages like C to add that programming model.  The vision for CUDA is laid out in the Lindholm et al., 2008 white paper. We highly recommend this paper, which is the original source for many claims, diagrams, and even specific turns of phrase in NVIDIA's documentation.  Here, we focus on the CUDA _software platform_.  The CUDA software platform is a collection of software for developing CUDA programs. Though CUDA software platforms exist for other languages, like FORTRAN, we will focus on the dominant CUDA C++ version.  This platform can be roughly divided into the components used to _build_ applications, like the NVIDIA CUDA Compiler Driver toolchain, and the components used _within_ or _from_ applications, like the CUDA Driver API and the CUDA Runtime API, diagrammed below.  !The CUDA Toolkit. Adapted from the *Professional CUDA C Programming Guide*.

## What is the NVIDIA CUDA Profiling Tools Interface?
abbreviation: CUPTI
The NVIDIA CUDA Profiling Tools Interface (CUPTI) provides a set of APIs for profiling execution of CUDA C++, PTX, and SASS code on GPUs. Critically, it synchronizes timestamps across the CPU host and the GPU device.  CUPTI's interfaces are consumed by, for example, the NSight Profiler and the PyTorch Profiler.  You can find its documentation here.  For details on using profiling tools for GPU applications running on Modal, see this example from our documentation.

## What is libcuda.so?
The typical name for the binary shared object file that implements the CUDA Driver API on Linux systems. It is dynamically linked by CUDA programs. If it is missing, the drivers are generally improperly installed.

## What is libcudart.so?
The typical name for the binary shared object file that implements the CUDA Runtime API on Linux systems. Deployed CUDA binaries often statically link this file, but libraries and frameworks built on the CUDA Toolkit, like PyTorch, typically load it dynamically.

## What is libnvml.so?
The typical name for the binary shared object file that implements the features of NVML on Linux systems.

## What is NVIDIA Nsight Systems?
NVIDIA Nsight Systems is a performance debugging tool for CUDA C++ programs. It combines profiling, tracing, and expert systems analysis in a GUI.  No one wakes up and says "today I want to write a program that runs on a hard to use, expensive piece of hardware using a proprietary software stack". Instead, GPUs are selected when normal computing hardware doesn't perform well enough to solve a computing problem. So almost all GPU programs are performance-sensitive, and the performance debugging workflows supported by Nsight Systems or other tools built on top of the CUDA Profiling Tools Interface are mission-critical.  You can find its documentation here, but watching someone use the tool is usually more helpful. For details on how to profile GPU applications on Modal, see our documentation.

## What is the NVIDIA CUDA Compiler Driver?
abbreviation: nvcc
The NVIDIA CUDA Compiler Driver is a toolchain for compiling CUDA C/C++ programs. It outputs binary executables that conform to the host ABI and include PTX and/or SASS to be executed on the GPU � a so-called "fat binary". These binaries are inspectable with the same tools used for other binaries, like `readelf` on Linux, but can be additionally manipulated with the specialized CUDA Binary Utilities.  The included PTX code is versioned by Compute Capability, configured by passing `compute_XYz` values to the `--gpu-architecture` or `--gpu-code` options.  The included SASS code is versioned by SM architecture version, configured by passing `sm_XYz` values to the `--gpu-architecture` or `--gpu-code` options. Passing `compute_XYz` to `--gpu-code` will also trigger the generation of SASS code with the same version as the PTX.  Compilation of host/CPU code is done using the host system's compiler driver, e.g. the `gcc` compiler driver. Note that compiler drivers are not to be confused with hardware drivers, like the NVIDIA GPU Drivers.  The documentation for `nvcc` can be found here.

## What are the NVIDIA GPU Drivers?
The NVIDIA GPU drivers mediate the interaction between host programs or the host operating system and the GPU device. The primary interfaces to the GPU drivers for applications are, in order, the CUDA Runtime API and the CUDA Driver API.  !The CUDA Toolkit. The NVIDIA GPU Driver, is the only component that communicates directly with the GPU. Adapted from the *Professional CUDA C Programming Guide*.  NVIDIA has released the source for their Linux Open GPU Kernel Module.

## What is nvidia.ko?
`nvidia.ko` is a binary kernel module file at the core of the NVIDIA GPU drivers for Linux.  Like other kernel modules, it executes in privileged mode and communicates directly with hardware on behalf of the user -- in this case, the GPU.  The Linux Open GPU Kernel Module is open source.

## What is nvidia-smi?
This command line utility is used to query and manage the state of the GPU exposed by the NVML management libraries. Its outputs, a sample of which appears below, are familiar to users of NVIDIA GPUs to the point of being a meme.  ``` +-----------------------------------------------------------------------------------------+ | NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     | |-----------------------------------------+------------------------+----------------------+ | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC | | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. | |                                         |                        |               MIG M. | |=========================================+========================+======================| |   0  NVIDIA H100 80GB HBM3          On  |   00000000:53:00.0 Off |                    0 | | N/A   25C    P0             92W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ |   1  NVIDIA H100 80GB HBM3          On  |   00000000:64:00.0 Off |                    0 | | N/A   27C    P0             93W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ |   2  NVIDIA H100 80GB HBM3          On  |   00000000:75:00.0 Off |                    0 | | N/A   26C    P0             96W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ |   3  NVIDIA H100 80GB HBM3          On  |   00000000:86:00.0 Off |                    0 | | N/A   27C    P0             93W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ |   4  NVIDIA H100 80GB HBM3          On  |   00000000:97:00.0 Off |                    0 | | N/A   27C    P0             95W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ |   5  NVIDIA H100 80GB HBM3          On  |   00000000:A8:00.0 Off |                    0 | | N/A   25C    P0             91W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ |   6  NVIDIA H100 80GB HBM3          On  |   00000000:B9:00.0 Off |                    0 | | N/A   26C    P0             91W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ |   7  NVIDIA H100 80GB HBM3          On  |   00000000:CA:00.0 Off |                    0 | | N/A   24C    P0             91W /  700W |       1MiB /  81559MiB |      0%      Default | |                                         |                        |             Disabled | +-----------------------------------------+------------------------+----------------------+ ```

## What is the NVIDIA Management Library?
abbreviation: NVML
The NVIDIA Management Library (NVML) is used for monitoring and managing the state of NVIDIA GPUs. It exposes, for example, the power draw and temperature of the GPU, the allocated memory, and the device's power limit and power limiting state.  The function of NVML are frequently accessed via the nvidia-smi command line utility, but are also accessible to programs via wrappers, like pynvml in Python and nvml_wrapper in Rust.

## What is the NVIDIA Runtime Compiler?
abbreviation: nvrtc
The NVIDIA Runtime Compiler (`nvrtc`) is a runtime compilation library for CUDA C. It compiles CUDA C++ to PTX without requiring a separate launch of the NVIDIA CUDA Compiler Driver (`nvcc`) in another process. It is used by some libraries or frameworks to, for example, map generated C/C++ code to PTX code that can run on a GPU.  Note that this PTX is then further JIT-compiled from the PTX IR to the SASS assembly. This is done by the NVIDIA GPU drivers and is distinct from the compilation done by NVRTC. CUDA binaries that contain PTX, as required for forward compatibility, also pass through this compilation step.  NVRTC is closed source. You can find its documentation here."""

import os
import time
import warnings
from uuid import uuid4
from fastapi.responses import StreamingResponse
import modal
import requests
import random

GPU_TYPE = os.environ.get("GPU_TYPE", "A100-80GB")
GPU_COUNT = os.environ.get("GPU_COUNT", 4)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues

MINUTES = 60  # seconds

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODELS_DIR = "/llamas"

try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_llama.py")

sgl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(  # add sglang and some Python dependencies
        "transformers==4.47.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "starlette==0.41.2",
        "torch==2.4.0",
        "sglang[all]==0.4.1",
        # as per sglang website: https://sgl-project.github.io/start/install.html
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/",
    )
)

app = modal.App(f"") # Removed to prevent Backend URL Leak

MODEL_CHAT_TEMPLATE = "llama-3-instruct"

class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    RED = "\033[0;31m"
    BOLD = "\033[1m"
    END = "\033[0m"

@app.cls(
    gpu=GPU_CONFIG,
    timeout=5 * MINUTES,
    container_idle_timeout=5 * MINUTES,
    allow_concurrent_inputs=100,
    image=sgl_image,
    volumes={MODELS_DIR: volume},
    keep_warm=1
)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl

        self.runtime = sgl.Runtime(
            model_path=MODELS_DIR + "/" + MODEL_NAME,
            tokenizer_path=MODELS_DIR + "/" + MODEL_NAME,
            tp_size=GPU_COUNT,  # t_ensor p_arallel size, number of GPUs to split the model over
            log_level=SGL_LOG_LEVEL,
        )
        print("Chat Template:", MODEL_CHAT_TEMPLATE, sgl.lang.chat_template.get_chat_template(MODEL_CHAT_TEMPLATE))
        print
        self.runtime.endpoint.chat_template = (
            sgl.lang.chat_template.get_chat_template(MODEL_CHAT_TEMPLATE)
        )
        sgl.set_default_backend(self.runtime)

    @modal.web_endpoint(method="POST", docs=True)
    def generate(self, request: dict):
        import sglang as sgl

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        query = request.get("text")

        request_max_tokens = request.get("max_tokens")

        @sgl.function
        def run_through_model(s, system_prompt, user_prompt, max_tokens):
            s += sgl.system(system_prompt)
            s += sgl.user(user_prompt)
            s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens))

        state = run_through_model.run(
            system_prompt = SYSTEM_PROMPT,
            user_prompt = query,
            max_tokens = request_max_tokens,
            stream=True
        )

        def generate_sub_fn():
            for out in state.text_iter(var_name="response"):
                yield out  # Stream each part of the response

        # Measure total time taken for the entire request
        def wrapper():
            yield from generate_sub_fn()

        # Return the streamed response
        return StreamingResponse(wrapper(), media_type="text/plain")   

        # LOGS FROM NON-STREAMING MODE
        # model_generation = state["response"]

        
        # # show the question, image, and response in the terminal for demonstration purposes
        # print(Colors.BOLD, Colors.GRAY, "System Prompt:", SYSTEM_PROMPT, Colors.END, sep=" ")
        # print(Colors.BOLD, Colors.BLUE, "User Prompt:", query, Colors.END, sep=" ")
        # print(Colors.BOLD, Colors.GREEN, "Answer:", model_generation, Colors.END, sep=" ")
        # print(Colors.BOLD, Colors.RED, "Full Text:", state.text, Colors.END, sep=" ")
        
        # print(
        #     f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        # )

        # return {
        #     "response": model_generation,
        #     "text": state.text,
        # }

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()