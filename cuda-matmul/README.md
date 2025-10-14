# CIS 5650 CUDA Introduction

The CUDA Introduction exercises are design to get you accustomed to CUDA Programming using simple operations:

1. SAXPY (`z[] = a * x[] + y[]`)
2. Matrix Transpose
3. Matrix Multiplication

## Clone, Build and Run

Clone this repository from Github and then run the instructions below.

1. From the `cuda-introduction` directory, run:
    * Windows: `cmake -B build -S . -G "Visual Studio 17 2022"` to generate the Visual Studio project. You may choose a different Visual Studio version.
    * Linux: `cmake -B build -S . -G "Unix Makefiles"` to generate the makefiles.
2. Open the generated `CUDAIntroduction` project from the `build` directory.
3. Build the project in Visual Studio. (Note that there are Debug and Release configuration options.)
4. Run. Make sure you run the actual project as target (not `ALL_BUILD`) by right-clicking it and selecting "Set as StartUp Project".

Please ask me or the TAs ahead of time if you have trouble compiling the code. We want to be ready to go at the start of the lab.

## Complete Intoduction Exercises

Start with **SAXPY**. Follow the `TODO`s, including taking a look at `common.h` and `common.cpp`. The `TODO`s are numbered in order to help guide you.

The `LOOK` comments are designed to show you best practices for CUDA Programming. You can copy these snippets into future projects if needed.

Once you have completed each the the exercises, follow the `TODO Optional`s for testing different configurations of sizes.

Repeat the same for **Matrix Transpose**, then **Matrix Multiplication**. In each file, follow the `TODO`s, which are numbered in order.

### Use Nsight Debuggers

With each implementation, run the NSight Debugger. Walk through the steps, go to different threads, warps, blocks, inspect the variables. Try to thoroughly understand your code as well as the debugging tools available to you.

### Unblocking

If you get stuck at any point, follow this order:

1. Use the Nsight debugger to understand the problem. Use paper and pencil to write down the equations especially with regards to indexing.
2. Search on the internet. Try to understand the code others have written and use that to solve your problem.
3. Use the [cuda-introduction-solutions branch](https://github.com/CIS5650-Fall-2025/Project0-Getting-Started/tree/cuda-introduction-solutions/cuda-introduction). The solutions for all the exercises are provided. For best learning, try to solve the problems on your own and only use this as reference to compare your implementation.

## Learn by breaking your programs

Following that, try to break your own implementations to familiarize yourself with common CUDA errors. Some examples include:

* Pass invalid pointers - either null, or pass the host pointer to device.
* Out of bounds access in CUDA functions like `cudaMemcpy` as well as in kernels.
* Use incorrect sizes in CUDA APIs, for example set the size parameter to 0.
* Launch kernels with bad configurations, including exceeding device limits.
* Flip indices to force bad access patterns.

When doing the above actions also use Nsight to debug.

The goal is to not just understand how to correctly implement CUDA programs, but also identify when you are doing incorrect actions. This way, when you see similar errors in your subsequent projects, you'll know where to look.

## Third Party Code

This repository includes code from [termcolor](https://github.com/ikalnytskyi/termcolor) licensed under the BSD 3 Clause.
