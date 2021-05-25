#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "definedGPUFunctions.cuh";
#include "gpuMath.h";

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ __forceinline__ float sigmoidPrime(float x) {
    float temp = sigmoid(x);
    return temp * (1 - temp);
}

__global__ void sigmoid_kernel(const float* __restrict__ src, float* __restrict__ dst, int len) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = sigmoid(src[i]);
    }
}

__global__ void sigmoidPrime_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = sigmoidPrime(src[i]);
    }
}

__global__ void addKernel(const float *a, const float *b, float *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void subKernel(const float* a, const float* b, float* c) {
    c[blockIdx.x] = a[blockIdx.x] - b[blockIdx.x];
}

__global__ void multCompKernel(const float* a, const float* b, float* c) {
    c[blockIdx.x] = a[blockIdx.x] * b[blockIdx.x];
}

__global__ void multScalarCompKernel(const float* a, float f, float* c) {
    c[blockIdx.x] = a[blockIdx.x] * f;
}

void definedGPUFunctions::addMatCWiseGPUMem(float *a, float *b, float *c) {
    addKernel<<<1,1>>>(a, b, c);
}

void definedGPUFunctions::subMatCWiseGPUMem(float *a, float *b, float *c) {
    subKernel<<<1,1>>>(a, b, c);
}

void definedGPUFunctions::multCompCWiseGPUMem(float *a, float *b, float *c) {
    multCompKernel<<<1,1>>>(a, b, c);
}

void definedGPUFunctions::multCompCWiseGPUMemScalar(float* a, float f, float* c) {
    multScalarCompKernel<<<1, 1>>>(a, f, c);
}

void definedGPUFunctions::sigmoidMatCWiseGPUMem(float* A, float* B, int len) {
    dim3 dimBlock(256);
    int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);
    sigmoid_kernel<<<dimGrid, dimBlock>>>(A, B, len);
}

void definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(float* A, float* B, int len) {
    dim3 dimBlock(256);
    int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);
    sigmoidPrime_kernel<<<dimGrid, dimBlock>>>(A, B, len);
}

/*
int main() {
    //this main method tests the functionality of the sigmoid and sigmoidPrime computations on the GPU
    int m = 3;
    int n = 3;
    float* cpuA = (float*)malloc(sizeof(float) * m * n);
    float* cpuB = (float*)malloc(sizeof(float) * m * n);
    float* gpuA, * gpuB;
    cudaMalloc(&gpuA, sizeof(float) * m * n);
    cudaMalloc(&gpuB, sizeof(float) * m * n);
    gpuMath::blasOp::randMatCPUMem(cpuA, m, n);
    cudaMemcpy(gpuA, cpuA, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    std::cout << "A=" << std::endl;
    gpuMath::blasOp::print_matrix(cpuA, m, n);
    definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(gpuA, gpuB, sizeof(float) * m * n);
    cudaMemcpy(cpuB, gpuB, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    std::cout << "B=" << std::endl;
    gpuMath::blasOp::print_matrix(cpuB, m, n);
}
*/