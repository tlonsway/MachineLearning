#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "ActivationFunctions.cuh"
#include <math.h>



//DEFINE SIGMOID KERNEL
__device__ __forceinline__ float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}
__device__ __forceinline__ float sigmoidPrime(float x) {
	float temp = sigmoid(x);
	return temp * (1 - temp);
}
__global__ void sigmoid_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
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

//DEFINE RELU KERNEL
__device__ __forceinline__ float reLu(float x) {
	if (x > 0) {
		return x;
	}
	return 0;
}
__device__ __forceinline__ float reLuPrime(float x) {
	if (x > 0) {
		return 1;
	}
	return 0;
}
__global__ void reLu_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < len; i += stride) {
		dst[i] = reLu(src[i]);
	}
}
__global__ void reLuPrime_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < len; i += stride) {
		dst[i] = reLuPrime(src[i]);
	}
}

//DEFINE HYPTAN KERNEL
__device__ __forceinline__ float hyptan(float x) {
	float epx = exp(x);
	float enx = exp(-x);
	return (epx - enx) / (epx + enx);
}
__device__ __forceinline__ float hyptanPrime(float x) {
	float htx = hyptan(x);
	return 1 - (htx * htx);
}
__global__ void hyptan_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < len; i += stride) {
		dst[i] = hyptan(src[i]);
	}
}
__global__ void hyptanPrime_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < len; i += stride) {
		dst[i] = hyptanPrime(src[i]);
	}
}

//Define abstract ActivationFunctions
ActivationFunction::ActivationFunction() {
	name = "N/A";
}
std::string ActivationFunction::getName() {
	return name;
}

//Define Sigmoid
Sigmoid::Sigmoid() {
	name = "Sigmoid";
}
void Sigmoid::eval(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	sigmoid_kernel<<<dimGrid, dimBlock>>>(x, y, len);
}
void Sigmoid::evalPrime(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	sigmoidPrime_kernel<<<dimGrid, dimBlock>>>(x, y, len);
}

//Define ReLu
ReLu::ReLu() {
	name = "ReLu";
}
void ReLu::eval(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	reLu_kernel << <dimGrid, dimBlock >> > (x, y, len);
}
void ReLu::evalPrime(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	reLuPrime_kernel << <dimGrid, dimBlock >> > (x, y, len);
}

//Define Hyperbolic Tangent
HypTan::HypTan() {
	name = "HypTan";
}
void HypTan::eval(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	hyptan_kernel << <dimGrid, dimBlock >> > (x, y, len);
}
void HypTan::evalPrime(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	hyptanPrime_kernel << <dimGrid, dimBlock >> > (x, y, len);
}