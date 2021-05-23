#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>
#include "gpuMath.h";
using namespace std::chrono;

/*
int main() {
	//this main() verifies that matrices can be left in the GPU memory for any future computation to avoid the overhead of transferring data between host&device
	int m = 3;
	int k = 3;
	int n = 3;
	float* cpuA = (float*)malloc(sizeof(float) * m * k);
	float* cpuB = (float*)malloc(sizeof(float) * k * n);
	float* cpuC = (float*)malloc(sizeof(float) * m * n);
	float* gpuA, * gpuB, * gpuC;
	cudaMalloc(&gpuA, sizeof(float) * m * k);
	cudaMalloc(&gpuB, sizeof(float) * k * n);
	cudaMalloc(&gpuC, sizeof(float) * m * n);
	gpuMath::blasOp::randMatCPUMem(cpuA, m, k);
	gpuMath::blasOp::randMatCPUMem(cpuB, k, n);
	cudaMemcpy(gpuA, cpuA, sizeof(float) * m * k, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuB, cpuB, sizeof(float) * k * n, cudaMemcpyHostToDevice);
	std::cout << "A=" << std::endl;
	gpuMath::blasOp::print_matrix(cpuA, m, k);
	std::cout << "B=" << std::endl;
	gpuMath::blasOp::print_matrix(cpuB, k, n);
	gpuMath::blasOp gpu;
	gpu.gemmStandardFromGPUMem(gpuA, gpuB, gpuC, m, k, n);
	gpu.gemmStandardFromGPUMem(gpuA, gpuC, gpuC, m, k, n);
	cudaMemcpy(cpuC, gpuC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
	std::cout << "C=" << std::endl;
	gpuMath::blasOp::print_matrix(cpuC, m, n);
}
*/
/*
int mainHH() {
	int m = 1000; //m=200,k=200,n=1 will show about equal CPU and GPU performance, higher values will always be faster on the GPU.
	int k = 1000;
	int n = 1;
	float* gpuA = (float*)malloc(sizeof(float) * m * k);
	float* gpuB = (float*)malloc(sizeof(float) * k * n);
	float* gpuC = (float*)malloc(sizeof(float) * m * n);
	gpuMath::randMatCPUMem(gpuA, m, k);
	gpuMath::randMatCPUMem(gpuB, k, n);
	float* AGPUMem, * BGPUMem, * CGPUMem;
	//cudaMalloc(&AGPUMem, m * k * sizeof(float));
	//cudaMalloc(&BGPUMem, k * n * sizeof(float));
	//cudaMalloc(&CGPUMem, m * n * sizeof(float));
	//cudaMemcpy(AGPUMem, gpuA, m * k * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(BGPUMem, gpuB, k * n * sizeof(float), cudaMemcpyHostToDevice);
	float* A = gpuA;
	float* B = gpuC;
	float* C = (float*)malloc(sizeof(float) * m * n); 
	gpuMath mathOp;
	auto start = high_resolution_clock::now();
	mathOp.gemmStandardFromCPUMem(gpuA, gpuB, gpuC, m, k, n);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "GPU gemm took: " << ((double)duration.count() / 1000000) << " seconds" << std::endl;
	long il, jl, kl;
	start = high_resolution_clock::now();
	for (il = 0; il < m; ++il) {
		for (jl = 0; jl < n; ++jl) {
			*(C + il * n + jl) = 0;
		}
	}
	for (il = 0; il < m; ++il) {
		for (jl = 0; jl < n; ++jl) {
			for (kl = 0; kl < m; ++kl) {
				*(C + il * n + jl) += *(A + il * k + kl) * *(B + kl * n + jl);
			}
		}
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << "CPU mmul took: " << ((double)duration.count() / 1000000) << " seconds" << std::endl;

}*/