#include "gpuMath.h"
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>

using namespace gpuMath;

blasOp::blasOp() {
	cublasCreate(&handle);
}

void blasOp::axpyStandardFromGPUMem(const float* x, float* y, int len) {
	//result will be stored in vector y
	const float alpha = 1;
	const float * alphaP = &alpha;
	cublasSaxpy(handle, len, alphaP, x, 1, y, 1);
}

void blasOp::gemmStandardFromCPUMem(const float* cpuA, const float* cpuB, float* cpuC, const int m, const int k, const int n) {
	//A,B,C are all pointers to host(CPU) memory, so transfers must occur between host and device(GPU), a function operating only on GPU memory would be more optimal
	float* A, * B, * C;
	cudaMalloc(&A, m * k * sizeof(float));
	cudaMalloc(&B, k * n * sizeof(float));
	cudaMalloc(&C, m * n * sizeof(float));
	cudaMemcpy(A, cpuA, m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B, cpuB, k * n * sizeof(float), cudaMemcpyHostToDevice);
	int lda, ldc, ldb;
	lda = ldc = m;
	ldb = k;
	//ldb = k;
	//lda = k;
	//ldb = ldc = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	cudaMemcpy(cpuC, C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
}
void blasOp::gemmStandardFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n) {
	//A,B,C are all pointers to device(GPU) memory, so transfers are not needed
	int lda, ldc, ldb;
	lda = ldc = m;
	ldb = k;
	//lda = k;
	//ldb = ldc = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}

void blasOp::gemmStandardTransposeAFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, int lda, int ldb, int ldc) {
	//int lda, ldc, ldb;
	//lda = ldc = m;
	//ldb = k;
	//lda = k;
	//ldb = ldc = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}

void blasOp::gemmStandardTransposeBFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, int lda, int ldb, int ldc) {
	//int lda, ldc, ldb;
	//lda = ldc = m;
	//ldb = k;
	//lda = k;
	//ldb = ldc = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}

void blasOp::gemmFullFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, const float alpha, const float beta) {
	int lda, ldc, ldb;
	lda = ldc = m;
	ldb = k;
	//lda = k;
	//ldb = ldc = n;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}

void blasOp::randMatGPUMem(float* A, int nr_rows_A, int nr_cols_A) {
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}
void blasOp::randMatCPUMem(float* A, int m, int n) {
	for (long i = 0; i < m * n; i++) {
		*(A + i) = (float)(rand() % 10)/10;
	}
}
void blasOp::print_matrix(const float* A, int nr_rows_A, int nr_cols_A) {
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
void blasOp::close() {
	cublasDestroy(handle);
}