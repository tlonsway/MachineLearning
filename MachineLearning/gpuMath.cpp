#include "gpuMath.h"
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <random>
#include "animations.h"

using namespace gpuMath;

blasOp::blasOp() {
	cublasCreate(&handle);
	srand(time(0));
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
	lda = k;
	ldc = m;
	ldb = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}
float* blasOp::gemmStandardFromGPUMemRet(const float* A, const float* B, const int m, const int k, const int n) {
	int lda, ldc, ldb;
	lda = k;
	ldc = m;
	ldb = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	float* C;
	cudaMalloc(&C, sizeof(float) * m * n);
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	return C;
}

void blasOp::gemmStandardTransposeAFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n) {
	//int lda, ldc, ldb;
	//lda = ldc = m;
	//ldb = k;
	//lda = k;
	//ldb = ldc = n;
	int lda, ldb, ldc;
	lda = m;
	ldc = m;
	ldb = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	//cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}
float* blasOp::gemmStandardTransposeAFromGPUMemRet(const float* A, const float* B, const int m, const int k, const int n) {
	int lda, ldb, ldc;
	lda = m;
	ldc = m;
	ldb = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	float* C;
	cudaMalloc(&C, sizeof(float) * m * n);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	return C;
}

void blasOp::gemmStandardTransposeBFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n) {
	//int lda, ldc, ldb;
	//lda = ldc = m;
	//ldb = k;
	//lda = k;
	//ldb = ldc = n;
	//lda = k;
	//ldb = n;
	//ldc = m;
	int lda, ldb, ldc;
	lda = k;
	ldc = m;
	ldb = k;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}
float* blasOp::gemmStandardTransposeBFromGPUMemRet(const float* A, const float* B, const int m, const int k, const int n) {
	int lda, ldb, ldc;
	lda = k;
	ldc = m;
	ldb = k;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	float* C;
	cudaMalloc(&C, sizeof(float) * m * n);
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	return C;
}

void blasOp::gemmStandardTransposeABFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, int lda, int ldb, int ldc) {
	//int lda, ldc, ldb;
	//lda = ldc = m;
	//ldb = k;
	//lda = k;
	//ldb = ldc = n;
	const float alpha = 1;
	const float beta = 0;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	//cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}

void blasOp::gemmFullFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, const float alpha, const float beta) {
	int lda, ldc, ldb;
	lda = ldc = m;
	ldb = k;
	//lda = k;
	//ldb = ldc = n;
	const float* alphaP = &alpha;
	const float* betaP = &beta;
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
}

void blasOp::geamTransposeSingleGPUMem(float* A, float* B, int m, int n) {
	float alpha = 1;
	float beta = 0;
	float* alphaP = &alpha;
	float* betaP = &beta;
	//cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, alphaP, A, n, betaP, B, m, B, m);
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, alphaP, A, m, betaP, B, n, B, n);
}

void blasOp::randMatGPUMem(float* A, int nr_rows_A, int nr_cols_A) {
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}
void blasOp::randMatCPUMem(float* A, int m, int n) {
	long len = (long)m * (long)n;
	long i;
	/*for (i=0; i < len; i += 1) {
		*(A + i) = ((float)(rand()));
		progress_bar(i, len, "Generating Random Values");
	}*/
	
	for (i = 0; i < len % 10; i++) {
		float num = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i) = num;
	}
	for (; i < len; i+=10) {
		*(A + i) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i+1) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 2) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 3) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 4) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 5) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 6) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 7) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 8) = ((float)(rand() % 200) - 100.0) / 100.0;
		*(A + i + 9) = ((float)(rand() % 200) - 100.0) / 100.0;
		progress_bar(i, len, "Generating Random Values");
	}
	std::cout << std::endl;
}
void blasOp::randMatCPUMemNormal(float* A, long len, float mean, float variance) {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, variance);
	for (long l = 0; l < len; l++) {
		*(A + l) = distribution(generator);
		progress_bar(l, len, "Generating Gaussian Random Distribution");
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