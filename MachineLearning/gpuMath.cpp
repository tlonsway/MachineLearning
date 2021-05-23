#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;


class gpuMath {
public:
	cublasHandle_t handle;
	gpuMath() {
		cublasCreate(&handle);
	}

	void gemmStandardFromCPUMem(const float* cpuA, const float* cpuB, float* cpuC, const int m, const int k, const int n) {
		//A,B,C are all pointers to host(CPU) memory, so transfers must occur between host and device(GPU), a function operating only on GPU memory would be more optimal
		float* A, * B, * C;
		cudaMalloc(&A, m * k * sizeof(float));
		cudaMalloc(&B, k * n * sizeof(float));
		cudaMalloc(&C, m * n * sizeof(float));
		cudaMemcpy(A, cpuA, m * k * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(B, cpuB, k * n * sizeof(float),cudaMemcpyHostToDevice);
		int lda, ldc, ldb;
		lda = ldc = m;
		ldb = k;
		const float alpha = 1;
		const float beta = 0;
		const float* alphaP = &alpha;
		const float* betaP = &beta;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
		cudaMemcpy(cpuC, C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	}
	void gemmStandardFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n) {
		//A,B,C are all pointers to device(GPU) memory, so transfers are not needed
		int lda, ldc, ldb;
		lda = ldc = m;
		ldb = k;
		const float alpha = 1;
		const float beta = 0;
		const float* alphaP = &alpha;
		const float* betaP = &beta;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaP, A, lda, B, ldb, betaP, C, ldc);
	}
	static void randMatGPUMem(float* A, int nr_rows_A, int nr_cols_A) {
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
		curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
	}
	static void randMatCPUMem(float* A, int m, int n) {
		for (int i = 0; i < m * n; i++) {
			A[i] = (float)(rand() % 1000);
		}
	}
	void close() {
		cublasDestroy(handle);
	}
};

int main() {
	int m = 10000;
	int k = 10000;
	int n = 1;
	float* gpuA = (float*)malloc(sizeof(float) * m * k);
	float* gpuB = (float*)malloc(sizeof(float) * k * n);
	float* gpuC = (float*)malloc(sizeof(float) * m * n);

	gpuMath::randMatCPUMem(gpuA, m, n);
	gpuMath::randMatCPUMem(gpuB, n, k);
	float* AGPUMem, * BGPUMem, * CGPUMem;
	cudaMalloc(&AGPUMem, m * k * sizeof(float));
	cudaMalloc(&BGPUMem, k * n * sizeof(float));
	cudaMalloc(&CGPUMem, m * n * sizeof(float));
	cudaMemcpy(AGPUMem, gpuA, m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(BGPUMem, gpuB, k * n * sizeof(float), cudaMemcpyHostToDevice);
	float* A = gpuA;
	float* B = gpuC;
	float* C = (float*)malloc(sizeof(float) * m * n); 
	gpuMath mathOp;
	auto start = high_resolution_clock::now();
	mathOp.gemmStandardFromGPUMem(AGPUMem, BGPUMem, CGPUMem, m, k, n);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "GPU gemm took: " << ((double)duration.count() / 1000000) << " seconds" << std::endl;
	int il, jl, kl;
	start = high_resolution_clock::now();
	for (il = 0; il < m; ++il)
	{
		for (jl = 0; jl < n; ++jl)
		{
			*(C + il * n + jl) = 0;
		}
	}
	for (il = 0; il < m; ++il)
	{
		for (jl = 0; jl < n; ++jl)
		{
			for (kl = 0; kl < m; ++kl)
			{
				*(C + il * n + jl) += *(A + il * k + kl) * *(B + kl * n + jl);
			}
		}
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << "CPU mmul took: " << ((double)duration.count() / 1000000) << " seconds" << std::endl;

}