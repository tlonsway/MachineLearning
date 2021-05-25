#pragma once
#include <cublas_v2.h>

namespace gpuMath {
	class blasOp {
	public:
		cublasHandle_t handle;
		blasOp();
		void axpyStandardFromGPUMem(const float* x, float* y, int len);
	    void gemmStandardFromCPUMem(const float* cpuA, const float* cpuB, float* cpuC, const int m, const int k, const int n);
		void gemmStandardFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n);
		void gemmStandardTransposeAFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, int lda, int ldb, int ldc);
		void gemmStandardTransposeBFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, int lda, int ldb, int ldc);
		void gemmFullFromGPUMem(const float* A, const float* B, float* C, const int m, const int k, const int n, const float alpha, const float beta);
		void geamTransposeSingleGPUMem(float* A, float* B, int m, int n);
		static void randMatGPUMem(float* A, int nr_rows_A, int nr_cols_A);
		static void randMatCPUMem(float* A, int m, int n);
		static void print_matrix(const float* A, int nr_rows_A, int nr_cols_A);
		void close();
	};
}