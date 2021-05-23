#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

#include <chrono>
using namespace std::chrono;

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t handle, const float* A, const float* B, float* C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float* alpha = &alf;
	const float* beta = &bet;

	// Create a handle for CUBLAS
	

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float* A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
/*
int main() {
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 1000;

	float* h_A = (float*)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float* h_B = (float*)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float* h_C = (float*)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU
	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	// cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	// cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A, d_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << "A =" << std::endl;
	//print_matrix(h_A, nr_rows_A, nr_cols_A);
	//std::cout << "B =" << std::endl;
	//print_matrix(h_B, nr_rows_B, nr_cols_B);

	
	
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Multiply A and B on GPU
	auto start = high_resolution_clock::now();
	gpu_blas_mmul(handle,d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << "C =" << std::endl;

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "GPU gemm took: " << ((double)duration.count()/1000000) << " seconds for two 10,000x10,000 matrices" << std::endl;

	//print_matrix(h_C, nr_rows_C, nr_cols_C);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}*/