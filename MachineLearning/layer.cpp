#include "layer.h"
#include <cublas_v2.h>
#include <cstdlib>
#include <curand.h>
#include <iostream>
#include "definedGPUFunctions.cuh"

using namespace layer;

FullyConnected::FullyConnected(int* lys, int lysN, float lr) {
	layers = lys;
	layerNum = lysN;
	lRate = lr;
	long wMatSizeTot = 0;
	long bMatSizeTot = 0;
	gpuMath::blasOp blas();
	for (int i = 0; i < lysN-1; i++) {
		wMatSizeTot += lys[i + 1] * lys[i];
		bMatSizeTot += lys[i + 1];
	}
	cudaMalloc(&wMat, sizeof(float) * wMatSizeTot);
	cudaMalloc(&bMat, sizeof(float) * bMatSizeTot);
	float* tWm = (float*)malloc(sizeof(float) * wMatSizeTot);
	float* tBm = (float*)malloc(sizeof(float) * bMatSizeTot);
	gpuMath::blasOp::randMatCPUMem(tWm, wMatSizeTot, 1);
	gpuMath::blasOp::randMatCPUMem(tBm, bMatSizeTot, 1);
	cudaMemcpy(wMat, tWm, sizeof(float) * wMatSizeTot, cudaMemcpyHostToDevice);
	cudaMemcpy(bMat, tBm, sizeof(float) * bMatSizeTot, cudaMemcpyHostToDevice);
}
float* FullyConnected::feedForward(const float* x) {
	//input and output are both pointers to vector on the CPU, function automatically loads and unloads data from GPU
	float* xG;
	cudaMalloc(&xG, sizeof(float) * layers[0]);
	cudaMemcpy(xG, x, sizeof(float) * layers[0], cudaMemcpyHostToDevice);
	int wMOffset = 0;
	int bMOffset = 0;
	for (int i = 0; i < layerNum - 1; i++) {
		float* xGnext;
		cudaMalloc(&xGnext, sizeof(float) * layers[i + 1]);
		float* lyrWMat; 
		float* lyrBMat;
		cudaMalloc(&lyrWMat, sizeof(float) * layers[i] * layers[i + 1]);
		cudaMalloc(&lyrBMat, sizeof(float) * layers[i + 1]);
		cudaMemcpy(lyrWMat, (wMat + wMOffset), sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaMemcpy(lyrBMat, (bMat + bMOffset), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		wMOffset += (layers[i] * layers[i + 1]);
		bMOffset += (layers[i + 1]);
		blas.gemmStandardFromGPUMem(lyrWMat, xG, xGnext, layers[i + 1], layers[i], 1);
		blas.axpyStandardFromGPUMem(lyrBMat, xGnext, layers[i + 1]);
		definedGPUFunctions::sigmoidMatCWiseGPUMem(xGnext, xGnext, layers[i + 1]);
		cudaFree(lyrWMat);
		cudaFree(lyrBMat);
		cudaFree(xG);
		cudaMalloc(&xG, sizeof(float) * layers[i + 1]);
		cudaMemcpy(xG, xGnext, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(xGnext);
	}
	float* y = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(y, xG, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
	cudaFree(xG);
	return y;
}

void FullyConnected::backProp(const float* x, const float* y) {
	//hard-coded using quadratic cost function

	
	//forward pass


	//calculate output error


	//perform gradient descent


}

