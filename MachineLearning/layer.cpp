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
	float* nodes;
	float* activations;
	float* layerError;
	int lTot = 0;
	for (int i = 1; i < layerNum; i++) {
		lTot += layers[i];
	}
	cudaMalloc(&nodes, sizeof(float) * lTot);
	cudaMalloc(&activations, sizeof(float) * lTot);
	cudaMalloc(&layerError, sizeof(float) * lTot);
	int nodesOffset = 0;
	int activationsOffset = 0;
	int layerErrorOffset = 0;
	//forward pass - xG will hold the result of forward propagation after this
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
		//SET NODES[layer] TO THIS BEFORE ACTIVATION FUNCTION IS APPLIED
		cudaMemcpy((nodes+nodesOffset), xGnext, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		nodesOffset += layers[i + 1];
		definedGPUFunctions::sigmoidMatCWiseGPUMem(xGnext, xGnext, layers[i + 1]);
		//SET ACTIVATIONS[layer] TO THIS AFTER ACTIVATION FUNCTION IS APPLIED
		cudaMemcpy((activations + activationsOffset), xGnext, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		activationsOffset += layers[i + 1];
		cudaFree(lyrWMat);
		cudaFree(lyrBMat);
		cudaFree(xG);
		cudaMalloc(&xG, sizeof(float) * layers[i + 1]);
		cudaMemcpy(xG, xGnext, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(xGnext);
	}
	//calculate output error
	float* tempComp;
	cudaMalloc(&tempComp, sizeof(float) * layers[layerNum - 1]);
	float* yG; //stores the "ideal" output of the network on the GPU memory 
	cudaMalloc(&yG, sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(yG, y, sizeof(float) * layers[layerNum - 1], cudaMemcpyHostToDevice);
	definedGPUFunctions::subMatCWiseGPUMem(xG, yG, xG);
	cudaMemcpy(tempComp, (nodes+lTot-layers[layerNum-1]), sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToDevice);
	definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(tempComp, tempComp, layers[layerNum - 1]); //tempComp now stores sigmoidPrime output of last layer	
	definedGPUFunctions::multCompCWiseGPUMem(xG, tempComp, xG); //xG now stores the vector for the error on the output layer of the network
	cudaMemcpy((layerError+lTot-layers[layerNum-1]), xG, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToDevice);
	cudaFree(tempComp);
	cudaFree(yG);
	cudaFree(xG);
	//backward pass
	wMOffset = 0;
	int eMOffset = lTot - layers[layerNum - 1] - layers[layerNum - 2];
	int nMOffset = lTot - layers[layerNum - 1] - layers[layerNum - 2] - layers[layerNum - 3];
	int lMOffset = lTot - layers[layerNum - 1] - layers[layerNum - 2] - layers[layerNum - 3];
	for (int i = 0; i < layerNum - 2; i++) {
		wMOffset += layers[i] * layers[i + 1];
	}
	for (int i = layerNum - 3; i >= 0; i--) {
		float* lyrWMat; //stores wMatrixArr[i+1]
		cudaMalloc(&lyrWMat, sizeof(float) * layers[i] * layers[i + 1]);
		cudaMemcpy(lyrWMat, (wMat + wMOffset), sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToDevice);
		wMOffset -= (layers[i] * layers[i + 1]);
		float* lyrErrT; //stores layerError[i+1]
		cudaMalloc(&lyrErrT, sizeof(float) * layers[i + 1]);
		cudaMemcpy(lyrErrT, (layerError + eMOffset), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		eMOffset -= layers[i + 1];
		float* lyrNodT; //stores nodes[i]
		cudaMalloc(&lyrNodT, sizeof(float) * layers[i]);
		cudaMemcpy(lyrNodT, (nodes+nMOffset), sizeof(float) * layers[i], cudaMemcpyDeviceToDevice);
		nMOffset -= layers[i];
		cudaMalloc(&tempComp, sizeof(float) * layers[i]); //tempComp stores af.evalPrimeMatrix(nodes[i])
		definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(lyrNodT, tempComp, layers[i]);
		float* tempComp2; //stores (wMatrixArr[i+1].transpose().mmul(layerError[i+1]))
		cudaMalloc(&tempComp2, sizeof(float) * layers[i]);
		blas.gemmStandardTransposeAFromGPUMem(lyrWMat, lyrErrT, tempComp2, layers[i], layers[i+1], 1);
		definedGPUFunctions::multCompCWiseGPUMem(tempComp2, tempComp, tempComp); //tempComp now stores (wMatrixArr[i+1].transpose().mmul(layerError[i+1])).mul(af.evalPrimeMatrix(nodes[i]))
		cudaMemcpy((layerError + lMOffset), tempComp, sizeof(float) * layers[i], cudaMemcpyDeviceToDevice);
		cudaFree(lyrWMat);
		cudaFree(lyrErrT);
		cudaFree(lyrNodT);
		cudaFree(tempComp);
		cudaFree(tempComp2);
	}
	//perform gradient descent


}

