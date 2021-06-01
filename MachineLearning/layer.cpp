#include "layer.h"
#include <cublas_v2.h>
#include <cstdlib>
#include <curand.h>
#include <iostream>
#include "definedGPUFunctions.cuh"
#include <windows.h>
#include "ProgressBar.h"
#include <string>

using namespace layer;

FullyConnected::FullyConnected(int* lys, int lysN, float lr, ActivationFunction *activationFunction) {
	std::string initStr = "Initializing Fully-Connected Network: [Nodes(";
	for (int i = 0; i < lysN; i++) {
		initStr = initStr + std::to_string(lys[i]);
		if (i != lysN - 1) {
			initStr = initStr + ":";
		}
	}
	initStr = initStr + "),lRate(" + std::to_string(lr) + "),af(" + activationFunction->getName()+ ")]";
	vbOut(initStr);
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	layers = lys;
	layerNum = lysN;
	lRate = lr;
	af = activationFunction;
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
	//Xavier initialization for weights
	long wMatOffset = 0;
	vbOut("Using Xavier initialization to set initial weights");
	for (int i = 0; i < layerNum - 1; i++) {
		float stdev = sqrt(1/(float)layers[i]);
		gpuMath::blasOp::randMatCPUMemNormal(tWm + wMatOffset, layers[i] * layers[i + 1], 0,stdev);
		wMatOffset += layers[i] * layers[i + 1];
	}
	for (int i = 0; i < bMatSizeTot; i++) {
		tBm[i] = 0;
	}
	vbOut("Transferring intial matrices from CPU to GPU memory");
	cudaMemcpy(wMat, tWm, sizeof(float) * wMatSizeTot, cudaMemcpyHostToDevice);
	cudaMemcpy(bMat, tBm, sizeof(float) * bMatSizeTot, cudaMemcpyHostToDevice);
	free(tWm);
	free(tBm);
	vbOut("Finished initializing network");
}

void FullyConnected::vbOut(std::string s) {
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	std::string vBStr = "[FCNET]: ";
	vBStr = vBStr + s;
	SetConsoleTextAttribute(hConsole, 10);
	std::cout << vBStr << std::endl;
	SetConsoleTextAttribute(hConsole, 15);
}

void FullyConnected::end() {
	blas.close();
	cudaFree(wMat);
	cudaFree(bMat);
	free(layers);
}

float* FullyConnected::getwMatAtIndex(int i) {
	//returns a pointer to where wMat[i] begins
	int wMOffset = 0;
	for (int j = 0; j < i; j++) {
		wMOffset += layers[j] * layers[j + 1];
	}
	return (wMat + wMOffset);
}
int* FullyConnected::getwMatDimsAtIndex(int i) {
	//returns the dimensions of wMat[i] in {row,column} format
	return new int[2]{layers[i + 1],layers[i]};

}
float* FullyConnected::getbMatAtIndex(int i) {
	int bMOffset = 0;
	for (int j = 0; j < i; j++) {
		bMOffset += layers[j + 1];
	}
	return (bMat + bMOffset);

}
int* FullyConnected::getbMatDimsAtIndex(int i) {
	return new int[2] {layers[i + 1], 1};
}
float* FullyConnected::getnVecAtIndex(int i, float* nodeVec) {
	int nMOffset = 0;
	for (int j = 0; j < i; j++) {
		nMOffset += layers[j + 1];
	}
	return (nodeVec + nMOffset);
}
int* FullyConnected::getnVecDimsAtIndex(int i) {
	return new int[2] {layers[i + 1],1};
}
float* FullyConnected::getaVecAtIndex(int i, float* activationVec) {
	int aMOffset = 0;
	for (int j = 0; j < i; j++) {
		aMOffset += layers[j + 1];
	}
	return (activationVec + aMOffset);
}
int* FullyConnected::getaVecDimsAtIndex(int i) {
	return new int[2] {layers[i + 1], 1};
}
float* FullyConnected::geteVecAtIndex(int i, float* errorVec) {
	int eMOffset = 0;
	for (int j = 0; j < i; j++) {
		eMOffset += layers[j + 1];
	}
	return (errorVec + eMOffset);
}
int* FullyConnected::geteVecDimsAtIndex(int i) {
	return new int[2] {layers[i + 1], 1};
}

void printGPUMat(float* GPUMem, int m, int n) {
	std::cout << std::endl;
	float* tCPUMem = (float*)malloc(sizeof(float) * m * n);
	cudaMemcpy(tCPUMem, GPUMem, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
	int incr = 0;
	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			std::cout << tCPUMem[incr] << " ";
			incr++;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	free(tCPUMem);
}
float getVecMagGPUMem(float* vec, int len) {
	float runSum = 0;
	float* vCPU = (float*)malloc(sizeof(float) * len);
	cudaMemcpy(vCPU, vec, sizeof(float) * len, cudaMemcpyDeviceToHost);
	for (int i = 0; i < len; i++) {
		runSum += vCPU[i] * vCPU[i];
	}
	return sqrt(runSum);
}


float* FullyConnected::feedForward(const float* x) {
	float* xG;
	cudaMalloc(&xG, sizeof(float) * layers[0]);
	cudaMemcpy(xG, x, sizeof(float) * layers[0], cudaMemcpyHostToDevice); //xG is now in GPU memory
	for (int i = 0; i < layerNum - 1; i++) {
		float* tempComp;
		float* tempCompFP2;
		cudaMalloc(&tempComp, sizeof(float) * layers[i + 1]);
		cudaMalloc(&tempCompFP2, sizeof(float) * getwMatDimsAtIndex(i)[0] * 1);
		blas.gemmStandardFromGPUMem(getwMatAtIndex(i), xG, tempCompFP2, getwMatDimsAtIndex(i)[0], getwMatDimsAtIndex(i)[1], 1);
		definedGPUFunctions::addMatCWiseGPUMem(tempCompFP2, getbMatAtIndex(i), tempComp, layers[i + 1]);
		cudaFree(xG);
		cudaMalloc(&xG, sizeof(float) * layers[i + 1]);
		//definedGPUFunctions::sigmoidMatCWiseGPUMem(tempComp, xG, layers[i + 1]);
		//definedGPUFunctions::reLuMatCWiseGPUMem(tempComp, xG, layers[i + 1]);
		af->eval(tempComp, xG, layers[i + 1]);
	}
	float* y = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(y, xG, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
	cudaFree(xG);
	return y;
}
void FullyConnected::backProp(const float* x, const float* y) {
	//initializing memory and variables
	float* xG;
	float* yG;
	cudaMalloc(&xG, sizeof(float) * layers[0]);
	cudaMalloc(&yG, sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(xG, x, sizeof(float) * layers[0], cudaMemcpyHostToDevice); //x is now in GPU memory as xG
	cudaMemcpy(yG, y, sizeof(float) * layers[layerNum - 1], cudaMemcpyHostToDevice); //y is now in GPU memory as yG
	float* xGOriginal; //hosts an unmodified copy of xG, since xG is modified during the forward pass
	cudaMalloc(&xGOriginal, sizeof(float) * layers[0]);
	cudaMemcpy(xGOriginal, xG, sizeof(float) * layers[0], cudaMemcpyDeviceToDevice); 
	float* nodes;
	float* activations;
	float* layerError;
	int lTot = 0;
	for (int i = 0; i < layerNum-1; i++) {
		lTot += layers[i+1];
	}
	cudaMalloc(&nodes, sizeof(float) * lTot);
	cudaMalloc(&activations, sizeof(float) * lTot);
	cudaMalloc(&layerError, sizeof(float) * lTot);
	//forward pass
	for (int i = 0; i < layerNum - 1; i++) {
		float* tempCompFP1;
		float* tempCompFP2;
		cudaMalloc(&tempCompFP1, sizeof(float) * layers[i + 1]);
		cudaMalloc(&tempCompFP2, sizeof(float) * getwMatDimsAtIndex(i)[0] * 1);
		blas.gemmStandardFromGPUMem(getwMatAtIndex(i), xG,tempCompFP2,getwMatDimsAtIndex(i)[0], getwMatDimsAtIndex(i)[1], 1);		
		definedGPUFunctions::addMatCWiseGPUMem(tempCompFP2, getbMatAtIndex(i), tempCompFP1, layers[i + 1]);
		cudaMemcpy(getnVecAtIndex(i,nodes), tempCompFP1, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(xG);
		cudaMalloc(&xG, sizeof(float) * layers[i + 1]);
		//definedGPUFunctions::sigmoidMatCWiseGPUMem(tempCompFP1, xG, layers[i + 1]);
		//definedGPUFunctions::reLuMatCWiseGPUMem(tempCompFP1, xG, layers[i + 1]);
		af->eval(tempCompFP1, xG, layers[i + 1]);
		cudaMemcpy(getaVecAtIndex(i, activations), xG, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(tempCompFP1);
		cudaFree(tempCompFP2);
		//printGPUMat(xG, 1, layers[i + 1]);
	}
	cudaFree(xG);
	//calculate output error
	float* tempCompOE1;
	float* tempCompOE2;
	cudaMalloc(&tempCompOE1, sizeof(float) * layers[layerNum - 1]);
	cudaMalloc(&tempCompOE2, sizeof(float) * layers[layerNum - 1]);
	definedGPUFunctions::subMatCWiseGPUMem(getaVecAtIndex(layerNum-2,activations), yG, tempCompOE1, layers[layerNum - 1]);
	//definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(getnVecAtIndex(layerNum - 2, nodes), tempCompOE2, layers[layerNum - 1]);
	//definedGPUFunctions::reLuPrimeMatCWiseGPUMem(getnVecAtIndex(layerNum - 2, nodes), tempCompOE2, layers[layerNum - 1]);
	af->evalPrime(getnVecAtIndex(layerNum - 2, nodes), tempCompOE2, layers[layerNum - 1]);
	definedGPUFunctions::multCompCWiseGPUMem(tempCompOE1, tempCompOE2, geteVecAtIndex(layerNum - 2, layerError), layers[layerNum - 1]);
	cudaFree(tempCompOE1);
	cudaFree(tempCompOE2);
	//backward pass
	for (int i = layerNum - 3; i >= 0; i--) {
		float* tempCompBP1;
		float* tempCompBP2;
		int* wMatDims = getwMatDimsAtIndex(i + 1);
		cudaMalloc(&tempCompBP1, sizeof(float) * wMatDims[1]);
		cudaMalloc(&tempCompBP2, sizeof(float) * wMatDims[1]);
		blas.gemmStandardTransposeAFromGPUMem(getwMatAtIndex(i + 1), geteVecAtIndex(i + 1, layerError), tempCompBP1, wMatDims[1], wMatDims[0], 1);
		//definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(getnVecAtIndex(i,nodes), tempCompBP2, wMatDims[1]);
		//definedGPUFunctions::reLuPrimeMatCWiseGPUMem(getnVecAtIndex(i, nodes), tempCompBP2, wMatDims[1]);
		af->evalPrime(getnVecAtIndex(i, nodes), tempCompBP2, wMatDims[1]);
		definedGPUFunctions::multCompCWiseGPUMem(tempCompBP1, tempCompBP2, geteVecAtIndex(i,layerError), wMatDims[1]);
		free(wMatDims);
		cudaFree(tempCompBP1);
		cudaFree(tempCompBP2);
	}
	//perform gradient descent
	for (int i = 0; i < layerNum - 1; i++) {
		float* tempCompGD1; 
		float* tempCompGD2;
		float* tempCompGD4;
		int* wMatDims = getwMatDimsAtIndex(i);
		cudaMalloc(&tempCompGD1, sizeof(float) * wMatDims[0] * wMatDims[1]);
		cudaMalloc(&tempCompGD2, sizeof(float) * wMatDims[0] * wMatDims[1]);
		cudaMalloc(&tempCompGD4, sizeof(float) * wMatDims[0] * wMatDims[1]);
		if (i == 0) {
			blas.gemmStandardTransposeBFromGPUMem(geteVecAtIndex(i, layerError), xGOriginal, tempCompGD4, wMatDims[0], 1, wMatDims[1]);
			blas.geamTransposeSingleGPUMem(tempCompGD4, tempCompGD1, wMatDims[0], wMatDims[1]);
		} else {
			blas.gemmStandardFromGPUMem(geteVecAtIndex(i, layerError), getaVecAtIndex(i - 1, activations), tempCompGD4, wMatDims[0], 1, wMatDims[1]);
			blas.geamTransposeSingleGPUMem(tempCompGD4, tempCompGD1, wMatDims[0], wMatDims[1]);
		}
		definedGPUFunctions::multCompCWiseGPUMemScalar(tempCompGD1, lRate, tempCompGD2, wMatDims[0] * wMatDims[1]);
		definedGPUFunctions::subMatCWiseGPUMem(getwMatAtIndex(i), tempCompGD2, getwMatAtIndex(i), wMatDims[0] * wMatDims[1]);
		cudaFree(tempCompGD1);
		cudaFree(tempCompGD2);
		cudaFree(tempCompGD4);
		float* tempCompGD3;
		int* bMatDims = getbMatDimsAtIndex(i);
		cudaMalloc(&tempCompGD3, sizeof(float) * bMatDims[0]);
		definedGPUFunctions::multCompCWiseGPUMemScalar(geteVecAtIndex(i, layerError), lRate, tempCompGD3, bMatDims[0]);
		definedGPUFunctions::subMatCWiseGPUMem(getbMatAtIndex(i), tempCompGD3, getbMatAtIndex(i), bMatDims[0]);
		free(wMatDims);
		free(bMatDims);
		cudaFree(tempCompGD3);
	}
	cudaFree(xGOriginal);
	cudaFree(yG);
	cudaFree(nodes);
	cudaFree(activations);
	cudaFree(layerError);
}
void FullyConnected::backPropGAN(const float* x, const float error) {
	//initializing memory and variables
	float* xG;
	cudaMalloc(&xG, sizeof(float) * layers[0]);
	cudaMemcpy(xG, x, sizeof(float) * layers[0], cudaMemcpyHostToDevice); //x is now in GPU memory as xG
	float* xGOriginal; //hosts an unmodified copy of xG, since xG is modified during the forward pass
	cudaMalloc(&xGOriginal, sizeof(float) * layers[0]); 
	cudaMemcpy(xGOriginal, xG, sizeof(float) * layers[0], cudaMemcpyDeviceToDevice);
	float* errorGPU;
	cudaMalloc(&errorGPU, sizeof(float));

	float* nodes;
	float* activations;
	float* layerError;
	int lTot = 0;
	for (int i = 0; i < layerNum - 1; i++) {
		lTot += layers[i + 1];
	}
	cudaMalloc(&nodes, sizeof(float) * lTot);
	cudaMalloc(&activations, sizeof(float) * lTot);
	cudaMalloc(&layerError, sizeof(float) * lTot);
	//forward pass
	for (int i = 0; i < layerNum - 1; i++) {
		float* tempCompFP1;
		float* tempCompFP2;
		cudaMalloc(&tempCompFP1, sizeof(float) * layers[i + 1]);
		cudaMalloc(&tempCompFP2, sizeof(float) * getwMatDimsAtIndex(i)[0] * 1);
		blas.gemmStandardFromGPUMem(getwMatAtIndex(i), xG, tempCompFP2, getwMatDimsAtIndex(i)[0], getwMatDimsAtIndex(i)[1], 1);
		definedGPUFunctions::addMatCWiseGPUMem(tempCompFP2, getbMatAtIndex(i), tempCompFP1, layers[i + 1]);
		cudaMemcpy(getnVecAtIndex(i, nodes), tempCompFP1, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(xG);
		cudaMalloc(&xG, sizeof(float) * layers[i + 1]);
		//definedGPUFunctions::sigmoidMatCWiseGPUMem(tempCompFP1, xG, layers[i + 1]);
		//definedGPUFunctions::reLuMatCWiseGPUMem(tempCompFP1, xG, layers[i + 1]);
		af->eval(tempCompFP1, xG, layers[i + 1]);
		cudaMemcpy(getaVecAtIndex(i, activations), xG, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(tempCompFP1);
		cudaFree(tempCompFP2);
		//printGPUMat(xG, 1, layers[i + 1]);
	}
	cudaFree(xG);
	//calculate output error
	float* tempCompOE1;
	float* tempCompOE2;
	cudaMalloc(&tempCompOE1, sizeof(float) * layers[layerNum - 1]);
	cudaMalloc(&tempCompOE2, sizeof(float) * layers[layerNum - 1]);
	//definedGPUFunctions::subMatCWiseGPUMem(getaVecAtIndex(layerNum - 2, activations), yG, tempCompOE1, layers[layerNum - 1]);
	//definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(getnVecAtIndex(layerNum - 2, nodes), tempCompOE2, layers[layerNum - 1]);
	//definedGPUFunctions::reLuPrimeMatCWiseGPUMem(getnVecAtIndex(layerNum - 2, nodes), tempCompOE2, layers[layerNum - 1]);
	af->evalPrime(getnVecAtIndex(layerNum - 2, nodes), tempCompOE2, layers[layerNum - 1]);
	definedGPUFunctions::multCompCWiseGPUMemScalar(tempCompOE2, error, geteVecAtIndex(layerNum - 2, layerError), layers[layerNum - 1]);
	cudaFree(tempCompOE1);
	cudaFree(tempCompOE2);
	//backward pass
	for (int i = layerNum - 3; i >= 0; i--) {
		float* tempCompBP1;
		float* tempCompBP2;
		int* wMatDims = getwMatDimsAtIndex(i + 1);
		cudaMalloc(&tempCompBP1, sizeof(float) * wMatDims[1]);
		cudaMalloc(&tempCompBP2, sizeof(float) * wMatDims[1]);
		blas.gemmStandardTransposeAFromGPUMem(getwMatAtIndex(i + 1), geteVecAtIndex(i + 1, layerError), tempCompBP1, wMatDims[1], wMatDims[0], 1);
		//definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(getnVecAtIndex(i,nodes), tempCompBP2, wMatDims[1]);
		//definedGPUFunctions::reLuPrimeMatCWiseGPUMem(getnVecAtIndex(i, nodes), tempCompBP2, wMatDims[1]);
		af->evalPrime(getnVecAtIndex(i, nodes), tempCompBP2, wMatDims[1]);
		definedGPUFunctions::multCompCWiseGPUMem(tempCompBP1, tempCompBP2, geteVecAtIndex(i, layerError), wMatDims[1]);
		free(wMatDims);
		cudaFree(tempCompBP1);
		cudaFree(tempCompBP2);
	}
	//perform gradient descent
	for (int i = 0; i < layerNum - 1; i++) {
		float* tempCompGD1;
		float* tempCompGD2;
		float* tempCompGD4;
		int* wMatDims = getwMatDimsAtIndex(i);
		cudaMalloc(&tempCompGD1, sizeof(float) * wMatDims[0] * wMatDims[1]);
		cudaMalloc(&tempCompGD2, sizeof(float) * wMatDims[0] * wMatDims[1]);
		cudaMalloc(&tempCompGD4, sizeof(float) * wMatDims[0] * wMatDims[1]);
		if (i == 0) {
			blas.gemmStandardTransposeBFromGPUMem(geteVecAtIndex(i, layerError), xGOriginal, tempCompGD4, wMatDims[0], 1, wMatDims[1]);
			blas.geamTransposeSingleGPUMem(tempCompGD4, tempCompGD1, wMatDims[0], wMatDims[1]);
		}
		else {
			blas.gemmStandardFromGPUMem(geteVecAtIndex(i, layerError), getaVecAtIndex(i - 1, activations), tempCompGD4, wMatDims[0], 1, wMatDims[1]);
			blas.geamTransposeSingleGPUMem(tempCompGD4, tempCompGD1, wMatDims[0], wMatDims[1]);
		}
		definedGPUFunctions::multCompCWiseGPUMemScalar(tempCompGD1, lRate, tempCompGD2, wMatDims[0] * wMatDims[1]);
		definedGPUFunctions::subMatCWiseGPUMem(getwMatAtIndex(i), tempCompGD2, getwMatAtIndex(i), wMatDims[0] * wMatDims[1]);
		cudaFree(tempCompGD1);
		cudaFree(tempCompGD2);
		cudaFree(tempCompGD4);
		float* tempCompGD3;
		int* bMatDims = getbMatDimsAtIndex(i);
		cudaMalloc(&tempCompGD3, sizeof(float) * bMatDims[0]);
		definedGPUFunctions::multCompCWiseGPUMemScalar(geteVecAtIndex(i, layerError), lRate, tempCompGD3, bMatDims[0]);
		definedGPUFunctions::subMatCWiseGPUMem(getbMatAtIndex(i), tempCompGD3, getbMatAtIndex(i), bMatDims[0]);
		free(wMatDims);
		free(bMatDims);
		cudaFree(tempCompGD3);
	}
	cudaFree(xGOriginal);
	cudaFree(nodes);
	cudaFree(activations);
	cudaFree(layerError);
}