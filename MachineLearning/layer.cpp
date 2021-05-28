#include "layer.h"
#include <cublas_v2.h>
#include <cstdlib>
#include <curand.h>
#include <iostream>
#include "definedGPUFunctions.cuh"
#include <windows.h>

using namespace layer;

FullyConnected::FullyConnected(int* lys, int lysN, float lr) {
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
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
	free(tWm);
	free(tBm);
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
		definedGPUFunctions::sigmoidMatCWiseGPUMem(tempComp, xG, layers[i + 1]);
	}
	float* y = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(y, xG, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
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

		//tempCompFP2 = blas.gemmStandardFromGPUMemRet(getwMatAtIndex(i), xG, getwMatDimsAtIndex(i)[0], getwMatDimsAtIndex(i)[1], 1);
		blas.gemmStandardFromGPUMem(getwMatAtIndex(i), xG,tempCompFP2,getwMatDimsAtIndex(i)[0], getwMatDimsAtIndex(i)[1], 1);
		
		definedGPUFunctions::addMatCWiseGPUMem(tempCompFP2, getbMatAtIndex(i), tempCompFP1, layers[i + 1]);
		cudaMemcpy(getnVecAtIndex(i,nodes), tempCompFP1, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(xG);
		cudaMalloc(&xG, sizeof(float) * layers[i + 1]);
		definedGPUFunctions::sigmoidMatCWiseGPUMem(tempCompFP1, xG, layers[i + 1]);
		cudaMemcpy(getaVecAtIndex(i, activations), xG, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaFree(tempCompFP1);
		cudaFree(tempCompFP2);
	}
	cudaFree(xG);
	//calculate output error
	float* tempCompOE1;
	float* tempCompOE2;
	cudaMalloc(&tempCompOE1, sizeof(float) * layers[layerNum - 1]);
	cudaMalloc(&tempCompOE2, sizeof(float) * layers[layerNum - 1]);
	definedGPUFunctions::subMatCWiseGPUMem(getaVecAtIndex(layerNum-2,activations), yG, tempCompOE1, layers[layerNum - 1]);
	definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(getnVecAtIndex(layerNum - 2, nodes), tempCompOE2, layers[layerNum - 1]);
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
		
		//tempCompBP1 = blas.gemmStandardTransposeAFromGPUMemRet(getwMatAtIndex(i + 1), geteVecAtIndex(i + 1, layerError),wMatDims[1],wMatDims[0],1);
		blas.gemmStandardTransposeAFromGPUMem(getwMatAtIndex(i + 1), geteVecAtIndex(i + 1, layerError), tempCompBP1, wMatDims[1], wMatDims[0], 1);
		
		definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(getnVecAtIndex(i,nodes), tempCompBP2, wMatDims[1]);
		definedGPUFunctions::multCompCWiseGPUMem(tempCompBP1, tempCompBP2, geteVecAtIndex(i,layerError), wMatDims[1]);
		free(wMatDims);
		cudaFree(tempCompBP1);
		cudaFree(tempCompBP2);
	}
	//perform gradient descent
	for (int i = 0; i < layerNum - 1; i++) {
		float* tempCompGD1;
		float* tempCompGD2;
		int* wMatDims = getwMatDimsAtIndex(i);
		cudaMalloc(&tempCompGD1, sizeof(float) * wMatDims[0] * wMatDims[1]);
		cudaMalloc(&tempCompGD2, sizeof(float) * wMatDims[0] * wMatDims[1]);
		if (i == 0) {
			//tempCompGD1 = blas.gemmStandardTransposeBFromGPUMemRet(geteVecAtIndex(i, layerError), xGOriginal, wMatDims[0], 1, wMatDims[1]);
			blas.gemmStandardTransposeBFromGPUMem(geteVecAtIndex(i, layerError), xGOriginal, tempCompGD1, wMatDims[0], 1, wMatDims[1]);
		} else {
			//tempCompGD1 = blas.gemmStandardTransposeBFromGPUMemRet(geteVecAtIndex(i, layerError), getaVecAtIndex(i-1,activations), wMatDims[0], 1, wMatDims[1]);
			blas.gemmStandardTransposeBFromGPUMem(geteVecAtIndex(i, layerError), getaVecAtIndex(i - 1, activations), tempCompGD1, wMatDims[0], 1, wMatDims[1]);
		}
		definedGPUFunctions::multCompCWiseGPUMemScalar(tempCompGD1, lRate, tempCompGD2, wMatDims[0] * wMatDims[1]);
		definedGPUFunctions::subMatCWiseGPUMem(getwMatAtIndex(i), tempCompGD2, getwMatAtIndex(i), wMatDims[0] * wMatDims[1]);
		cudaFree(tempCompGD1);
		cudaFree(tempCompGD2);
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



/*
float* FullyConnected::feedForwardOld(const float* x) {
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

void FullyConnected::backPropOld(const float* x, const float* y) {
	bool veryVerbose = false;
	if (veryVerbose) {
		std::cout << "Weight Matrix Before: " << std::endl;
		int rTotTemp = 0;
		for (int i = 0; i < layerNum - 1; i++) {
			float* tCPUVal = (float*)malloc(sizeof(float) * layers[i] * layers[i + 1]);
			cudaMemcpy(tCPUVal, (wMat + rTotTemp), sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToHost);
			std::cout << "Layer " << i << std::endl;
			blas.print_matrix(tCPUVal, layers[i + 1], layers[i]);
			std::cout << std::endl;
			rTotTemp += layers[i] * layers[i + 1];
		}
		std::cout << std::endl;
	}
	if (veryVerbose) {
		std::cout << "Bias Matrix Before: " << std::endl;
		int rTotTemp = 0;
		for (int i = 0; i < layerNum - 1; i++) {
			float* tCPUVal = (float*)malloc(sizeof(float) * layers[i + 1]);
			cudaMemcpy(tCPUVal, (bMat + rTotTemp), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
			std::cout << "Layer " << i << std::endl;
			blas.print_matrix(tCPUVal, layers[i + 1], 1);
			std::cout << std::endl;
			rTotTemp += layers[i + 1];
		}
		std::cout << std::endl;
	}
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
	float* xGOr; //stores the original value of xG and x, since xG is changed in the forward pass
	cudaMalloc(&xG, sizeof(float) * layers[0]);
	cudaMalloc(&xGOr, sizeof(float) * layers[0]);
	cudaMemcpy(xG, x, sizeof(float) * layers[0], cudaMemcpyHostToDevice);
	cudaMemcpy(xGOr, xG, sizeof(float) * layers[0], cudaMemcpyDeviceToDevice);
	//cudaMemcpy(xGOr, x, sizeof(float) * layers[0], cudaMemcpyHostToDevice);
	int wMOffset = 0;
	int bMOffset = 0;
	for (int i = 0; i < layerNum - 1; i++) {
		float* xGnext;
		cudaMalloc(&xGnext, sizeof(float) * layers[i + 1]);
		float* lyrWMat;
		float* lyrBMat;
		cudaMalloc(&lyrWMat, sizeof(float) * layers[i] * layers[i + 1]);
		cudaMalloc(&lyrBMat, sizeof(float) * layers[i + 1]);
		cudaMemcpy(lyrWMat, (wMat + wMOffset), sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToDevice); //step works correctly
		cudaMemcpy(lyrBMat, (bMat + bMOffset), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		wMOffset += (layers[i] * layers[i + 1]);
		bMOffset += (layers[i + 1]);
		//float* tCPUVal = (float*)malloc(sizeof(float) * layers[i]);
		//cudaMemcpy(tCPUVal, xG, sizeof(float) * layers[i], cudaMemcpyDeviceToHost);
		//blas.print_matrix(tCPUVal, layers[i], 1);
		blas.gemmStandardFromGPUMem(lyrWMat, xG, xGnext, layers[i + 1], layers[i], 1);
		blas.axpyStandardFromGPUMem(lyrBMat, xGnext, layers[i + 1]);
		//SET NODES[layer] TO THIS BEFORE ACTIVATION FUNCTION IS APPLIED
		cudaMemcpy((nodes + nodesOffset), xGnext, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
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
	//calculate output error - this step appears to be working correctly
	float* tCPUVal2;
	bool oeVerbose = false;

	//std::cout << "Calculating output error" << std::endl;

	float* tempComp;
	cudaMalloc(&tempComp, sizeof(float) * layers[layerNum - 1]);
	float* tempComp3; //tempComp3 stores activations[activations.length-1]
	cudaMalloc(&tempComp3, sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(tempComp3, (activations + lTot - layers[layerNum - 1]), sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToDevice);

	if (oeVerbose) { //tempComp3 is storing the correct value
		std::cout << "Output Activations=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
		cudaMemcpy(tCPUVal2, tempComp3, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[layerNum - 1], 1);
		std::cout << std::endl;
	}

	float* yG; //stores the "ideal" output of the network on the GPU memory 
	cudaMalloc(&yG, sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(yG, y, sizeof(float) * layers[layerNum - 1], cudaMemcpyHostToDevice);
	
	if (oeVerbose) { //yG is storing the correct value
		std::cout << "Y=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
		cudaMemcpy(tCPUVal2, yG, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[layerNum - 1], 1);
		std::cout << std::endl;
	}
	
	
	definedGPUFunctions::subMatCWiseGPUMem(tempComp3, yG, tempComp3, layers[layerNum - 1]);

	if (oeVerbose) {
		std::cout << "subMatCWiseGPUMem=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
		cudaMemcpy(tCPUVal2, tempComp3, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[layerNum - 1], 1);
		std::cout << std::endl;
	}

	cudaMemcpy(tempComp, (nodes + lTot - layers[layerNum - 1]), sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToDevice);
	definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(tempComp, tempComp, layers[layerNum - 1]); //tempComp now stores sigmoidPrime output of last layer	

	if (oeVerbose) {
		std::cout << "sigmoidPrime(nodes of output)=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
		cudaMemcpy(tCPUVal2, tempComp, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[layerNum - 1], 1);
		std::cout << std::endl;
	}

	definedGPUFunctions::multCompCWiseGPUMem(tempComp3, tempComp, tempComp3, layers[layerNum - 1]); //xG now stores the vector for the error on the output layer of the network

	if (oeVerbose) {
		std::cout << "multCompCWiseGPUMem=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
		cudaMemcpy(tCPUVal2, tempComp3, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[layerNum - 1], 1);
		std::cout << std::endl;
	}


	cudaMemcpy((layerError + lTot - layers[layerNum - 1]), tempComp3, sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToDevice);
	cudaFree(tempComp3);
	cudaFree(tempComp);
	cudaFree(yG);
	cudaFree(xG);

	
	std::cout << "Output Error=" << std::endl;
	tCPUVal2 = (float*)malloc(sizeof(float) * layers[layerNum - 1]);
	cudaMemcpy(tCPUVal2, layerError + lTot - layers[layerNum - 1], sizeof(float) * layers[layerNum - 1], cudaMemcpyDeviceToHost);
	blas.print_matrix(tCPUVal2, layers[layerNum - 1], 1);
	std::cout << std::endl;

	//backward pass
	bool backwardVerbose = false;
	if (backwardVerbose) {
		std::cout << "Starting backward pass" << std::endl;
	}
	wMOffset = 0;
	int lTot2 = 0;
	for (int i = 0; i < layerNum; i++) {
		lTot2 += layers[i];
	}

	int eMOffset = lTot - layers[layerNum - 1];
	//printf("lTot: %d, layers[lnum-1]: %d, layers[lnum-2]: %d, eMoffset: %d\n", lTot, layers[layerNum - 1], layers[layerNum - 2], eMOffset);
	//std::cout << "eMOffset: " << eMOffset << std::endl;
	int nMOffset = lTot - layers[layerNum - 1] - layers[layerNum-2];
	//std::cout << "nMOffset: " << eMOffset << std::endl;
	int lMOffset = lTot - layers[layerNum - 1] - layers[layerNum - 2];
	for (int i = 0; i < layerNum - 2; i++) {
		wMOffset += layers[i] * layers[i + 1];
	}
	if (backwardVerbose) {
		int tRunTotal = 0;
		std::cout << "Weight Matrix=" << std::endl;
		for (int i = 0; i < layerNum - 1; i++) {
			float* tCPUVal2 = (float*)malloc(sizeof(float) * layers[i] * layers[i + 1]);
			cudaMemcpy(tCPUVal2, (wMat + tRunTotal), sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal2, layers[i + 1], layers[i]);
			tRunTotal += layers[i] * layers[i + 1];
			cudaFree(tCPUVal2);
		}
		std::cout << std::endl;
	}
	//std::cout << "Backward Pass" << std::endl;
	for (int i = layerNum - 3; i >= 0; i--) {
		float* lyrWMat; //stores wMatrixArr[i+1]
		//cudaMalloc(&lyrWMat, sizeof(float) * layers[i] * layers[i + 1]);
		cudaMalloc(&lyrWMat, sizeof(float) * layers[i + 1] * layers[i + 2]); //lyrWMat is storing the correct value
		cudaMemcpy(lyrWMat, (wMat + wMOffset), sizeof(float) * layers[i+1] * layers[i + 2], cudaMemcpyDeviceToDevice);

		if (backwardVerbose) {
			std::cout << "lyrWMat=" << std::endl;
			float* tCPUVal2 = (float*)malloc(sizeof(float) * layers[i + 1] * layers[i + 2]);
			cudaMemcpy(tCPUVal2, lyrWMat, sizeof(float) * layers[i + 1] * layers[i + 2], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal2, layers[i + 2], layers[i + 1]);
			std::cout << std::endl;
			cudaFree(tCPUVal2);
		}

		//std::cout << "wMOffset before: " << wMOffset << std::endl;

		wMOffset -= (layers[i] * layers[i + 1]);

		//std::cout << "wMOffset after: " << wMOffset << std::endl;

		float* lyrErrT; //stores layerError[i+1]
		cudaMalloc(&lyrErrT, sizeof(float) * layers[i + 2]);
		cudaMemcpy(lyrErrT, (layerError + eMOffset), sizeof(float) * layers[i + 2], cudaMemcpyDeviceToDevice);
		//std::cout << "EMOffset before: " << eMOffset << std::endl;
		

		if (backwardVerbose) {
			std::cout << "lyrErrT=" << std::endl;
			tCPUVal2 = (float*)malloc(sizeof(float) * layers[i + 2]);
			cudaMemcpy(tCPUVal2, lyrErrT, sizeof(float) * layers[i + 2], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal2, layers[i + 2], 1);
			std::cout << std::endl;
			cudaFree(tCPUVal2);
		}

		float* lyrNodT; //stores nodes[i]
		cudaMalloc(&lyrNodT, sizeof(float) * layers[i+1]);
		cudaMemcpy(lyrNodT, (nodes + nMOffset), sizeof(float) * layers[i+1], cudaMemcpyDeviceToDevice);
		nMOffset -= layers[i+1];

		if (backwardVerbose) {
			std::cout << "lyrNodT=" << std::endl;
			tCPUVal2 = (float*)malloc(sizeof(float) * layers[i+1]);
			cudaMemcpy(tCPUVal2, lyrNodT, sizeof(float) * layers[i+1], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal2, layers[i+1], 1);
			std::cout << std::endl;
			cudaFree(tCPUVal2);
		}

		cudaMalloc(&tempComp, sizeof(float) * layers[i+1]); //tempComp stores af.evalPrimeMatrix(nodes[i])
		definedGPUFunctions::sigmoidPrimeMatCWiseGPUMem(lyrNodT, tempComp, layers[i+1]);

		if (backwardVerbose) {
			std::cout << "af.evalPrimeMatrix(nodes[i])= " << std::endl;
			tCPUVal2 = (float*)malloc(sizeof(float) * layers[i+1]);
			cudaMemcpy(tCPUVal2, tempComp, sizeof(float) * layers[i+1], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal2, layers[i+1], 1);
			std::cout << std::endl;
			cudaFree(tCPUVal2);
		}

		float* tempComp2; //stores (wMatrixArr[i+1].transpose().mmul(layerError[i+1]))
		cudaMalloc(&tempComp2, sizeof(float) * layers[i+1]);
		//blas.gemmStandardTransposeAFromGPUMem(lyrWMat, lyrErrT, tempComp2, layers[i], layers[i+1], 1);
		//blas.gemmStandardTransposeAFromGPUMem(lyrWMat, lyrErrT, tempComp2, layers[i], layers[i + 1], 1, layers[i + 1], layers[i+1], layers[i + 1]);
		
		//blas.gemmStandardTransposeAFromGPUMem(lyrWMat, lyrErrT, tempComp2, layers[i], layers[i + 1], 1, layers[i + 1], layers[i+1], layers[i + 1]);
		//blas.gemmStandardTransposeAFromGPUMem(lyrWMat, lyrErrT, tempComp2, layers[i+1], layers[i + 2], 1, layers[i + 2], layers[i + 2], layers[i + 2]);
		//blas.gemmStandardTransposeAFromGPUMem(lyrWMat, lyrErrT, tempComp2, layers[i + 1], layers[i + 2], 1, layers[i + 2], layers[i + 2], layers[i + 1]);
		blas.gemmStandardTransposeAFromGPUMem(lyrWMat, lyrErrT, tempComp2, layers[i + 1], layers[i + 2], 1, layers[i + 2], layers[i + 2], layers[i + 1]);

		//blas.gemmStandardFromGPUMem(lyrWMat, lyrErrT, tempComp2, layer[])

		if (backwardVerbose) {
			std::cout << "(wMatrixArr[i+1].transpose().mmul(layerError[i+1]))=" << std::endl;
			tCPUVal2 = (float*)malloc(sizeof(float) * layers[i + 1]);
			cudaMemcpy(tCPUVal2, tempComp2, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal2, layers[i + 1], 1);
			std::cout << std::endl;
			cudaFree(tCPUVal2);
		}

		//definedGPUFunctions::multCompCWiseGPUMem(tempComp2, tempComp, tempComp2,layers[i+1]); //tempComp now stores (wMatrixArr[i+1].transpose().mmul(layerError[i+1])).mul(af.evalPrimeMatrix(nodes[i]))
		
		//VERY VERY VERY IMPORTANT!!! I COMMENTED OUT THIS ABOVE LINE FOR DEBUGGING, IT SHOULD NOT BE COMMENTED!!!!! IT WILL CAUSE PROBLEMS!!!!!


		if (backwardVerbose) {
			std::cout << " (wMatrixArr[i+1].transpose().mmul(layerError[i+1])).mul(af.evalPrimeMatrix(nodes[i]))=" << std::endl;
			tCPUVal2 = (float*)malloc(sizeof(float) * layers[i + 1]);
			cudaMemcpy(tCPUVal2, tempComp2, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal2, layers[i + 1], 1);
			std::cout << std::endl;
			cudaFree(tCPUVal2);
		}
		eMOffset -= layers[i + 1];
		//std::cout << "eMOffset: " << eMOffset << std::endl;
		//cudaMemcpy((layerError + lMOffset), tempComp, sizeof(float) * layers[i], cudaMemcpyDeviceToDevice); this line might work, but lMOffset is never modified?
		cudaMemcpy((layerError + eMOffset), tempComp2, sizeof(float) * layers[i+1], cudaMemcpyDeviceToDevice); //eMOffset might not be the correct value
		//std::cout << "eMOffset: " << eMOffset << std::endl;
		
		if (backwardVerbose) {
			float* tCPUVal = (float*)malloc(sizeof(float) * layers[i + 1]);
			cudaMemcpy(tCPUVal, tempComp2, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
			blas.print_matrix(tCPUVal, layers[i + 1], 1);
		}

		//eMOffset -= layers[i + 2];

		cudaFree(lyrWMat);
		cudaFree(lyrErrT);
		cudaFree(lyrNodT);
		cudaFree(tempComp);
		cudaFree(tempComp2);
	}

	//perform gradient descent
	bool gradVerbose = false;
	wMOffset = 0;
	bMOffset = 0;
	eMOffset = 0;
	int aMOffset = 0;
	float* lyrErrT;
	float* lyrWMat;
	float* lyrBMat;
	cudaMalloc(&lyrErrT, sizeof(float) * layers[1]);
	cudaMalloc(&lyrWMat, sizeof(float) * layers[0] * layers[1]);
	cudaMalloc(&lyrBMat, sizeof(float) * layers[1]);
	cudaMemcpy(lyrErrT, (layerError + eMOffset), sizeof(float) * layers[1], cudaMemcpyDeviceToDevice);

	if (gradVerbose) {
		std::cout << "lyrErrT=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[1]);
		cudaMemcpy(tCPUVal2, lyrErrT, sizeof(float) * layers[1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[1], 1);
		std::cout << std::endl;
		cudaFree(tCPUVal2);
	}

	cudaMemcpy(lyrWMat, (wMat + wMOffset), sizeof(float) * layers[0] * layers[1], cudaMemcpyDeviceToDevice);
	
	if (gradVerbose) {
		std::cout << "lyrWMat=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[1]*layers[0]);
		cudaMemcpy(tCPUVal2, lyrWMat, sizeof(float) * layers[1]*layers[0], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[1], layers[0]);
		std::cout << std::endl;
		cudaFree(tCPUVal2);
	}

	cudaMemcpy(lyrBMat, (bMat + bMOffset), sizeof(float) * layers[1], cudaMemcpyDeviceToDevice);

	if (gradVerbose) {
		std::cout << "lyrBMat=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[1]);
		cudaMemcpy(tCPUVal2, lyrBMat, sizeof(float) * layers[1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[1], 1);
		std::cout << std::endl;
		cudaFree(tCPUVal2);
	}

	float* tempComp2;
	cudaMalloc(&tempComp, sizeof(float) * layers[0] * layers[1]); //tempComp stores layerError[0].mmul(x.transpose())
	cudaMalloc(&tempComp2, sizeof(float) * layers[0] * layers[1]); //tempComp2 stores (layerError[0].mmul(x.transpose())).mul(lRate)

	//blas.gemmStandardTransposeBFromGPUMem(lyrErrT, xGOr, tempComp, layers[1], 1, layers[0]);
	blas.gemmStandardTransposeBFromGPUMem(lyrErrT, xGOr, tempComp, layers[1], 1, layers[0],layers[1],layers[0],layers[1]);
	//blas.gemmStandardFromGPUMem(lyrErrT, xGOr, tempComp, layers[1], 1, layers[0]);

	if (gradVerbose) {
		std::cout << "xGOr=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[0]);
		cudaMemcpy(tCPUVal2, xGOr, sizeof(float) * layers[0], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[0], 1);
		std::cout << std::endl;
		cudaFree(tCPUVal2);
	}

	if (gradVerbose) {
		std::cout << "layerError[0].mmul(x.transpose())=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[1]*layers[0]);
		cudaMemcpy(tCPUVal2, tempComp, sizeof(float) * layers[1]*layers[0], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[1], layers[0]);
		std::cout << std::endl;
		cudaFree(tCPUVal2);
	}


	//float* tCPUVal = (float*)malloc(sizeof(float) * layers[1]);
	//cudaMemcpy(tCPUVal, lyrErrT, sizeof(float)* layers[1], cudaMemcpyDeviceToHost);
	//blas.print_matrix(tCPUVal, layers[1],1);

	definedGPUFunctions::multCompCWiseGPUMemScalar(tempComp, lRate, tempComp2,layers[0]*layers[1]);

	if (gradVerbose) {
		std::cout << "(layerError[0].mmul(x.transpose())).mul(lRate)=" << std::endl;
		tCPUVal2 = (float*)malloc(sizeof(float) * layers[1] * layers[0]);
		cudaMemcpy(tCPUVal2, tempComp2, sizeof(float) * layers[1] * layers[0], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal2, layers[1], layers[0]);
		std::cout << std::endl;
		cudaFree(tCPUVal2);
	}


	definedGPUFunctions::subMatCWiseGPUMem(lyrWMat, tempComp2, tempComp,layers[0]*layers[1]); //tempComp now stores wMatrixArr[0].sub((layerError[0].mmul(x.transpose())).mul(lRate));

	//cudaMalloc(&tempComp2, sizeof(float)* layers[0] * layers[1]);
	//blas.geamTransposeSingleGPUMem(tempComp, tempComp2, layers[1], layers[0]);

	cudaMemcpy(wMat, tempComp, sizeof(float) * layers[0] * layers[1], cudaMemcpyDeviceToDevice);

	std::cout << "Layer " << 0 << " Weight Changed: " << std::endl;
	float* tCPUVal = (float*)malloc(sizeof(float) * layers[0] * layers[1]);
	cudaMemcpy(tCPUVal, tempComp2, sizeof(float)* layers[0] * layers[1], cudaMemcpyDeviceToHost);
	blas.print_matrix(tCPUVal, layers[1], layers[0]);
	std::cout << std::endl;

	cudaFree(tempComp); //stores layerError[0].mul(lRate)
	cudaFree(tempComp2); //stores bMatrixArr[0].sub(layerError[0].mul(lRate))
	cudaMalloc(&tempComp, sizeof(float) * layers[1]);
	cudaMalloc(&tempComp2, sizeof(float) * layers[1]);
	definedGPUFunctions::multCompCWiseGPUMemScalar(lyrErrT, lRate, tempComp,layers[1]);
	definedGPUFunctions::subMatCWiseGPUMem(lyrBMat, tempComp, tempComp2,layers[1]);
	cudaMemcpy(bMat, tempComp2, sizeof(float) * layers[1], cudaMemcpyDeviceToDevice);
	eMOffset += layers[1];
	wMOffset += layers[0] * layers[1];
	bMOffset += layers[1];
	cudaFree(lyrErrT);
	cudaFree(lyrWMat);
	cudaFree(lyrBMat);
	cudaFree(tempComp);
	cudaFree(tempComp2);
	for (int i = 1; i < layerNum - 1; i++) {
		float* lyrErrT;
		float* lyrWMat;
		float* lyrBMat;
		float* lyrAMat;
		cudaMalloc(&lyrErrT, sizeof(float) * layers[i + 1]);
		cudaMalloc(&lyrWMat, sizeof(float) * layers[i] * layers[i + 1]);
		cudaMalloc(&lyrBMat, sizeof(float) * layers[i + 1]);
		cudaMalloc(&lyrAMat, sizeof(float) * layers[i]);
		cudaMemcpy(lyrErrT, (layerError + eMOffset), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaMemcpy(lyrWMat, (wMat + wMOffset), sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaMemcpy(lyrBMat, (bMat + bMOffset), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		cudaMemcpy(lyrAMat, (activations + aMOffset), sizeof(float) * layers[i], cudaMemcpyDeviceToDevice);

		//wMOffset += layers[i] * layers[i + 1];
		//bMOffset += layers[i + 1];

		float* tempComp2;
		cudaMalloc(&tempComp, sizeof(float) * layers[i] * layers[i + 1]); //tempComp stores layerError[0].mmul(x.transpose())
		cudaMalloc(&tempComp2, sizeof(float) * layers[i] * layers[i + 1]); //tempComp2 stores (layerError[0].mmul(x.transpose())).mul(lRate)

		//blas.gemmStandardTransposeBFromGPUMem(lyrErrT, lyrAMat, tempComp, layers[i+1], 1, layers[i],);
		blas.gemmStandardTransposeBFromGPUMem(lyrErrT, lyrAMat, tempComp, layers[i + 1], 1, layers[i],layers[i+1],layers[i],layers[i+1]);
		//blas.gemmStandardFromGPUMem(lyrErrT, lyrAMat, tempComp, layers[i + 1], layers[i], 1);


		definedGPUFunctions::multCompCWiseGPUMemScalar(tempComp, lRate, tempComp2,layers[i]*layers[i+1]);
		definedGPUFunctions::subMatCWiseGPUMem(lyrWMat, tempComp2, tempComp,layers[i]*layers[i+1]); //tempComp now stores wMatrixArr[0].sub((layerError[0].mmul(x.transpose())).mul(lRate));
		
		//cudaMalloc(&tempComp2, sizeof(float)* layers[i] * layers[i + 1]);
		//blas.geamTransposeSingleGPUMem(tempComp, tempComp2, layers[i+1], layers[i]);
		
		cudaMemcpy((wMat + wMOffset), tempComp, sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToDevice);
		

		std::cout << "Layer " << i << " Weight Changed: " << std::endl;
		float* tCPUVal = (float*)malloc(sizeof(float) * layers[i] * layers[i + 1]);
		cudaMemcpy(tCPUVal, tempComp2, sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal, layers[i + 1], layers[i]);
		std::cout << std::endl;
		
		
		
		cudaFree(tempComp); //stores layerError[0].mul(lRate)
		cudaFree(tempComp2); //stores bMatrixArr[0].sub(layerError[0].mul(lRate))
		cudaMalloc(&tempComp, sizeof(float) * layers[i + 1]);
		cudaMalloc(&tempComp2, sizeof(float) * layers[i + 1]);
		definedGPUFunctions::multCompCWiseGPUMemScalar(lyrErrT, lRate, tempComp,layers[i+1]);
		definedGPUFunctions::subMatCWiseGPUMem(lyrBMat, tempComp, tempComp2,layers[i+1]);

		

		std::cout << "Layer " << i << " Bias Changed: " << std::endl;
		tCPUVal = (float*)malloc(sizeof(float) * layers[i + 1]);
		cudaMemcpy(tCPUVal, tempComp2, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
		blas.print_matrix(tCPUVal, layers[i + 1], 1);
		std::cout << std::endl;
		

		
		//float* tCPUVal = (float*)malloc(sizeof(float) * layers[i + 1]);
		//cudaMemcpy(tCPUVal, lyrErrT, sizeof(float)* layers[i + 1], cudaMemcpyDeviceToHost);
		//blas.print_matrix(tCPUVal, layers[i + 1],1);
		//std::cout << i << std::endl;


		cudaMemcpy((bMat + bMOffset), tempComp2, sizeof(float) * layers[i + 1], cudaMemcpyDeviceToDevice);
		eMOffset += layers[i + 1];
		wMOffset += layers[i] * layers[i + 1];
		bMOffset += layers[i + 1];
		aMOffset += layers[i];
		cudaFree(lyrErrT);
		cudaFree(lyrWMat);
		cudaFree(lyrBMat);
		cudaFree(lyrAMat);
		cudaFree(tempComp);
		cudaFree(tempComp2);
	}
	veryVerbose = false;
	if (veryVerbose) {
		std::cout << "Weight matrix after: " << std::endl;
		int rTotTemp = 0;
		for (int i = 0; i < layerNum - 1; i++) {
			float* tCPUVal = (float*)malloc(sizeof(float) * layers[i] * layers[i + 1]);
			cudaMemcpy(tCPUVal, (wMat + rTotTemp), sizeof(float) * layers[i] * layers[i + 1], cudaMemcpyDeviceToHost);
			std::cout << "Layer " << i << std::endl;
			blas.print_matrix(tCPUVal, layers[i + 1], layers[i]);
			std::cout << std::endl;
			rTotTemp += layers[i] * layers[i + 1];
		}
		std::cout << std::endl;
	}

	bool finalVerbose = false;
	
	if (finalVerbose) {
		std::cout << "Printing nodes:" << std::endl;
		int nOff = 0;
		for (int i = 0; i < layerNum - 1; i++) {
			float* tCPUVal = (float*)malloc(sizeof(float) * layers[i + 1]);
			cudaMemcpy(tCPUVal, (nodes + nOff), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
			nOff += layers[i + 1];
			std::cout << "Layer " << i << std::endl;
			blas.print_matrix(tCPUVal, layers[i + 1], 1);
			std::cout << std::endl;
		}
		std::cout << "Printing activations:" << std::endl;
		int aOff = 0;
		for (int i = 0; i < layerNum - 1; i++) {
			float* tCPUVal = (float*)malloc(sizeof(float) * layers[i + 1]);
			cudaMemcpy(tCPUVal, (activations + aOff), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
			aOff += layers[i + 1];
			std::cout << "Layer " << i << std::endl;
			blas.print_matrix(tCPUVal, layers[i + 1], 1);
			std::cout << std::endl;
		}
		int eOff = 0;
		std::cout << "Printing errors:" << std::endl;
		for (int i = 0; i < layerNum - 1; i++) {
			float* tCPUVal = (float*)malloc(sizeof(float) * layers[i + 1]);
			cudaMemcpy(tCPUVal, (layerError + eOff), sizeof(float) * layers[i + 1], cudaMemcpyDeviceToHost);
			eOff += layers[i + 1];
			std::cout << "Layer " << i << std::endl;
			blas.print_matrix(tCPUVal, layers[i + 1], 1);
			std::cout << std::endl;
		}
	}
*/	


