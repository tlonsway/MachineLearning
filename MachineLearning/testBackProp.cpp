#include <iostream>
#include "layer.h"
#include <cuda_runtime.h>
#include "animations.h"

using namespace layer;

/*
int main() {
	gpuMath::blasOp blas;
	float in1[8] = { 1,2,3,4,5,6,7,8 };
	float in2[6] = { 1,2,3,4,5,6 };
	float* out = (float*)malloc(sizeof(float) * 4 * 3);
	blas.gemmStandardFromCPUMem(in1, in2, out, 4, 2, 3);
	gpuMath::blasOp::print_matrix(out,4,3);
}
*/



void main23() {
	
	int layerNum = 6;
	int* layers = (int*)malloc(sizeof(int) * layerNum);
	layers[0] = 2;
	layers[1] = 512;
	//layers[2] = 2;
	layers[2] = 256;
	layers[3] = 128;
	layers[4] = 64;
	layers[5] = 2;
	float lRate = .05;
	FullyConnected net(layers, layerNum, lRate);
	
	//train network
	int totalBackprops = 20000;
	for (int i = 0; i < totalBackprops; i++) {
		float* x = (float*)malloc(sizeof(float) * 2);
		float* y = (float*)malloc(sizeof(float) * 2);
		float x1 = ((float)(rand()%1000))/1000.0;
		float x2 = ((float)(rand()%1000))/1000.0;
		if (x1 > x2) {
			y[0] = 1.0;
			y[1] = 0.0;
		}
		else {
			y[0] = 0.0;
			y[1] = 1.0;
		}
		x[0] = x1;
		x[1] = x2;
		//std::cout << "x0: " << x[0] << std::endl;
		//std::cout << "x1: " << x[1] << std::endl;
		//std::cout << "y0: " << y[0] << std::endl;
		//std::cout << "y1: " << y[1] << std::endl;
		net.backProp(x, y);
		progress_bar(i,totalBackprops, "Training");
	}
	std::cout << std::endl;
	//test network
	int numTest = 10000;
	int numCorrect = 0;

	for (int i = 0; i < numTest; i++) {
		float* x = (float*)malloc(sizeof(float) * 2);
		float x1 = ((float)(rand() % 1000)) / 1000.0;
		float x2 = ((float)(rand() % 1000)) / 1000.0;
		x[0] = x1;
		x[1] = x2;
		float* output = net.feedForward(x);
		bool correct = false;
		//std::cout << "x0: " << x[0] << std::endl;
		//std::cout << "x1: " << x[1] << std::endl;
		//std::cout << "out0: " << output[0] << std::endl;
		//std::cout << "out1: " << output[1] << std::endl;
		if (x1 > x2 && output[0] > output[1]) {
			correct = true;
		}
		else if (x1 < x2 && output[0] < output[1]) {
			correct = true;
		}
		else {
			correct = false;
		}
		if (correct) {
			//std::cout << "Correct" << std::endl;
			numCorrect++;
		} else {
			//std::cout << "Wrong" << std::endl;
		}
		progress_bar(i, numTest, "Testing");
	}
	std::cout << std::endl;
	std::cout << "Percent correct: " << (100 * (float)numCorrect / (float)numTest) << "%" << std::endl;
}