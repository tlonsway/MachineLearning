#include <iostream>
#include "layer.h";


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




int main() {
	int* layers = (int*)malloc(sizeof(int) * 3);
	layers[0] = 2;
	layers[1] = 3;
	layers[2] = 2;
	float lRate = 0.1;
	int layerNum = 3;
	FullyConnected net(layers, layerNum, lRate);
	//train network
	for (int i = 0; i < 10000; i++) {
		float* x = (float*)malloc(sizeof(float) * 2);
		float* y = (float*)malloc(sizeof(float) * 2);
		float x1 = ((float)(rand()%100))/100.0;
		float x2 = ((float)(rand()%100))/100.0;
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
		net.backProp(x, y);
	}
	//test network
	int numTest = 1000;
	int numCorrect = 0;

	for (int i = 0; i < numTest; i++) {
		float* x = (float*)malloc(sizeof(float) * 2);
		float x1 = ((float)(rand() % 100)) / 100.0;
		float x2 = ((float)(rand() % 100)) / 100.0;
		x[0] = x1;
		x[1] = x2;
		float* output = net.feedForward(x);
		bool correct = false;
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
			numCorrect++;
		}
	}
	std::cout << "Percent correct: " << (100 * (float)numCorrect / (float)numTest) << "%" << std::endl;
}