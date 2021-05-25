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
	
	
	//double[][] wMat1 = new double[][] {{0.2,0.3},{0.4,0.5},{0.6,0.7}};
	//double[][] wMat2 = new double[][]{ {0.8,0.9,1.0},{1.1,1.2,1.3} };
	//double[] bMat1 = new double[] {1,2,3};
	//double[] bMat2 = new double[] {4, 5};
	float* wMat = (float*)malloc(sizeof(float) * 12);
	float* bMat = (float*)malloc(sizeof(float) * 5);
	wMat[0] = 0.2; wMat[1] = 0.3; wMat[2] = 0.4; wMat[3] = 0.5; wMat[4] = 0.6; wMat[5] = 0.7; wMat[6] = 0.8; wMat[7] = 0.9; wMat[8] = 1.0; wMat[9] = 1.1; wMat[10] = 1.2; wMat[11] = 1.3;
	bMat[0] = 1; bMat[1] = 2; bMat[2] = 3; bMat[3] = 4; bMat[4] = 5;
	net.setWMat(wMat);
	net.setBMat(bMat);
	float* xt = (float*)malloc(sizeof(float) * 2);
	float* yt = (float*)malloc(sizeof(float) * 2);
	xt[0] = 0.3; xt[1] = 0.7;
	yt[0] = 0; yt[1] = 1;
	net.backProp(xt, yt);
	float* testX = (float*)malloc(sizeof(float) * 2);
	testX[0] = 0.1;
	testX[1] = 0.8;
	float* y = net.feedForward(testX);
	for (int i = 0; i < 2; i++) {
		std::cout << "Output " << i << " = " << y[i] << std::endl;
	}
	exit(-7);
	
	//train network
	for (int i = 0; i < 1000; i++) {
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