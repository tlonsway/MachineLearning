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

	int layerNum = 4;
	int* layers = (int*)malloc(sizeof(int) * layerNum);
	layers[0] = 3;
	layers[1] = 5;
	layers[2] = 5;
	layers[3] = 1;
	float lRate = 5;
	FullyConnected net(layers, layerNum, lRate);
	for (int i = 0; i < 500; i++) {
		float* x = (float*)malloc(sizeof(float) * 3);
		float* y = (float*)malloc(sizeof(float));
		y[0] = 0;
		for (int j = 0; j < 3; j++) {
			x[j] = (float)(rand() % 1000) / 1000.0;
			y[0] = y[0]+x[j];
		}
		y[0] = y[0] / 3;
		net.backProp(x, y);
	}
	std::cout << std::endl;
	for (int i = 0; i < 100; i++) {
		float* x = (float*)malloc(sizeof(float) * 3);
		float* y = (float*)malloc(sizeof(float));
		y[0] = 0;
		for (int j = 0; j < 3; j++) {
			x[j] = (float)(rand() % 1000) / 1000.0;
			y[0] = y[0] + x[j];
		}
		y[0] = y[0] / 3;
		float* outP = (float*)malloc(sizeof(float));
		outP = net.feedForward(x);
		std::cout << "Net out: " << outP[0] << " Real Average: " << y[0] << std::endl;
	}





	//double[][] wMat1 = new double[][] {{0.2,0.3},{0.4,0.5},{0.6,0.7}};
	//double[][] wMat2 = new double[][]{ {0.8,0.9,1.0},{1.1,1.2,1.3} };
	//double[] bMat1 = new double[] {1,2,3};
	//double[] bMat2 = new double[] {4, 5};
	/*float* wMat = (float*)malloc(sizeof(float) * 12);
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
		std::cout << "Output 1 " << i << " = " << y[i] << std::endl;
	}
	xt[0] = 0.9; xt[1] = 0.1;
	yt[0] = 1; yt[1] = 0;
	net.backProp(xt, yt);
	testX[0] = 0.7;
	testX[1] = 0.4;
	y = net.feedForward(testX);
	for (int i = 0; i < 2; i++) {
		std::cout << "Output 2 " << i << " = " << y[i] << std::endl;
	}
	

	exit(26);
	
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
		std::cout << "x0: " << x[0] << std::endl;
		std::cout << "x1: " << x[1] << std::endl;
		std::cout << "y0: " << y[0] << std::endl;
		std::cout << "y1: " << y[1] << std::endl;
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
		std::cout << "x0: " << x[0] << std::endl;
		std::cout << "x1: " << x[1] << std::endl;
		std::cout << "out0: " << output[0] << std::endl;
		std::cout << "out1: " << output[1] << std::endl;
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
			std::cout << "Correct" << std::endl;
			numCorrect++;
		} else {
			std::cout << "Wrong" << std::endl;
		}
	}
	std::cout << "Percent correct: " << (100 * (float)numCorrect / (float)numTest) << "%" << std::endl;*/
}