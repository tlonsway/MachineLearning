#include <iostream>
#include "layer.h";

using namespace layer;


void main233() {
	int* layers = (int*)malloc(sizeof(int) * 3);
	layers[0] = 2;
	layers[1] = 5;
	layers[2] = 3;
	float lRate = 0.1;
	int layerNum = 3;
	FullyConnected net(layers, layerNum, lRate);
	float* sampleIn = (float*)malloc(sizeof(float) * 2);
	sampleIn[0] = 3.6;
	sampleIn[1] = -1.7;
	float* out = net.feedForward(sampleIn);
	for (int i = 0; i < layers[layerNum - 1]; i++) {
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
}
