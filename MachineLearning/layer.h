#pragma once
#include "gpuMath.h";

namespace layer {
	class FullyConnected {
	public:
		float* wMat;
		float* bMat;
		int* layers;
		int layerNum;
		float lRate;
		gpuMath::blasOp blas;
		FullyConnected(int* lys, int lysN, float lr);
		void setWMat(float* nwMat);
		void setBMat(float* nbMat);
		float* feedForward(const float* x);
		void backProp(const float* x, const float* y);
	};
}
