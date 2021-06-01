#pragma once
#include "gpuMath.h"
#include <string>
#include "ActivationFunctions.cuh";

namespace layer {
	class FullyConnected {
	public:
		float* wMat;
		float* bMat;
		int* layers;
		int layerNum;
		float lRate;
		gpuMath::blasOp blas;
		ActivationFunction *af;
		FullyConnected(int* lys, int lysN, float lr, ActivationFunction *activationFunction);
		void end();
		float* feedForward(const float* x);
		void backProp(const float* x, const float* y);
		void backPropGAN(const float* x, const float error);
		float* getwMatAtIndex(int i);
		int* getwMatDimsAtIndex(int i);
		float* getbMatAtIndex(int i);
		int* getbMatDimsAtIndex(int i);
		float* getnVecAtIndex(int i, float* nodeVec);
		int* getnVecDimsAtIndex(int i);
		float* getaVecAtIndex(int i, float* activationVec);
		int* getaVecDimsAtIndex(int i);
		float* geteVecAtIndex(int i, float* errorVec);
		int* geteVecDimsAtIndex(int i);
		void vbOut(std::string s);
	};
}
