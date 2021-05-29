#pragma once
#include "gpuMath.h"
#include <string>

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
		void end();
		float* feedForward(const float* x);
		void backProp(const float* x, const float* y);
		float* feedForwardOld(const float* x);
		void backPropOld(const float* x, const float* y);
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
