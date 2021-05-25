#pragma once

namespace definedGPUFunctions {
	void sigmoidMatCWiseGPUMem(float* A, float* B, int len);
	void sigmoidPrimeMatCWiseGPUMem(float* A, float* B, int len);
	void addMatCWiseGPUMem(float* a, float* b, float* c);
	void subMatCWiseGPUMem(float* a, float* b, float* c);
	void multCompCWiseGPUMem(float* a, float* b, float* c);
	void multCompCWiseGPUMemScalar(float* a, float f, float* c);
}