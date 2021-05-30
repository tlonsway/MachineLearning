#pragma once

namespace definedGPUFunctions {
	void sigmoidMatCWiseGPUMem(float* A, float* B, int len);
	void sigmoidPrimeMatCWiseGPUMem(float* A, float* B, int len);
	void reLuMatCWiseGPUMem(float* A, float* B, int len);
	void reLuPrimeMatCWiseGPUMem(float* A, float* B, int len);
	void addMatCWiseGPUMem(float* a, float* b, float* c, int len);
	void subMatCWiseGPUMem(float* a, float* b, float* c, int len);
	void multCompCWiseGPUMem(float* a, float* b, float* c, int len);
	void multCompCWiseGPUMemScalar(float* a, float f, float* c, int len);
}