#pragma once

namespace definedGPUFunctions {
	void sigmoidMatCWiseGPUMem(float* A, float* B, int len);
	void sigmoidPrimeMatCWiseGPUMem(float* A, float* B, int len);
}