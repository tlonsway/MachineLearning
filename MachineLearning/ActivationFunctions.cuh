#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <string>

class ActivationFunction {
public:
	ActivationFunction();
	std::string getName();
	virtual void eval(float* x, float* y, int len) = 0;
	virtual void evalPrime(float* x, float* y, int len) = 0;
	std::string name;
};
class Sigmoid : public ActivationFunction {
public:
	Sigmoid();
	void eval(float* x, float* y, int len);
	void evalPrime(float* x, float* y, int len);
};
class ReLu : public ActivationFunction {
public:
	ReLu();
	void eval(float* x, float* y, int len);
	void evalPrime(float* x, float* y, int len);
};
class HypTan : public ActivationFunction {
public:
	HypTan();
	void eval(float* x, float* y, int len);
	void evalPrime(float* x, float* y, int len);
};