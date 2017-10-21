#include "SigmoidFunction.h"
#include <complex>

double SigmoidFunction::forward(double input)
{
	return 1.0 / (1.0 + std::exp(-input));
}

double SigmoidFunction::backward(double output, double input, double valueFromOut)
{
	return valueFromOut * (1.0 - output) * output;
}
