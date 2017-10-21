#pragma once
#include "ActivationFunction.h"
class SigmoidFunction :
	public ActivationFunction
{
private:
public:
	virtual double forward(double input);
	virtual double backward(double output, double input, double valueFromOut);
};
