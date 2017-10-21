#pragma once
class ActivationFunction
{
private:
public:
	virtual double forward(double input) = 0;
	virtual double backward(double output, double input, double valueFromOut) = 0;
};
