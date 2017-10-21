#pragma once
#include "ActivationFunction.h"

class NeuralNetwork
{
private:
	int fLayerNum;
	int* fNodeNum;
	int fWeightNum;
	double* fWeight;
	int fBiasNum;
	double* fBias;
	ActivationFunction* fActivationFunction;

	int fInputValueNum;
	double* fInputValue;
	int fOutputValueNum;
	double* fOutputValue;
	int fInputToOutputValueNum;
	double* fInputToOutputValue;

	double* fGradientWeight;
	double* fGradientBias;
	int fBackPropagationValueNum;
	double* fBackPropagationValue;
	int fBackPropagationValueFromOutNum;
	double* fBackPropagationValueFromOut;

	int fNetworkOutputValueNum;
	double* fNetworkOutputValue;
public:
	NeuralNetwork(int layerNum, int& nodeNum, ActivationFunction& function);
	~NeuralNetwork();
	void setWeight(double& weight);
	void setBias(double& bias);
	int getBiasNum();
	int getWeightNum();
	double& getOutPutValue(double& input);
	void doBackPropagation(double& input);
	double& getGradientWeight();
	double& getGradientBias();

};

