#include "NeuralNetwork.h"
#include "ActivationFunction.h"


NeuralNetwork::NeuralNetwork(int layerNum, int& nodeNum, ActivationFunction& function)
{
	fLayerNum = layerNum;
	fNodeNum = new int[fLayerNum];
	for (int i = 0; i < fLayerNum; ++i)
	{
		fNodeNum[i] = (&nodeNum)[i];
	}
	int wDim = 0, bDim = 0, iDim = 0;
	for (int i = 0; i < fLayerNum - 1; ++i)
	{
		wDim += fNodeNum[i] * fNodeNum[i + 1];
		bDim += fNodeNum[i + 1];
		iDim += fNodeNum[i];
	}
	fWeightNum = wDim;
	fWeight = new double[fWeightNum];
	fBiasNum = bDim;
	fBias = new double[fBiasNum];
	fActivationFunction = &function;

	fInputValueNum = iDim;
	fInputValue = new double[fInputValueNum];
	fOutputValueNum = bDim;
	fOutputValue = new double[fOutputValueNum];
	fInputToOutputValueNum = fOutputValueNum;
	fInputToOutputValue = new double[fInputToOutputValueNum];

	fGradientWeight = new double[fWeightNum];
	fGradientBias = new double[fBiasNum];
	fBackPropagationValueNum = fOutputValueNum;
	fBackPropagationValue = new double[fBackPropagationValueNum];
	fBackPropagationValueFromOutNum = fOutputValueNum;
	fBackPropagationValueFromOut = new double[fBackPropagationValueFromOutNum];

	fNetworkOutputValueNum = fNodeNum[fLayerNum - 1];
	fNetworkOutputValue = new double[fNetworkOutputValueNum];
}

NeuralNetwork::~NeuralNetwork()
{
	delete fNodeNum;
	delete fWeight;
	delete fBias;
	delete fInputValue;
	delete fOutputValue;
	delete fInputToOutputValue;
	delete fGradientWeight;
	delete fGradientBias;
	delete fBackPropagationValue;
	delete fBackPropagationValueFromOut;
	delete fNetworkOutputValue;
}

void NeuralNetwork::setWeight(double& weight)
{
	for (int i = 0; i < fWeightNum; ++i)
	{
		fWeight[i] = (&weight)[i];
	}
}

void NeuralNetwork::setBias(double& bias)
{
	for (int i = 0; i < fBiasNum; ++i)
	{
		fBias[i] = (&bias)[i];
	}
}

int NeuralNetwork::getBiasNum()
{
	return fBiasNum;
}

int NeuralNetwork::getWeightNum()
{
	return fWeightNum;
}

double& NeuralNetwork::getOutPutValue(double& input)
{
	int setOutputIdx = 0, setInputIdx = 0, getInputIdx = 0, weightIdx = 0, networkOutputIdx = 0;
	for (int layer = 0; layer < fLayerNum - 1; ++layer)
	{
		if (layer == 0)
		{
			//set inputValue
			for (int in = 0; in < fNodeNum[0]; ++in)
			{
				fInputValue[setInputIdx] = (&input)[in];
				++setInputIdx;
			}
		}
		// set outputValue
		for (int out = 0; out < fNodeNum[layer + 1]; ++out)
		{
			fInputToOutputValue[setOutputIdx] = fBias[setOutputIdx];
			for (int in = 0; in < fNodeNum[layer]; ++in)
			{
				fInputToOutputValue[setOutputIdx] += fInputValue[getInputIdx + in] * fWeight[weightIdx];
				++weightIdx;
			}

			fOutputValue[setOutputIdx] = fActivationFunction->forward(fInputToOutputValue[setOutputIdx]);
			// set inputValue
			if (layer + 1 < fLayerNum - 1)
			{
				fInputValue[setInputIdx] = fOutputValue[setOutputIdx];
				++setInputIdx;
			}
			else if (layer + 1 == fLayerNum - 1)
			{
				fNetworkOutputValue[networkOutputIdx] = fOutputValue[setOutputIdx];
				++networkOutputIdx;
			}
			++setOutputIdx;
		}
		getInputIdx += fNodeNum[layer];
	}
	return *fNetworkOutputValue;
}

void NeuralNetwork::doBackPropagation(double& input)
{
	int setBackPropIdx = fBackPropagationValueNum - 1;
	int getBackPropIdx = fBackPropagationValueNum - 1;
	int setBackPropFromOutIdx = fBackPropagationValueFromOutNum - 1;
	int getBackPropFromOutIdx = fBackPropagationValueFromOutNum - 1;
	int getInputIdx = fInputToOutputValueNum - 1;
	int setBiasIdx = fBiasNum - 1;
	int setWeightIdx = fWeightNum - 1;
	int getOutputIdx = fOutputValueNum - 1;
	int getInputForWeightIdx = fInputValueNum - 1;
	int getWeightIdx = fWeightNum - 1;

	// set fBackPropagetionValueFromOutNum
	for (int i = fNodeNum[fLayerNum - 1] - 1; i >= 0; --i ) {
		fBackPropagationValueFromOut[setBackPropFromOutIdx] = (&input)[i];
		--setBackPropFromOutIdx;
	}
	for (int layer = fLayerNum - 1; layer >= 1; --layer)
	{
		for (int out = fNodeNum[layer] - 1; out >= 0; --out)
		{
			double tmp = fInputToOutputValue[getInputIdx];
			--getInputIdx;
			double outputValue = fOutputValue[getOutputIdx];
			--getOutputIdx;
			double backFromOut = fBackPropagationValueFromOut[getBackPropFromOutIdx];
			--getBackPropFromOutIdx;
			fBackPropagationValue[setBackPropIdx] = fActivationFunction->backward(outputValue, tmp, backFromOut);
			double backPropValue = fBackPropagationValue[setBackPropIdx];
			--setBackPropIdx;
			// bias gradient
			fGradientBias[setBiasIdx] = backPropValue;
			--setBiasIdx;
			// weight gradient
			int inIdx = 0;
			for (int in = fNodeNum[layer - 1] - 1; in >= 0; --in)
			{
				fGradientWeight[setWeightIdx] = backPropValue * fInputValue[getInputForWeightIdx + inIdx];
				--setWeightIdx;
				--inIdx;
			}
		}
		getInputForWeightIdx -= fNodeNum[layer - 1];
		if (layer > 1) {
			// set fBackPropagationValueFromOut
			int inIdx = 0;
			for (int in = fNodeNum[layer - 1] - 1; in >= 0; --in)
			{
				fBackPropagationValueFromOut[setBackPropFromOutIdx + inIdx] = 0;
				--inIdx;
			}
			int outIdx = 0;
			for (int out = fNodeNum[layer] - 1; out >= 0; --out)
			{
				inIdx = 0;
				for (int in = fNodeNum[layer - 1] - 1; in >= 0; --in)
				{
					fBackPropagationValueFromOut[setBackPropFromOutIdx + inIdx] +=
						fWeight[getWeightIdx] * fBackPropagationValue[getBackPropIdx + outIdx];
					--getWeightIdx;
					--inIdx;
				}
				--outIdx;
			}
			setBackPropFromOutIdx -= fNodeNum[layer - 1];
			getBackPropIdx -= fNodeNum[layer];
		}
	}
	return;
}

double& NeuralNetwork::getGradientWeight() {
	return *fGradientWeight;
}

double& NeuralNetwork::getGradientBias() {
	return *fGradientBias;
}

