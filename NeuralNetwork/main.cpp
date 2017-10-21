#include "src/SigmoidFunction.h"
#include "src/NeuralNetwork.h"

#include <cstdio>
#include <string> // std::string
#include <iostream> // cout とか
#include <random> // std::random
#include <chrono> // 時間計測

using namespace std;

int main()
{
	// 活性化関数の設定
	SigmoidFunction sigmoid;
	ActivationFunction& function = sigmoid;
	// ネットワークの設定
	int layerNum = 9;
	int *nodeNum = new int[layerNum] { 2, 3, 4, 5, 6, 5, 4, 3, 2 };
	NeuralNetwork neuralNetwork(
		layerNum,
		*nodeNum,
		function
	);
	delete nodeNum;

	// random
	random_device rnd;
	mt19937 mt(rnd());
	uniform_real_distribution<> realRand01(0.0, 1.0);

	// バイアス項と重みの設定
	int biasNum = neuralNetwork.getBiasNum();
	int weightNum = neuralNetwork.getWeightNum();
	double *bias = new double[biasNum];
	for (int i = 0; i < biasNum; ++i)
	{
		bias[i] = 0.1 * i;
	}
	double *weight = new double[weightNum];
	for (int i = 0; i < weightNum; ++i)
	{
		weight[i] = 0.01 * i;
	}
	neuralNetwork.setBias(*bias);
	neuralNetwork.setWeight(*weight);
	delete bias;
	delete weight;

	// 入力の設定
	double input[] = { 1.2, 1.4 };

	double* output;
	cout << "test: ";
	output = &(neuralNetwork.getOutPutValue(*input));
	cout << "output: ";
	for (int i = 0; i < 2; ++i) {
		cout << output[i] << ",";
	}
	cout << endl;

	// バックプロパゲーションの初期設定
	double *backInit = new double[2]{ 1.0, 1.0 };
	cout << "start" << endl;
	chrono::system_clock::time_point start, end;
	start = chrono::system_clock::now();
	for (int num = 0; num < 100000; ++num)
	{
		input[0] = realRand01(mt);
		input[1] = realRand01(mt);
		output = &(neuralNetwork.getOutPutValue(*input));
		neuralNetwork.doBackPropagation(*backInit);

	}
	end = chrono::system_clock::now();
	double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "time: " << time << "[ms]" << endl;
	cout << "end" << endl;
	
/*
	double *backInit = new double[2]{ 1.0, 1.0 };
	neuralNetwork.doBackPropagation(*backInit);

	double* gradient_weight;
	gradient_weight = &(neuralNetwork.getGradientWeight());
	cout << "gradient_weight" << endl;
	for (int i = 0; i < weightNum; ++i) {
		cout << "" << i << ": " << gradient_weight[i] << endl;
	}
	double* gradient_bias;
	gradient_bias = &(neuralNetwork.getGradientBias());
	cout << "gradient_bias" << endl;
	for (int i = 0; i < biasNum; ++i) {
		cout << "" << i << ": " << gradient_bias[i] << endl;
	}
*/
	return 0;
}