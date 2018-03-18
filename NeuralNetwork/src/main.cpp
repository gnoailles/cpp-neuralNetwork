#include <iostream>
#include "NeuralNetwork.h"
#include <array>


int main(int argc, char* argv[])
{
	srand(static_cast<unsigned>(time(nullptr)));
	arma::arma_rng::set_seed_random();

	NeuralNetwork nn = NeuralNetwork::LoadNew("network_saves/save01.bin");

	std::array<std::array<arma::vec, 2>, 4> trainingData;
	trainingData[0][0] = { 0, 1 };
	trainingData[0][1] = { 1 };

	trainingData[1][0] = { 1, 0 };
	trainingData[1][1] = { 1 };

	trainingData[2][0] = { 0, 0 };
	trainingData[2][1] = { 0 };

	trainingData[3][0] = { 1, 1 };
	trainingData[3][1] = { 0 };

	for (int i = 0; i < 500000; ++i)
	{
		const int data =rand() % trainingData.size();
		nn.Train(trainingData[data][0], trainingData[data][1]);
	}

	nn.Save("network_saves/save01.bin");


	arma::vec guess = nn.Guess({ 0,1 });
	guess.print();
	guess = nn.Guess({ 1,0 });
	guess.print();
	guess = nn.Guess({ 0,0 });
	guess.print();
	guess = nn.Guess({ 1,1 });
	guess.print();

	system("pause");
	return 0;
}
