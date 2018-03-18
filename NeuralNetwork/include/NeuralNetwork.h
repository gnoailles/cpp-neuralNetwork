#pragma once
#include <armadillo>
#include <vector>
#include "NeuralNetworkLayer.h"

class NeuralNetwork
{
private:

	float m_learningRate;
	ActivationFunction m_activationFunction;

	unsigned int m_inputCount;
	std::vector<NeuralNetworkLayer> m_hiddenLayers;
	NeuralNetworkLayer m_outputLayer;

public:

	NeuralNetwork(const unsigned int p_inputCount, const arma::uvec p_hiddenCounts, const unsigned int p_outputCount, const float p_learningRate = 0.01f);
	NeuralNetwork(const NeuralNetwork& p_other);
	~NeuralNetwork();

	arma::vec Guess(const arma::vec& p_inputs);
	void Train(const arma::vec& p_inputs, const arma::vec& p_targets);

	inline void SetLearningRate(const float p_learningRate)
	{
		if (p_learningRate > 0.0f)
			this->m_learningRate = p_learningRate;
	}

	inline void SetActivationFunction(ActivationFunction& p_activationFunction)
	{
		if (p_activationFunction)
		{
			this->m_activationFunction = p_activationFunction;
		}
	}

	void Save(const std::string&& p_filepath);
	void Load(const std::string&& p_filepath);
	static NeuralNetwork LoadNew(const std::string&& p_filepath);

	static void Sigmoid(arma::mat::elem_type& n);
};
