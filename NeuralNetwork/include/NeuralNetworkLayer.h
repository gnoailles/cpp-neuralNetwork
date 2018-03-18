#pragma once
#include <armadillo>

using ActivationFunction = std::function<void(arma::mat::elem_type&)>;

class NeuralNetworkLayer
{
private:
	float m_learningRate;

	unsigned int m_inputCount;
	unsigned int m_nodesCount;

	ActivationFunction m_activationFunction;

	arma::vec m_bias;
	arma::vec m_lastOutput;
	arma::mat m_weights;

public:
	NeuralNetworkLayer(const unsigned p_inputCount, const unsigned p_nodesCount, ActivationFunction& p_activationFunction,
	                   const float p_learningRate = 0.01f);

	NeuralNetworkLayer(const NeuralNetworkLayer& p_other);

	arma::vec FeedForward(const arma::vec& p_inputs);

	arma::vec Train(const arma::vec& p_inputs, const arma::vec& p_errors);

	void Save(std::ofstream& p_file);

	void Load(std::ifstream& p_file);

	static NeuralNetworkLayer LoadNew(std::ifstream& p_file);

	inline const arma::vec& GetLastGuess() const
	{
		return m_lastOutput;
	}
};
