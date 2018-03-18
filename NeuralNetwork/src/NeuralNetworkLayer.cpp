#include "NeuralNetworkLayer.h"
#include "NeuralNetwork.h"

NeuralNetworkLayer::NeuralNetworkLayer(const unsigned p_inputCount, const unsigned p_nodesCount,
                                       ActivationFunction& p_activationFunction, const float p_learningRate):
	m_learningRate(p_learningRate),
	m_inputCount(p_inputCount),
	m_nodesCount(p_nodesCount),
	m_activationFunction(p_activationFunction),
	m_bias(arma::randu<arma::vec>(m_nodesCount)),
	m_weights(arma::randu<arma::mat>(m_nodesCount, m_inputCount))
{
	m_lastOutput.zeros();
	m_weights *= 2;
	m_weights -= 1;

	m_bias *= 2;
	m_bias -= 1;
}

NeuralNetworkLayer::NeuralNetworkLayer(const NeuralNetworkLayer& p_other):
	m_learningRate(p_other.m_learningRate),
	m_inputCount(p_other.m_inputCount),
	m_nodesCount(p_other.m_nodesCount),
	m_activationFunction(
		p_other.m_activationFunction),
	m_bias(p_other.m_bias),
	m_lastOutput(p_other.m_lastOutput),
	m_weights(p_other.m_weights)
{
}

arma::vec NeuralNetworkLayer::FeedForward(const arma::vec& p_inputs)
{
	arma::mat output = m_weights * p_inputs;
	output += m_bias;
	output.for_each(m_activationFunction);
	m_lastOutput = arma::vectorise(output);
	return output;
}

arma::vec NeuralNetworkLayer::Train(const arma::vec& p_inputs, const arma::vec& p_errors)
{
	arma::mat gradients = m_lastOutput % (1 - m_lastOutput);
	gradients %= p_errors;
	gradients *= m_learningRate;

	const arma::mat weights_dt = gradients * arma::trans(p_inputs);

	m_weights += weights_dt;
	m_bias += gradients;

	return arma::trans(m_weights) * p_errors;
}

void NeuralNetworkLayer::Save(std::ofstream& p_file)
{
	if (p_file.is_open())
	{
		p_file.write((char*)&m_learningRate, sizeof(m_learningRate));
		p_file.write((char*)&m_inputCount, sizeof(m_inputCount));
		p_file.write((char*)&m_nodesCount, sizeof(m_nodesCount));

		m_bias.save(p_file, arma::arma_binary);
		m_lastOutput.save(p_file, arma::arma_binary);
		m_weights.save(p_file, arma::arma_binary);
	}
}

void NeuralNetworkLayer::Load(std::ifstream& p_file)
{
	if (p_file.is_open())
	{
		p_file.read((char*)&m_learningRate, sizeof(m_learningRate));
		p_file.read((char*)&m_inputCount, sizeof(m_inputCount));
		p_file.read((char*)&m_nodesCount, sizeof(m_nodesCount));

		m_bias.load(p_file, arma::arma_binary);
		m_lastOutput.load(p_file, arma::arma_binary);
		m_weights.load(p_file, arma::arma_binary);
	}
}

NeuralNetworkLayer NeuralNetworkLayer::LoadNew(std::ifstream& p_file)
{
	float learningRate;

	unsigned int inputCount;
	unsigned int nodesCount;

	ActivationFunction activationFunction = NeuralNetwork::Sigmoid;

	if (p_file.is_open())
	{
		p_file.read((char*)&learningRate, sizeof(m_learningRate));
		p_file.read((char*)&inputCount, sizeof(m_inputCount));
		p_file.read((char*)&nodesCount, sizeof(m_nodesCount));

		NeuralNetworkLayer layer(inputCount, nodesCount, activationFunction, learningRate);

		layer.m_bias.load(p_file, arma::arma_binary);
		layer.m_lastOutput.load(p_file, arma::arma_binary);
		layer.m_weights.load(p_file, arma::arma_binary);
		return layer;
	}
	throw std::ifstream::badbit;
}
