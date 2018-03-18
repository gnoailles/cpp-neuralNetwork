#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const unsigned int p_inputCount, const arma::uvec p_hiddenCounts, const unsigned int p_outputCount, const float p_learningRate) :
	m_learningRate(p_learningRate),
	m_activationFunction(Sigmoid), 
	m_inputCount(p_inputCount),
	m_outputLayer(p_hiddenCounts(p_hiddenCounts.n_elem-1), p_outputCount, m_activationFunction, m_learningRate)
{
	m_hiddenLayers.emplace_back(m_inputCount, p_hiddenCounts(0), m_activationFunction, m_learningRate);
	for(arma::uword i = 1; i < p_hiddenCounts.n_elem; ++i )
	{
			m_hiddenLayers.emplace_back(p_hiddenCounts(i-1), p_hiddenCounts(i), m_activationFunction, m_learningRate);
	}
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& p_other):
	m_learningRate(p_other.m_learningRate),
	m_activationFunction(p_other.m_activationFunction),
	m_inputCount(p_other.m_inputCount),
	m_outputLayer(p_other.m_outputLayer),
	m_hiddenLayers(p_other.m_hiddenLayers)
{
}

NeuralNetwork::~NeuralNetwork()
{
}

arma::vec NeuralNetwork::Guess(const arma::vec& p_inputs)
{
	arma::mat output = p_inputs;

	for (auto& hidden : m_hiddenLayers)
	{
		output = hidden.FeedForward(output);
	}

	return m_outputLayer.FeedForward(output);
}

void NeuralNetwork::Train(const arma::vec& p_inputs, const arma::vec& p_targets)
{
	const arma::vec outputs = Guess(p_inputs);


	arma::vec outputErrors = p_targets - outputs;

	outputErrors = m_outputLayer.Train(m_hiddenLayers.back().GetLastGuess(), outputErrors);
	for (int i = m_hiddenLayers.size() - 1; i >= 0; --i)
	{
		if (i > 0)
		{
			outputErrors = m_hiddenLayers[i].Train(m_hiddenLayers[i-1].GetLastGuess(),outputErrors);
		}
		else
		{
			outputErrors = m_hiddenLayers[i].Train(p_inputs, outputErrors);
		}
	}
}

void NeuralNetwork::Save(const std::string&& p_filepath)
{
	std::ofstream file;
	file.open(p_filepath, std::ios::binary | std::ios::out);
	if(file.is_open())
	{
		file.write((char*)&m_learningRate, sizeof(m_learningRate));
		file.write((char*)&m_inputCount, sizeof(m_inputCount));
		
		size_t layersCount = m_hiddenLayers.size();
		file.write((char*)&layersCount, sizeof(layersCount));

		for (auto& layer : m_hiddenLayers)
		{
			layer.Save(file);
		}
		m_outputLayer.Save(file);
	}
	file.close();
}


void NeuralNetwork::Load(const std::string&& p_filepath)
{
	std::ifstream file;
	file.open(p_filepath, std::ios::binary | std::ios::in);
	if(file.is_open())
	{
		file.read((char*)&m_learningRate, sizeof(m_learningRate));
		file.read((char*)&m_inputCount, sizeof(m_inputCount));

		size_t layersCount;
		file.read((char*)&layersCount, sizeof(layersCount));

		while (m_hiddenLayers.size() > layersCount)
		{
			m_hiddenLayers.pop_back();
		}
		for (auto& layer : m_hiddenLayers)
		{
			layer.Load(file);
		}

		while (m_hiddenLayers.size() < layersCount)
		{
			m_hiddenLayers.push_back(NeuralNetworkLayer::LoadNew(file));
		}


		m_outputLayer.Load(file);
	}
	file.close();
}

NeuralNetwork NeuralNetwork::LoadNew(const std::string&& p_filepath)
{
	std::ifstream file;
	file.open(p_filepath, std::ios::binary | std::ios::in);
	if (file.is_open())
	{
		float learningRate;

		unsigned int inputCount;
		std::vector<NeuralNetworkLayer> hiddenLayers;

		file.read((char*)&learningRate, sizeof(learningRate));
		file.read((char*)&inputCount, sizeof(inputCount));

		size_t layersCount;
		file.read((char*)&layersCount, sizeof(layersCount));
		NeuralNetwork nn(inputCount, { 0 }, 0);

		nn.m_hiddenLayers.clear();
		while (nn.m_hiddenLayers.size() < layersCount)
		{
			nn.m_hiddenLayers.push_back(NeuralNetworkLayer::LoadNew(file));
		}

		nn.m_outputLayer.Load(file);

		file.close();
		return nn;
	}
	file.close();
	throw std::ifstream::badbit;
}

void NeuralNetwork::Sigmoid(arma::mat::elem_type& x)
{
	x = (1 / (1 + exp(-x)));
}
