# cpp-neuralNetwork
This is my first and very naive neural network library.
Linear algebra operations are using [Armadillo](http://arma.sourceforge.net/ "Armadillo's website").

This library supports :
* Multiple hidden layers
* Save & Load network to binary


## How To Use

Constructor : `NeuralNetwork(inputs, { hiddenLayer1, hiddenLayer2, ... }, output)`

Train : `Train(inputs, targets)`

Guess : `Guess(inputs)`

Save : `Save(file)`

Load : `Load(file)`


## Example

The included main tests on XOR problem :
  * Load previous network state
  * Train
  * Save state
  * Guess
  
```c++
  NeuralNetwork nn = NeuralNetwork::LoadNew("network_saves/save01.bin");
    
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
```
