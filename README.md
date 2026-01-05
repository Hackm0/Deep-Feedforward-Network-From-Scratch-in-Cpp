# Deep Feedforward Neural Network From Scratch in C++

A simple deep feedforward neural network implementation built entirely from scratch in C++, without any external machine learning libraries.

## Features

- **Custom Matrix Class**: Lightweight matrix implementation for weight storage and operations
- **Multi-layer Architecture**: Configurable input, two hidden layers, and output layer
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output layer
- **He Initialization**: Proper weight initialization for deep networks
- **Backpropagation**: Full gradient descent implementation with MSE loss

## Project Structure

```
├── lib/
│   ├── include/
│   │   ├── FNN.h          # Neural network class header
│   │   ├── Matrix.h       # Matrix class header
│   │   └── activation.h   # Activation functions
│   └── src/
│       ├── FNN.cpp        # Neural network implementation
│       └── Matrix.cpp     # Matrix implementation
├── examples/
│   └── src/
│       └── main.cpp       # Example: Circle classification
├── CMakeLists.txt
└── README.md
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Creating a Network

```cpp
#include "FNN.h"

// Create a network with:
// - 2 input neurons
// - 8 neurons in first hidden layer
// - 4 neurons in second hidden layer
// - 1 output neuron
NeuralNetwork nn(2, 8, 4, 1);
```

### Training

```cpp
std::vector<std::vector<double>> inputs = { /* your training data */ };
std::vector<std::vector<double>> targets = { /* your labels */ };

// Train with learning rate 0.01 for 1000 epochs
nn.train(inputs, targets, 0.01, 1000);
```

### Inference

```cpp
std::vector<double> input = {0.5, 0.5};
std::vector<double> output = nn.forward(input);
```

## Example: Circle Classification

The included example trains the network to classify points as inside or outside a unit circle (x² + y² < 1).

```bash
./example
```

Sample output:
```
EPOCH : 0 MSE : 0.19659
EPOCH : 100 MSE : 0.0116126
...
EPOCH : 900 MSE : 0.00408221
Training Time : 3637 ms

 Test results (1 : inside, 0 : outside) :
 point ( 0,0 ) :0.985105 ( actual : 1 , error : 0.0148954)
 point ( 1,1 ) :1.26827e-08 ( actual : 0 , error : 1.26827e-08)
 point ( 0.5,0.5 ) :0.98192 ( actual : 1 , error : 0.0180802)
 point ( 2,2 ) :2.94023e-37 ( actual : 0 , error : 2.94023e-37)
```

## Requirements

- C++17 or later
- CMake 3.16+

## License

See [LICENSE](LICENSE) file.
