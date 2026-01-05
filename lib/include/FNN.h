#include <vector>
#include <random>
#include "Matrix.h"

class NeuralNetwork
{
public:
    NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize);

    std::vector<double> forward(const std::vector<double> &input);

    void train(const std::vector<std::vector<double>> &inputs,
                const std::vector<std::vector<double>> &targets,
                double learningRate, int epochs);
    
private:
    std::vector<int> layerSize_;
    Matrix weights1_, weights2_, weights3_;
    std::vector<double> bias1_, bias2_, bias3_; 
    std::mt19937 gen;


    void initializeWeights();
    
    

};