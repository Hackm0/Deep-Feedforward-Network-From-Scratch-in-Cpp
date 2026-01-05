#include <iostream>
#include <vector>
#include <cmath>
#include <random> //to initialize weigth
#include <stdexcept>
#include <chrono>
#include "FNN.h"
#include "activation.h"

NeuralNetwork::NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize):
    layerSize_{inputSize, hidden1Size, hidden2Size, outputSize},
    weights1_(inputSize, hidden1Size),
    weights2_(hidden1Size, hidden2Size),
    weights3_(hidden2Size, outputSize),
    bias1_(hidden1Size),
    bias2_(hidden2Size),
    bias3_(outputSize),
    gen(std::random_device{}())
    {
        if (inputSize <= 0 || hidden1Size <= 0 || hidden2Size <= 0 || outputSize <= 0)
        {
            throw std::invalid_argument("Layer sizes must be positive");
        }
        initializeWeights();
        
    }

    std::vector<double> NeuralNetwork::forward(const std::vector<double> &input)
    {
        if (input.size() != layerSize_[0])
        {
            throw std::runtime_error("Input size mismatch");
        }

        std::vector<double> hidden1(layerSize_[1]);

        for (uint8_t i = 0; i < layerSize_[1]; i++)
        {
            double sum = bias1_[i];
            for (uint8_t j = 0; j < layerSize_[0]; j++)
            {
                sum += input[j] * weights1_(j,i);
            }

            hidden1[i] = lib::relu(sum);
        }


        std::vector<double> hidden2(layerSize_[2]);

        for (uint8_t i = 0; i < layerSize_[2]; i++)
        {
            double sum = bias2_[i];
            for (uint8_t j = 0; j < layerSize_[1]; j++)
            {
                sum += hidden1[j] * weights2_(j,i);
            }

            hidden2[i] = lib::relu(sum);
        }

        std::vector<double> output(layerSize_[3]);

        for (uint8_t i = 0; i < layerSize_[3]; i++)
        {
            double sum = bias3_[i];
            for (uint8_t j = 0; j < layerSize_[2]; j++)
            {
                sum += hidden2[j] * weights3_(j,i);
            }

            output[i] = lib::sigmoid(sum);
        }
        return output;
    }


    void NeuralNetwork::train(const std::vector<std::vector<double>> &inputs,
                const std::vector<std::vector<double>> &targets,
                double learningRate, int epochs)
        {
            if (inputs.size() != targets.size())
            {
                throw std::runtime_error("Input and Target sizes don't match");
            }
            
            for (uint32_t epoch = 0 ; epoch < epochs; epoch++)
            {
                double totalError = 0.0;

                for (size_t k = 0; k < inputs.size(); k++)
                {
                    std::vector<double> hidden1(layerSize_[1]);
                    std::vector<double> hidden1Pre(layerSize_[1]);
                    for (uint32_t j = 0; j < layerSize_[1]; j++)
                    {
                        double sum = bias1_[j];
                        for (uint32_t i = 0; i < layerSize_[0]; i++)
                        {
                            sum+=inputs[k][i] * weights1_(i,j);
                        }
                        hidden1Pre[j] = sum;
                        hidden1[j] = lib::relu(sum);
                    }


                    std::vector<double> hidden2(layerSize_[2]);
                    std::vector<double> hidden2Pre(layerSize_[2]);
                    for (uint32_t j = 0; j < layerSize_[2]; j++)
                    {
                        double sum = bias2_[j];
                        for (uint32_t i = 0; i < layerSize_[1]; i++)
                        {
                            sum+=hidden1[i] * weights2_(i,j);
                        }
                        hidden2Pre[j] = sum;
                        hidden2[j] = lib::relu(sum);
                    }

                    std::vector<double> output(layerSize_[3]);
                    std::vector<double> outputPre(layerSize_[3]);
                    for (uint32_t j = 0; j < layerSize_[3]; j++)
                    {
                        double sum = bias3_[j];
                        for (uint32_t i = 0; i < layerSize_[2]; i++)
                        {
                            sum+=hidden2[i] * weights3_(i,j);
                        }
                        outputPre[j] = sum;
                        output[j] = lib::sigmoid(sum);
                    }

                    //error
                    for (uint32_t i = 0; i < layerSize_[3]; i++)
                    {
                        double error = targets[k][i] - output[i];
                        //MSE
                        totalError += error * error;
                    }


                    //gradients
                    std::vector<double> outputGradients(layerSize_[3]);
                    for (uint32_t i = 0; i< layerSize_[3]; i++)
                    {
                        outputGradients[i] = (output[i] - targets[k][i]) * lib::sigmoidDerivative(outputPre[i]);
                    }

                    std::vector<double> hidden2Gradients(layerSize_[2]);
                    for (uint32_t i = 0; i < layerSize_[2]; i++)
                    {
                        double error = 0;
                        for (uint32_t j = 0; j < layerSize_[3]; j++)
                        {
                            error += outputGradients[j] * weights3_(i,j);
                        }

                        hidden2Gradients[i] = error * lib::reluDerivative(hidden2Pre[i]);
                    }

                    std::vector<double> hidden1Gradients(layerSize_[1]);
                    for (uint32_t i = 0; i < layerSize_[1]; i++)
                    {
                        double error = 0;
                        for (uint32_t j = 0; j < layerSize_[2]; j++)
                        {
                            error += hidden2Gradients[j] * weights2_(i,j);
                        }

                        hidden1Gradients[i] = error * lib::reluDerivative(hidden1Pre[i]);
                    }

                    //update
                    for (uint32_t i = 0; i < layerSize_[2]; i++)
                    {
                        for (uint32_t j = 0; j < layerSize_[3]; j++)
                        {
                            weights3_(i,j) -= learningRate * outputGradients[j] * hidden2[i];
                        }
                    }

                    for (uint32_t j = 0; j < layerSize_[3]; j++)
                    {
                        bias3_[j] -= learningRate * outputGradients[j];
                    }

                    for (uint32_t i = 0; i < layerSize_[1]; i++)
                    {
                        for (uint32_t j = 0; j < layerSize_[2]; j++)
                        {
                            weights2_(i,j) -= learningRate * hidden2Gradients[j] * hidden1[i];
                        }
                    }

                    for (uint32_t j = 0; j < layerSize_[2]; j++)
                    {
                        bias2_[j] -= learningRate * hidden2Gradients[j];
                    }
                    for (uint32_t i = 0; i < layerSize_[0]; i++)
                    {
                        for (uint32_t j = 0; j < layerSize_[1]; j++)
                        {
                            weights1_(i,j) -= learningRate * hidden1Gradients[j] * inputs[k][i];
                        }
                    }

                    for (uint32_t j = 0; j < layerSize_[1]; j++)
                    {
                        bias1_[j] -= learningRate * hidden1Gradients[j];
                    }

                }
                //every 100 epochs, print  
                if (epoch % 100 == 0)
                {
                    std::cout << "EPOCH : " << epoch << " MSE : "
                              << totalError / inputs.size() << "\n";
                }
            }
        }


void NeuralNetwork::initializeWeights()
{
    std::normal_distribution<> dist(0.0, 1.0);

    //He Initializing
    double scale1 = sqrt(2.0/layerSize_[0]);
    double scale2 = sqrt(2.0/layerSize_[1]);
    double scale3 = sqrt(2.0/layerSize_[2]);

    for (uint8_t i = 0; i < weights1_.getRows(); i++)
    {
        for (uint8_t j = 0; j < weights1_.getCols(); j++)
        {
            weights1_(i,j) = dist(gen) * scale1;
        }
    }

    for (uint8_t i = 0; i < weights2_.getRows(); i++)
    {
        for (uint8_t j = 0; j < weights2_.getCols(); j++)
        {
            weights2_(i,j) = dist(gen) * scale2;
        }
    }

    for (uint8_t i = 0; i < weights3_.getRows(); i++)
    {
        for (uint8_t j = 0; j < weights3_.getCols(); j++)
        {
            weights3_(i,j) = dist(gen) * scale3;
        }
    }

    fill(bias1_.begin(), bias1_.end(), 0.0);
    fill(bias2_.begin(), bias2_.end(), 0.0);
    fill(bias3_.begin(), bias3_.end(), 0.0);
    

}