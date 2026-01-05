#include <iostream>
#include <chrono>
#include "FNN.h"


int main()
{
    try{

        NeuralNetwork nn(2,8,4,1);
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> dist(-2.0, 2.0);
        const int numSamples = 1000;
        
        std::vector<std::vector<double>> inputs(numSamples);
        std::vector<std::vector<double>> targets(numSamples);

        for (uint32_t i = 0; i < numSamples; i++)
        {
            double x = dist(gen);
            double y = dist(gen);
            inputs[i] = {x, y};
            double distance = sqrt(x * x + y * y); //euclidian
            targets[i] = {distance < 1.0 ? 1.0 : 0.0};
        }

        auto start = std::chrono::high_resolution_clock::now();
        nn.train(inputs, targets, 0.01, 10000);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Training Time : " << 
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() <<
            " ms\n";
        
        std::vector<std::vector<double>> testPoints = {
            {0.0,0.0},
            {1.0,1.0},
            {0.5,0.5},
            {2.0,2.0}};

        std::cout << "\n Test results (1 : inside, 0 : outside) : \n";

        for (const auto &point : testPoints)
        {
            auto output = nn.forward(point);
            double actual = sqrt(point[0] * point[0] + point[1] * point[1]) < 1.0 ? 1.0 : 0.0; //x² + y² = 1.0
            std::cout << " point ( " << point[0] << "," << point[1] << " ) :"
                      << output[0] << " ( actual : " << actual
                      << " , error : " << std::abs(output[0]- actual) << ") \n";
        }   
        


    } catch (const std::exception &e) {
        std::cerr << "ERROR : " << e.what() << "\n";

    }
}