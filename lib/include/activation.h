#ifndef LIB_ACTIVATION_H
#define LIB_ACTIVATION_H

#include <algorithm>
#include <cmath>

namespace lib {
    
/// Rectified Linear Unit activation function (ReLU)
inline double relu(double x) noexcept {return std::max(0.0, x);}

/// Derivative of ReLU
inline double reluDerivative(double x) noexcept {return x > 0.0 ? 1.0 : 0.0;}

/// Sigmoid Function
inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x));}

/// Derivative of Sigmoid Function
inline double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

}

#endif //LIB_ACTIVATION_H