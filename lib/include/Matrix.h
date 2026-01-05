#ifndef LIB_MATRIX_H
#define LIB_MATRIX_H

#include <vector>

class Matrix {
public:
    Matrix(std::size_t r, std::size_t c);

    double& operator()(std::size_t i, std::size_t j);

    const double& operator()(std::size_t i, std::size_t j) const;


    std::size_t getRows() const;
    std::size_t getCols() const;

private:
    std::vector<std::vector<double>> data_;
    std::size_t rows_, cols_;

};

#endif //LIB_MATRIX_H