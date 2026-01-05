#include "Matrix.h"

Matrix::Matrix(std::size_t r, std::size_t c):
    rows_(r),
    cols_(c),
    data_(r, std::vector<double>(c, 0.0)) {
        //OK
    }

double& Matrix::operator()(std::size_t i, std::size_t j){
    return data_[i][j];
}

const double& Matrix::operator()(std::size_t i, std::size_t j) const{
    return data_[i][j];
}

std::size_t Matrix::getRows() const {
    return rows_;
}

std::size_t Matrix::getCols() const {
    return cols_;
}
