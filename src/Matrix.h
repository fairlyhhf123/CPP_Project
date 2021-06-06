//
// Created by 胡鸿飞 on 2021/5/30.
//

#ifndef CPROJECT_MATRIX_H
#define CPROJECT_MATRIX_H

#include <vector>
#include <ostream>

using namespace std;
template<typename T>
class Matrix {
private:
    vector<vector<T>> vec;
public:
    explicit Matrix() : Matrix(0, 0) {};
    explicit Matrix(int rows, int cols);
    //Copy Constructor
    Matrix(const Matrix<T> &matrix);

    // 移动构造函数
    Matrix(Matrix<T> &&matrix) noexcept ;

    //Copy Assignment operator
    Matrix<T> &operator=(const Matrix<T> &matrix);

    // Move Assignment operator
    Matrix<T> &operator=(Matrix<T> &&matrix) noexcept;

    explicit Matrix<T>(const vector<vector<T>> &vec);

    explicit Matrix<T>(vector<vector<T>> &&vec);

    static Matrix<T> values(int rows = 0, int cols = 0, T = static_cast<T>(0));

    static Matrix<T> eye(int s);

    static Matrix<T> eye_value(int s, T t);

    friend std::ostream &operator<<(std::ostream &output, const Matrix<T> &matrix) {
        for (const auto &i : matrix.vec) {
            for (const auto &j : i) {
                output << j << " ";
            }
            output << endl;
        }
        return output;
    }
    inline int rows() const;

    inline int cols() const;
};

template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
    this->vec = vector<vector<T >>(rows, vector<T>(cols));
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &matrix) {
    this->vec = vector<vector<T >>(matrix.vec);
}

template<typename T>
Matrix<T>::Matrix(Matrix<T> &&matrix) noexcept: vec(matrix.vec) {
    matrix.vec = vector<vector<T>>{0, vector<T>{0, static_cast<T>(0)}};
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &matrix) {
    this->vec = matrix.vec;
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&matrix) noexcept {
    this->vec = std::move(matrix.vec);
    matrix.vec = vector<vector<T>>{0, vector<T>{0, static_cast<T>(0)}};
    return *this;
}

template<typename T>
Matrix<T>::Matrix(const vector<vector<T>> &vec) {
    this->vec = vec;

}

template<typename T>
Matrix<T>::Matrix(vector<vector<T>> &&vec) {
    this->vec = std::move(vec);
}

template<typename T>
Matrix<T> Matrix<T>::values(int rows, int cols, T t) {
    Matrix<T> will_return(rows, cols);
    for (auto &i : will_return.vec) {
        i = vector<T>(cols, t);
    }
    return will_return;
}

template<typename T>
inline int Matrix<T>::rows() const {
    return static_cast<int>(this->vec.size());
}

template<typename T>
inline int Matrix<T>::cols() const {
    if (this->rows() == 0) {
        return 0;
    }
    return static_cast<int>(this->vec.front().size());
}
#endif //CPROJECT_MATRIX_H
