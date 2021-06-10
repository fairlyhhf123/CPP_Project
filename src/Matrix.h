#pragma once
//
// Created by 胡鸿飞 on 2021/5/30.
//
//
// Created by 胡鸿飞 on 2021/5/30.
//

#ifndef CPROJECT_MATRIX_H
#define CPROJECT_MATRIX_H
#include <vector>
#include <complex>
#include <typeinfo>
#include <cmath>
#include <map>
#include <ostream>
#include <iostream>
#include <limits>
#include "climits"
#include <opencv2/opencv.hpp>
using namespace std;
struct Size_Matching_Exception : public exception {
    char result[100];
    explicit Size_Matching_Exception(const char *message, int line) {
        sprintf(result, "%s, in line: %d", message, line);
    }
    const char *what() const noexcept override {
        return result;
    }
};
using namespace cv;
template<typename T>
class Vector;
template<typename T>
class Matrix {
    friend class Vector<T>;
private:
    vector<vector<T>> vec;
public:
    explicit Matrix() : Matrix(1, 1) {};
    explicit Matrix(int rows, int cols);

    void setVecValue(int rows, int cols, T value);

    T getvecvalue(int rows, int cols);

    //Copy Constructor
    Matrix(const Matrix<T>& matrix);

    // 移动构造函数
    Matrix(Matrix<T>&& matrix) noexcept;

    //Copy Assignment operator
    Matrix<T>& operator=(const Matrix<T>& matrix);

    // Move Assignment operator
    Matrix<T>& operator=(Matrix<T>&& matrix) noexcept;

    explicit Matrix<T>(const vector<vector<T>>& vec);

    explicit Matrix<T>(vector<vector<T>>&& vec);

    static Matrix<T> Assign(int rows = 0, int cols = 0, T = static_cast<T>(0));

    friend std::ostream& operator<<(std::ostream& output, const Matrix<T>& matrix) {
        for (const auto& i : matrix.vec) {
            for (const auto& j : i) {
                output << j << " ";
            }
            output << endl;
        }
        return output;
    }

    Vector<T> pickRowvalue( int row){
        Vector<T> m1(vec[0].size());
        for(int i= 0 ; i < vec[0].size() ; i++){
            m1.setValue(i , getvecvalue(row , i));
        }
        return m1;
    }
    Vector<T> pickColvalue(int col){
        Vector<T> m1(vec.size());
        for(int i= 0 ; i < vec.size() ; i++){
            m1.setValue(i , getvecvalue(i , col));
        }
        return m1;
    }


    Matrix operator+(Matrix<T> &right) {
        if (vec.size()!= right.vec.size() || vec[0].size()!= right.vec[0].size()) {
            throw Size_Matching_Exception("Two adding matrix must have the same size!", __LINE__);
        }
        Matrix<T> m1(vec.size(), vec[0].size());
        if (vec.size() != right.vec.size() && vec[0].size() != right.vec[0].size()) {
            std::cout << "The dimension is wrong!" << endl;
            return m1;
        }

        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, vec[i][j] + right.vec[i][j]);
            }
        }
        return m1;
    }//add

    Matrix operator-(Matrix<T> &right) {
        if (vec.size()!= right.vec.size() || vec[0].size()!= right.vec[0].size()) {
            throw Size_Matching_Exception("Two subtract matrix must have the same size!", __LINE__);
        }
        Matrix<T> m1(vec.size(), vec[0].size());
        if (vec.size() != right.vec.size() && vec[0].size() != right.vec[0].size()) {
            std::cout << "The dimension is wrong!" << endl;
            return m1;
        }
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, vec[i][j] - right.vec[i][j]);
            }
        }
        return m1;
    }//sub

    Matrix operator*(int num) {
        Matrix<T> m1(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, getvecvalue(i, j) * num);
            }
        }
        return m1;
    }//scalar multiplication

    Matrix operator/(int num) {
        Matrix<T> m1(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, this->getvecvalue(i, j) / num);
            }
        }
        return m1;
    }//scalar division

    Matrix Transpose();

    Matrix Conjugation();

    Matrix<T> Element_Wise(const Matrix<T> &right);

    Matrix operator*(Matrix<T> &right) {
        if (vec[0].size() != right.vec.size()) {
            throw Size_Matching_Exception("Two multiplying matrix have size error!", __LINE__);
        }

        Matrix<T> m1(vec.size(), right.vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < right.vec[0].size(); j++) {
                T newnum = 0;
                for (int k = 0; k < vec[0].size(); k++) {
                    newnum += getvecvalue(i, k) * right.getvecvalue(k, j);
                }
                m1.setVecValue(i, j, newnum);
            }
        }
        return m1;
    }//matrix-matrix multiplication

    Vector<T> operator*(Vector<T> &right) {
        if (vec[0].size() != right.value.size()) {
            throw Size_Matching_Exception("The dimension of the vector multiplying matrix is wrong!", __LINE__);
        }
        Vector<T> m1(vec.size());
        Vector<T> m2(vec.size());
        for (int i = 0; i < right.value.size(); i++) {
            m2 = pickColvalue(i) * right.getValue(i);
            m1 = m1 + m2;
        }
        return m1;
    }//matrix-vector multiplication

    static inline T hypot(T a, T b)
    {
        a = abs(a);
        b = abs(b);
        if (a > b) {
            b /= a;
            return a * sqrt(1 + b*b);
        }
        if (b > 0) {
            a /= b;
            return b * sqrt(1 + a*a);
        }
        return 0;
    }

    void eigen(vector<T>& eigenvalues, vector<vector<T>>& eigenvectors)
    {
        auto n = vec.size();
        for (const auto& m : vec) {
            if (m.size() != n) {
                throw Size_Matching_Exception("matrix must be square and it should be a real symmetric matrix!", __LINE__);
            }
        }

        eigenvalues.resize(n, (T)0);//初始化为单位向量
        vector<T> V(n * n, (T)0);
        for (int i = 0; i < n; ++i) {
            V[n * i + i] = (T)1;
            eigenvalues[i] = vec[i][i];
        }
        //初始化为矩阵主对角线元素
        const T eps = numeric_limits<T>::epsilon();
        int maxIters{ (int)n * (int)n * 30 };
        T mv{(T)0 };

        vector<T> indR(n, 0), indC(n, 0);

        vector<T> A;
        for (int i = 0; i < n; ++i) {
            A.insert(A.begin() + i * n, vec[i].begin(), vec[i].end());
        }//二维矩阵赋给一维向量

        for (int k = 0; k < n; ++k) {
            int m, i;
            if (k < n - 1) {
                for (i = k + 2; i < n; i++) {
                    m = k + 1;
                    mv = abs(A[n*k + m]);
                    T val = abs(A[n * k + i]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indR[k] = m;
            }
            if (k > 0) {
                for (i = 1; i < k; i++) {
                    m = 0;
                    mv = abs(A[k]);
                    T val = abs(A[n * i + k]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indC[k] = m;
            }
        }//把非绝对值的最大元素赋给indR,并把第K列，前K个元素中绝对值最大的元素赋给indC.

        if (n > 1) for (int iters = 0; iters < maxIters; iters++) {
                int k, i, m;
                // find index (k,l) of pivot p
                for (i = 1; i < n - 1; i++) {
                    k = 0;
                    mv = abs(A[indR[0]]);
                    T val = abs(A[n * i + indR[i]]);
                    if (mv < val)
                        mv = val, k = i;
                }
                int l = indR[k];
                for (i = 1; i < n; i++) {
                    T val = abs(A[n * indC[i] + i]);
                    if (mv < val)
                        mv = val, k = indC[i], l = i;
                }
                T p = A[n * k + l];
                if (abs(p) <= eps)
                    break;
                T y = ((eigenvalues[l] - eigenvalues[k]) * 0.5);
                T t = abs(y) + hypot(p, y);
                T s = hypot(p, t);
                T c = t / s;
                s = p / s; t = (p / t) * p;
                if (y < 0)
                    s = -s, t = -t;
                A[n*k + l] = 0;
                eigenvalues[k] -= t;
                eigenvalues[l] += t;
                T a0, b0;
#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c
                // rotate rows and columns k and l
                for (i = 0; i < k; i++)
                    rotate(A[n*i + k], A[n*i + l]);
                for (i = k + 1; i < l; i++)
                    rotate(A[n*k + i], A[n*i + l]);
                for (i = l + 1; i < n; i++)
                    rotate(A[n*k + i], A[n*l + i]);
                // rotate eigenvectors
                for (i = 0; i < n; i++)
                    rotate(V[n*k+i], V[n*l+i]);
#undef rotate
                for (int j = 0; j < 2; j++) {
                    int idx = j == 0 ? k : l;
                    if (idx < n - 1) {
                        for (m = idx + 1, mv = std::abs(A[n*idx + m]), i = idx + 2; i < n; i++) {
                            T val = std::abs(A[n * idx + i]);
                            if (mv < val)
                                mv = val, m = i;
                        }
                        indR[idx] = m;
                    }
                    if (idx > 0) {
                        for (m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++) {
                            T val = std::abs(A[n * i + idx]);
                            if (mv < val)
                                mv = val, m = i;
                        }
                        indC[idx] = m;
                    }
                }
            }
        // sort eigenvalues & eigenvectors
        for (int k = 0; k < n - 1; k++) {
            int m = k;
            for (int i = k + 1; i < n; i++) {
                if (eigenvalues[m] < eigenvalues[i])
                    m = i;
            }
            if (k != m) {
                swap(eigenvalues[m], eigenvalues[k]);
                for (int i = 0; i < n; i++)
                    swap(V[n*m+i], V[n*k+i]);
            }
        }
        eigenvectors.resize(n);
        for (int i = 0; i < n; ++i) {
            eigenvectors[i].resize(n);
            eigenvectors[i].assign(V.begin() + i * n, V.begin() + i * n + n);
        }
    }//eigenvalues and eigenvectors

    T Traces(){
        T traces;
        for (int i = 0; i < vec.size(); i++) {
            traces += vec[i][i];
        }
        return traces;
    }//traces

    T determinant(vector<vector<T>> det , int size)
    {
        if (this->rows() != this->cols()){
            throw Size_Matching_Exception("Matrix is not square, no determinant!", __LINE__);
        }
        T detVal = 0;
        if(size == 1)
            return det[0][0];
        vector<vector<T>> tempdet = vector<vector<T >>(size - 1 , vector<T>( size - 1 ));

        for(int i=0;i < size ;i++)
        {
            for(int j = 0 ; j < size -1 ; j++)
                for(int k=0;k < size - 1 ;k++)
                {
                    if(k <i)
                        tempdet[j][k] = det[j+1][k] ;
                    else
                        tempdet[j][k] = det[j+1][k+1];
                }
            detVal += det[0][i] * pow(-1 , i ) * determinant(tempdet, size - 1 );
        }
        return detVal;
    }//determinant

    vector<vector<T>> getAdjugate()
    {
        vector<vector<T>> ans = vector<vector<T>>(vec.size() , vector<T>(vec.size()));
        if(vec.size() == 1)
        {
            ans[0][0] = 1;
            return ans;
        }
        int i,j,k,t;
        vector<vector<T>> temp = vector<vector<T>>(vec.size() , vector<T>(vec.size()));
        for(i = 0;i < vec.size(); i++)
        {
            for(j = 0;j < vec.size();j++)
            {
                for(k=0; k < vec.size() - 1;k++)
                {
                    for(t=0;t < vec.size() - 1;t++)
                    {
                        temp[k][t] = vec[k>=i?k+1:k][t>=j?t+1:t];
                    }
                }
                ans[j][i]  =  determinant(temp,vec.size() - 1);
                if(( i + j ) % 2 == 1)
                {
                    ans[j][i] = - ans[j][i];
                }
            }
        }
        return ans;
    }

    //得到给定矩阵src的逆矩阵保存到des中。
    void GetMatrixInverse(vector<vector<T>>& des)
    {
        des = vector<vector<T>>(vec.size()  , vector<T>(vec.size()));
        vector<vector<T>> t = vector<vector<T>>(vec.size()  , vector<T>(vec.size()));
        double flag = determinant(vec , vec.size());
        if(0==flag)
        {
            throw Size_Matching_Exception("The determinant is 0!", __LINE__);

        }
        else
        {
            t = getAdjugate();
            for(int i = 0;i < vec.size();i++)
            {
                for(int j = 0;j < vec.size();j++)
                {
                    des[i][j]=t[i][j]/flag;
                }
            }
        }
    }//inverse

    T min_all() {
        T min = INT_MAX;
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                if (vec[i][j] < min) {
                    min = vec[i][j];
                }
            }
        }
        return min;
    }

    T min_row(int row) {
        if (row <= 0 || row > this->rows()) {
            throw invalid_argument("Row out of range!");
        }
        T min = INT_MAX;
        for (int j = 0; j < vec[0].size(); j++) {
            if (vec[row][j] < min) {
                min = vec[row][j];
            }
        }

        return min;
    }

    T min_col(int col) {
        if (col <= 0 || col > this->cols()) {
            throw invalid_argument("col out of range");
        }
        T min = INT_MAX;
        for (int j = 0; j < vec[0].size(); j++) {
            if (vec[j][col] < min) {
                min = vec[j][col];
            }
        }

        return min;
    }

    T max_all() {
        T max = INT_MIN;
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                if (vec[i][j] > max) {
                    max = vec[i][j];
                }
            }
        }
        return max;
    }

    T max_row(int row) {
        if (row <= 0 || row > this->rows()) {
            throw invalid_argument("row out of range");
        }
        T max = INT_MIN;
        for (int j = 0; j < vec[0].size(); j++) {
            if (vec[row][j] > max) {
                max = vec[row][j];
            }
        }

        return max;
    }

    T max_col(int col) {
        if (col <= 0 || col > this->cols()) {
            throw invalid_argument("col out of range");
        }
        T max = INT_MIN;
        for (int j = 0; j < vec[0].size(); j++) {
            if (vec[j][col] > max) {
                max = vec[j][col];
            }
        }

        return max;
    }

    T sum_all() {
        T sum = 0;
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                sum += vec[i][j];
            }
        }
        return sum;
    }

    T sum_row(int row) {
        if (row <= 0 || row > this->rows()) {
            throw invalid_argument("row out of range");
        }
        T sum = 0;
        for (int j = 0; j < vec[row].size(); j++) {
            sum += vec[row][j];
        }

        return sum;
    }

    T sum_col(int col) {
        if (col <= 0 || col > this->cols()) {
            throw invalid_argument("col out of range");
        }
        T sum = 0;
        for (int j = 0; j < vec.size(); j++) {
            sum += vec[j][col];
        }

        return sum;
    }

    T avg_all() {
        T sum = 0;
        int he;
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                sum += vec[i][j];
                he++;
            }
        }
        T avg = sum / he;
        return avg;
    }

    T avg_row(int row) {
        if (row <= 0 || row > this->rows()) {
            throw invalid_argument("row out of range");
        }
        T sum = 0;
        int he;
        for (int j = 0; j < vec[row].size(); j++) {
            sum += vec[row][j];
            he++;
        }

        T avg = sum / he;
        return avg;
    }

    T avg_col(int col) {
        if (col <= 0 || col > this->cols()) {
            throw invalid_argument("col out of range");
        }
        T sum = 0;
        int he;
        for (int j = 0; j < vec.size(); j++) {
            sum += vec[j][col];
            he++;
        }

        T avg = sum / he;
        return avg;
    }

    static Matrix<T> reshape(Matrix<T> m, int r, int c);
    static Matrix<T> slicing(Matrix<T> m, int rb, int re, int cb, int ce);
    static Matrix<T> convolution(Matrix<T> m1, Matrix<T> m2);


    void zhuanhuan(Matrix<T> m);
    static Matrix<T> nzh();


    inline int rows() const;

    inline int cols() const;

    inline bool is_empty() const;
};

template<typename T>
class Sparse : public Matrix<T> {
private:
    int terms;
    T *rowArray;   //存非零元素的三元数组
    T *colArray;    //三元组最大可容纳的元素个数
    T *valArray;
public:
    explicit Sparse();  //构造函数
    Sparse(Sparse& b);      //赋值构造函数
    ~Sparse(); //析构函数

    void Assign();
//    T operator+(T num); //赋值运算符重载
//    Sparse& Add(Sparse& b, Sparse& c);      //矩阵的加法
//    Sparse Multiply(Sparse& b);      //矩阵的乘法
    void Show();
};

template<typename T>
Sparse<T>::Sparse(){
    terms = 0;
    rowArray = new T[1];
    colArray = new T[1];
    valArray = new T[1];
}

template<typename T>
Sparse<T>::Sparse(Sparse<T>& b){  //复制构造函数
    this->rows()=b.rows();      //赋值矩阵的性质
    this->cols()=b.cols();
    terms=b.terms;
    for(int i=0;i<terms;i++){
        rowArray[i] = b.rowArray[i];
        colArray[i] = b.colArray[i];
        valArray[i] = b.valArray[i];
    }
}
template<typename T>
void Sparse<T>::Assign() {
    int t;
    cout << "how many terms in this sparse matrix: ";
    cin >> t;
    terms = t;
    rowArray = new int[t];
    colArray = new int[t];
    valArray = new T[t];
    for (int i = 0; i < t; ++i) {
        cout << "Input the position and value of the " << i + 1 << "th term:" << endl;
        cin >> rowArray[i];
        cin >> colArray[i];
        cin >> valArray[i];
    }
}

template<typename T>
Sparse<T>::~Sparse(){   //析构函数：释放所有存储
    delete[] rowArray;
    delete[] colArray;
    delete[] valArray;
}

template<typename T>
void Sparse<T>::Show() {
    cout << "Sparse Matrix has " << terms << " non-zero numbers" << endl;
    for (int i = 0; i < terms; ++i) {
        cout << "Row: " << rowArray[i] << " , " << "Col: " << colArray[i] << " , " << "value: " << valArray[i] << endl;
    }
}

template<typename T>
class Vector {
public:

    Vector()= default;;
    Vector(Vector& source);
    explicit Vector(int size);
    explicit Vector<T>(const vector<T>& vec);
    explicit Vector<T>(vector<T>&& vec);
    void setValue(int i,T newvalue);
    T getValue(int i);

    Vector<T> operator +(Vector<T>& right)
    {
        Vector<T> m1(value.size());
        if(value.size() != right.value.size())
        {
            cout << "The dimension is wrong!" << endl;
            return m1;
        }
        for(int i = 0 ; i < value.size() ; i++){
            m1.setValue(i,getValue(i) + right.getValue(i));
        }
        return m1;
    }
    Vector<T> operator -(Vector<T>& right)
    {
        Vector<T> m1(value.size());
        if(value.size() != right.value.size())
        {
            std::cout << "The dimension is wrong!" << endl;
            return m1;
        }
        for(int i = 0 ; i < value.size() ; i++){
            m1.setValue(i,getValue(i) - right.getValue(i));
        }
        return m1;
    }

    Vector<T> operator *(T right)
    {
        Vector<T> m1(value.size());
        for(int i = 0 ; i < value.size() ; i++){
            m1.setValue(i,getValue(i) * right);
        }
        return m1;
    }

    T operator *(Vector<T>& right)
    {
        T ans = 0;
        for(int i = 0 ; i < value.size() ; i++){
            ans += getValue(i) * right.getValue(i);
        }
        return ans;
    }//dot product

    Vector<T> operator *(Matrix<T>& right)
    {
        if(value.size() != right.vec.size()){
            throw Size_Matching_Exception("Dimension of vector is wrong!", __LINE__);
        }
        Vector<T> m1(right.vec[0].size());
        Vector<T> m2(right.vec[0].size());
        for(int i = 0 ; i < right.value.size() ; i++){
            m2 = right.pickColvalue(i) * right[i];
            m1 = m1 + m2;
        }
        return m1;
    }//matrix-vector multiplication

    friend std::ostream &operator<<(std::ostream &output, const Vector<T>& v) {
        for (const auto &i : v.value) {
            output << i << " ";
        }
        output << endl;
        return output;
    }

    vector<T> value;
};

template<typename T>
void Vector<T>::setValue(int i, T newvalue) {
    value[i] = newvalue;
}

template<typename T>
Vector<T>::Vector(Vector &source) {
    for(int i = 0 ; i < source.value.size() ; i++){
        value[i] = source.value[i];
    }
}

template<typename T>
Vector<T>::Vector(const vector<T>& vec) {
    this->value = vec;
}

template<typename T>
Vector<T>::Vector(vector<T>&& vec) {
    this->value = move(vec);
}


template<typename T>
T Vector<T>::getValue(int i) {
    return value[i];
}

template<typename T>
Vector<T>::Vector(int size) {
    this->value = vector<T>(size);
}


template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
    this->vec = vector<vector<T >>(rows, vector<T>(cols));
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& matrix) {
    this->vec = vector<vector<T >>(matrix.vec);
}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& matrix) noexcept : vec(matrix.vec) {
    matrix.vec = vector<vector<T>>{ 0, vector<T>{0, static_cast<T>(0)} };
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& matrix) {
    this->vec = matrix.vec;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& matrix) noexcept {
    this->vec = std::move(matrix.vec);
    matrix.vec = vector<vector<T>>{ 0, vector<T>{0, static_cast<T>(0)} };
    return *this;
}

template<typename T>
Matrix<T>::Matrix(const vector<vector<T>>& vec) {
    this->vec = vec;

}

template<typename T>
Matrix<T>::Matrix(vector<vector<T>>&& vec) {
    this->vec = std::move(vec);
}

template<typename T>
Matrix<T> Matrix<T>::Assign(int rows, int cols, T t) {
    Matrix<T> will_return(rows, cols);
    for (auto& i : will_return.vec) {
        i = vector<T>(cols, t);
    }
    return will_return;
}

template<typename T>
Matrix<T> Matrix<T>::Transpose() {
    Matrix<T> m1(vec[0].size(), vec.size());
    for (int i = 0; i < vec[0].size(); i++) {
        for (int j = 0; j < vec.size(); j++) {
            m1.setVecValue(i, j, this->getvecvalue(j, i));
        }
    }
    return m1;
}//transpose

template<typename T>
Matrix<T> Matrix<T>::Conjugation() {
    Matrix<T> m1(vec.size(), vec[0].size());
    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[0].size(); j++) {
            T m2;
            m2.real(real(this->getvecvalue(i, j)));
            m2.imag(imag(this->getvecvalue(i, j)) * -1);
            m1.setVecValue(i, j, m2);
        }
    }
    return m1;
}//Conjugation

template<typename T>
Matrix<T> Matrix<T>::Element_Wise(const Matrix<T> &right) {
    if (vec.size() != right.vec.size() || vec[0].size() != right.vec[0].size()) {
        throw Size_Matching_Exception("Matrix is not the same size, no element_wise!", __LINE__);
    }
    Matrix<T> m1(vec.size(), vec[0].size());
    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[0].size(); j++) {
            m1.setVecValue(i, j, vec[i][j] * right.vec[i][j]);
        }
    }
    return m1;
}//Element_Wise

template<typename T>
T Matrix<T>::getvecvalue(int rows, int cols) {
    return vec[rows][cols];
}

template<typename T>
void Matrix<T>::setVecValue(int rows, int cols , T value) {
    vec[rows][cols] = value;
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


template<typename T>
Matrix<T> reshape(Matrix<T> m, int row, int col) {
    int row1 = m.rows();
    int col1 = m.cols();
    if (row1 * col1 < row * col) {
        return m;
    }
    else {
        vector<T> array;
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col1; j++) {
                array.push_back(m.getvecvalue(i, j));
            }
        }
        int now = 0;
        Matrix<T> s(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                s.setVecValue(i, j, array[now]);
                now++;


            }
        }
        now = 0;
        return s;
    }
}

template<typename T>
Matrix<T> slicing(Matrix<T> m, int rowbegin, int rowend, int colbegin, int colend) {
    int row1 = rowend - rowbegin + 1;
    int col1 = colend - colbegin + 1;
    Matrix<T> s(row1, col1);
    if (rowbegin > rowend || colbegin > colend) {
        throw Size_Matching_Exception("dimension has something wrong, no slicing!", __LINE__);
        return s;
    }
    else {
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col1; j++) {
                s.setVecValue(i, j, m.getvecvalue(i + rowbegin - 1, j + colbegin - 1));

            }
        }
    }
    return s;
}



template<typename T>
Matrix<T> convolution(Matrix<T> m1, Matrix<T> m2) {
    if(m1.rows() != m1.cols() || m2.rows() != m2.cols()){
        throw Size_Matching_Exception("Matrix is not square, no convolution!", __LINE__);
    }
    Matrix<T> big;
    Matrix<T> small;
    if (m1.rows() * m1.cols() > m2.rows() * m2.cols()) {
        big = m1;
        small = m2;
    }
    else {
        big = m2;
        small = m1;
    }

    Matrix<T> extend(big.rows() + 2 * (small.rows() - 1), big.cols() + 2 * (small.cols() - 1));
    Matrix<T> ans(big.rows() + small.rows() - 1, big.cols() + small.cols() - 1);
    Matrix<T> daosmall(small.rows(), small.cols());
    for (int i = 0; i < small.rows(); ++i) {
        for (int j = 0; j < small.cols(); ++j) {
            daosmall.setVecValue(small.rows() - i - 1, small.cols() - j - 1, small.getvecvalue(i, j));
        }
    }//将小矩阵逆转
    for (int i = 0; i < big.rows(); ++i) {
        for (int j = 0; j < big.cols(); ++j) {
            extend.setVecValue(small.rows() - 1 + i, small.cols() - 1 + j, big.getvecvalue(i, j));

        }
    }//将大矩阵扩大至完全能够涵盖小矩阵的运算空间

    for (int i = 0; i < ans.rows(); ++i) {
        for (int j = 0; j < ans.cols(); ++j) {
            T sum = 0;
            for (int k = 0; k < daosmall.rows(); ++k) {
                for (int l = 0; l < daosmall.cols(); ++l) {
                    sum += daosmall.getvecvalue(k, l) * extend.getvecvalue(i + k, j + l);
                }
            }
            ans.setVecValue(i, j, sum);

        }
    } //daosmall矩阵起点在extend矩阵的位置，对应ans矩阵的i,j
    return ans;
}

template<typename T>
inline bool Matrix<T>::is_empty() const {
    return vec.empty() || vec.front().empty();
}


template<typename T>
void zhuanhuan(Matrix<T> m) {
    int rows = m.rows();
    int cols = m.cols();
    Mat testMat1 = Mat(Size(cols, rows), CV_32FC1);
    string name;
    cout << "请输入储存文件名：";
    cin >> name;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            testMat1.at<float>(i, j) = m.chakan(i,j);
        }
    }
    string name1 = "./" + name + ".xml";

    FileStorage fs(name1, FileStorage::WRITE);
    fs << "mat1" << testMat1;
    fs.release();
}

template<typename T>
Matrix<T>  Matrix<T>::nzh() {
    cout << "请输入储存文件名称： ";
    string name;
    cin >> name;
    string name1 = "./" + name + ".xml";

    FileStorage fsRead(name1, FileStorage::READ);
    Mat readMat1;
    fsRead["mat1"] >> readMat1;

    Matrix<T> m(readMat1.rows, readMat1.cols);
    for (int i = 0; i < readMat1.rows; i++)
    {
        for (int j = 0; j < readMat1.cols; j++)
        {
            m.xiugai(i, j, readMat1.at<float>(i, j));
        }
    }
    return m;
}

#endif //MATRIX_H
