#include "Matrix.h"
#include "iostream"
using namespace std;
int main(){
    Matrix<int> m1(2,3);
    cout << m1;

    Matrix<int> m2 = m1;
    cout << m2;

    Matrix<int> m3(m1);
    cout << m3;
    vector<vector<int32_t>> v1 = {{1, 2},
                                  {3, 4},
                                  {5, 6},
                                  {7, 8},
                                  {9, 10}};
    Matrix<int> m4(v1);
    cout << m4;

    Matrix<int> m5 = m4;
    cout << m5;
}