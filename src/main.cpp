#include "iostream"
#include "matr.h"
using namespace std;

int main()
{
    //default Constructor
    cout << "Default Constructor:" << endl;
    Matrix<int> default_int_m;
    cout << default_int_m;
    Matrix<float> default_float_m;
    cout << default_float_m;

    //parameterized Constructor
    cout << "Parameterized Constructor:" << endl;
    Matrix<int> para_m1(2, 2);
    cout << para_m1;
    Matrix<int> para_m2(4, 5);
    cout << para_m2;
    Matrix<complex<int>> para_com_m(3, 3);
    cout << para_com_m;

    //Copy Constructor
    cout << "Copy Constructor: " << endl;
    Matrix<int> copy_m1(2, 2);
    Matrix<int> copy_m2(5, 4);
    Matrix<int> copied_m1(copy_m1);
    cout << copied_m1;
    Matrix<int> copied_m2 = copy_m2;
    cout << copied_m2;

    //Move Constructor
    cout << "Move Constructor:" << endl;
    Matrix<int> move_m1(1, 3);
    Matrix<int> move_m2(4, 1);
    move_m2 = move(move_m1);
    cout << move_m2;

//    //No vectors negative
//    cout << "no vectors negative:" << endl;
//    Matrix<complex<int>> m1(5, 2);
//    Matrix<int> m2(4, 6);
//    Matrix<int> m3(1, 4);
//    Matrix<double> m4(0, -1);
//    Matrix<float> m5(-1, -1);
//    Matrix<int> m6(-2, 9);

    //empty
    cout << "empty test: " << endl;
    Matrix<int> empty_m(0,0);
    if (empty_m.is_empty()){
        cout << "This is an empty matrix!" << endl;
    }else {
        cout << "Not empty." << endl;
    }

    vector<vector<complex<int>>> com_v1 = {{complex<int>(2, 3),  complex<int>(3, 5)},
                                   {complex<int>(-2, 3), complex<int>(-2, -3)},
                                   {complex<int>(2, -3), complex<int>(2, 3)}};
    vector<vector<complex<float>>> com_v2 = {{complex<float>(2.5f, -3.0f), complex<float>(3.0f, 5.0f)},
                                       {complex<float>(-2.3f, 3.2f), complex<float>(-2.0f, -3.0f)},
                                       {complex<float>(2.0f, -3.0f), complex<float>(2.5f, 3.0f)}};
    vector<vector<int>> v1 = {{1, 2},
                              {3, 4},
                              {5, 6},
                              {7, 8},
                              {9, 10}};
    vector<vector<double>> v2 = {{1.1, 1.2, 4.5},
                              {5.3, 1.6, 4.4}};
    vector<vector<int>> v3 = {{1, 1, 9},
                              {1, 2, 0},
                              {1, 1, 4}};
    vector<vector<int>> v4 = {{2, 3},
                              {2, 3},
                              {4, 5},
                              {5, 6},
                              {7, 8}};
    vector<vector<double>> v5 = {{2.1, 2.2, 5.4},
                                 {5.9, 5.8, 7.3}};

    //Addition test
    cout << "Addition test:" << endl;
    Matrix<int> add_m1(v1);
    Matrix<int> add_m2(v1);
    Matrix<int> add_m3;
    Matrix<complex<int>> add_com_m1(com_v1);
    Matrix<complex<int>> add_com_m2 = add_com_m1 + add_com_m1;
    add_m3 = add_m1 + add_m2;
    cout << add_m3;
    cout << add_com_m2;

    //subtraction test
    cout << "Subtraction test:" << endl;
    Matrix<int> sub_m1(v1);
    Matrix<int> sub_m2(v4);
    Matrix<int> sub_m3 = sub_m1 - sub_m2;
    Matrix<double> sub_m4(v2);
    Matrix<double> sub_m5(v5);
    Matrix<double> sub_m6 = sub_m5 - sub_m4;
    cout << sub_m3;


    //scalar multiplication
    cout << "scalar multiplication test:" << endl;
    Matrix<int> mul_m1(v1);
    Matrix<int> mul_m2 = mul_m1 * 3;
    cout << mul_m2;


    //scalar division
    cout << "scalar division test: " << endl;
    Matrix<double> div_m1(v2);
    Matrix<double> div_m2 = div_m1 / 4;
    cout << div_m2;

    //transposition
    cout << "transposition test: " << endl;
    Matrix<double> tran_m1(v2);
    Matrix<double> tran_m2 = tran_m1.Transpose();
    cout << tran_m2;


    //conjugation
    cout << "conjugation test: " << endl;
    Matrix<complex<float>> conj_m1(com_v2);
    Matrix<complex<float>> conj_m2 = conj_m1.Conjugation();
    cout << tran_m2;
    return 0;
}
