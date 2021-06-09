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
    vector<vector<int>> v6 = {{2, 3}, {2, 3}, {4, 5},
                              {5, 6}, {4, 6}, {1, 4}};
    vector<vector<double>> v5 = {{2.1, 2.2, 5.4},
                                 {5.9, 5.8, 7.3}};

    //Addition test
    cout << "Addition test:" << endl;
    Matrix<int> add_m1(v1);
    Matrix<int> add_m2(v1);
    Matrix<int> add_m3;
    Matrix<complex<int>> add_com_m1(com_v1);
    Matrix<complex<int>> add_com_m2 = add_com_m1 + add_com_m1;
    cout << "Before: " << endl;
    cout << add_m1;
    cout << add_m2;
    cout << "After: " << endl;
    add_m3 = add_m1 + add_m2;
    cout << add_m3;
    cout << "Before: " << endl;
    cout << add_com_m1;
    cout << "After: " << endl;
    cout << add_com_m2;

    //subtraction test
    cout << "Subtraction test:" << endl;
    Matrix<int> sub_m1(v1);
    Matrix<int> sub_m2(v4);
    Matrix<int> sub_m3 = sub_m1 - sub_m2;
    Matrix<double> sub_m4(v2);
    Matrix<double> sub_m5(v5);
    Matrix<double> sub_m6 = sub_m5 - sub_m4;
    cout << "Before: " << endl;
    cout << sub_m1;
    cout << sub_m2;
    cout << "After: " << endl;
    cout << sub_m3;


    //scalar multiplication
    cout << "scalar multiplication test:" << endl;
    Matrix<int> mul_m1(v1);
    Matrix<int> mul_m2 = mul_m1 * 3;
    cout << "Before: " << endl;
    cout << mul_m1;
    cout << "After: " << endl;
    cout << mul_m2;


    //scalar division
    cout << "scalar division test: " << endl;
    Matrix<double> div_m1(v2);
    Matrix<double> div_m2 = div_m1 / 4;
    cout << "Before: " << endl;
    cout << div_m1;
    cout << "After: " << endl;
    cout << div_m2;

    //transposition
    cout << "transposition test: " << endl;
    Matrix<double> tran_m1(v2);
    Matrix<double> tran_m2 = tran_m1.Transpose();
    cout << "Before: " << endl;
    cout << tran_m1;
    cout << "After: " << endl;
    cout << tran_m2;


    //conjugation
    cout << "conjugation test: " << endl;
    Matrix<complex<int>> conj_m1(com_v1);
    Matrix<complex<int>> conj_m2 = conj_m1.Conjugation();
    cout << "Before conjugation: " << endl;
    cout << conj_m1;
    cout << "After conjugation: " << endl;
    cout << conj_m2;

    //Element_Wise
    cout << "Element_Wise test: " << endl;
    Matrix<int> ele_m1(v1);
    Matrix<int> ele_m2(v4);
    Matrix<int> ele_m3 = ele_m1.Element_Wise(ele_m2);
    cout << "Before two: " << endl;
    cout << ele_m1;
    cout << ele_m2;
    cout << "After: " << endl;
    cout << ele_m3;

//    //matrix-matrix multiplication
//    cout << " matrix-matrix multiplication test: " << endl;
//    Matrix<int> mm_m1(v1);
//    Matrix<int> mm_m2(v6);
//    Matrix<int> mm_m3 = mm_m1 * mm_m2;
//
//
//    //matrix-vector multiplication
//    cout << "matrix-vector multiplication test: " << endl;
//    Matrix<int> mv_m1(v1);
//    Matrix<int> mv_m2 = mv_m1 * v6;
//
//
//    //dot product
//    cout << "dot product test: " << endl;
//    vector<vector<int>> d_v1 = v1 * v6;


    //cross product



    //max
    cout << "max, min, sum, avg, test: " << endl;
    Matrix<double> cal_m1(v2);
    cout << "matrix: " << endl;
    cout << cal_m1;
    cout << "max of all: " << cal_m1.max_all() << endl;
    cout << "min of all: " << cal_m1.min_all() << endl;
    cout << "max of row1: " << cal_m1.max_row(1) << endl;
    cout << "min of col1: " << cal_m1.min_col(1) << endl;
    cout << "avg of all: " << cal_m1.avg_all() << endl;






    //slicing and reshape
    cout << "slicing and reshape test: " << endl;
    vector<vector<double>> s(4, vector<double>(4));
    s = { {1,2,3,4},{5,6,7,8},{1,3,5,7},{2,4,6,8} };
    Matrix<double> S(s);
    cout << "Before slicing and reshape: " << endl;
    cout << S;
    Matrix<double> m = slicing(S, 1, 2, 3, 4);
    Matrix<double> b = slicing(S, 3, 4, 1, 2);
    Matrix<double> c = reshape(S, 3, 3);
    Matrix<double> d = reshape(b, 1, 4);
    cout << "slicing 1: " << endl;
    cout << m;
    cout << "slicing 2: " << endl;
    cout << b;
    cout << "reshape 1: " << endl;
    cout << c;
    cout << "reshape 2: " << endl;
    cout << d;


    //convolution
    cout << "convolution test: " << endl;
    vector<vector<double>> s1 = {{17,24,1,8,15},{23,5,7,14,16},{4,6,13,20,22},{10,12,19,21,3},{11,18,25,2,9}};
    Matrix<double> S1(s1);
    vector<vector<double>> e = {{1,3,1},{0,5,0},{2,1,2}};
    Matrix<double> E(e);
    cout << "Before convolution: " << endl;
    cout << "m1: " << endl;
    cout << S1;
    cout << "m2: " << endl;
    cout << E;
    Matrix<double> f = convolution(S1, E);
    cout << "After convolution: " << endl;
    cout << f;

//    zhuanhuan(f);

//    Matrix<double> g = Matrix<double>::nzh();
//    cout << g;

    return 0;
}
