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
    vector<vector<int>> v6 = {{2, 3},
                              {2, 3}};
    Vector<int> vector1{{1,1}};


    vector<vector<double>> v5 = {{2.1, 2.2, 5.4},
                                 {5.9, 5.8, 7.3}};

    //Addition test
    cout << "Addition test:" << endl;
    Matrix<int> add_m1(v1);
    Matrix<int> add_m2(v3);
    Matrix<int> add_m3;
    Matrix<complex<int>> add_com_m1(com_v1);
    Matrix<complex<int>> add_com_m2 = add_com_m1 + add_com_m1;
    cout << "Before: " << endl;
    cout << add_m1;
    cout << add_m2;
    cout << "After adding itself: " << endl;
    add_m3 = add_m1 + add_m1;
    cout << add_m3;
    cout << "Before: " << endl;
    cout << add_com_m1;
    cout << "After adding itself: " << endl;
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
    cout << "m1: " << endl;
    cout << sub_m1;
    cout << "m2: " << endl;
    cout << sub_m2;
    cout << "After: " << endl;
    cout << sub_m3;


    //scalar multiplication
    cout << "scalar multiplication test:" << endl;
    Matrix<int> mul_m1(v1);
    Matrix<int> mul_m2 = mul_m1 * 3;
    cout << "Before: " << endl;
    cout << mul_m1;
    cout << "After multiplying by 3: " << endl;
    cout << mul_m2;


    //scalar division
    cout << "scalar division test: " << endl;
    Matrix<double> div_m1(v2);
    Matrix<double> div_m2 = div_m1 / 4;
    cout << "Before: " << endl;
    cout << div_m1;
    cout << "After dividing 4: " << endl;
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
    cout << "Before: " << endl;
    cout << "m1: " << endl;
    cout << ele_m1;
    cout << "m2: " << endl;
    cout << ele_m2;
    cout << "After: " << endl;
    cout << ele_m3;

    //matrix-matrix multiplication
    cout << " matrix-matrix multiplication test: " << endl;
    Matrix<int> mm_m1(v1);
    Matrix<int> mm_m2(v6);
    cout << "Before multiply: " << endl;
    Matrix<int> mm_m3 = mm_m1 * mm_m2;

    cout << "m1: " << endl;
    cout << mm_m1;
    cout << "m2: " << endl;
    cout << mm_m2;
    cout << "After multiply: " << endl;
    cout << mm_m3;


    //matrix-vector multiplication
    cout << "matrix-vector multiplication test: " << endl;
    Matrix<int> mv_m1(v1);
    Vector<int> mv_m2(v1[0].size());
    cout << "m1: " << endl;
    cout << mv_m1;
    cout << "Vector: " << endl;
    cout << vector1;
    cout << "After multiply: " << endl;
    mv_m2 = mv_m1 * vector1;
    cout << mv_m2;


    //dot product
    cout << "dot product test: " << endl;
    int dot1 = vector1 * vector1;
    cout << dot1 << endl;


    //cross product
//    cout << "cross product test: " << endl;
//    Vector<int> cross_v1(vector1.value.size());
//    cross_v1 = vector1 * vector1;


    //max
    cout << "max, min, sum, avg, test: " << endl;
    Matrix<double> cal_m1(v2);
    cout << "matrix: " << endl;
    cout << cal_m1;
    cout << "max of all: " << cal_m1.max_all() << endl;
    cout << "min of all: " << cal_m1.min_all() << endl;
    cout << "max of row1: " << cal_m1.max_row(1) << endl;
    cout << "min of row1: " << cal_m1.min_row(1) << endl;
    cout << "avg of all: " << cal_m1.avg_all() << endl;

    //trace and determinant
    cout << "trace and determinant test: " << endl;
    vector<vector<int>> trace_v1 = {{1, 2},
                                    {3, 1}};
    vector<vector<int>> trace_v2 = {{0, 3},
                                    {2, 1}};
    vector<vector<int>> trace_v3 = {{0, 3, 0, 6},
                                   {2, 1, 4, 2},
                                   {0, 9, 0, 3},
                                   {6, 3, 2, 1}};
    vector<vector<int>> trace_v4 = {{1 , 2},
                                    {-1 , -3}};
    vector<vector<double>> trace_v5 = {{1 , 2 , -4},
                                    {2 , -5 , 8},
                                    {-4 , 8 , 7}};
    Matrix<int> trace_m1(trace_v1);
    Matrix<int> trace_m2(trace_v2);
    Matrix<int> trace_m3(trace_v3);

    cout << "trace of m1: ";
    cout << trace_m1.Traces() << "\n";
    cout << "determinant of m1: ";
    cout << trace_m1.determinant(trace_v1, 2) << "\n";
    cout << "trace of m2: ";
    cout << trace_m2.Traces() << "\n";
    cout << "determinant of m2: ";
    cout << trace_m2.determinant(trace_v2, 2) << "\n";
    cout << "trace of m3: ";
    cout << trace_m3.Traces() << "\n";
    cout << "determinant of m3: ";
    cout << trace_m3.determinant(trace_v3, 4) << "\n";


    //inverse
    cout << "inverse test: " << endl;
    cout << "Before inverse: " << endl;
    cout << "m1: " << endl;
    Matrix<int> inverse_m1(trace_v4);
    cout << inverse_m1;
    vector<vector<int>> inverse_v1;
    inverse_m1.GetMatrixInverse(inverse_v1);
    cout << "After inverse: " << endl;
    Matrix<int> inverse_m2(inverse_v1);
    cout << inverse_m2;


    //eigenvalue eigenvector
    cout << "eigen test: " << endl;
    cout << "Before eigen: " << endl;
    cout << "m1: " << endl;
    Matrix<double> eigen_m1(trace_v5);
    cout << eigen_m1;
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
    eigen_m1.eigen(eigenvalues , eigenvectors);
    cout << "After eigen: " << endl;
    cout << "eigen values: " << endl;
    for(int eigenvalue : eigenvalues){
        cout << eigenvalue << " ";
    }
    cout << endl;
    cout << "eigen vectors: " << endl;
    for(int i = 0 ; i < eigenvectors[0].size() ; i++){
        for(int j = 0 ; j < eigenvectors.size() ; j++){
            cout << eigenvectors[i][j] << " ";
        }
        cout << endl;
    }



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

    zhuanhuan(f);
    Matrix<double> g = Matrix<double>::nzh();
    cout << g;

    return 0;
}
