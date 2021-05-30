                                      Project description：

Building a library for matrix computation

Matrix is an important concept introduced in linear algebra. Matrix calculation is widely used in many practical applications, such as image processing and machine learning. Programmers can indeed use many different existing libraries, and in certain cases, programmers are required to design their own matrix calculation libraries for specific implementations. This project will build a new library (do not attempt to directly copy codes from other existing library) that can perform the following operations on the matrix:
1) It supports all matrix sizes, from small fixed-size matrices to arbitrarily large dense matrices, and even sparse matrices (Add: try to use efficient ways to store the sparse matrices). (10 points)
2) It supports all standard numeric types, including std::complex, integers, and is easily extensible to custom numeric types. (10 points)
3) It supports matrix and vector arithmetic, including addition, subtraction, scalar multiplication, scalar division, transposition, conjugation, element-wise multiplication, matrix-matrix multiplication, matrix-vector multiplication, dot product and cross product. (20 points)
4) It supports basic arithmetic reduction operations, including finding the maximum value, finding the minimum value, summing all items, calculating the average value (all supporting axis-specific and all items). (10 points)
5) It supports computing eigenvalues and eigenvectors, calculating traces, computing inverse and computing determinant. (10 points)
6) It supports the operations of reshape and slicing. (10 points)
7) It supports convolutional operations of two matrices. (10 points)
8) It supports to transfer the matrix from OpenCV to the matrix of this library and vice versa. (10 points)
9) It should process likely exceptions as much as possible. (10 points)

建立矩阵计算库

矩阵是线性代数中引入的一个重要概念。矩阵计算在图像处理、机器学习等许多实际应用中有着广泛的应用。程序员确实可以使用许多不同的现有库，在某些情况下，程序员需要为特定的实现设计自己的矩阵计算库。此项目将构建一个新库（不要试图直接从其他现有库复制代码），该库可以对矩阵执行以下操作：

1.支持所有的矩阵大小，从小的固定大小的矩阵到任意大的密集矩阵，甚至稀疏矩阵（添加：尝试使用有效的方法来存储稀疏矩阵）(10分）

2.支持所有标准数字类型，包括std：：complex、integers，并且易于扩展到自定义数字类型(10分）

3.支持矩阵和向量运算，包括加法、减法、标量乘、标量除、换位、共轭、元素乘、矩阵乘、矩阵向量乘、点积和叉积(20分）

4.支持基本的算术归约运算，包括求最大值、求最小值、求和所有项目、计算平均值（所有支持特定轴和所有项目）(10分）

5.支持计算特征值和特征向量，计算轨迹，计算逆和计算行列式(10分）

6.支持整形和切片操作(10分）

7.支持两个矩阵的卷积运算(10分）

8.支持将矩阵从OpenCV传输到这个库的矩阵，反之亦然(10分）

9.尽可能多地处理可能的异常(10分）
