
# Lecture 9：Independence, Basis and Dimension

$v_1,\ v_2,\ \cdots,\ v_n$是$m\times n$矩阵$A$的列向量：

如果$A$零空间中有且仅有$0$向量，则各向量线性无关，$rank(A)=n$。

如果存在非零向量$c$使得$Ac=0$，则存在线性相关向量，$rank(A)\lt n$。

向量空间$S$中的一组基（basis），具有两个性质：

1. 他们线性无关；
2. 他们可以生成$S$。

对于向量空间$\mathbb{R}^n$，如果$n$个向量组成的矩阵为可逆矩阵，则这$n$个向量为该空间的一组基，而数字$n$就是该空间的维数（dimension）。

举例：
$
A=
\begin{bmatrix}
1 & 2 & 3 & 1 \\
1 & 1 & 2 & 1 \\
1 & 2 & 3 & 1 \\
\end{bmatrix}
$
，A的列向量线性相关，其零空间中有非零向量，所以$rank(A)=2=主元存在的列数=列空间维数$。

可以很容易的求得$Ax=0$的两个解，如
$
x_1=
\begin{bmatrix}
-1 \\
-1 \\
1 \\
0 \\
\end{bmatrix}, 
x_2=
\begin{bmatrix}
-1 \\
0 \\
0 \\
1 \\
\end{bmatrix}
$，根据前几讲，我们知道特解的个数就是自由变量的个数，所以$n-rank(A)=2=自由变量存在的列数=零空间维数$

我们得到：列空间维数$dim C(A)=rank(A)$，零空间维数$dim N(A)=n-rank(A)$
