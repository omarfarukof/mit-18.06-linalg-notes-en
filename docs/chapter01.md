
# Lecture 1: The Geometry of Linear Equations

We begin the class by solving the system of linear equations，Start with a common example：The system of equations has$2$unknowns, with$2$equations.Look at the "row image" and "column image" of the equations respectively.

System of equations are: $\begin{cases}2x&-y&=0\\-x&+2y&=3\end{cases}$， We write matrix as: $\begin{bmatrix}2&-1\\-1&2\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}$

Usually we call the first matrix a coefficient matrix : $A$ ，The second matrix is called a vector : $x$ ，The third matrix is called a vector : $b$.

Thus the system of linear equations can be expressed as : $Ax=b$

Let's look at the line image, the image in the rectangular coordinate system ：


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

x = [-2, 2, -2, 2]
y = [-4, 4, 0.5, 2.5]

fig = plt.figure()
plt.axhline(y=0, c='black')
plt.axvline(x=0, c='black')

plt.plot(x[:2], y[:2], x[2:], y[2:])

plt.draw()
```


![png](img/chapter01_1_0.png)



```python
plt.close(fig)
```

The figure above shows the intersection of two straight lines in a rectangular coordinate system that we are all familiar with. Next, we'll look at the equations in columns: $x\begin{bmatrix}2\\-1\end{bmatrix}+y\begin{bmatrix}-1\\2\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}$

(We will call the first vector as $col_1$，The second vector as $col_2$，to represent the first column vector and the second column vector）.

To make the formula hold, you need to add twice the second vector to the first vector, i.e

 $1\begin{bmatrix}2\\-1\end{bmatrix}+2\begin{bmatrix}-1\\2\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}$

Now look at the column image and draw the column vector above on a 2D plane：


```python
from functools import partial

fig = plt.figure()
plt.axhline(y=0, c='black')
plt.axvline(x=0, c='black')
ax = plt.gca()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-3, 4)

arrow_vector = partial(plt.arrow, width=0.01, head_width=0.1, head_length=0.2, length_includes_head=True)

arrow_vector(0, 0, 2, -1, color='g')
arrow_vector(0, 0, -1, 2, color='c')
arrow_vector(2, -1, -2, 4, color='b')
arrow_vector(0, 0, 0, 3, width=0.05, color='r')

plt.draw()
```


![png](img/chapter01_4_0.png)



```python
plt.close(fig)
```

Green vector, as shown $col_1$blue-green vector (double blue-green vector $col_2$）red vector$b$.

Now, we'll continue our observation. $x\begin{bmatrix}2\\-1\end{bmatrix}+y\begin{bmatrix}-1\\2\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}$，$col_1,col_2$some linear combination of , yields the vector$b$，So$col_1,col_2$What can be achieved by all linear combinations? They will be spread over the entire plane.

Let's move on to three unknown equations：$\begin{cases}2x&-y&&=0\\-x&+2y&-z&=-1\\&-3y&+4z&=4\end{cases}$，writing matrix form $A=\begin{bmatrix}2&-1&0\\-1&2&-1\\0&-3&4\end{bmatrix},\ b=\begin{bmatrix}0\\-1\\4\end{bmatrix}$。

In a three-dimensional rectangular coordinate system, each equation will determine a plane, and the three planes in the example will intersect at one point, which is the solution of the system of equations.

Similarly, write the system of equations as a linear combination of column vectors and observe column images：$x\begin{bmatrix}2\\-1\\0\end{bmatrix}+y\begin{bmatrix}-1\\2\\-3\end{bmatrix}+z\begin{bmatrix}0\\-1\\4\end{bmatrix}=\begin{bmatrix}0\\-1\\4\end{bmatrix}$. In the example specially arranged, the last column vector happens to be equal to the $b$ vector on the right side of the equation，So the linear combination we need is $x=0,y=0,z=1$. Suppose we let $b=\begin{bmatrix}1\\1\\-3\end{bmatrix}$，Then the required linear combination is $x=1,y=1,z=0$。

We can't always find the right linear combination so easily, so we'll talk about elimination method, a systematic solution to linear equations.

Now, we need to consider that for any arbitrary $b$，Can it be solved $Ax=b$？
From the point of view of linear combination of column vectors, can the linear combination of column vectors cover the whole 3D vector space? For the above example, the answer is yes. In this example, $A$ is our preferred type of matrix, but for others, the answer is no. So under what circumstances can a linear combination of three vectors not get $b$?

If, the three vectors are in the same plane, the problem arises, then their linear combinations must also all be in this plane. For example, such as $col_3=col_1+col_2$, then no matter how the combination is made, the results of these three vectors cannot escape the plane, so when $b$ is in the plane, the equation system has a solution, and when $b$ is not in the plane, these three column vectors cannot construct $b$. In later lessons, we will learn that this situation is called **singular**, **matrix irreversibility**.

Next, we extend it to the nine-dimensional space. Each equation has nine unknowns, a total of nine equations. At this time, it is impossible to describe the problem from the coordinate image, but we can still solve the problem from the perspective of finding a linear combination of nine-dimensional column vectors. Still the above question, is it always possible to get $b$? Of course, it still depends on these nine vectors. If we take some vectors that are not independent of each other, the answer is no. For example, if we take nine columns, but it is only equivalent to eight columns, one column has no contribution (this column is the previous column). some linear combination of columns), there will be a part of $b$ that cannot be obtained.

Next, we will introduce the matrix form of the equation $Ax=b$, which is a multiplication operation. For example, take $A=\begin{bmatrix}2&5\\1&3\end{bmatrix},\ x=\begin{bmatrix }1\\2\end{bmatrix}$, let's see how to calculate matrix multiplied by vector.

* We still use the linear combination of column vectors, calculating one column at a time，$\begin{bmatrix}2&5\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix}=1\begin{bmatrix}2\\1\end{bmatrix}+2\begin{bmatrix}5\\3\end{bmatrix}=\begin{bmatrix}12\\7\end{bmatrix}$
* Another method, using the vector inner product, the first row of the matrix is dot-multiplication by the $x$ vector$\begin{bmatrix}2&5\end{bmatrix}\cdot\begin{bmatrix}1&2\end{bmatrix}^T=12,\ \begin{bmatrix}1&3\end{bmatrix}\cdot\begin{bmatrix}1&2\end{bmatrix}^T=7$.

It is suggested using the first method, treating $Ax$ as a linear combination of $A$ column vectors.
