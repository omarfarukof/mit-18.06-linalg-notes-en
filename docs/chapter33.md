
# Lecture 33：Quiz 3 review

在上一次复习中，我们已经涉及了求特征值与特征向量（通过解方程$\det(A-\lambda I)=0$得出$\lambda$，再将$\lambda$带入$A-\lambda I$求其零空间得到$x$）。

接下的章节来我们学习了：

* 解微分方程$\frac{\mathrm{d}u}{\mathrm{d}t}=Au$，并介绍了指数矩阵$e^{At}$；
* 介绍了对称矩阵的性质$A=A^T$，了解了其特征值均为实数且总是存在足量的特征向量（即使特征值重复特征向量也不会短缺，总是可以对角化）；同时对称矩阵的特征向量正交，所以对称矩阵对角化的结果可以表示为$A=Q\Lambda Q^T$；
* 接着我们学习了正定矩阵；
* 然后学习了相似矩阵，$B=M^{-1}AM$，矩阵$A,B$特征值相同，其实相似矩阵是用不同的基表示相同的东西；
* 最后我们学习了Singular value decomposition$A=U\varSigma V^T$。

现在，我们继续通过例题复习这些知识点。

1. *解方程$\frac{\mathrm{d}u}{\mathrm{d}t}=Au=\begin{bmatrix}0&-1&0\\1&0&-1\\0&1&0\end{bmatrix}u$*。

    首先通过$A$的特征值/向量求通解$u(t)=c_1e^{\lambda_1t}x_1+c_2e^{\lambda_2t}x_2+c_3e^{\lambda_3t}x_3$，很明显矩阵是奇异的，所以有$\lambda_1=0$；
    
    继续观察矩阵会发现$A^T=-A$，这是一个反对称矩阵（anti-symmetric）或斜对陈矩阵（skew-symmetric），这与我们在第二十一讲介绍过的旋转矩阵类似，它的特征值应该为纯虚数（特征值在虚轴上），所以我们猜测其特征值应为$0\cdot i,\ b\cdot i,\ -b\cdot i$。通过解$\det(A-\lambda I)=0$验证一下：$\begin{bmatrix}-\lambda&-1&0\\1&-\lambda&-1\\0&1&\lambda\end{bmatrix}=\lambda^3+2\lambda=0, \lambda_2=\sqrt 2i, \lambda_3=-\sqrt 2i$。
    
    此时$u(t)=c_1+c_2e^{\sqrt 2it}x_2+c_3e^{-\sqrt 2it}x_3$，$e^{\sqrt 2it}$始终在复平面单位圆上，所以$u(t)$及不发散也不收敛，它只是具有周期性。当$t=0$时有$u(0)=c_1+c_2+c_3$，如果使$e^{\sqrt 2iT}=1$即$\sqrt 2iT=2\pi i$则也能得到$u(T)=c_1+c_2+c_3$，周期$T=\pi\sqrt 2$。
    
    另外，反对称矩阵同对称矩阵一样，具有正交的特征向量。当矩阵满足什么条件时，其特征向量相互正交？答案是必须满足$AA^T=A^TA$。所以对称矩阵$A=A^T$满足此条件，同时反对称矩阵$A=-A^T$也满足此条件，而正交矩阵$Q^{-1}=Q^T$同样满足此条件，这三种矩阵的特征向量都是相互正交的。
    
    上面的解法并没有求特征向量，进而通过$u(t)=e^{At}u(0)$得到通解，现在我们就来使用指数矩阵来接方程。如果矩阵可以对角化（在本例中显然可以），则$A=S\Lambda S^{-1}, e^{At}=Se^{\Lambda t}S^{-1}=S\begin{bmatrix}e^{\lambda_1t}&&&\\&e^{\lambda_1t}&&\\&&\ddots&\\&&&e^{\lambda_1t}\end{bmatrix}S^{-1}$，这个公式在能够快速计算$S,\lambda$时很方便求解。

2. 已知矩阵的特征值$\lambda_1=0,\lambda_2=c,\lambda_3=2$，特征向量$x_1=\begin{bmatrix}1\\1\\1\end{bmatrix},x_2=\begin{bmatrix}1&-1&0\end{bmatrix},x_3=\begin{bmatrix}1\\1\\-2\end{bmatrix}$：
    
    *$c$如何取值才能保证矩阵可以对角化？*其实可对角化只需要有足够的特征向量即可，而现在特征向量已经足够，所以$c$可以取任意值。
    
    *$c$如何取值才能保证矩阵对称？*我们知道，对称矩阵的特征值均为实数，且注意到给出的特征向量是正交的，有了实特征值及正交特征向量，我们就可以得到对称矩阵。
    
    *$c$如何取值才能使得矩阵正定？*已经有一个零特征值了，所以矩阵不可能是正定的，但可以是半正定的，如果$c$去非负实数。
    
    *$c$如何取值才能使得矩阵是一个马尔科夫矩阵？*在第二十四讲我们知道马尔科夫矩阵的性质：必有特征值等于$1$，其余特征值均小于$1$，所以$A$不可能是马尔科夫矩阵。
    
    *$c$取何值才能使得$P=\frac{A}{2}$是一个投影矩阵？*我们知道投影矩阵的一个重要性质是$P^2=P$，所以有对其特征值有$\lambda^2=\lambda$，则$c=0,2$。
    
    题设中的正交特征向量意义重大，如果没有正交这个条件，则矩阵$A$不会是对称、正定、投影矩阵。因为特征向量的正交性我们才能直接去看特征值的性质。

3. 复习Singular value decomposition，$A=U\varSigma V^T$：

    先求正交矩阵$V$：$A^TA=V\varSigma^TU^TU\varSigma V^T=V\left(\varSigma^T\varSigma\right)V^T$，所以$V$是矩阵$A^TA$的特征向量矩阵，而矩阵$\varSigma^T\varSigma$是矩阵$A^TA$的特征值矩阵，即$A^TA$的特征值为$\sigma^2$。
    
    接下来应该求正交矩阵$U$：$AA^T=U\varSigma^TV^TV\varSigma U^T=U\left(\varSigma^T\varSigma\right)U^T$，但是请注意，我们在这个式子中无法确定特征向量的符号，我们需要使用$Av_i=\sigma_iu_i$，通过已经求出的$v_i$来确定$u_i$的符号（因为$AV=U\varSigma$），进而求出$U$。
    
    *已知$A=\bigg[u_1\ u_2\bigg]\begin{bmatrix}3&0\\0&2\end{bmatrix}\bigg[v_1\ v_2\bigg]^T$*
    
    从已知的$\varSigma$矩阵可以看出，$A$矩阵是非奇异矩阵，因为它没有零奇异值。另外，如果把$\varSigma$矩阵中的$2$改成$-5$，则题目就不再是Singular value decomposition了，因为奇异值不可能为负；如果将$2$变为$0$，则$A$是奇异矩阵，它的秩为$1$，零空间为$1$维，$v_2$在其零空间中。

4. *$A$是正交对称矩阵，那么它的特征值具有什么特点*？

    首先，对于对称矩阵，有特征值均为实数；然后是正交矩阵，直觉告诉我们$|\lambda|=1$。来证明一下，对于$Qx=\lambda x$，我们两边同时取模有$\|Qx\|=|\lambda|\|x\|$，而**正交矩阵不会改变向量长度**，所以有$\|x\|=|\lambda|\|x\|$，因此$\lambda=\pm1$。
    
    *$A$是正定的吗？*并不一定，因为特征向量可以取$-1$。
    
    *$A$的特征值没有重复吗？*不是，如果矩阵大于$2$阶则必定有重复特征值，因为只能取$\pm1$。
    
    *$A$可以被对角化吗？*是的，任何对称矩阵、任何正交矩阵都可以被对角化。
    
    *$A$是非奇异矩阵吗？*是的，正交矩阵都是非奇异矩阵。很明显它的特征值都不为零。
    
    *证明$P=\frac{1}{2}(A+I)$是投影矩阵*。
    
    我们使用投影矩阵的性质验证，首先由于$A$是对称矩阵，则$P$一定是对称矩阵；接下来需要验证$P^2=P$，也就是$\frac{1}{4}\left(A^2+2A+I\right)=\frac{1}{2}(A+I)$。来看看$A^2$是什么，$A$是正交矩阵则$A^T=A^{-1}$，而$A$又是对称矩阵则$A=A^T=A^{-1}$，所以$A^2=I$。带入原式有$\frac{1}{4}(2A+2I)=\frac{1}{2}(A+I)$，得证。
    
    我们可以使用特征值验证，$A$的特征值可以取$\pm1$，则$A+I$的特征值可以取$0,2$，$\frac{1}{2}(A+I)$的特征值为$0,1$，特征值满足投影矩阵且它又是对称矩阵，得证。
