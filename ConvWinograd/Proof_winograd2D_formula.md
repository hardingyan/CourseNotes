# Winograd算法从1D推广到2D的形式化证明

## 1. 引言：
Winograd算法是CNN网络中Conv2D算子的一种常用加速方式，原论文直接给出了公式的1D形式和2D形式，并没有给出从1D形式推广到2D形式的推导过程。本文将给出Winograd算法从1D推广到2D的形式化证明。

## 2. Winograd算法的1D形式
记 $d = (d_0,d_1,d_2,d_3) ^ T$, $g = (w_0,w_1,w_2)^T$，以d为输入，g为卷积核的卷积运算结果$Conv(d,g)$是$y = (y_0,y_1)^T$
根据卷积算法的定义，有
$$y_0 = d_0w_0+d_1w_1+d_2w_2 \\
y_1 = d_1w_0+d_2w_1+d_3w_2$$
Winograd算法的1D形式是
$$y = A^T[(Gg)⊙(B^Td)] \tag{2.1} $$
式中，
$$B^T=\left[ \begin{matrix} 
1 &0  &-1 &0 \\ 
0 &1 &1 &0 \\ 
0 &-1 &1 &0 \\ 
0 &1 &0 &-1  
\end{matrix} \right]$$
$$G=\left[ \begin{matrix} 
1 &0  &0 \\ 
1/2 &1/2 &1/2 \\ 
1/2 &-1/2 &1/2 \\ 
0 &0 &1 
\end{matrix} \right]$$
$$A^T=\left[ \begin{matrix} 
1 &1  &1 &0 \\ 
0 &1 &-1 &-1 
\end{matrix} \right]$$

## 3. 从1D形式推广到2D形式的形式化证明
为从1D形式推广到2D形式，首先需要写出Winograd公式的行向量形式。延续上一节$y,d,g$都是列向量的约定，则$y^T,g^T,d^T$都是行向量。
$$\because y = A^T[(Gg)⊙(B^Td)]$$
$$\therefore y^T = (A^T[(Gg)⊙(B^Td)])^T$$
$$\therefore y^T = [(Gg)⊙(B^Td)]^TA$$
$$\therefore y^T = [(Gg)^T⊙(B^Td)^T]A$$
$$\therefore y^T = [(g^TG^T)⊙(d^TB)]A \tag{3.1} $$
$y^T = [(g^TG^T)⊙(d^TB)]A$就是Winograd 1D公式的行向量形式了。

下面来考虑2D情形。
记
$$D=\left[ \begin{matrix} 
d_0 &d_1  &d_2 &d_3 \\ 
d_4 &d_5 &d_6 &d_7 \\ 
d_8 &d_9 &d_{10} &d_{11} \\ 
d_{12} &d_{13} &d_{14} &d_{15}  
\end{matrix} \right]$$

$$W=\left[ \begin{matrix} 
w_0 &w_1  &w_2 \\ 
w_3 &w_4 &w_5 \\ 
w_6 &w_7 &w_8 
\end{matrix} \right]$$

以D为输入，W为卷积核的卷积运算结果$Conv(D,W)$是
$$Y = \left[ \begin{matrix} 
y_0 &y_1 \\ 
y_2 &y_3 
\end{matrix} \right]$$

借助im2col,把2D卷积写成矩阵乘法形式，

$$
\left[ \begin{matrix} 
y_0 \\ 
y_1 \\ 
y_2 \\ 
y_3 
\end{matrix} \right] 
= 
\left[ \begin{matrix} 
d_0 &d_1 &d_2 &d_4 &d_5 &d_6 &d_8 &d_9 &d_{10} \\ 
d_1 &d_2 &d_3 &d_5 &d_6 &d_7 &d_9 &d_10 &d_{11} \\ 
d_4 &d_5 &d_6 &d_8 &d_9 &d_{10} &d_{12} &d_{13} &d_{14} \\ 
d_5 &d_6 &d_7 &d_9 &d_{10} &d_{11} &d_{13} &d_{14} &d_{15} \\ 
\end{matrix} \right]
\cdot
\left[ \begin{matrix} 
w_0 \\ 
w_1 \\ 
w_2 \\ 
w_3 \\ 
w_4 \\ 
w_5 \\ 
w_6 \\ 
w_7 \\ 
w_8 \\ 
\end{matrix} \right] 
\tag{3.2} 
$$

引入符号
$$K_0 = \left[ \begin{matrix} 
d_0 &d_1 &d_2 \\ 
d_1 &d_2 &d_3 
\end{matrix} \right]$$

$$K_1 = \left[ \begin{matrix} 
d_4 &d_5 &d_6 \\ 
d_5 &d_6 &d_7 
\end{matrix} \right]$$

$$K_2 = \left[ \begin{matrix} 
d_8 &d_9 &d_{10} \\ 
d_{9} &d_{10} &d_{11} 
\end{matrix} \right]$$

$$K_3 = \left[ \begin{matrix} 
d_{12} &d_{13} &d_{14} \\ 
d_{13} &d_{14} &d_{15} 
\end{matrix} \right]$$

$$W_0 = \left[ \begin{matrix} 
w_0 \\ 
w_1 \\ 
w_2 \\ 
\end{matrix} \right]$$

$$W_1 = \left[ \begin{matrix} 
w_3 \\ 
w_4 \\ 
w_5 \\ 
\end{matrix} \right]$$

$$W_2 = \left[ \begin{matrix} 
w_6 \\ 
w_7 \\ 
w_8 \\ 
\end{matrix} \right]$$

$$R_0 = \left[ \begin{matrix} 
y_0 \\ 
y_1 \\ 
\end{matrix} \right]$$

$$R_1 = \left[ \begin{matrix} 
y_2 \\ 
y_3 \\ 
\end{matrix} \right]$$

那么公式$(3.2)$就可以写成
$$
\left[ \begin{matrix} 
R_0 \\ 
R_1 \\ 
\end{matrix} \right] 
= 
\left[ \begin{matrix} 
K_0 &K_1 &K_2 \\ 
K_1 &K_2 &K_3 \\ 
\end{matrix} \right]
\cdot
\left[ \begin{matrix} 
W_0 \\ 
W_1 \\ 
W_2 \\ 
\end{matrix} \right] 
\tag{3.3} 
$$

把式子$(3.3)$写成算数形式

$$
\left[ \begin{matrix} 
R_0 \\ 
R_1 \\ 
\end{matrix} \right] 
= 
\left[ \begin{matrix} 
K_0W_0 + K_1W_1 + K_2W_2 \\ 
K_1W_0 + K_2W_1 + K_3W_2 \\ 
\end{matrix} \right]
\tag{3.4} 
$$

引入符号
$$D_0 = \left[ \begin{matrix} 
d0 &d1 &d2 &d3
\end{matrix} \right] $$

$$D_1 = \left[ \begin{matrix} 
d4 &d5 &d6 &d7
\end{matrix} \right] $$

$$D_2 = \left[ \begin{matrix} 
d_8 &d_9 &d_{10} &d_{11} 
\end{matrix} \right] $$

$$D_3 = \left[ \begin{matrix} 
d_{12} &d_{13} &d_{14} &d_{15} 
\end{matrix} \right] $$

注意到
$$(K_0W_0)^T = Conv(D_0, {W_0}^T)$$
$$(K_1W_1)^T = Conv(D_1, {W_1}^T)$$
$$(K_2W_1)^T = Conv(D_2, {W_2}^T)$$
$$(K_1W_0)^T = Conv(D_1, {W_0}^T)$$
$$(K_2W_1)^T = Conv(D_2, {W_1}^T)$$
$$(K_3W_2)^T = Conv(D_3, {W_2}^T)$$

所以
$$
\left[ \begin{matrix} 
{R_0}^T \\ 
{R_1}^T \\ 
\end{matrix} \right] 
=
\left[ \begin{matrix} 
(K_0W_0)^T + (K_1W_1)^T + (K_1W_1)^T \\ 
(K_1W_0)^T + (K_2W_1)^T + (K_3W_2)^T \\ 
\end{matrix} \right]
=
\left[ \begin{matrix} 
Conv(D_0, {W_0}^T) + Conv(D_1, {W_1}^T) + Conv(D_2, {W_2}^T) \\ 
Conv(D_1, {W_0}^T) + Conv(D_2, {W_1}^T) + Conv(D_3, {W_2}^T) \\ 
\end{matrix} \right]
\tag{3.5} 
$$
注意到$D_0$和${W_0}^T$都是1D的行向量，所以这里可以使用1D的行向量形式的Winograd公式，即
$$Conv(D_0, {W_0}^T) = [({W_0}^TG^T)⊙(D_0B)]A$$
$$Conv(D_1, {W_1}^T) = [({W_1}^TG^T)⊙(D_1B)]A$$
$$Conv(D_2, {W_2}^T) = [({W_2}^TG^T)⊙(D_2B)]A$$
$$Conv(D_1, {W_0}^T) = [({W_0}^TG^T)⊙(D_1B)]A$$
$$Conv(D_2, {W_1}^T) = [({W_1}^TG^T)⊙(D_2B)]A$$
$$Conv(D_3, {W_2}^T) = [({W_2}^TG^T)⊙(D_3B)]A$$

代入到式子$(3.5)$，得
$$
\left[ \begin{matrix} 
{R_0}^T \\ 
{R_1}^T \\ 
\end{matrix} \right] 
=
\left[ \begin{matrix} 
[({W_0}^TG^T)⊙(D_0B)]A + [({W_1}^TG^T)⊙(D_1B)]A + [({W_2}^TG^T)⊙(D_2B)]A \\ 
[({W_0}^TG^T)⊙(D_1B)]A + [({W_1}^TG^T)⊙(D_2B)]A + [({W_2}^TG^T)⊙(D_3B)]A \\ 
\end{matrix} \right]
$$
把右乘$A$提出，得
$$
\left[ \begin{matrix} 
{R_0}^T \\ 
{R_1}^T \\ 
\end{matrix} \right] 
=
\left[ \begin{matrix} 
[({W_0}^TG^T)⊙(D_0B)] + [({W_1}^TG^T)⊙(D_1B)] + [({W_2}^TG^T)⊙(D_2B)] \\ 
[({W_0}^TG^T)⊙(D_1B)] + [({W_1}^TG^T)⊙(D_2B)] + [({W_2}^TG^T)⊙(D_3B)] \\ 
\end{matrix} \right]A
\tag{3.6} 
$$

注意到式$(3.6)$大括号内的部分，恰好是以列向量$({W_0}^T, {W_1}^T, {W_2}^T)^TG^T$作为卷积核，以列向量$(D_0,D_1,D_2,D_3)^TB$为输入的卷积运算。所以
$$
\left[ \begin{matrix} 
{R_0}^T \\ 
{R_1}^T \\ 
\end{matrix} \right] 
=
A^T
\left[ \begin{matrix} 
(G(({W_0}^T, {W_1}^T, {W_2}^T)^TG^T))⊙(B^T((D_0,D_1,D_2,D_3)^TB)) 
\end{matrix} \right]A
\tag{3.7} 
$$
注意到$({W_0}^T, {W_1}^T, {W_2}^T)^T$正是卷积核$W$，$(D_0,D_1,D_2,D_3)^T$正是输入$D$，所以有
$$
\left[ \begin{matrix} 
{R_0}^T \\ 
{R_1}^T \\ 
\end{matrix} \right] 
=
A^T
\left[ \begin{matrix} 
(GWG^T)⊙(B^TDB) 
\end{matrix} \right]A
\tag{3.8} 
$$
带入
$R_0 = \left[ \begin{matrix} 
y_0 \\ 
y_1 \\ 
\end{matrix} \right]$，
$R_1 = \left[ \begin{matrix} 
y_2 \\ 
y_3 \\ 
\end{matrix} \right]$
，得
$$Y 
=
A^T
\left[ \begin{matrix} 
(GWG^T)⊙(B^TDB) 
\end{matrix} \right]A$$

Winograd公式的二维形式得证。
