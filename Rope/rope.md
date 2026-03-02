## ROPE（Rotary Position Embedding）计算公式

### 1. 旋转频率计算

对于维度为 `d` 的向量，计算每个维度对的旋转频率：

$$
\theta_j = 10000^{-2j/d}, \quad j \in [0, d/2-1]
$$

- `j`：维度索引（共 `d/2` 个）
- `d`：向量总维度（通常为偶数）
- `10000`：频率衰减系数。超参数，控制频率的衰减速度（较大的值会使高频成分衰减更快）

### 2. 复数表示

将向量分为实部和虚部，其中，$[0:d/2-1]$ 部分为实部，$[d/2:d-1]$ 部分为虚部：

$$
\begin{aligned}
q_{\text{complex}} &= [q_0 + iq_{d/2}, q_1 + iq_{d/2+1}, ..., q_{d/2-1} + iq_{d-1}] \\
k_{\text{complex}} &= [k_0 + ik_{d/2}, k_1 + ik_{d/2+1}, ..., k_{d/2-1} + ik_{d-1}]
\end{aligned}
$$

### 3. Query 旋转公式（位置 m）

$$
q_{\text{rotated}_j} = (q_j + iq_{j+d/2}) \cdot (\cos(m\theta_j) + i\sin(m\theta_j))
$$

根据复数乘法公式 $(a + ib) \cdot (c + id) = (ac - bd) + i(ad + bc)$

**实数形式展开：**

$$
\begin{cases}
\text{Re}(q_{\text{rotated}_j}) = q_j \cos(m\theta_j) - q_{j+d/2} \sin(m\theta_j) \\
\text{Im}(q_{\text{rotated}_j}) = q_j \sin(m\theta_j) + q_{j+d/2} \cos(m\theta_j)
\end{cases}
$$

### 4. Key 旋转公式（位置 n）

key 的旋转计算与 query **完全相同**，同样使用正向旋转（而非共轭）：

$$
k_{\text{rotated}_j} = (k_j + ik_{j+d/2}) \cdot (\cos(n\theta_j) + i\sin(n\theta_j))
$$

**实数形式展开：**

$$
\begin{cases}
\text{Re}(k_{\text{rotated}_j}) = k_j \cos(n\theta_j) - k_{j+d/2} \sin(n\theta_j) \\
\text{Im}(k_{\text{rotated}_j}) = k_j \sin(n\theta_j) + k_{j+d/2} \cos(n\theta_j)
\end{cases}
$$

### 5. 合并实数结果

把 Re 部分和 Im 部分合并到一起，组成旋转后的向量 $[\text{Re}, \text{Im}]$，则最终旋转后的向量：

$$
\begin{aligned}
q_{\text{rotated}} &= [\text{Re}(q_{\text{rotated}_0}), ..., \text{Re}(q_{\text{rotated}_{d/2-1}}), \text{Im}(q_{\text{rotated}_0}), ..., \text{Im}(q_{\text{rotated}_{d/2-1}})] \\
k_{\text{rotated}} &= [\text{Re}(k_{\text{rotated}_0}), ..., \text{Re}(k_{\text{rotated}_{d/2-1}}), \text{Im}(k_{\text{rotated}_0}), ..., \text{Im}(k_{\text{rotated}_{d/2-1}})]
\end{aligned}
$$

### 6. 相对位置编码性质

q 和 k 均使用正向旋转，点积结果自然包含相对位置信息。推导如下：

$$
\langle q_{\text{rotated}}(m),\ k_{\text{rotated}}(n) \rangle
= \text{Re}\left[ q_{\text{complex}} \cdot \bar{k}_{\text{complex}} \cdot e^{i(m-n)\theta} \right]
= f(m-n)
$$

其中，内积运算本身会对 $k$ 取共轭，因此 q 和 k 各自正向旋转后，点积中自然出现 $e^{i(m-n)\theta}$，从而编码了相对位置 $m - n$，无需对 k 做反向旋转。