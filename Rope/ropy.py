import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """将输入的后半部分取负"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rotate_query(q, m, theta):
    """
    q: [d]  query向量
    m: 位置（标量）
    theta: [d//2] 旋转频率
    """

    # 计算 cos(mθ) 和 sin(mθ)
    m_theta = m * theta  # [d/2]
    cos = torch.cos(m_theta)  # [d/2]
    sin = torch.sin(m_theta)  # [d/2]

    # 取出前后半部分
    # d = q.shape[-1]
    # half_d = d // 2
    # q_half1 = q[..., :half_d]
    # q_half2 = q[..., half_d:]
    # 计算旋转后的 q
    # q_rotated_half1 = q_half1 * cos - q_half2 * sin  # Re部分
    # q_rotated_half2 = q_half1 * sin + q_half2 * cos  # Im部分
    # 合并回 [d] 维向量
    # q_rotated = torch.cat([q_rotated_half1, q_rotated_half2], dim=-1)
    q_rotated = q * cos + rotate_half(q) * sin

    return q_rotated


def rotate_key(k, n, theta):
    """
    k: [d]  key向量
    n: 位置（标量）
    theta: [d//2] 旋转频率
    """

    # 计算 cos(nθ) 和 sin(nθ)
    n_theta = n * theta  # [d/2]
    cos = torch.cos(n_theta)  # [d/2]
    sin = torch.sin(n_theta)  # [d/2]

    # 取出前后半部分
    # d = k.shape[-1]
    # half_d = d // 2
    # k_half1 = k[..., :half_d]
    # k_half2 = k[..., half_d:]
    # 计算旋转后的 k
    # k_rotated_half1 = k_half1 * cos + k_half2 * sin  # Re部分
    # k_rotated_half2 = -k_half1 * sin + k_half2 * cos  # Im部分
    # 合并回 [d] 维向量
    # k_rotated = torch.cat([k_rotated_half1, k_rotated_half2], dim=-1)
    k_rotated = k * cos - rotate_half(k) * sin

    return k_rotated


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置嵌入（逐元素乘法形式）
    Args:
        q: [batch_size, seq_len, n_head, head_dim]
        k: [batch_size, seq_len, n_head, head_dim]
        cos: [1, seq_len, 1, head_dim]
        sin: [1, seq_len, 1, head_dim]
    Returns:
        旋转后的qk
    """
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) - (rotate_half(k) * sin)

    return q_rotated, k_rotated 

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 构建缓存
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        # 生成位置序列
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)

        # 计算所有位置的外积 [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # 复制一份形成复数表示 [seq_len, dim]
        freqs = torch.cat((freqs, freqs), dim=-1)

        # 转换为复数形式并应用旋转
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # 重塑为 [1, seq_len, 1, dim] 用于广播
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q, k, offset=0):
        """
        Args:
            q: [batch_size, seq_len, n_head, head_dim]
            k: [batch_size, seq_len, n_head, head_dim]
            offset: 用于缓存的位置偏移量
        """
        seq_len = q.size(1)

        # 动态扩展缓存（如果需要）
        if seq_len + offset > self.max_seq_len:
            new_max_len = max(seq_len + offset, self.max_seq_len * 2)
            self._build_cache(new_max_len)
            self.max_seq_len = new_max_len

        # 获取对应的位置频率
        cos = self.cos[:, offset:offset+seq_len]
        sin = self.sin[:, offset:offset+seq_len]

        # 应用旋转位置嵌入
        return apply_rotary_pos_emb(q, k, cos, sin)

if __name__ == "__main__":
    head_dim = 128
    batch_size = 2
    seq_len = 10
    n_heads = 4

    # 创建ROPE模块
    rope = RotaryPositionEmbedding(dim=head_dim)

    # 随机生成query和key
    query = torch.randn(batch_size, seq_len, n_heads, head_dim)
    key = torch.randn(batch_size, seq_len, n_heads, head_dim)

    # 应用旋转位置编码
    q_rotated, k_rotated = rope(query, key)

    print(f"{query.shape=}, {key.shape=}")
    print(f"{q_rotated.shape=}, {k_rotated.shape=}")
