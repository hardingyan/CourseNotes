import math
import torch

from Ref import repeat_kv
from diff_tensors import compute_diff, print_diff

def gqa_trick(seq_len, num_heads, qk_head_size, kv_len, num_kv_heads, v_head_size):
    def ref(q, k ,v):
        q = q.clone()
        k = k.clone()
        v = v.clone()

        k, v = repeat_kv(k, v, 1, num_heads // num_kv_heads)

        q = q.transpose(0, 1).view(num_heads, seq_len, qk_head_size)
        k = k.transpose(0, 1).view(num_heads, kv_len, qk_head_size)
        v = v.transpose(0, 1).view(num_heads, kv_len, v_head_size)

        kT = k.transpose(1, 2).view(num_heads, qk_head_size, kv_len)

        qkT = torch.bmm(q, kT)  # (num_heads,seq_len, kv_len)

        o = torch.matmul(qkT, v)
        o = o.transpose(0, 1).view(seq_len, num_heads, v_head_size)

        return o

    def act(q, k, v):
        q = q.clone()
        k = k.clone()
        v = v.clone()

        group_num = num_kv_heads
        group_size = num_heads // group_num

        q = q.view(seq_len, group_num, group_size, qk_head_size)
        k = k.view(kv_len, group_num, 1, qk_head_size)
        v = v.view(kv_len, group_num, 1, v_head_size)

        out = torch.zeros(
            (seq_len, group_num, group_size, qk_head_size), dtype=dtype, device=device
        )

        for g0 in range(group_num):
            group_slice = slice(g0, g0 + 1)
            q_slice = q[:, group_slice, :, :].reshape(seq_len * group_size, qk_head_size)
            k_slice = k[:, group_slice, :, :].reshape(kv_len, qk_head_size)
            v_slice = v[:, group_slice, :, :].reshape(kv_len, v_head_size)

            kT = k_slice.transpose(0, 1).view(qk_head_size, kv_len)

            qkT = torch.matmul(q_slice, kT)  # (seq_len*group_size, kv_len)

            o = torch.matmul(qkT, v_slice)  # (seq_len*group_size, v_head_size)
            o = o.view(seq_len, 1, group_size, v_head_size)

            out[:, group_slice, :, :] = o

        out = out.view(seq_len, num_heads, qk_head_size)

        return out

    dtype = torch.float16
    device = "cpu"

    q = torch.randn(seq_len, num_heads, qk_head_size, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_kv_heads, qk_head_size, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_kv_heads, v_head_size, dtype=dtype, device=device)

    out_act = act(q, k, v)
    out_ref = ref(q, k, v)

    out_act = out_act.detach().numpy()
    out_ref = out_ref.detach().numpy()

    abs_error, rel_error, top_indices = compute_diff(out_act, out_ref, idx_cnt=20)
    print_diff(
        out_act, out_ref, "act", "ref", abs_error, rel_error, top_indices, eps=1e-1
    )

def paged_flash_attention_v2(q, k_cache, v_cache, page_table, seq_lengths_host, kv_lengths_host, mask, qk_scale=None):
    """
        q: [batch_seq_len, num_heads, qk_head_size]
        k_cache: [num_pages, page_size, num_kv_heads, qk_head_size]
        v_cache: [num_pages, page_size, num_kv_heads, v_head_size]
        page_table: [num_batches, num_pages]
        seq_lengths_host: [num_batches]
        kv_lengths_host: [num_batches]
    """
    num_batches = len(seq_lengths_host)

    outs = []
    offset = 0

    return outs


if __name__ == "__main__":
    # seq_len, num_heads, qk_head_size, kv_len, num_kv_heads, v_head_size
    gqa_trick(10, 16, 128, 20, 2, 128)
