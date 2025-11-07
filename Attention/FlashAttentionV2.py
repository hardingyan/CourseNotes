import math
import torch

from Ref import get_kv, repeat_kv


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
