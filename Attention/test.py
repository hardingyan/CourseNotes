import numpy as np

import torch
from diff_tensors import compute_diff, print_diff

from Ref import paged_attention, attention, flash_attention_v2
from FlashAttentionV2 import paged_flash_attention_v2


def generate_seq_lengths(seq_len, num_batches):
    """
    Randomly distributes a total sequence length (seq_len)
    across multiple requests, returning sorted lengths that sum to seq_len.

    Example: generate_seq_lengths(6, 2) might return [2, 4]
    (two requests with 2,4 tokens each)
    """
    if seq_len == 0:
        return [0] * num_batches
    seq_lengths = list(
        np.random.choice(range(1, seq_len), size=num_batches - 1, replace=False)
    )
    seq_lengths.append(0)
    seq_lengths.append(seq_len)
    seq_lengths = sorted(seq_lengths)
    seq_lengths = [seq_lengths[i + 1] - seq_lengths[i] for i in range(num_batches)]
    seq_lengths = sorted(seq_lengths)
    return seq_lengths


def get_prefill_mask(mask_size: int, dtype=torch.float16, device="cpu") -> torch.Tensor:
    mask = torch.tril(torch.ones(mask_size, mask_size), diagonal=0).to(dtype).to(device)
    mask[mask == 0] = float("-inf")
    mask[mask == 1] = 0

    return mask


def test_paged_attention(
    num_batches,
    batch_seq_len,
    batch_kv_len,
    num_heads,
    num_kv_heads,
    head_size,
    **kwargs,
):
    # q: [batch_seq_len, num_heads, qk_head_size]
    # k_cache: [num_pages, page_size, num_kv_heads, qk_head_size]
    # v_cache: [num_pages, page_size, num_kv_heads, v_head_size]
    # page_table: [num_batches, num_pages]
    # seq_lengths_host: [num_batches]
    # kv_lengths_host: [num_batches]

    assert batch_kv_len >= batch_seq_len

    dtype = torch.float16
    device = "cpu"

    seq_lengths_host = generate_seq_lengths(batch_seq_len, num_batches)
    history_kv_len = batch_kv_len - batch_seq_len
    history_kv_lengths = generate_seq_lengths(history_kv_len, num_batches)
    kv_lengths_host = [
        seq + hist for seq, hist in zip(seq_lengths_host, history_kv_lengths)
    ]

    seq_lengths_host = kwargs.get("seq_lengths_host", seq_lengths_host)
    kv_lengths_host = kwargs.get("kv_lengths_host", kv_lengths_host)

    page_size = 16
    num_pages = sum(
        [(kv_len + page_size - 1) // page_size for kv_len in kv_lengths_host]
    )

    q = torch.randn(batch_seq_len, num_heads, head_size, dtype=dtype, device=device)
    k_cache = torch.randn(
        (num_pages, page_size, num_kv_heads, head_size),
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn(
        (num_pages, page_size, num_kv_heads, head_size),
        dtype=dtype,
        device=device,
    )

    page_table = torch.stack(
        [
            torch.randperm(num_pages, dtype=torch.int32, device=device)
            for _ in range(num_batches)
        ]
    )

    mask_size = 2048
    mask = get_prefill_mask(mask_size, dtype=dtype, device=device)

    print(f"{seq_lengths_host=}")
    print(f"{kv_lengths_host=}")
    print(f"{q.shape=}")
    print(f"{k_cache.shape=}")
    print(f"{v_cache.shape=}")
    print(f"{page_table.shape=}")
    print(f"{mask.shape=}")

    out_ref = paged_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lengths_host=seq_lengths_host,
        kv_lengths_host=kv_lengths_host,
        mask=mask,
        qk_scale=None,
        attentionFunc=flash_attention_v2,
    )

    out_act = paged_flash_attention_v2(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lengths_host=seq_lengths_host,
        kv_lengths_host=kv_lengths_host,
        mask=mask,
        qk_scale=None,
    )

    print(f"{out_act.shape=}")

    out_act = out_act.detach().numpy()
    out_ref = out_ref.detach().numpy()

    abs_error, rel_error, top_indices = compute_diff(out_act, out_ref, idx_cnt=20)
    print_diff(
        out_act, out_ref, "act", "ref", abs_error, rel_error, top_indices, eps=1e-1
    )


def main():
    """
    kv_lengths contains the total sequence length for each request,
    including both the context length (historical tokens) and the current query.
    For example, if a sequence has 5 previous tokens and is currently decoding
    the 6th token, then kv_lengths = 5 + 1 = 6.
    """
    # num_batches, batch_seq_len, batch_kv_len, num_heads, num_kv_heads, head_size

    print("===Test prefill===")
    test_paged_attention(2, 100, 100, 16, 4, 128)

    print("===Test chunk prefill===")
    test_paged_attention(2, 100, 200, 16, 4, 128)

    print("===Test decode===")
    test_paged_attention(2, 2, 100, 16, 4, 128)


if __name__ == "__main__":
    main()
