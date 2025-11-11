import math

import torch

import config_logger
from loguru import logger


def get_kv(
    batch_idx: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    kv_lengths_host: list,
):
    """
    Get k and v tensors from the cache based on the page_table and kv lengths.
    """
    k = []
    v = []

    kv_len = kv_lengths_host[batch_idx]
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    qk_head_size = k_cache.shape[3]
    v_head_size = v_cache.shape[3]

    # map from kv id to page id, eg: [0, 2, 11, 7, ...]
    # means the 0th kv page is stored in page 0, 1st kv page is stored in page 2, etc.
    mapping_kv_id_2_page_id = page_table[batch_idx]

    for kv_len0 in range(0, kv_len, page_size):
        kv_idx = kv_len0 // page_size
        page_id = mapping_kv_id_2_page_id[kv_idx]

        k_cache_page = k_cache[page_id]
        v_cache_page = v_cache[page_id]

        k_cache_page = k_cache_page.view(page_size, num_kv_heads, qk_head_size)
        v_cache_page = v_cache_page.view(page_size, num_kv_heads, v_head_size)

        k.append(k_cache_page)
        v.append(v_cache_page)

    k = torch.stack(k, dim=0)
    v = torch.stack(v, dim=0)

    k = k.view(-1, num_kv_heads, qk_head_size)
    v = v.view(-1, num_kv_heads, v_head_size)

    k = k[:kv_len].contiguous()
    v = v[:kv_len].contiguous()

    return k, v


def repeat(hidden_states: torch.Tensor, dim: int, n_rep: int) -> torch.Tensor:
    """
    1. Expands kv_head_num to attn_head_num for Grouped Query Attention (GQA).
    2. Equivalent to hidden_states.repeat_interleave(dim=dim, repeats=n_rep),
    but creates views instead of copying data, avoiding additional memory allocation.
    """
    if n_rep == 1:
        return hidden_states

    org_shape = list(hidden_states.size())

    expand_shape = org_shape.copy()
    expand_shape.insert(dim + 1, 1)
    hidden_states = hidden_states.view(*expand_shape)

    expand_shape = [-1] * len(expand_shape)
    expand_shape[dim + 1] = n_rep

    hidden_states = hidden_states.expand(*expand_shape)

    final_shape = list(org_shape)
    final_shape[dim] = final_shape[dim] * n_rep

    return hidden_states.reshape(*final_shape)


def repeat_kv(
    k: torch.Tensor, v: torch.Tensor, dim: int, n_rep: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = repeat(k, dim, n_rep)
    v = repeat(v, dim, n_rep)

    return k, v


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    qk_scale: float = None,
    alibi_mask: bool = False,
):
    """
    q: (seq_len, num_heads, qk_head_size)
    k: (kv_len, num_kv_heads, qk_head_size)
    v: (kv_len, num_kv_heads, v_head_size)
    mask: (max_col_size, max_col_size)
    """

    seq_len, num_heads, qk_head_size = q.size()
    kv_len = k.size(0)
    num_kv_heads = k.size(1)
    v_head_size = v.size(2)

    if qk_scale is None:
        qk_scale = 1 / math.sqrt(qk_head_size)

    q = q.transpose(0, 1).view(num_heads, seq_len, qk_head_size)
    k = k.transpose(0, 1).view(num_kv_heads, kv_len, qk_head_size)
    v = v.transpose(0, 1).view(num_kv_heads, kv_len, v_head_size)

    k, v = repeat_kv(k, v, dim=0, n_rep=num_heads // num_kv_heads)

    kT = k.transpose(1, 2)  # [num_heads, qk_head_size, kv_len]

    weight = torch.bmm(q, kT)  # [num_heads, seq_len, kv_len]
    weight = weight * qk_scale

    if seq_len != 1 and mask is not None:
        mask_start = kv_len - seq_len
        mask_slice = mask[mask_start:kv_len, :kv_len]
        weight = weight + mask_slice.unsqueeze(0)

    weight = torch.nn.functional.softmax(weight, dim=-1)

    out = torch.bmm(weight, v).transpose(0, 1).view(seq_len, num_heads, v_head_size)
    return out


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    qk_scale: float = None,
    alibi_mask: bool = False,
):
    """
    q: (seq_len, num_heads, qk_head_size)
    k_cache: (kv_len, num_kv_heads, qk_head_size)
    v_cache: (kv_len, num_kv_heads, v_head_size)
    mask: (max_col_size, max_col_size)
    qk_scale: float
    """

    seq_len = q.shape[0]
    num_heads = q.shape[1]
    qk_head_size = q.shape[2]
    kv_len = k.shape[0]
    num_kv_heads = k.shape[1]
    v_head_size = v.shape[2]

    dtype = q.dtype
    device = q.device

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(qk_head_size)

    # M = 20 * 1024 * 1024  # SRAM, A100 20M
    # Bc = math.ceil(M / (num_heads * head_size))
    # Br = math.ceil(M / (num_kv_heads * head_size))

    Br_size, Bc_size = 8, 4  # For testing purposes, set to small values

    Br_num = (seq_len + Br_size - 1) // Br_size
    Bc_num = (kv_len + Bc_size - 1) // Bc_size

    out = torch.zeros(seq_len, num_heads, v_head_size, dtype=dtype, device=device)

    for i in range(Br_num):
        o_slice = torch.zeros(
            num_heads, Br_size, v_head_size, dtype=dtype, device=device
        )
        l = torch.zeros(num_heads, Br_size, dtype=dtype, device=device)
        m = (
            torch.ones(num_heads, Br_size, dtype=dtype, device=device)
            * torch.finfo(dtype).min
        )

        row_slice = slice(i * Br_size, (i + 1) * Br_size)

        if (i + 1) * Br_size > seq_len:
            # |---row_slice_real---|--row_slice_padding--|
            row_slice_real = slice(i * Br_size, seq_len)
            row_slice_padding = (i + 1) * Br_size - seq_len
        else:
            row_slice_real = None

        if row_slice_real is not None:
            q_slice = torch.cat(
                [
                    q[row_slice_real, :, :],
                    torch.zeros(
                        (row_slice_padding, num_heads, qk_head_size),
                        dtype=dtype,
                        device=device,
                    ),
                ],
                dim=0,
            )
        else:
            q_slice = q[row_slice, :, :]

        q_slice = q_slice.view(Br_size, num_heads, qk_head_size)
        q_slice = q_slice.transpose(0, 1).view(num_heads, Br_size, qk_head_size)

        for j in range(Bc_num):
            col_slice = slice(j * Bc_size, (j + 1) * Bc_size)

            if (j + 1) * Bc_size > kv_len:
                # |---col_slice_real---|--col_slice_padding--|
                col_slice_real = slice(j * Bc_size, kv_len)
                col_slice_padding = (j + 1) * Bc_size - kv_len
            else:
                col_slice_real = None

            if col_slice_real is not None:
                k_slice = torch.cat(
                    [
                        k[col_slice_real, :, :],
                        torch.zeros(
                            (col_slice_padding, num_kv_heads, qk_head_size),
                            dtype=dtype,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
            else:
                k_slice = k[col_slice, :, :]

            if col_slice_real is not None:
                v_slice = torch.cat(
                    [
                        v[col_slice_real, :, :],
                        torch.zeros(
                            (col_slice_padding, num_kv_heads, v_head_size),
                            dtype=dtype,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
            else:
                v_slice = v[col_slice, :, :]

            k_slice = (
                k_slice.transpose(0, 1)
                .contiguous()
                .view(num_kv_heads, Bc_size, qk_head_size)
            )
            v_slice = (
                v_slice.transpose(0, 1)
                .contiguous()
                .view(num_kv_heads, Bc_size, v_head_size)
            )

            k_slice, v_slice = repeat_kv(
                k_slice, v_slice, dim=0, n_rep=num_heads // num_kv_heads
            )

            # (num_heads, Br_size, Bc_size)
            S_ij = torch.bmm(q_slice, k_slice.transpose(1, 2))
            S_ij = S_ij * qk_scale

            # Note: For decoding, set padded columns values to -inf.
            if (seq_len != 1 or col_slice_real is not None) and mask is not None:
                mask_row_slice_start = row_slice.start + (kv_len - seq_len)
                mask_row_slice_stop = row_slice.stop + (kv_len - seq_len)
                mask_row_slice = slice(mask_row_slice_start, mask_row_slice_stop)
                mask_slice = mask[mask_row_slice, col_slice]
                S_ij = S_ij + mask_slice.unsqueeze(0)

            m_ij, _ = torch.max(S_ij, dim=-1, keepdim=False)
            m_new = torch.max(m, m_ij)

            # (num_heads, Br_size, Bc_size)
            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
            l_ij = torch.sum(P_ij, dim=-1, keepdim=False) + 1e-10
            l_new = torch.exp(m - m_new) * l + l_ij

            # (num_heads, Br_size, v_head_size)
            o_slice = (torch.exp(m - m_new).unsqueeze(-1) * o_slice) + (
                torch.bmm(P_ij, v_slice)
            )

            l = l_new
            m = m_new

        o_slice = o_slice / l.unsqueeze(-1)

        o_slice = (
            o_slice.transpose(0, 1).contiguous().view(Br_size, num_heads, v_head_size)
        )

        if row_slice_real is not None:
            out[row_slice_real, :, :] = o_slice[: (Br_size - row_slice_padding), :, :]
        else:
            out[row_slice, :, :] = o_slice

    return out


def paged_attention(
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lengths_host,
    kv_lengths_host,
    mask,
    qk_scale=None,
    attentionFunc=flash_attention_v2,
):
    """
    q: (batch_seq_len, num_heads, qk_head_size)
    k_cache: (num_pages, page_size, num_kv_heads, qk_head_size)
    v_cache: (num_pages, page_size, num_kv_heads, v_head_size)
    page_table: (num_batches, num_pages)
    seq_lengths_host: (num_batches)
    kv_lengths_host: (num_batches)
    attentionFunc: attention function
    """
    num_batches = len(seq_lengths_host)

    outs = []
    offset = 0
    for batch_idx in range(num_batches):
        seq_len = seq_lengths_host[batch_idx]
        batch_slice = slice(offset, offset + seq_len)

        q_batch = q[batch_slice, :, :]
        k_batch, v_batch = get_kv(
            batch_idx,
            k_cache,
            v_cache,
            page_table,
            kv_lengths_host,
        )

        out = attentionFunc(q_batch, k_batch, v_batch, mask.clone(), qk_scale)

        outs.append(out)
        offset += seq_len

    outs = torch.cat(outs, dim=0)
    return outs
