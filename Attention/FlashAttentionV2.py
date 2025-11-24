import math
import torch

from Ref import repeat_kv
from diff_tensors import compute_diff, print_diff

import config_logger
from loguru import logger

logger.remove()
logger.disable(__name__)


def flash_attention_v2_single_batch(
    q, k_cache, v_cache, page_table, kv_len, mask, qk_scale=None
):
    """
    q: (seq_len, num_heads, qk_head_size)
    k_cache: (num_pages, page_size, num_kv_heads, qk_head_size)
    v_cache: (num_pages, page_size, num_kv_heads, v_head_size)
    page_table: (num_pages)
    kv_len:
    """

    seq_len = q.shape[0]
    num_heads = q.shape[1]
    qk_head_size = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    v_head_size = v_cache.shape[3]

    page_size = k_cache.shape[1]

    dtype = q.dtype
    device = q.device

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(qk_head_size)

    Br_size, Bc_size = 8, page_size

    assert Bc_size == page_size

    Br_num = (seq_len + Br_size - 1) // Br_size
    Bc_num = (kv_len + Bc_size - 1) // Bc_size

    out = torch.zeros(seq_len, num_heads, v_head_size, dtype=dtype, device=device)

    # TODO: (group, Br, Bc) May BE  better

    num_groups = num_kv_heads
    group_size = num_heads // num_groups

    out = out.view(seq_len, num_groups, group_size, v_head_size)

    logger.debug(f"{Br_num=}, {Bc_num=}")

    for Br_idx in range(Br_num):
        logger.debug(f"{Br_idx=}")

        row_slice = slice(Br_idx * Br_size, (Br_idx + 1) * Br_size)
        if (Br_idx + 1) * Br_size > seq_len:
            # |---row_slice_real---|--row_slice_padding--|
            row_slice_real = slice(Br_idx * Br_size, seq_len)
            row_slice_padding = (Br_idx + 1) * Br_size - seq_len
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
        q_slice = q_slice.view(Br_size, num_groups, group_size, qk_head_size)

        for group_idx in range(num_groups):
            logger.debug(f"{group_idx=}")
            group_slice = slice(group_idx, group_idx + 1)

            o = torch.zeros(
                Br_size, group_size, v_head_size, dtype=dtype, device=device
            )
            l = torch.zeros(Br_size, group_size, dtype=dtype, device=device)
            m = (
                torch.ones(Br_size, group_size, dtype=dtype, device=device)
                * torch.finfo(dtype).min
            )

            for Bc_idx in range(Bc_num):
                logger.debug(f"{Bc_idx=}")
                col_slice = slice(Bc_idx * Bc_size, (Bc_idx + 1) * Bc_size)
                if (Bc_idx + 1) * Bc_size > kv_len:
                    # |---col_slice_real---|--col_slice_padding--|
                    col_slice_real = slice(Bc_idx * Bc_size, kv_len)
                    col_slice_padding = (Bc_idx + 1) * Bc_size - kv_len
                else:
                    col_slice_real = None

                page_idx = page_table[Bc_idx]

                k_page = k_cache[page_idx].view(Bc_size, num_kv_heads, qk_head_size)
                v_page = v_cache[page_idx].view(Bc_size, num_kv_heads, v_head_size)

                # For gqa group

                k_page = k_page.view(Bc_size, num_groups, 1, qk_head_size)
                v_page = v_page.view(Bc_size, num_groups, 1, v_head_size)

                q_group_slice = q_slice[:, group_slice, :, :].reshape(
                    Br_size * group_size, qk_head_size
                )
                k_group_slice = k_page[:, group_slice, :, :].reshape(
                    Bc_size, qk_head_size
                )
                v_group_slice = v_page[:, group_slice, :, :].reshape(
                    Bc_size, v_head_size
                )

                # (Br_size*group_size, Bc_size)
                S_ij = torch.matmul(q_group_slice, k_group_slice.transpose(0, 1))
                S_ij = S_ij * qk_scale
                S_ij = S_ij.view(Br_size, group_size, Bc_size)

                logger.debug(f"{S_ij=}")

                if (seq_len != 1 or col_slice_real is not None) and mask is not None:
                    mask_row_slice_start = row_slice.start + (kv_len - seq_len)
                    mask_row_slice_stop = row_slice.stop + (kv_len - seq_len)
                    mask_row_slice = slice(mask_row_slice_start, mask_row_slice_stop)
                    mask_slice = mask[mask_row_slice, col_slice]
                    # (Br_size, group_size, Bc_size) + (Br_size, 1, Bc_size)
                    S_ij = S_ij + mask_slice.reshape(Br_size, 1, Bc_size)

                logger.debug(f"{S_ij=}")

                # (Br_size, group_size)
                m_ij, _ = torch.max(S_ij, dim=-1, keepdim=False)
                m_new = torch.max(m, m_ij)

                # (Br_size, group_size, Bc_size)
                P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))

                logger.debug(f"{P_ij=}")

                # (Br_size, group_size)
                l_ij = torch.sum(P_ij, dim=-1, keepdim=False) + 1e-10
                l_new = torch.exp(m - m_new) * l + l_ij

                # (Br_size, group_size, v_head_size)
                o = (torch.exp(m - m_new).unsqueeze(-1) * o) + (
                    torch.matmul(P_ij, v_group_slice)
                )
                l = l_new
                m = m_new

                logger.debug(f"{o=}")

            o = o / l.unsqueeze(-1)
            o = o.reshape(Br_size, 1, group_size, v_head_size)

            if row_slice_real is not None:
                out[row_slice_real, group_slice, :, :] = o[
                    : (Br_size - row_slice_padding), :, :, :
                ]
            else:
                out[row_slice, group_slice, :, :] = o

    out = out.view(seq_len, num_heads, v_head_size)

    return out


def paged_flash_attention_v2(
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lengths_host,
    kv_lengths_host,
    mask,
    qk_scale=None,
):
    """
    q: (batch_seq_len, num_heads, qk_head_size)
    k_cache: (num_pages, page_size, num_kv_heads, qk_head_size)
    v_cache: (num_pages, page_size, num_kv_heads, v_head_size)
    page_table: (num_batches, num_pages)
    seq_lengths_host: (num_batches)
    kv_lengths_host: (num_batches)
    """
    num_batches = len(seq_lengths_host)

    outs = []
    offset = 0
    for batch_idx in range(num_batches):
        seq_len = seq_lengths_host[batch_idx]
        kv_len = kv_lengths_host[batch_idx]
        page_table_cur_batch = page_table[batch_idx]

        batch_slice = slice(offset, offset + seq_len)
        q_batch = q[batch_slice, :, :]

        out = flash_attention_v2_single_batch(
            q=q_batch,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table_cur_batch,
            kv_len=kv_len,
            mask=mask,
            qk_scale=qk_scale,
        )

        outs.append(out)
        offset += seq_len

    outs = torch.cat(outs, dim=0)
    return outs


def gqa_trick(seq_len, num_heads, qk_head_size, kv_len, num_kv_heads, v_head_size):
    def ref(q, k, v):
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

        num_groups = num_kv_heads
        group_size = num_heads // num_groups

        q = q.view(seq_len, num_groups, group_size, qk_head_size)
        k = k.view(kv_len, num_groups, 1, qk_head_size)
        v = v.view(kv_len, num_groups, 1, v_head_size)

        out = torch.zeros(
            (seq_len, num_groups, group_size, qk_head_size), dtype=dtype, device=device
        )

        for g0 in range(num_groups):
            group_slice = slice(g0, g0 + 1)
            q_slice = q[:, group_slice, :, :].reshape(
                seq_len * group_size, qk_head_size
            )
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


if __name__ == "__main__":
    # seq_len, num_heads, qk_head_size, kv_len, num_kv_heads, v_head_size
    gqa_trick(10, 16, 128, 20, 2, 128)
