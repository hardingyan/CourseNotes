
import torch

from Ref import attention, get_kv

class AttentionOp(torch.nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, Wqkv, Wo, qk_scale=None, qknorm=False):
        """
        :param Wqkv: shape is (q_hidden_size, (q_hidden_size + k_hidden_size + v_hidden_size))
        :param Wunifyheads: shape is (v_hidden_size, q_hidden_size)

        :param kqnorm: Whether to apply layer normalization to the keys and queries.
        :param qk_scale: Multiplier for the attention weights.
        """

        super().__init__()

        assert qknorm == False, "kqnorm=True not supported in reference implementation."
        self.qk_scale = qk_scale

        in_features, out_features = Wqkv.shape
        self.qkv_proj = torch.nn.Linear(in_features, out_features, bias=False)
        self.qkv_proj.weight = torch.nn.Parameter(Wqkv)

        in_features, out_features = Wo.shape
        self.out_proj = torch.nn.Linear(in_features, out_features, bias=False)
        self.out_proj.weight = torch.nn.Parameter(Wo)


    def forward(self, x, k_cache, v_cache, block_table, seq_lengths_host, kv_lengths_host, mask):
        """
          q: (batch_seq_len, num_heads, qk_head_size)
          k_cache: (num_pages, page_size, num_kv_heads, qk_head_size)
          v_cache: (num_pages, page_size, num_kv_heads, v_head_size)
          block_table: (num_batches, num_pages)
          seq_lengths_host: (num_batches)
          kv_lengths_host: (num_batches)
        """

        batch_seq_len = x.size(0)
        num_heads = x.size(1)
        qk_head_size = x.size(2)
        num_kv_heads = k_cache.size(2)
        v_head_size = v_cache.size(3)

        q_hidden_size = num_heads * qk_head_size
        k_hidden_size = num_kv_heads * qk_head_size
        v_hidden_size = num_kv_heads * v_head_size

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(
            [
                q_hidden_size,
                k_hidden_size,
                v_hidden_size,
            ],
            dim=-1,
        )

        q = q.view(batch_seq_len, num_heads, qk_head_size)
        k = k.view(batch_seq_len, num_kv_heads, qk_head_size)
        v = v.view(batch_seq_len, num_kv_heads, v_head_size)

        num_batches = len(seq_lengths_host)
        outs = []
        for batch_idx in range(num_batches):
            kv_len = kv_lengths_host[batch_idx]

            k_history, v_history = get_kv(
                batch_idx,
                k_cache,
                v_cache,
                block_table,
                kv_lengths_host,
            )

            k = torch.cat([k_history, k], dim=0)
            v = torch.cat([v_history, v], dim=0)

            out = attention(q, k, v, mask, self.qk_scale)
            outs.append(out)

        outs = torch.cat(outs, dim=0)
        outs = self.out_proj(outs)

        return outs