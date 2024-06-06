from flash_attn_v1 import attention
from fused_layernorm_flash import attention as fused_key_attention


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads, variant="standard", causal=False):
        super(CustomMultiHeadSelfAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.causal = causal

        assert (
            self.head_dim * num_heads == emb_size
        ), "Embedding size needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, emb_size)

        if variant == "queries_keys":
            self.norm_keys = nn.LayerNorm(self.head_dim)
            self.norm_queries = nn.LayerNorm(self.head_dim)

        self.variant = variant

    def forward(self, values, keys, queries, mask, sm_scale=1.0):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        if self.variant == "queries_keys":
            keys = self.norm_keys(self.keys(keys))
            queries = self.norm_queries(self.queries(queries))
        else:
            keys = self.keys(keys)
            queries = self.queries(queries)

        values = self.values(values)

        out = attention(queries, keys, values, self.causal, sm_scale)

        out = out.reshape(N, query_len, self.num_heads * self.head_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads,
        dropout,
        forward_expansion,
        variant="standard",
        causal=False,
    ):
        super(TransformerBlock, self).__init__()
        self.attention = CustomMultiHeadSelfAttention(
            emb_size, num_heads, variant=variant, causal=causal
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.variant = variant

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        if self.variant == "standard":
            x = self.dropout(self.norm1(attention + query))
            forward = self.feed_forward(x)
            out = self.dropout(self.norm2(forward + x))
        elif self.variant == "queries_keys":
            x = self.dropout(attention + query)
            forward = self.feed_forward(x)
            out = self.dropout(self.norm2(forward + x))
        elif self.variant == "no_layer_norm":
            x = self.dropout(attention + query)
            forward = self.feed_forward(x)
            out = self.dropout(forward + x)

        return out


class GPTModel(nn.Module):
    def __init__(
        self,
        emb_size,
        num_layers,
        num_heads,
        vocab_size,
        max_length,
        forward_expansion,
        dropout,
        variant="standard",
        causal=False,
    ):
        super(GPTModel, self).__init__()
        self.emb_size = emb_size
        self.word_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_length, emb_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    variant=variant,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return self.fc_out(out)


@torch.inference_mode()
def benchmark_model(model, input_data, num_trials=100):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_trials):
            _ = model(input_data)
        end_time = time.time()

    avg_time_per_inference = (end_time - start_time) / num_trials
    throughput = input_data.size(0) / avg_time_per_inference

    return avg_time_per_inference, throughput


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    emb_size = 512
    num_layers = 32
    num_heads = 4
    vocab_size = 10000
    max_length = 512
    forward_expansion = 4
    dropout = 0.0

    input_data = torch.randint(0, vocab_size, (4, max_length)).to(
        device
    )  # Batch size of 32

    variants = ["standard", "queries_keys", "no_layer_norm"]
    results = {}

    for variant in variants:
        model = (
            GPTModel(
                emb_size,
                num_layers,
                num_heads,
                vocab_size,
                max_length,
                forward_expansion,
                dropout,
                variant=variant,
                causal=True,
            )
            .to(device)
            .half()
        )

        avg_time, throughput = benchmark_model(model, input_data)
        results[variant] = {
            "avg_time_per_inference": avg_time,
            "throughput": throughput,
        }

    return results


results = run_benchmark()
print(results)
