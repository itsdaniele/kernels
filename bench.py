from flash_attn_v1 import attention
from fused_layernorm_flash import attention as fused_query_attention

import torch
import torch.nn as nn
import time


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

        self.values = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.keys = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.queries = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, emb_size)

        if variant in ["queries_keys", "fused_q_norm"]:
            self.norm_keys = nn.LayerNorm(self.head_dim)
            self.norm_queries = nn.LayerNorm(self.head_dim)
        self.variant = variant

    def forward(self, x, mask, sm_scale=1.0):
        N, L = x.shape[:2]

        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        values = (
            values.reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        keys = (
            keys.reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        queries = (
            queries.reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if self.variant == "queries_keys":
            keys = self.norm_keys(keys)
            queries = self.norm_queries(queries)
        elif self.variant == "fused_q_norm":
            keys = self.norm_keys(keys)
            weight_key = self.norm_keys.weight
            bias_key = self.norm_keys.bias
            weight_query = self.norm_queries.weight
            bias_query = self.norm_queries.bias

        if self.variant == "fused_q_norm":
            out = fused_query_attention(
                queries,
                keys,
                values,
                self.causal,
                sm_scale,
                weight_query,
                bias_query,
                weight_key,
                bias_key,
            )
        else:
            out = attention(
                queries,
                keys,
                values,
                self.causal,
                sm_scale,
            )

        out = (
            out.transpose(1, 2)
            .reshape(N, L, self.num_heads * self.head_dim)
            .contiguous()
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads,
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
        self.variant = variant

    def forward(self, x, mask):
        if self.variant == "standard":
            x = x + self.attention(self.norm1(x), mask)
            out = x + self.feed_forward(self.norm2(x))
        else:
            x = x + self.attention(x, mask)
            out = x + self.feed_forward(x)

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
                    forward_expansion=forward_expansion,
                    variant=variant,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.max_length = max_length

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, mask)

        return self.fc_out(out)


@torch.inference_mode()
def benchmark_model(
    model, input_data_shape, vocab_size, num_trials=100, warmup_trials=10
):
    model.eval()

    input_data = torch.randint(0, vocab_size, input_data_shape).to(model.device)
    for _ in range(warmup_trials):
        _ = model(input_data)

    total_time = 0
    total_tokens = 0

    for _ in range(num_trials):
        input_data = torch.randint(0, vocab_size, input_data_shape).to(model.device)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = model(input_data)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event) / 1000

        batch_size, seq_length = input_data.shape
        total_time += elapsed_time
        total_tokens += batch_size * seq_length

    avg_time_per_token = total_time / total_tokens
    tokens_per_second = total_tokens / total_time

    return avg_time_per_token, tokens_per_second


def run_benchmark():
    device = torch.device("cuda")

    # Hyperparameters
    emb_size = 512
    num_layers = 12
    num_heads = 4
    vocab_size = 512
    max_length = 4096
    forward_expansion = 2

    batch_sizes = [1,4]

    variants = ["standard", "queries_keys", "no_layer_norm", "fused_q_norm"]
    # variants = ["queries_keys", "fused_q_norm"]
    results = {}

    for batch_size in batch_sizes:
        input_data_shape = (batch_size, max_length)
        for variant in variants:
            model = (
                GPTModel(
                    emb_size,
                    num_layers,
                    num_heads,
                    vocab_size,
                    max_length,
                    forward_expansion,
                    variant=variant,
                    causal=True,
                )
                .to(device)
                .half()
            )
            model.device = device
            avg_time_per_token, tokens_per_second = benchmark_model(
                model, input_data_shape, vocab_size, num_trials=50
            )

            # Make avg time per token in ms
            avg_time_per_token = avg_time_per_token * 1000
            if variant not in results:
                results[variant] = {}
            results[variant][f"batch_size_{batch_size}"] = {
                "avg_time_per_token": avg_time_per_token,
                "tokens_per_second": tokens_per_second,
            }

    return results


results = run_benchmark()
print(results)
