"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import pytest
import torch

import triton
import triton.language as tl

from flash_attn_v1 import attention as attention_v1


# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=8),
#         triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=3, num_warps=8),
#         triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_stages=3, num_warps=8),
#         triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_stages=3, num_warps=4),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=4, num_warps=4),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=4),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=7, num_warps=8),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=7, num_warps=8),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=6, num_warps=8),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=5, num_warps=8),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=4, num_warps=8),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=6, num_warps=4),
#     ],
#     key=["N_CTX"],
# )
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    ######### Layer norm stuff.
    W_query,  # pointer to the layer norm weights
    B_query,  # pointer to the  layer norm biases
    Mean_query,  # pointer to the mean
    Rstd_query,  # pointer to the 1/std
    stride_layernorm_query,  # how much to increase the pointer when moving by 1 row for layer norm
    N_query,  # number of columns in X
    W_key,  # pointer to the layer norm weights
    B_key,  # pointer to the  layer norm biases
    Mean_key,  # pointer to the mean
    Rstd_key,  # pointer to the 1/std
    stride_layernorm_key,  # how much to increase the pointer when moving by 1 row for layer norm
    N_key,  # number of columns in X
    eps,  # epsilon to avoid division by zero in Layer norm.
    ###########
    sm_scale,
    L,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,  # how many queries on a query block?
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,  # how many keys on a key block?
    MODE: tl.constexpr,
):
    start_m = tl.program_id(0)  # which query are we starting from?
    off_hz = tl.program_id(1)  # which head are we processing in the batch?

    # offset that takes us from the the pointer to the query, key or value tensor to the current query/key/value.
    # stride_qh is the number of bytes it takes to get to the next head.
    qvk_offset = off_hz * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,  # start from current head.
        shape=(
            N_CTX,
            BLOCK_DMODEL,
        ),  # This is the shape of the underlying tensor. We are going to be processing BLOCK_M queries at a time.
        strides=(stride_qm, stride_qk),
        offsets=(
            start_m * BLOCK_M,
            0,
        ),  # start from the current query, depending on the program id.
        block_shape=(
            BLOCK_M,
            BLOCK_DMODEL,
        ),  # each block of query has shape (BLOCK_M, BLOCK_DMODEL)
        order=(1, 0),
    )

    # Here, I am note sure why the shape is (BLOCK_DMODEL, N_CTX) and not (N_CTX, BLOCK_DMODEL). I supposed there is some optimization purpose.
    # In any case, this will work because order is (0,1), so triton knows that the blog is laid out in memoory with the first axis being the inner dimension.

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,  # start from current head.
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # [0,1,...,127]
    # initialize pointer to m and l
    m_i = (
        tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    )  # shape is (128,). In the context of attention m is the maximum value of the query-key dot product.
    l_i = tl.zeros(
        [BLOCK_M], dtype=tl.float32
    )  # in flash attention, l is the sum of the softmax of the query-key dot product.
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # causal check on every loop iteration can be expensive
    # and peeling the last iteration of the loop does not work well with ptxas
    # so we have a mode to do the causal check in a separate kernel entirely
    if MODE == 0:  # entire non-causal attention
        lo, hi = 0, N_CTX
    if MODE == 1:  # entire causal attention
        lo, hi = (
            0,
            (start_m + 1) * BLOCK_M,
        )  # if working with causal attention, we only need to look at the first start_m blocks.
    if MODE == 2:  # off band-diagonal
        lo, hi = 0, start_m * BLOCK_M
    if MODE == 3:  # on band-diagonal
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        m_i = tl.load(m_ptrs)
        l_i = tl.load(l_ptrs)
        acc += tl.load(O_block_ptr).to(tl.float32)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M

    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(
        Q_block_ptr
    )  # load block of queries. This has shape (BLOCK_M, BLOCK_DMODEL).

    ######### Layer-normalize the queries.

    # q has shape (BLOCK_M, BLOCK_DMODEL). We want to normalize along the second axis.

    mean = tl.sum(q, axis=1) / N_query
    diff = q - mean[:, None]
    diff_squared = diff * diff
    var = tl.sum(diff_squared, axis=1) / N_query

    rstd = 1 / tl.sqrt(var + eps)

    # store mean and rstd.
    tl.store(Mean_query + off_hz * N_CTX + offs_m, mean)
    tl.store(Rstd_query + off_hz * N_CTX + offs_m, rstd)

    # normalize q and apply linear transformation.

    # load W and B

    W_query = tl.load(W_query + tl.arange(0, BLOCK_DMODEL))
    B_query = tl.load(B_query + tl.arange(0, BLOCK_DMODEL))
    q = (q - mean[:, None]) * rstd[:, None] * W_query + B_query

    ###########

    q = (q * qk_scale).to(tl.float16)
    # advance block pointers to first iteration of the loop
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)  # load block of keys. Shape is (BLOCK_DMODEL, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if MODE == 1 or MODE == 3:  # causal masking within the block
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf")
            )  # if we are in the causal mode, we need to mask the values that are in the future.
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i *= alpha
        l_i_new = l_i + beta * l_ij
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        causal,
        sm_scale,
        weight_query,
        bias_query,
        weight_key,
        bias_key,
        eps=1e-5,
    ):
        # BLOCK = 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        BLOCK_M = 128
        BLOCK_N = 64 if Lk <= 64 else 32
        num_stages = 4 if Lk <= 64 else 3
        num_warps = 4

        o = torch.empty_like(q)

        # According to this, we have ceil(N_CTX/128) programs running, and tl.program_id(0) tells us on which one we are in.
        # For each of those, we have one program running for every head in the batch. tl.program_id(1) allows us to access those.
        grid = (triton.cdiv(q.shape[2], 128), q.shape[0] * q.shape[1], 1)
        L = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        m = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        num_warps = 4 if Lk <= 64 else 8
        if causal:
            modes = [1] if q.shape[2] <= 2048 else [2, 3]
        else:
            modes = [0]

        q_arg = q.reshape(-1, q.shape[-1])
        M_query, N_query = q_arg.shape
        mean_query = torch.empty((M_query,), dtype=torch.float32, device="cuda")
        rstd_query = torch.empty((M_query,), dtype=torch.float32, device="cuda")

        # same for k
        k_arg = k.reshape(-1, k.shape[-1])
        M_key, N_key = k_arg.shape
        mean_key = torch.empty((M_key,), dtype=torch.float32, device="cuda")
        rstd_key = torch.empty((M_key,), dtype=torch.float32, device="cuda")

        for mode in modes:
            _fwd_kernel[grid](
                q,
                k,
                v,
                ######### Layer norm stuff.
                weight_query,  # pointer to the layer norm weights
                bias_query,  # pointer to the  layer norm biases
                mean_query,  # pointer to the mean
                rstd_query,  # pointer to the 1/std
                q_arg.stride(
                    0
                ),  # how much to increase the pointer when moving by 1 row for layer norm
                N_query,  # number of columns in X
                weight_key,  # pointer to the layer norm weights
                bias_key,  # pointer to the  layer norm biases
                mean_key,  # pointer to the mean
                rstd_key,  # pointer to the 1/std
                k_arg.stride(
                    0
                ),  # how much to increase the pointer when moving by 1 row for layer norm
                N_key,
                eps,  # epsilon to avoid division by zero in Layer norm.
                ###########
                sm_scale,
                L,
                m,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=Lk,
                MODE=mode,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(6, 9, 1024, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_op(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    w_shape = (D_HEAD,)
    weight_query = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias_query = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    weight_key = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias_key = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)

    # initialized a layernorm layer and assign the weights and biases, eps to the layer.
    q_norm = torch.nn.LayerNorm(D_HEAD, eps=eps).to("cuda")

    # assign the weights and biases to the layer.
    q_norm.weight = torch.nn.Parameter(weight_query)
    q_norm.bias = torch.nn.Parameter(bias_query)

    sm_scale = 0.5
    # reference implementation

    # mean = q.mean(dim=-1, keepdim=True)
    # var = (q - mean).pow(2).mean(dim=-1, keepdim=True)
    # std = 1 / (var + eps).sqrt()
    # q = (q - mean) * std

    q_normalized = q_norm(q)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q_normalized, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = attention(
        q, k, v, causal, sm_scale, weight_query, bias_query, weight_key, bias_key, eps
    ).half()
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)


BATCH, N_HEADS, N_CTX, D_HEAD = 1, 16, 2048, 64
# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton", "flash"],
        line_names=["Fused Q norm", "Not fused Q norm"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-fwd",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "dtype": torch.float16,
            "mode": "fwd",
            "causal": causal,
        },
    )
    for causal in [False, True]
]


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH,
    H,
    N_CTX,
    D_HEAD,
    causal,
    mode,
    provider,
    dtype=torch.float16,
):
    assert mode == "fwd"
    warmup = 25
    rep = 100

    q = torch.randn(
        (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
    )
    k = torch.randn(
        (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
    )
    v = torch.randn(
        (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
    )

    sm_scale = 1.3
    weight = torch.rand(D_HEAD, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.rand(D_HEAD, dtype=dtype, device="cuda", requires_grad=True)

    def fn_notfused(q):
        q = torch.nn.functional.layer_norm(
            q, (D_HEAD,), weight=weight, bias=bias, eps=1e-5
        )
        return attention_v1(q, k, v, causal, sm_scale)

    def fn_fused(q):
        return attention(q, k, v, causal, sm_scale, weight, bias, weight, bias, 1e-5)

    if provider == "triton":
        fn = lambda: fn_fused(q)
    else:
        fn = lambda: fn_notfused(q)

    # Benchmark
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul

    # add layernorm flops
    total_flops += 4 * BATCH * H * N_CTX * D_HEAD

    if causal:
        total_flops *= 0.5

    # compute tokens per second.

    return total_flops / ms * 1e-9


# test_op(6, 9, 1024, 64, False, dtype=torch.float16)
# bench_flash_attention.run(save_path=".", print_data=True)

# run test
