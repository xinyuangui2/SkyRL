"""QSA sparse attention: SkyRL's FlexAttention path vs cuDNN frontend qsa_forward/qsa_backward.

One B200 (SM100). Run from a SkyRL checkout of qwen38-flash-next, with a cudnn-frontend that has
cudnn.ops.qsa_forward (>= 1.32.0.dev, GitHub release wheel):

    uv run --isolated --extra megatron --with <nvidia_cudnn_frontend-1.32.0.dev*.whl> \
        python bench_qsa_cudnn.py > /tmp/bench_qsa.log 2>&1

Per pack, times forward and forward+backward for both paths (CUDA events, median), checks that
cuDNN's out / dq / dk / dv match the Flex path, and reports the cuDNN metadata build time
separately (it is per layer: the selection differs per layer).
"""

import argparse
import statistics

import torch

from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
    QSASelection,
    QSASequenceSelection,
    qsa_sparse_attention,
)

R, TOPK, D = 4, 512, 256
DENSE_MAX = TOPK * R + R - 1  # 2051


def random_block_mask(length, local_bias, gen):
    """Top-512 complete blocks per query (as Qwen4ExpQSAIndexer.select_sequence), random scores.
    local_bias > 0 favors recent blocks, closer to real indexer selections."""
    nb = length // R
    blocks = torch.arange(nb, device="cuda")
    visible = (torch.arange(length, device="cuda") + 1) // R
    complete = blocks < visible[:, None]
    scores = torch.rand(length, nb, device="cuda", generator=gen)
    scores += local_bias * (blocks[None, :] / max(nb, 1) - visible[:, None] / max(nb, 1)).clamp(-1, 0)
    top = scores.masked_fill(~complete, float("-inf")).topk(min(TOPK, nb), dim=-1).indices
    selected = torch.zeros_like(complete).scatter_(-1, top, True)
    return torch.where((visible > TOPK)[:, None], selected, complete)


def make_pack(lengths, align, local_bias, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    padded = [-(-n // align) * align for n in lengths]
    cu_pad = torch.tensor([0] + padded, device="cuda").cumsum(0)
    seqs = []
    for start, n in zip(cu_pad[:-1].tolist(), lengths):
        mask = random_block_mask(n, local_bias, gen) if n > DENSE_MAX else None
        seqs.append(QSASequenceSelection(start, n, mask))
    return QSASelection(seqs, R), padded, int(cu_pad[-1])


def cudnn_metadata(selection, padded, total):
    """The cuDNN bitmap contract (docs/fe-oss-apis/qsa.md) for a SkyRL QSASelection.

    Every padded sequence span is one partition with its own block numbering. Real queries see
    [seq start, t]; padding queries get empty rows (lo > hi). The incomplete causal tail block
    is added explicitly (the kernel does not add it)."""
    dev = "cuda"
    nbs = [-(-p // R) for p in padded]
    NB = sum(nbs)
    sel = torch.zeros(total, NB, dtype=torch.uint8, device=dev)
    lo = torch.empty(total, dtype=torch.int32, device=dev)
    hi = torch.empty(total, dtype=torch.int32, device=dev)
    blk_base = torch.empty(total, dtype=torch.int32, device=dev)
    tok_base = torch.empty(total, dtype=torch.int32, device=dev)
    block_first = torch.empty(NB, dtype=torch.int32, device=dev)
    block_last = torch.empty(NB, dtype=torch.int32, device=dev)
    b0 = 0
    for seq, p, nb in zip(selection.sequences, padded, nbs):
        s, n = seq.start, seq.length
        u = torch.arange(p, device=dev)
        lo[s : s + p] = s + u.clamp_max(n)  # padding rows: lo = s + n ...
        hi[s : s + p] = torch.where(u < n, s + u, s + n - 1)  # ... > hi = s + n - 1
        lo[s : s + n] = s
        blk_base[s : s + p] = b0
        tok_base[s : s + p] = s
        first = s + R * torch.arange(nb, device=dev)
        block_first[b0 : b0 + nb] = first
        block_last[b0 : b0 + nb] = (first + R - 1).clamp_max(s + p - 1)
        rows = torch.arange(n, device=dev)
        if seq.block_mask is None:  # dense: every block up to the query's own
            sel[s : s + n, b0 : b0 + nb] = (torch.arange(nb, device=dev)[None, :] <= (rows // R)[:, None]).to(
                torch.uint8
            )
        else:
            cb = seq.block_mask.shape[1]
            sel[s : s + n, b0 : b0 + cb] = seq.block_mask.to(torch.uint8)
            tail = rows[(rows + 1) % R != 0]  # queries with a non-empty incomplete tail
            sel[s + tail, b0 + tail // R] = 1
        b0 += nb
    return sel, lo, hi, blk_base, tok_base, block_first, block_last


class CudnnQSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, meta, scale):
        from cudnn.ops import qsa_forward

        out, out_fp32, lse = qsa_forward(q, k, v, *meta, scale)
        ctx.save_for_backward(q, k, v, out_fp32, lse, *meta)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from cudnn.ops import qsa_backward

        q, k, v, out_fp32, lse, *meta = ctx.saved_tensors
        dq, dk, dv = qsa_backward(grad_out.contiguous(), q, k, v, out_fp32, lse, *meta, ctx.scale)
        return dq, dk, dv, None, None


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def run(lengths, hq, hkv, args):
    selection, padded, total = make_pack(lengths, args.align, args.local_bias, args.seed)
    scale = D**-0.5
    torch.manual_seed(args.seed)
    q, k, v = (torch.randn(total, h, D, device="cuda", dtype=torch.bfloat16) for h in (hq, hkv, hkv))
    grad = torch.randn(total, hq * D, device="cuda", dtype=torch.bfloat16)
    leaves = [t.requires_grad_() for t in (q, k, v)]

    def flex_fwd():
        with torch.no_grad():
            return qsa_sparse_attention(q, k, v, selection, scale)

    def flex_fwd_bwd():
        out = qsa_sparse_attention(q, k, v, selection, scale)
        return out, torch.autograd.grad(out, leaves, grad)

    meta_ms = timed(lambda: cudnn_metadata(selection, padded, total), args.iters, 2)
    meta = cudnn_metadata(selection, padded, total)

    def cudnn_fwd():
        with torch.no_grad():
            return CudnnQSA.apply(q, k, v, meta, scale).reshape(total, -1)

    def cudnn_fwd_bwd():
        out = CudnnQSA.apply(q, k, v, meta, scale).reshape(total, -1)
        return out, torch.autograd.grad(out, leaves, grad)

    # Correctness (real tokens only; padding rows are zero in both).
    ref_out, ref_g = flex_fwd_bwd()
    out, g = cudnn_fwd_bwd()
    real = torch.zeros(total, dtype=torch.bool, device="cuda")
    for seq in selection.sequences:
        real[seq.start : seq.start + seq.length] = True
    errs = {
        name: ((a.float() - b.float())[real].abs().max().item(), b.float()[real].abs().max().item())
        for name, a, b in zip(("out", "dq", "dk", "dv"), (out, *g), (ref_out, *ref_g))
    }

    res = {
        "flex_fwd": timed(flex_fwd, args.iters, args.warmup),
        "cudnn_fwd": timed(cudnn_fwd, args.iters, args.warmup),
        "flex_fwd_bwd": timed(flex_fwd_bwd, args.iters, args.warmup),
        "cudnn_fwd_bwd": timed(cudnn_fwd_bwd, args.iters, args.warmup),
    }
    print(
        f"pack={lengths} heads={hq}Q/{hkv}KV T={total} | "
        f"fwd flex {res['flex_fwd']:.2f} ms cudnn {res['cudnn_fwd']:.2f} ms ({res['flex_fwd'] / res['cudnn_fwd']:.2f}x) | "
        f"fwd+bwd flex {res['flex_fwd_bwd']:.2f} ms cudnn {res['cudnn_fwd_bwd']:.2f} ms "
        f"({res['flex_fwd_bwd'] / res['cudnn_fwd_bwd']:.2f}x) | cudnn metadata {meta_ms:.2f} ms",
        flush=True,
    )
    print("   max|cudnn - flex| (max|flex|): " + ", ".join(f"{k} {e:.3g} ({m:.3g})" for k, (e, m) in errs.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--align", type=int, default=16)
    p.add_argument("--local-bias", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    major, minor = torch.cuda.get_device_capability()
    assert major == 10, f"cuDNN QSA needs SM100/SM103, got sm_{major}{minor}"
    packs = [
        [4096],
        [8192],
        [8192, 1500, 997],  # DAPO-like micro-batch: one long response + short ones (<= 10752 tokens)
        [10240],
        [32768],
    ]
    for hq, hkv in ((24, 2), (12, 1)):  # TP1, and TP2 per rank
        for lengths in packs:
            run(lengths, hq, hkv, args)


if __name__ == "__main__":
    main()
