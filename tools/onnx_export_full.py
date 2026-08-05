"""Export fish-speech 1.5 DualAR text2semantic transformer to ONNX for DirectML.

Produces three graphs (one-token step, explicit KV-cache I/O, all fp32):
  slow.onnx    - 24-layer semantic/audio transformer, one step, cache in/out
  fast.onnx    - 4-layer codebook transformer, one step over 8 positions
  fast_emb.onnx- fast codebook embedding lookup (Gather)

Cache update is done OUTSIDE the graph (numpy): the graph emits k_step/v_step
for the current token and attends over the cache input with an in-graph
Slice+Concat insert at position `pos`. RoPE freqs, causal mask row and the
semantic flag are precomputed/sliced in numpy and passed as inputs, so the
graph contains no data-dependent Reshape (which DirectML rejects).

Usage:
  python tools/onnx_export_full.py --checkpoint checkpoints/fish-speech-1.5 \
      --out-dir onnx_artifacts
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from fish_speech.models.text2semantic.llama import DualARTransformer, apply_rotary_emb
from fish_speech.models.text2semantic.inference import load_model

CACHE_LEN = 2048  # static KV cache length (config max_seq_len is 8192)


class SlowStep(nn.Module):
    """One-token step of the slow transformer with explicit KV-cache I/O."""

    def __init__(self, model: DualARTransformer):
        super().__init__()
        self.m = model
        cfg = model.config
        self.n_head = cfg.n_head
        self.n_local = cfg.n_local_heads
        self.head_dim = cfg.head_dim
        self.dim = cfg.dim
        self.kv_size = self.n_local * self.head_dim

    def embed(self, tok, is_semantic):
        # tok: [1, 9, 1] int64, is_semantic: [1] int32
        emb = self.m.embeddings(tok[:, 0])  # [1, 1, dim]
        cb = None
        for i in range(self.m.config.num_codebooks):
            e = self.m.codebook_embeddings(
                tok[:, i + 1] + i * self.m.config.codebook_size
            )
            cb = e if cb is None else cb + e
        flag = is_semantic.float().to(emb.dtype).view(1, 1, 1)
        return emb + cb * flag

    def attn(self, layer, x, freqs, mask, pos, k_in, v_in):
        bsz, seqlen, _ = x.shape
        q, k, v = layer.attention.wqkv(layer.attention_norm(x)).split(
            [self.dim, self.kv_size, self.kv_size], dim=-1
        )
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local, self.head_dim)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)
        q = q.transpose(1, 2)  # [1, H, 1, D]
        k = k.transpose(1, 2)  # [1, n_local, 1, D]
        v = v.transpose(1, 2)

        # insert current k/v at position `pos` in the cache: Slice+Concat
        k_eff = torch.cat([k_in[:, :, :pos, :], k, k_in[:, :, pos + 1:, :]], dim=2)
        v_eff = torch.cat([v_in[:, :, :pos, :], v, v_in[:, :, pos + 1:, :]], dim=2)

        # GQA: broadcast n_local -> n_head (Mul-by-ones trick, no Tile/Expand)
        ones = torch.ones(1, 1, self.n_head // self.n_local, 1, 1, dtype=x.dtype)
        k_full = (k_eff.unsqueeze(2) * ones).reshape(bsz, self.n_head, CACHE_LEN, self.head_dim)
        v_full = (v_eff.unsqueeze(2) * ones).reshape(bsz, self.n_head, CACHE_LEN, self.head_dim)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # [1,H,1,L]
        scores = scores + mask
        w = torch.softmax(scores, dim=-1)
        y = torch.matmul(w, v_full)  # [1,H,1,D]
        y = y.transpose(1, 2).reshape(bsz, seqlen, self.dim)
        return layer.attention.wo(y), k, v

    def forward(self, tok, pos, freqs, mask, is_semantic, k_ins, v_ins):
        x = self.embed(tok, is_semantic)
        k_steps, v_steps = [], []
        for layer, k_in, v_in in zip(self.m.layers, k_ins, v_ins):
            h, k_step, v_step = self.attn(layer, x, freqs, mask, pos, k_in, v_in)
            x = x + h
            x = x + layer.feed_forward(layer.ffn_norm(x))
            k_steps.append(k_step)
            v_steps.append(v_step)
        hidden = x  # pre-norm hidden feeds the fast transformer (matches torch)
        if self.m.config.tie_word_embeddings:
            logits = F.linear(self.m.norm(x), self.m.embeddings.weight)
        else:
            logits = self.m.output(self.m.norm(x))
        return logits, hidden, k_steps, v_steps


class Prefill(nn.Module):
    """Full-sequence causal forward over the prompt: one call replaces T
    stepwise slow_steps. Returns the last-position logits + hidden, which
    feed decode_one_token_ar's codebook sampling exactly like slow_step."""

    def __init__(self, model: DualARTransformer, cache_len: int = CACHE_LEN):
        super().__init__()
        self.m = model
        cfg = model.config
        self.n_head = cfg.n_head
        self.n_local = cfg.n_local_heads
        self.head_dim = cfg.head_dim
        self.dim = cfg.dim
        self.L = cache_len

    def attn(self, layer, x, freqs, m):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local * self.head_dim
        q, k, v = layer.attention.wqkv(layer.attention_norm(x)).split(
            [self.dim, kv_size, kv_size], dim=-1
        )
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local, self.head_dim)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ones = torch.ones(1, 1, self.n_head // self.n_local, 1, 1, dtype=x.dtype)
        k_full = (k.unsqueeze(2) * ones).reshape(
            bsz, self.n_head, seqlen, self.head_dim
        )
        v_full = (v.unsqueeze(2) * ones).reshape(
            bsz, self.n_head, seqlen, self.head_dim
        )
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        scores = scores + m
        w = torch.softmax(scores, dim=-1)
        y = torch.matmul(w, v_full)
        y = y.transpose(1, 2).reshape(bsz, seqlen, self.dim)
        return layer.attention.wo(y)

    def forward(self, inp, is_semantic, last_pos):
        emb = self.m.embeddings(inp[:, 0])  # [1, L, dim]
        cb = 0
        for i in range(self.m.config.num_codebooks):
            cb = cb + self.m.codebook_embeddings(
                inp[:, i + 1] + i * self.m.config.codebook_size
            )
        x = emb + cb * is_semantic.float().to(emb.dtype).view(1, self.L, 1)

        freqs = self.m.freqs_cis[: self.L].float()  # [L, n_elem, 2]
        causal = self.m.causal_mask[: self.L, : self.L]
        m = torch.where(causal, 0.0, float("-inf"))[None, None, :, :].to(x.dtype)

        for layer in self.m.layers:
            x = x + self.attn(layer, x, freqs, m)
            x = x + layer.feed_forward(layer.ffn_norm(x))
        hidden_all = self.m.norm(x)  # [1, L, dim]
        idx = last_pos.to(torch.int64).view(1, 1, 1).expand(1, 1, self.dim)
        hidden = torch.gather(hidden_all, 1, idx)  # [1, 1, dim]
        if self.m.config.tie_word_embeddings:
            logits = F.linear(hidden, self.m.embeddings.weight)
        else:
            logits = self.m.output(hidden)
        return logits, hidden


class FastStep(nn.Module):
    """One codebook-position step of the fast transformer (no embedding)."""

    def __init__(self, model: DualARTransformer):
        super().__init__()
        self.m = model
        cfg = model.config
        self.n_head = cfg.fast_n_head
        self.n_local = cfg.fast_n_local_heads
        self.head_dim = cfg.fast_head_dim
        self.dim = cfg.fast_dim
        self.kv_size = self.n_local * self.head_dim

    def attn(self, layer, x, freqs, mask, pos, k_in, v_in):
        bsz, seqlen, _ = x.shape
        q, k, v = layer.attention.wqkv(layer.attention_norm(x)).split(
            [self.dim, self.kv_size, self.kv_size], dim=-1
        )
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local, self.head_dim)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        n_cb = self.m.config.num_codebooks
        k_eff = torch.cat([k_in[:, :, :pos, :], k, k_in[:, :, pos + 1:, :]], dim=2)
        v_eff = torch.cat([v_in[:, :, :pos, :], v, v_in[:, :, pos + 1:, :]], dim=2)

        ones = torch.ones(1, 1, self.n_head // self.n_local, 1, 1, dtype=x.dtype)
        k_full = (k_eff.unsqueeze(2) * ones).reshape(bsz, self.n_head, n_cb, self.head_dim)
        v_full = (v_eff.unsqueeze(2) * ones).reshape(bsz, self.n_head, n_cb, self.head_dim)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        scores = scores + mask
        w = torch.softmax(scores, dim=-1)
        y = torch.matmul(w, v_full)
        y = y.transpose(1, 2).reshape(bsz, seqlen, self.dim)
        return layer.attention.wo(y), k, v

    def forward(self, h, pos, freqs, mask, k_ins, v_ins):
        x = h  # [1, 1, fast_dim]
        k_steps, v_steps = [], []
        for layer, k_in, v_in in zip(self.m.fast_layers, k_ins, v_ins):
            hh, k_step, v_step = self.attn(layer, x, freqs, mask, pos, k_in, v_in)
            x = x + hh
            x = x + layer.feed_forward(layer.ffn_norm(x))
            k_steps.append(k_step)
            v_steps.append(v_step)
        out = self.m.fast_norm(x)
        logits = self.m.fast_output(out)
        return logits, k_steps, v_steps


class FastEmb(nn.Module):
    def __init__(self, model: DualARTransformer):
        super().__init__()
        self.e = model.fast_embeddings

    def forward(self, ids):
        return self.e(ids)  # [1, 1] -> [1, 1, fast_dim]


def export(checkpoint: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(checkpoint, device="cpu", precision=torch.float32)
    assert isinstance(model, DualARTransformer), "expected DualARTransformer"

    torch.manual_seed(0)
    n_cb = model.config.num_codebooks

    # ---- slow ----
    slow = SlowStep(model).eval()
    tok = torch.cat(
        [
            torch.randint(0, model.config.vocab_size, (1, 1, 1)),
            torch.randint(0, model.config.codebook_size, (1, n_cb, 1)),
        ],
        dim=1,
    )
    pos = torch.tensor([5], dtype=torch.int32)
    freqs = torch.randn(1, 32, 2)
    mask = torch.randn(1, 1, 1, CACHE_LEN)
    is_sem = torch.tensor([1], dtype=torch.int32)
    k_ins = [torch.zeros(1, 2, CACHE_LEN, 64) for _ in model.layers]
    v_ins = [torch.zeros(1, 2, CACHE_LEN, 64) for _ in model.layers]
    ref = slow(tok, pos, freqs, mask, is_sem, k_ins, v_ins)

    torch.onnx.export(
        slow,
        (tok, pos, freqs, mask, is_sem, k_ins, v_ins),
        str(out / "slow.onnx"),
        opset_version=18,
        input_names=["tok", "pos", "freqs", "mask", "is_semantic"]
        + [f"k_in_{i}" for i in range(len(model.layers))]
        + [f"v_in_{i}" for i in range(len(model.layers))],
        output_names=["logits", "hidden"]
        + [f"k_step_{i}" for i in range(len(model.layers))]
        + [f"v_step_{i}" for i in range(len(model.layers))],
        do_constant_folding=False,
    )
    print(f"exported {out/'slow.onnx'}")

    # ---- prefill ----
    prefill = Prefill(model).eval()
    L = CACHE_LEN
    pf_inp = torch.cat(
        [
            torch.randint(0, model.config.vocab_size, (1, 1, L)),
            torch.randint(0, model.config.codebook_size, (1, n_cb, L)),
        ],
        dim=1,
    )
    pf_is_sem = torch.randint(0, 2, (1, L), dtype=torch.int32)
    pf_last_pos = torch.tensor([L - 1], dtype=torch.int32)
    prefill_ref = prefill(pf_inp, pf_is_sem, pf_last_pos)
    torch.onnx.export(
        prefill,
        (pf_inp, pf_is_sem, pf_last_pos),
        str(out / "prefill.onnx"),
        opset_version=18,
        input_names=["inp", "is_semantic", "last_pos"],
        output_names=["logits", "hidden"],
        do_constant_folding=False,
    )
    print(f"exported {out/'prefill.onnx'}")

    # ---- fast ----
    fast = FastStep(model).eval()
    h = torch.randn(1, 1, model.config.fast_dim)
    posf = torch.tensor([3], dtype=torch.int32)
    freqs_f = torch.randn(1, 32, 2)
    mask_f = torch.randn(1, 1, 1, n_cb)
    kf_ins = [torch.zeros(1, 2, n_cb, 64) for _ in model.fast_layers]
    vf_ins = [torch.zeros(1, 2, n_cb, 64) for _ in model.fast_layers]
    fast_ref = fast(h, posf, freqs_f, mask_f, kf_ins, vf_ins)

    torch.onnx.export(
        fast,
        (h, posf, freqs_f, mask_f, kf_ins, vf_ins),
        str(out / "fast.onnx"),
        opset_version=18,
        input_names=["h", "pos", "freqs", "mask"]
        + [f"k_in_{i}" for i in range(len(model.fast_layers))]
        + [f"v_in_{i}" for i in range(len(model.fast_layers))],
        output_names=["logits"]
        + [f"k_step_{i}" for i in range(len(model.fast_layers))]
        + [f"v_step_{i}" for i in range(len(model.fast_layers))],
        do_constant_folding=False,
    )
    print(f"exported {out/'fast.onnx'}")

    # ---- fast_emb ----
    ids = torch.tensor([[7]], dtype=torch.int64)
    torch.onnx.export(
        FastEmb(model).eval(),
        ids,
        str(out / "fast_emb.onnx"),
        opset_version=18,
        input_names=["ids"],
        output_names=["h"],
        do_constant_folding=False,
    )
    print(f"exported {out/'fast_emb.onnx'}")

    # ---- torch parity of exported graphs (onnxruntime CPU EP) ----
    import numpy as np
    import onnxruntime as ort

    def session(path):
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    sess = session(out / "slow.onnx")
    feeds = {
        "tok": tok.numpy(),
        "pos": pos.numpy(),
        "freqs": freqs.numpy(),
        "mask": mask.numpy(),
        "is_semantic": is_sem.numpy(),
        **{f"k_in_{i}": k.numpy() for i, k in enumerate(k_ins)},
        **{f"v_in_{i}": v.numpy() for i, v in enumerate(v_ins)},
    }
    onnx_out = sess.run(["logits", "hidden"], feeds)
    print("slow parity max-err:",
          float(np.abs(onnx_out[0] - ref[0].detach().numpy()).max()),
          float(np.abs(onnx_out[1] - ref[1].detach().numpy()).max()))

    sessf = session(out / "fast.onnx")
    feedsf = {
        "h": h.numpy(),
        "pos": posf.numpy(),
        "freqs": freqs_f.numpy(),
        "mask": mask_f.numpy(),
        **{f"k_in_{i}": k.numpy() for i, k in enumerate(kf_ins)},
        **{f"v_in_{i}": v.numpy() for i, v in enumerate(vf_ins)},
    }
    onnxf = sessf.run(["logits"], feedsf)
    print("fast parity max-err:",
          float(np.abs(onnxf[0] - fast_ref[0].detach().numpy()).max()))

    sembe = session(out / "fast_emb.onnx")
    print("fast_emb parity max-err:",
          float(np.abs(sembe.run(["h"], {"ids": ids.numpy()})[0]
                       - model.fast_embeddings(ids).detach().numpy()).max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="onnx_artifacts")
    args = ap.parse_args()
    export(args.checkpoint, args.out_dir)
