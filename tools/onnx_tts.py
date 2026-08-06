"""Fish-speech 1.5 TTS on ONNX Runtime + DirectML (AMD GPU, Windows).

Replicates generate_long / decode_one_token_ar / generate / decode_n_tokens
(fish_speech/models/text2semantic/inference.py) with the model forward steps
executed by ONNX graphs (slow.onnx / fast.onnx / fast_emb.onnx) on the
DirectML execution provider. KV-cache bookkeeping, RoPE freqs, causal masks,
and all sampling (top-p / temperature / repetition penalty) run in numpy.

Usage:
  python tools/onnx_tts.py \
      --checkpoint checkpoints/fish-speech-1.5 \
      --onnx-dir onnx_artifacts \
      --text "I am the master computer." \
      --reference_audio C:/path/ref.wav \
      --reference_text "reference transcript" \
      --output out.wav
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from fish_speech.models.text2semantic.inference import (
    encode_tokens,
    split_text,
    Conversation,
)
from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.utils.schema import Message, TextPart

CACHE_LEN = 1024
N_LAYER = 24
N_FAST = 4
N_CODEBOOK = 8
WIN_SIZE = 16


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def sample_logits(logits, previous_tokens, repetition_penalty, top_p, temperature):
    """numpy port of logits_to_probs + multinomial_sample_one_no_sync."""
    if previous_tokens is not None and previous_tokens.size:
        l = logits.copy()
        s = l[previous_tokens]
        l[previous_tokens] = np.where(s < 0, s * repetition_penalty, s / repetition_penalty)
    else:
        l = logits
    order = np.argsort(l)[::-1]
    sorted_l = l[order]
    cum = np.cumsum(softmax(sorted_l))
    remove = cum > top_p
    remove[0] = False
    drop = np.zeros_like(l, dtype=bool)
    drop[order[remove]] = True
    l = np.where(drop, -np.inf, l)
    l = l / max(temperature, 1e-5)
    probs = softmax(l)
    idx = np.random.choice(len(probs), p=probs)
    return idx, probs


class OnnxTTS:
    def __init__(
        self, checkpoint: str, onnx_dir: str, use_dml: bool = True, use_fp16: bool = False
    ):
        from fish_speech.models.text2semantic.inference import load_model

        self.checkpoint = checkpoint
        self.use_fp16 = use_fp16
        self._hdtype = np.float16 if use_fp16 else np.float32
        suffix = "16" if use_fp16 else ""
        self.model, _ = load_model(checkpoint, device="cpu", precision=torch.float32)
        assert isinstance(self.model, DualARTransformer)
        with torch.device("cpu"):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=torch.float32,
            )
        self.tok = self.model.tokenizer
        self.cfg = self.model.config
        self.semantic_ids = [
            self.tok.get_token_id(f"<|semantic:{i}|>") for i in range(1024)
        ]
        self.semantic_set = set(self.semantic_ids)
        self.semantic_begin = self.tok.semantic_begin_id
        self.im_end_id = self.tok.get_token_id("<|im_end|>")

        # RoPE + causal mask buffers (bit-exact to the torch buffers)
        self.freqs = self.model.freqs_cis[:CACHE_LEN].float().numpy()  # [2048,32,2]
        self.fast_freqs = self.model.fast_freqs_cis.float().numpy()  # [8,32,2]
        causal = self.model.causal_mask[:CACHE_LEN, :CACHE_LEN].numpy()  # [2048,2048]
        self.mask_rows = np.where(causal, 0.0, -np.inf).astype(self._hdtype)
        fast_causal = self.model.causal_mask[:N_CODEBOOK, :N_CODEBOOK].numpy()
        self.fast_mask_rows = np.where(fast_causal, 0.0, -np.inf).astype(self._hdtype)

        # ONNX sessions
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if use_dml
            else ["CPUExecutionProvider"]
        )
        d = Path(onnx_dir)
        self.onnx_dir = d
        self.use_dml = use_dml
        self.slow = ort.InferenceSession(
            str(d / f"slow{suffix}.onnx"), so, providers=providers
        )
        self.fast = ort.InferenceSession(
            str(d / f"fast{suffix}.onnx"), so, providers=providers
        )
        self.emb = ort.InferenceSession(str(d / "fast_emb.onnx"), so, providers=providers)
        self._uh2d_sess = None

        # VQGAN encoder/decoder, loaded once (loading is ~1s; decode is ~0.3s)
        from tools.vqgan.extract_vq import get_model
        from tools.export_onnx import Encoder, Decoder

        vqgan = get_model(
            "firefly_gan_vq",
            f"{checkpoint}/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            device="cpu",
        )
        self._vqgan_enc = Encoder(vqgan)
        self._vqgan_dec = Decoder(vqgan)

        # cache buffers
        self.k_cache = np.zeros((N_LAYER, 1, 2, CACHE_LEN, 64), self._hdtype)
        self.v_cache = np.zeros((N_LAYER, 1, 2, CACHE_LEN, 64), self._hdtype)
        self.fk_cache = np.zeros((N_FAST, 1, 2, N_CODEBOOK, 64), self._hdtype)
        self.fv_cache = np.zeros((N_FAST, 1, 2, N_CODEBOOK, 64), self._hdtype)

    def _uh2d(self):
        if self._uh2d_sess is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.enable_mem_pattern = False
            providers = (
                ["DmlExecutionProvider", "CPUExecutionProvider"]
                if self.use_dml
                else ["CPUExecutionProvider"]
            )
            self._uh2d_sess = ort.InferenceSession(
                str(self.onnx_dir / "uhdyn2d.onnx"), so, providers=providers
            )
        return self._uh2d_sess

    # ---- slow / fast steps ----
    def slow_step(self, tok_row, pos, is_semantic):
        feeds = {
            "tok": tok_row[None, :, None].astype(np.int64),  # [1,9,1]
            "pos": np.array([pos], np.int32),
            "freqs": self.freqs[pos : pos + 1],
            "mask": self.mask_rows[pos][None, None, None, :],
            "is_semantic": np.array([is_semantic], np.int32),
        }
        for i in range(N_LAYER):
            feeds[f"k_in_{i}"] = self.k_cache[i]
            feeds[f"v_in_{i}"] = self.v_cache[i]
        outs = self.slow.run(None, feeds)
        logits = np.reshape(outs[0], (102048,)).astype(np.float32)  # [102048]
        hidden = np.reshape(outs[1], (1, 1, 1024))
        for i in range(N_LAYER):
            self.k_cache[i][:, :, pos] = np.reshape(outs[2 + i], (1, 2, 64))
            self.v_cache[i][:, :, pos] = np.reshape(
                outs[2 + N_LAYER + i], (1, 2, 64)
            )
        return logits, hidden

    def fast_step(self, h, pos):
        feeds = {
            "h": h,
            "pos": np.array([pos], np.int32),
            "freqs": self.fast_freqs[pos : pos + 1],
            "mask": self.fast_mask_rows[pos][None, None, None, :],
        }
        for i in range(N_FAST):
            feeds[f"k_in_{i}"] = self.fk_cache[i]
            feeds[f"v_in_{i}"] = self.fv_cache[i]
        outs = self.fast.run(None, feeds)
        logits = np.reshape(outs[0], (1024,)).astype(np.float32)  # [1024]
        for i in range(N_FAST):
            self.fk_cache[i][:, :, pos] = np.reshape(outs[1 + i], (1, 2, 64))
            self.fv_cache[i][:, :, pos] = np.reshape(outs[1 + N_FAST + i], (1, 2, 64))
        return logits

    def fast_emb(self, ids):
        out = self.emb.run(["h"], {"ids": np.array([[ids]], np.int64)})[0]
        return np.reshape(out, (1, 1, 1024)).astype(self._hdtype)

    # ---- codebook sampling given slow-path logits + hidden ----
    def sample_codes(self, logits, hidden, previous_tokens, sampling_kwargs):
        # semantic token (row 0 history for rep penalty)
        sem = sample_logits(
            logits,
            previous_tokens[0] if previous_tokens is not None else None,
            **sampling_kwargs,
        )[0]

        # reset fast cache
        self.fk_cache.fill(0)
        self.fv_cache.fill(0)

        # warmup at position 0 with the semantic hidden state (discarded)
        self.fast_step(hidden, 0)

        a0 = sem - self.semantic_begin
        a0 = max(int(a0), 0)
        h = self.fast_emb(a0)
        codes = [int(sem), a0]
        for cb in range(1, N_CODEBOOK):
            logits_f = self.fast_step(h, cb)
            a = sample_logits(
                logits_f,
                previous_tokens[cb + 1] if previous_tokens is not None else None,
                **sampling_kwargs,
            )[0]
            h = self.fast_emb(a)
            codes.append(int(a))
        return np.array(codes, np.int64)  # [9]

    # ---- decode_one_token_ar (numpy port) ----
    def decode_one_token_ar(self, cur_token, pos, previous_tokens, sampling_kwargs):
        is_sem = 1 if int(cur_token[0]) in self.semantic_set else 0
        logits, hidden = self.slow_step(cur_token, pos, is_sem)
        return self.sample_codes(logits, hidden, previous_tokens, sampling_kwargs)

    # ---- generate ----
    def generate(self, prompt, max_new_tokens, sampling_kwargs):
        T = prompt.shape[1]
        if T >= CACHE_LEN:
            raise ValueError(f"prompt length {T} exceeds CACHE_LEN {CACHE_LEN}")
        max_new_tokens = min(max_new_tokens, CACHE_LEN - T)
        seq = np.zeros((1 + N_CODEBOOK, self.cfg.max_seq_len), np.int64)
        seq[:, :T] = prompt

        # prefill on torch CPU (oneDNN batched FFN ~0.9s) instead of DML
        # stepwise (~7s): forward_generate fills the torch KV caches, which we
        # copy into the engine's numpy caches so the DML decode steps continue
        # from the correct state. Logits + pre-norm hidden match slow.onnx.
        with torch.inference_mode():
            pt = torch.from_numpy(prompt.astype(np.int64)).unsqueeze(0)
            r = self.model.forward_generate(pt, torch.arange(T, dtype=torch.long))
            last_logits = r.logits[0, 0].numpy()
            last_hidden = (
                r.hidden_states[0, 0].float().numpy()[None, None].astype(self._hdtype)
            )
        for i, layer in enumerate(self.model.layers):
            kc = layer.attention.kv_cache.k_cache
            vc = layer.attention.kv_cache.v_cache
            self.k_cache[i][0, :, :T, :] = kc[0, :, :T, :].numpy().astype(self._hdtype)
            self.v_cache[i][0, :, :T, :] = vc[0, :, :T, :].numpy().astype(self._hdtype)

        previous_tokens = np.zeros((1 + N_CODEBOOK, self.cfg.max_seq_len), np.int64)
        # next_token is sampled from the LAST prompt position (matches torch
        # generate(): prefill_decode returns the first generated token)
        next_token = self.sample_codes(last_logits, last_hidden, None, sampling_kwargs)
        seq[:, T] = next_token
        input_pos = T + 1

        for i in range(max_new_tokens - 1):
            win = previous_tokens[:, max(0, i - WIN_SIZE) : i]
            nxt = self.decode_one_token_ar(next_token, input_pos, win, sampling_kwargs)
            previous_tokens[:, i] = nxt
            seq[:, input_pos] = nxt
            input_pos += 1
            if nxt[0] == self.im_end_id:
                break
            next_token = nxt

        y = seq[:, :input_pos]
        return y

    # ---- reference encoding (torch CPU, negligible cost) ----
    def encode_reference(self, audio):
        if isinstance(audio, (bytes, bytearray)):
            data, sr = sf.read(io.BytesIO(bytes(audio)), dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(audio, dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
        if sr != 22050:
            wave = torchaudio.functional.resample(wave, sr, 22050)
        codes = self._vqgan_enc(wave[None])  # [1, 8, L]
        return codes[0].to(torch.int32)  # [8, L]

    def synthesize(
        self,
        text,
        reference_audio=None,
        reference_text=None,
        max_new_tokens=0,
        top_p=0.7,
        repetition_penalty=1.5,
        temperature=0.7,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        tokenizer = self.tok
        encoded_prompts = [
            Conversation(
                messages=[
                    Message(
                        role="system",
                        parts=[TextPart(text="Speak out the provided text.")],
                        cal_loss=False,
                    )
                ]
            )
            .encode_for_inference(
                tokenizer=tokenizer, num_codebooks=self.cfg.num_codebooks
            )
            .to("cpu")
            .numpy()
        ]
        if reference_audio is not None:
            prompt_tokens = self.encode_reference(reference_audio)
            encoded_prompts.append(
                encode_tokens(
                    tokenizer,
                    string=reference_text,
                    device="cpu",
                    prompt_tokens=prompt_tokens,
                    num_codebooks=self.cfg.num_codebooks,
                )
                .numpy()
            )
        texts = split_text(text, 150)
        encoded = [
            encode_tokens(
                tokenizer, string=t, device="cpu", num_codebooks=self.cfg.num_codebooks
            ).numpy()
            for t in texts
        ]
        prompt = np.concatenate(encoded_prompts + encoded, axis=1)  # [9, S]

        sampling_kwargs = {
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
        }
        if max_new_tokens <= 0:
            max_new_tokens = self.cfg.max_seq_len - prompt.shape[1]

        y = self.generate(prompt, max_new_tokens, sampling_kwargs)
        codes = y[1:, prompt.shape[1] + 1 :]
        assert (codes >= 0).all()
        return codes

    def decode(self, codes):
        """[8, L] codes -> audio: FSQ lookup (torch) + conv stack on DirectML."""
        idx = torch.from_numpy(codes[None].astype(np.int64))
        with torch.inference_mode():
            ind = idx.view(1, 8, -1, idx.shape[-1]).permute(1, 0, 3, 2)
            dims = self._vqgan_dec.model.quantizer.residual_fsq.dim
            groups = self._vqgan_dec.model.quantizer.residual_fsq.groups
            dpg = dims // groups
            z_q = torch.empty((1, idx.shape[-1], dims))
            for i in range(groups):
                z_q[:, :, i * dpg : (i + 1) * dpg] = (
                    self._vqgan_dec.get_output_from_indices(i, ind[i])
                )
            z = z_q.transpose(1, 2).contiguous().numpy().astype(np.float32)
        try:
            out = self._uh2d().run(["audio"], {"z": z})[0]
            return out[0, 0]
        except Exception:
            with torch.inference_mode():
                audio = self._vqgan_dec.model.head(
                    self._vqgan_dec.model.quantizer.upsample(torch.from_numpy(z))
                )
            return audio[0, 0].numpy()

    def synthesize_request(self, req) -> np.ndarray:
        """Full TTS for a ServeTTSRequest; returns 22050 Hz float32 audio."""
        refs = list(req.references or [])
        kw = dict(
            max_new_tokens=req.max_new_tokens,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            seed=req.seed,
        )
        if refs and refs[0].audio:
            codes = self.synthesize(
                req.text, reference_audio=refs[0].audio, reference_text=refs[0].text, **kw
            )
        else:
            codes = self.synthesize(req.text, **kw)
        return self.decode(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/fish-speech-1.5")
    ap.add_argument("--onnx-dir", default="onnx_artifacts")
    ap.add_argument("--text", required=True)
    ap.add_argument("--reference_audio", required=True)
    ap.add_argument("--reference_text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-dml", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="use slow16/fast16 graphs")
    args = ap.parse_args()

    t0 = time.perf_counter()
    engine = OnnxTTS(
        args.checkpoint, args.onnx_dir, use_dml=not args.no_dml, use_fp16=args.fp16
    )
    print(f"[init] {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    codes = engine.synthesize(args.text, args.reference_audio, args.reference_text)
    print(f"[generate] {time.perf_counter()-t0:.1f}s -> codes {codes.shape}")

    t0 = time.perf_counter()
    audio = engine.decode(codes)
    print(f"[decode] {time.perf_counter()-t0:.1f}s -> {len(audio)/22050:.2f}s audio")
    sf.write(args.output, audio, 22050)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
