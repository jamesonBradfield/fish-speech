"""1999-era audio crunch for TTS output (System Shock 2 / late-90s game voice).

Recipe (numpy-only, no new deps):
  1. mono mix
  2. telephone/game bandpass 200-3400 Hz (FFT brickwall)
  3. downsample to 11025 Hz (classic 90s game voice rate)
  4. dynamics crush: tanh drive + normalize (flat monotone feel)
  5. 8-bit quantization + 1/2-LSB dither (codec grain)
  6. output at 11025 Hz

Usage:
  python tools/lofi_1999.py in.wav out.wav [--rate 11025] [--drive 1.5]
"""
import argparse

import numpy as np
import soundfile as sf


def bandpass(x, sr, lo=200.0, hi=3400.0):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    X[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(X, n)


def resample(x, sr_in, sr_out):
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x)) / sr_in
    t_out = np.arange(n_out) / sr_out
    return np.interp(t_out, t_in, x)


def crunch(x, sr, out_rate, drive):
    x = x - x.mean()
    x = bandpass(x, sr)
    x = resample(x, sr, out_rate)
    # dynamics crush: soft-clip drive, then full-scale normalize (flat)
    x = np.tanh(drive * x)
    x = x / (np.abs(x).max() + 1e-9)
    # 8-bit quantization with 1/2-LSB dither (codec grain)
    dither = (np.random.rand(len(x)) - 0.5) / 127.0
    x = np.round((x + dither) * 127.0) / 127.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--rate", type=int, default=11025)
    ap.add_argument("--drive", type=float, default=1.5)
    args = ap.parse_args()

    data, sr = sf.read(args.input, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    out = crunch(mono, sr, args.rate, args.drive)
    sf.write(args.output, out, args.rate)
    rms = float((out**2).mean() ** 0.5)
    print(
        f"{args.input}: {sr}Hz {len(mono)/sr:.2f}s "
        f"-> {args.output}: {args.rate}Hz {len(out)/args.rate:.2f}s rms={rms:.3f}"
    )


if __name__ == "__main__":
    main()
