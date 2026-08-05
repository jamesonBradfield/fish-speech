"""Cheap 1999 TTS: Microsoft SAPI voice (Zira female) + 1999 audio crunch.

No fish-speech, no GPU, no ONNX. Windows-only (SAPI 5 via cscript).
The Microsoft SAPI voice IS a TTS -- we just drive it and crunch the
result down to the 1999 game-audio aesthetic (11 kHz, 8-bit, bandpass,
dynamics crush).

Usage:
  python tools/cheap_tts.py --text "I am the master computer." --output out.wav
  python tools/cheap_tts.py --text "..." --output out.wav --rate 8000 --drive 2.0
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.lofi_1999 import crunch

import soundfile as sf

HERE = Path(__file__).parent
VBS = HERE / "sapi_ref.vbs"


def speak(text: str) -> Path:
    """Render text with the installed SAPI voice; return the raw WAV path."""
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "raw.wav"
    try:
        r = subprocess.run(
            ["cscript", "//nologo", str(VBS), str(out), text],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"SAPI failed: {r.stdout} {r.stderr}")
        return out
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rate", type=int, default=11025)
    ap.add_argument("--drive", type=float, default=1.5)
    args = ap.parse_args()

    tmp = None
    try:
        raw = speak(args.text)
        tmp = raw.parent
        data, sr = sf.read(raw, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        out = crunch(mono, sr, args.rate, args.drive)
        sf.write(args.output, out, args.rate)
        rms = float((out**2).mean() ** 0.5)
        print(f"voice line: {len(mono)/sr:.2f}s @ {sr}Hz")
        print(
            f"1999 output: {args.output} {args.rate}Hz "
            f"{len(out)/args.rate:.2f}s rms={rms:.3f}"
        )
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
