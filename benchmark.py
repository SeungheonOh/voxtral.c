#!/usr/bin/env python3
"""Fast benchmark runner for a representative subset of audio files.

By default it runs a small/medium/long mini-suite under samples/benchmark/night1968
for N repeats, then reports metrics compatible with SPEED.md summaries.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_SUITE = [
    "5s_dont_worry_about_him.wav",
    "45s_right_through_the_billboard.wav",
    "60s_i_dont_want_anyones_life_on_my_hands.wav",
]

AUDIO_RE = re.compile(r"Audio:\s+\d+\s+samples\s+\(([0-9.]+)\s+seconds\)")
DECODER_RE = re.compile(
    r"Decoder:\s+\d+\s+text tokens\s+\((\d+)\s+steps\)\s+in\s+(\d+)\s+ms\s+"
    r"\(prefill\s+(\d+)\s+ms\s+\+\s+([0-9.]+)\s+ms/step\)"
)
ENCODER_RE = re.compile(r"Encoder:\s+\d+\s+mel\s+->\s+\d+\s+tokens\s+\((\d+)\s+ms\)")


@dataclass
class RunResult:
    file: Path
    repeat: int
    audio_sec: float
    steps: int
    step_ms: float
    prefill_ms: int
    encoder_ms: int
    decoder_total_ms: int
    wall_sec: float


def parse_metrics(stderr_text: str) -> tuple[float, int, float, int, int, int]:
    audio_m = AUDIO_RE.search(stderr_text)
    dec_m = DECODER_RE.search(stderr_text)
    enc_m = ENCODER_RE.search(stderr_text)
    if not audio_m or not dec_m:
        raise ValueError("missing Audio/Decoder metrics in stderr")

    audio_sec = float(audio_m.group(1))
    steps = int(dec_m.group(1))
    decoder_total_ms = int(dec_m.group(2))
    prefill_ms = int(dec_m.group(3))
    step_ms = float(dec_m.group(4))
    encoder_ms = int(enc_m.group(1)) if enc_m else 0
    return audio_sec, steps, step_ms, prefill_ms, encoder_ms, decoder_total_ms


def resolve_files(samples_root: Path, files_arg: list[str] | None) -> list[Path]:
    if files_arg:
        files: list[Path] = []
        for item in files_arg:
            p = Path(item)
            if not p.is_absolute():
                p = samples_root / p
            files.append(p)
    else:
        files = [samples_root / name for name in DEFAULT_SUITE]

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("missing benchmark files:\n" + "\n".join(missing))
    return files


def weighted_step_ms(results: Iterable[RunResult]) -> float:
    total_steps = 0
    total = 0.0
    for r in results:
        total_steps += r.steps
        total += r.step_ms * r.steps
    return (total / total_steps) if total_steps else 0.0


def avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def run_one(binary: str, model_dir: str, wav: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    cmd = [binary, "-d", model_dir, "-i", str(wav)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    return proc, wall


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a compact, repeatable benchmark suite.")
    parser.add_argument("--binary", default="./voxtral", help="Path to voxtral binary (default: ./voxtral)")
    parser.add_argument("--model", default="voxtral-model", help="Model directory (default: voxtral-model)")
    parser.add_argument("--samples-root", default="samples/benchmark/night1968", help="Root dir for default sample set")
    parser.add_argument("--files", nargs="+", help="Override file list (relative to samples-root unless absolute)")
    parser.add_argument("-n", "--repeats", type=int, default=2, help="Repeats per file (default: 2)")
    parser.add_argument("--long-threshold", type=float, default=59.0, help="Seconds threshold for long clips (default: 59)")
    parser.add_argument("--mode", default="mini_bench", help="Mode label in summary")
    parser.add_argument("--log", help="Log path (default: bench/mini_<mode>_<ts>.log)")
    parser.add_argument("--capture-stdout", action="store_true", help="Append stdout transcripts to the log")
    args = parser.parse_args()

    if args.repeats <= 0:
        print("--repeats must be > 0", file=sys.stderr)
        return 2

    binary = Path(args.binary)
    if not binary.exists():
        print(f"binary not found: {binary}", file=sys.stderr)
        return 2
    binary_cmd = str(binary.resolve())

    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"model directory not found: {model_dir}", file=sys.stderr)
        return 2

    samples_root = Path(args.samples_root)
    try:
        files = resolve_files(samples_root, args.files)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log else Path("bench") / f"mini_{args.mode}_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    total_runs = len(files) * args.repeats
    run_idx = 0

    wall_start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for rep in range(1, args.repeats + 1):
            for wav in files:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] rep {rep}/{args.repeats}: {wav.name}")

                proc, run_wall = run_one(binary_cmd, str(model_dir), wav)

                log.write(f"RUN\trep={rep}\tfile={wav.name}\n")
                if proc.stderr:
                    log.write(proc.stderr)
                    if not proc.stderr.endswith("\n"):
                        log.write("\n")
                if args.capture_stdout and proc.stdout:
                    log.write("STDOUT\n")
                    log.write(proc.stdout)
                    if not proc.stdout.endswith("\n"):
                        log.write("\n")
                log.write(f"ENDRUN\trep={rep}\tfile={wav.name}\n")

                if proc.returncode != 0:
                    print(f"benchmark command failed for {wav} (exit {proc.returncode})", file=sys.stderr)
                    return 1

                try:
                    audio_sec, steps, step_ms, prefill_ms, enc_ms, dec_total_ms = parse_metrics(proc.stderr)
                except ValueError as e:
                    print(f"unable to parse metrics for {wav}: {e}", file=sys.stderr)
                    return 1

                results.append(
                    RunResult(
                        file=wav,
                        repeat=rep,
                        audio_sec=audio_sec,
                        steps=steps,
                        step_ms=step_ms,
                        prefill_ms=prefill_ms,
                        encoder_ms=enc_ms,
                        decoder_total_ms=dec_total_ms,
                        wall_sec=run_wall,
                    )
                )

    wall_total = time.time() - wall_start
    audio_total = sum(r.audio_sec for r in results)
    overall_rtf = (wall_total / audio_total) if audio_total > 0 else 0.0
    w_step = weighted_step_ms(results)
    short_steps = [r.step_ms for r in results if r.audio_sec < args.long_threshold]
    long_steps = [r.step_ms for r in results if r.audio_sec >= args.long_threshold]
    short_avg = avg(short_steps)
    long_avg = avg(long_steps)

    summary = [
        "SUMMARY",
        f"mode={args.mode}",
        f"log_file={log_path.resolve()}",
        f"files={len(files)}",
        f"repeats={args.repeats}",
        f"planned_runs={total_runs}",
        f"runs={len(results)}",
        f"audio_sec={audio_total:.1f}",
        f"runtime_sec={wall_total:.1f}",
        f"overall_rtf={overall_rtf:.4f}",
        f"weighted_step_ms={w_step:.2f}",
        f"short_avg_step_ms={short_avg:.2f}",
        f"long_avg_step_ms={long_avg:.2f}",
    ]

    with log_path.open("a", encoding="utf-8") as log:
        for line in summary:
            log.write(line + "\n")

    for line in summary:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
