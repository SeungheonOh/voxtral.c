#!/usr/bin/env python3
"""Quantize Voxtral BF16 safetensors model to Q8 (per-row symmetric int8)."""

import struct
import sys
import os
import json
import shutil
import numpy as np


def read_safetensors(path):
    """Read a safetensors file, return (header_dict, raw_data_bytes, data_offset)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
        data_start = 8 + header_size
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(data_start)
        raw_data = f.read(file_size - data_start)
    return header, raw_data, data_start


def bf16_to_f32(buf):
    """Convert raw BF16 bytes to float32 numpy array."""
    bf16 = np.frombuffer(buf, dtype=np.uint16)
    # BF16 is the upper 16 bits of float32
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def quantize_q8_row(row_f32):
    """Quantize a single row: returns (scale_f32, q8_int8)."""
    amax = np.max(np.abs(row_f32))
    if amax == 0:
        scale = 0.0
        q = np.zeros(len(row_f32), dtype=np.int8)
    else:
        scale = float(amax) / 127.0
        q = np.round(row_f32 / scale).clip(-128, 127).astype(np.int8)
    return np.float32(scale), q


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir> [output_dir]", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else model_dir.rstrip("/") + "-q8"

    input_path = os.path.join(model_dir, "consolidated.safetensors")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Copy auxiliary files
    for fname in ("tekken.json", "params.json"):
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"Copied {fname}")

    print(f"\nReading {input_path}...")
    header, raw_data, _ = read_safetensors(input_path)

    # Remove metadata entry if present
    metadata = header.pop("__metadata__", None)

    # Sort tensors by data offset for sequential reading
    tensors = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])

    original_size = len(raw_data)
    print(f"Original data size: {original_size / 1e9:.2f} GB")
    print(f"Tensors: {len(tensors)}\n")

    # First pass: quantize all tensors and collect output data
    out_entries = {}  # name -> {dtype, shape, data_bytes}
    total_q8 = 0
    total_f32 = 0

    for name, info in tensors:
        dtype_str = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]
        tensor_bytes = raw_data[start:end]
        ndim = len(shape)

        if ndim == 2:
            # Weight matrix -> quantize to Q8
            rows, cols = shape
            if dtype_str == "BF16":
                f32 = bf16_to_f32(tensor_bytes).reshape(rows, cols)
            elif dtype_str == "F32":
                f32 = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(rows, cols)
            elif dtype_str == "F16":
                f32 = np.frombuffer(tensor_bytes, dtype=np.float16).astype(np.float32).reshape(rows, cols)
            else:
                print(f"  {name}: unsupported dtype {dtype_str}, keeping as-is")
                out_entries[name] = {
                    "dtype": dtype_str,
                    "shape": shape,
                    "data": tensor_bytes,
                }
                continue

            # Per-row quantization
            scales = np.empty(rows, dtype=np.float32)
            quants = np.empty((rows, cols), dtype=np.int8)
            for r in range(rows):
                scales[r], quants[r] = quantize_q8_row(f32[r])

            q8_data = scales.tobytes() + quants.tobytes()
            orig_sz = len(tensor_bytes)
            new_sz = len(q8_data)
            ratio = new_sz / orig_sz * 100

            print(f"  Q8  {name:60s} [{rows:6d}, {cols:6d}]  {orig_sz:>12,} -> {new_sz:>12,}  ({ratio:.0f}%)")
            total_q8 += 1

            out_entries[name] = {
                "dtype": "Q8",
                "shape": shape,
                "data": q8_data,
            }
        else:
            # 1D or other: keep as F32
            if dtype_str == "BF16":
                f32 = bf16_to_f32(tensor_bytes)
                f32_bytes = f32.tobytes()
            elif dtype_str == "F32":
                f32_bytes = tensor_bytes
            elif dtype_str == "F16":
                f32 = np.frombuffer(tensor_bytes, dtype=np.float16).astype(np.float32)
                f32_bytes = f32.tobytes()
            else:
                f32_bytes = tensor_bytes

            orig_sz = len(tensor_bytes)
            new_sz = len(f32_bytes)
            print(f"  F32 {name:60s} {str(shape):20s}  {orig_sz:>12,} -> {new_sz:>12,}")
            total_f32 += 1

            out_entries[name] = {
                "dtype": "F32",
                "shape": shape,
                "data": f32_bytes,
            }

    # Build output safetensors file
    # Compute data offsets
    offset = 0
    out_header = {}
    ordered_names = list(out_entries.keys())

    for name in ordered_names:
        entry = out_entries[name]
        data_len = len(entry["data"])
        out_header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + data_len],
        }
        offset += data_len

    total_data_size = offset

    # Serialize header JSON
    header_json = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    header_size = len(header_json)

    output_path = os.path.join(output_dir, "consolidated.safetensors")
    print(f"\nWriting {output_path}...")

    with open(output_path, "wb") as f:
        # 8 bytes: header size
        f.write(struct.pack("<Q", header_size))
        # Header JSON
        f.write(header_json)
        # Tensor data (contiguous)
        for name in ordered_names:
            f.write(out_entries[name]["data"])

    output_file_size = 8 + header_size + total_data_size
    print(f"\nDone!")
    print(f"  Tensors quantized (Q8): {total_q8}")
    print(f"  Tensors kept (F32):     {total_f32}")
    print(f"  Original file size:     {os.path.getsize(input_path) / 1e9:.2f} GB")
    print(f"  Output file size:       {output_file_size / 1e9:.2f} GB")
    print(f"  Reduction:              {(1 - output_file_size / os.path.getsize(input_path)) * 100:.1f}%")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
