# Voxtral Speed Roadmap

## Scope
- Keep MPS backend.
- No quantization for now.
- Focus on significant speedups, especially on long clips.

## Latest Baseline (2026-02-13, post decoder-attention rewrite)
Hardware: Apple Silicon M3 Max, MPS enabled.

Corpus benchmark:
- Path: `/Users/antirez/hack/2026/qwen-asr/samples/night_of_the_living_dead_1968`
- Files: 20 wav clips, total 560.6s audio
- Raw log: `bench/night1968_20260213_205257.log`
- Overall RTF: `0.3998`
- Weighted decoder step: `26.43 ms/step` (`37.84 tok/s`)
- Short clips (<60s): `24.01 ms/step`
- Long clips (>=60s): `29.80 ms/step`

Interpretation:
- Decoder attention kernel change delivered the first major no-quantization win.
- Exit targets are now met on this corpus (`RTF <= 0.40`, long-clip `<= 30 ms/step`).

## Latest A/B (2026-02-13, fp16 KV cache)
Same-session corpus A/B on the same 20-file set:

- fp16 KV enabled (default):
  - Log: `bench/night1968_fp16kv_20260213_211157.log`
  - Overall RTF: `0.4074`
  - Weighted decoder step: `26.95 ms/step`
  - Short clips: `24.14 ms/step`
  - Long clips: `30.00 ms/step`
- fp16 KV disabled (`VOX_DECODER_KV_FP16=0`):
  - Log: `bench/night1968_fp32kv_20260213_211832.log`
  - Overall RTF: `0.4186`
  - Weighted decoder step: `27.67 ms/step`
  - Short clips: `24.76 ms/step`
  - Long clips: `31.30 ms/step`

Delta (fp16 vs fp32, same session):
- `~2.6%` faster weighted decoder step
- `~4.2%` faster long-clip step time
- `~2.7%` better overall RTF

## Latest A/B (2026-02-13, packed QKV path / no deinterleave copies)
Same 20-file corpus, compared against commit `bcd48e11` baseline binary:

- Baseline (`bcd48e11`):
  - Log: `bench/night1968_baseline_bcd48e11_20260213_234419.log`
  - Overall RTF: `0.4722`
  - Weighted decoder step: `30.07 ms/step`
  - Short clips: `27.94 ms/step`
  - Long clips: `31.75 ms/step`
- Packed-QKV (current):
  - Log: `bench/night1968_packedqkv_rerun_20260213_235220.log`
  - Overall RTF: `0.3956`
  - Weighted decoder step: `26.28 ms/step`
  - Short clips: `23.84 ms/step`
  - Long clips: `29.25 ms/step`

Delta (packed-QKV vs baseline):
- `~14.0%` better overall RTF
- `~11.6%` faster weighted decoder step
- `~13.2%` faster short-clip step time
- `~7.9%` faster long-clip step time

## Latest A/B (2026-02-14, decoder FFN gate fused kernel)
Same 20-file corpus, compared against commit `bd7bcaae` baseline binary:

- Baseline (`bd7bcaae`):
  - Log: `bench/night1968_baseline_bd7_20260214_112251.log`
  - Overall RTF: `0.3912`
  - Weighted decoder step: `26.05 ms/step`
  - Short clips: `23.58 ms/step`
  - Long clips: `28.90 ms/step`
- FFN gate fused kernel (current):
  - Log: `bench/night1968_ffn_gate_kernel_20260214_112843.log`
  - Overall RTF: `0.3847`
  - Weighted decoder step: `25.53 ms/step`
  - Short clips: `23.04 ms/step`
  - Long clips: `28.50 ms/step`

Delta (ffn gate fused kernel vs baseline):
- `~1.7%` better overall RTF
- `~2.0%` faster weighted decoder step
- `~2.3%` faster short-clip step time
- `~1.4%` faster long-clip step time

## Latest A/B (2026-02-14, mini benchmark, decoder W2+residual fused kernel)
Mini-suite benchmark (`benchmark.py`, 3 representative clips, `-n 2`) compared against
commit `eaf78039` baseline binary:

- Baseline (`eaf78039`):
  - Log: `bench/mini_baseline_eaf_20260214_121218.log`
  - Overall RTF: `0.5549`
  - Weighted decoder step: `24.94 ms/step`
  - Short clips: `23.25 ms/step`
  - Long clips: `25.50 ms/step`
- W2+residual fused kernel (current):
  - Log: `bench/mini_w2_residual_20260214_121423.log`
  - Overall RTF: `0.5427`
  - Weighted decoder step: `24.18 ms/step`
  - Short clips: `22.52 ms/step`
  - Long clips: `24.75 ms/step`

Delta (W2+residual fused kernel vs mini baseline):
- `~2.2%` better overall RTF
- `~3.0%` faster weighted decoder step
- `~3.1%` faster short-clip step time
- `~2.9%` faster long-clip step time

## Latest A/B (2026-02-14, mini benchmark, decoder WO+residual fused kernel)
Mini-suite benchmark (`benchmark.py`, 3 representative clips, `-n 2`) compared against
commit `6da60cb6` baseline binary:

- Baseline (`6da60cb6`):
  - Log: `bench/mini_baseline_6da_wofusion_n2_20260214_125124.log`
  - Overall RTF: `0.5422`
  - Weighted decoder step: `24.11 ms/step`
  - Short clips: `22.52 ms/step`
  - Long clips: `24.65 ms/step`
- WO+residual fused kernel (current):
  - Log: `bench/mini_wo_residual_fused_n2_20260214_125326.log`
  - Overall RTF: `0.5370`
  - Weighted decoder step: `23.78 ms/step`
  - Short clips: `22.15 ms/step`
  - Long clips: `24.35 ms/step`

Delta (WO+residual fused kernel vs mini baseline):
- `~1.0%` better overall RTF
- `~1.4%` faster weighted decoder step
- `~1.6%` faster short-clip step time
- `~1.2%` faster long-clip step time

## Latest A/B (2026-02-14, mini benchmark, vectorized decoder fused kernels)
Mini-suite benchmark (`benchmark.py`, 3 representative clips, `-n 2`) compared against
commit `7ace0254` baseline binary:

- Baseline (`7ace0254`):
  - Log: `bench/mini_baseline_7ace_vec_n2_20260214_131121.log`
  - Overall RTF: `0.5428`
  - Weighted decoder step: `23.89 ms/step`
  - Short clips: `22.25 ms/step`
  - Long clips: `24.40 ms/step`
- Vectorized fused kernels (current):
  - Log: `bench/mini_vec4_decoder_kernels_n2_20260214_131324.log`
  - Overall RTF: `0.5370`
  - Weighted decoder step: `23.66 ms/step`
  - Short clips: `22.02 ms/step`
  - Long clips: `24.25 ms/step`

Delta (vectorized fused kernels vs mini baseline):
- `~1.1%` better overall RTF
- `~1.0%` faster weighted decoder step
- `~1.0%` faster short-clip step time
- `~0.6%` faster long-clip step time

## Priority Plan (Next Work Only)

### Done: decoder single-token attention rewrite
- Implemented in:
  - `voxtral_shaders.metal` (`decoder_attention`)
  - `voxtral_metal.m` (`vox_metal_decoder_full_step` dispatch to 32 threads)
- Change:
  - 32-thread single-SIMD kernel, 4 dims per lane, no cross-SIMD barrier per KV step.
- Result:
  - Weighted step: `32.69 -> 26.43 ms/step` (~19.1% faster)
  - Long clips: `~40 -> 29.8 ms/step` (~25% faster)
  - Corpus RTF: `0.4906 -> 0.3998` (~18.5% faster)

### Done: decoder fp16 KV cache with fp32 compute fallback
- Implemented in:
  - `voxtral.h` (dual-format KV pointers + fp16 mode flag)
  - `voxtral_decoder.c` (fp16 KV alloc/grow/compact + switch-to-fp32 fallback)
  - `voxtral_metal.m` (fp16 KV kernel dispatch in decoder full-step/prefill)
  - `voxtral_shaders.metal` (fp16 KV copy + fp16 KV attention kernels)
- Runtime control:
  - `VOX_DECODER_KV_FP16=1` (default): fp16 KV cache
  - `VOX_DECODER_KV_FP16=0`: force fp32 KV cache
- Validation:
  - `make test` passed (`2 passed, 0 failed`)

### Done: remove deinterleave copies from prefill/encoder paths
- Implemented in:
  - `voxtral_metal.m` (`vox_metal_encoder_full_step`, `vox_metal_decoder_prefill_step`)
  - `voxtral_shaders.metal` (strided packed-QKV kernels + q-strided attention entrypoints)
- Change:
  - Removed explicit Q/K/V split buffers in monolithic encoder and decoder prefill.
  - Operate directly on packed merged QKV output (bias, RoPE, KV copy, attention input).
- Validation:
  - `make test` passed (`2 passed, 0 failed`)

### Done: decoder FFN gate fused kernel (M=1 path)
- Implemented in:
  - `voxtral_shaders.metal` (`decoder_ffn_gate`)
  - `voxtral_metal.m` (`encode_wo_ffn_steps` dispatch)
- Change:
  - Replaced decoder FFN gate path `x_norm @ [w1;w3]^T` + `silu` + elementwise multiply
    with one custom compute kernel for the token-by-token (`M=1`) decoder path.
  - Removed one large MPS matmul and one follow-up compute dispatch per decoder layer.
- Validation:
  - `make test` passed (`2 passed, 0 failed`)

### Done: decoder W2+residual fused kernel (M=1 path)
- Implemented in:
  - `voxtral_shaders.metal` (`decoder_w2_residual`)
  - `voxtral_metal.m` (`encode_wo_ffn_steps` dispatch)
- Change:
  - Replaced decoder FFN tail `gate @ w2^T` + `x += ffn_out`
    with one custom compute kernel in token-by-token (`M=1`) decoder path.
  - Removed one MPS matmul and one follow-up add dispatch per decoder layer.
- Validation:
  - `make test` passed (`2 passed, 0 failed`)

### Done: decoder WO+residual fused kernel (M=1 path)
- Implemented in:
  - `voxtral_shaders.metal` (`decoder_wo_residual`)
  - `voxtral_metal.m` (`encode_wo_ffn_steps` dispatch)
- Change:
  - Replaced decoder attention output projection + residual add
    (`proj = attn @ wo^T`, then `x += proj`) with one custom compute kernel
    for token-by-token (`M=1`) decoder path.
  - Removed one MPS matmul and one follow-up add dispatch per decoder layer.
- Validation:
  - `make test` passed (`2 passed, 0 failed`)

### Done: vectorized decoder fused kernels (`float4`/`half4`)
- Implemented in:
  - `voxtral_shaders.metal`
    - `decoder_ffn_gate`
    - `decoder_w2_residual`
    - `decoder_wo_residual`
- Change:
  - Reworked inner dot-product loops from scalar lanes to `float4`/`half4` chunks
    with scalar tail handling.
  - Reduced loop/control overhead and improved memory access efficiency
    in the decoder token-by-token fused kernels.
- Validation:
  - `make test` passed (`2 passed, 0 failed`)

### 1) Reduce command-encoder churn in monolithic steps
- Fuse adjacent small kernels in encoder/prefill where synchronization allows.
- Reuse one compute encoder for sequential lightweight ops per layer to trim CPU submission overhead.
- Files:
  - `voxtral_metal.m`
  - `voxtral_shaders.metal` (if new fused kernels are added)
- Target impact:
  - ~3-8% depending on sequence length and layer count.

### 2) Decoder KV write path fusion (RoPE-K + KV write)
- Revisit with a tighter kernel that minimizes extra reads and register pressure.
- Keep Q RoPE separate; focus only on K transform + K/V writes in one dispatch.
- Files:
  - `voxtral_shaders.metal`
  - `voxtral_metal.m`
- Target impact:
  - ~1-4% decoder step time.

## Benchmark Protocol (Run After Each Change)
0. Run and log raw output first, then parse the log (never re-run because of parser bugs).

1. Quick checks:
- `./voxtral -d voxtral-model -i samples/test_speech.wav`
- `./voxtral -d voxtral-model -i samples/jfk.wav`

2. Fast A/B check (default for each optimization iteration):
- `./benchmark.py -n 2 --mode <candidate>`
- Uses `samples/benchmark/night1968` representative mini-suite:
  - `5s_dont_worry_about_him.wav`
  - `45s_right_through_the_billboard.wav`
  - `60s_i_dont_want_anyones_life_on_my_hands.wav`
- Typical runtime: ~1-2 minutes.

3. Full corpus check (only for promoted wins):
- `/Users/antirez/hack/2026/qwen-asr/samples/night_of_the_living_dead_1968` (20 wav)
- Save stderr output to `bench/*.log`, then parse metrics from that log.

4. Report these metrics:
- `overall_rtf`
- `weighted_step_ms`
- short-clip avg step ms
- long-clip avg step ms (`>= 59s` for mini-suite accounting)

## Exit Criteria
- `overall_rtf <= 0.40` on the 20-file corpus.
- Long-clip decoder step `<= 30 ms/step`.
- No transcription quality regression in practical tests.
