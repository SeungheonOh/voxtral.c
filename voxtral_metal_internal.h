/*
 * voxtral_metal_internal.h - Shared Metal state for Q8 dispatch
 *
 * Internal header exporting globals from voxtral_metal.m so that
 * voxtral_metal_q8.m can use them.  Not part of the public API.
 */

#ifndef VOXTRAL_METAL_INTERNAL_H
#define VOXTRAL_METAL_INTERNAL_H

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

/* ---- Global Metal state (defined in voxtral_metal.m) ---- */

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;

/* Zero-copy mmap buffer (registered by vox_metal_register_mmap) */
extern id<MTLBuffer> g_mmap_buffer;
extern void *g_mmap_base;

/* Persistent GPU x buffer for cross-layer decoder fusion */
extern id<MTLBuffer> g_dec_x;

/* ---- Shared compute pipelines ---- */

extern id<MTLComputePipelineState> g_rms_norm_pipeline;
extern id<MTLComputePipelineState> g_silu_pipeline;
extern id<MTLComputePipelineState> g_add_inplace_pipeline;
extern id<MTLComputePipelineState> g_mul_inplace_pipeline;
extern id<MTLComputePipelineState> g_bias_add_pipeline;
extern id<MTLComputePipelineState> g_bias_add_strided_pipeline;
extern id<MTLComputePipelineState> g_ada_scale_mul_pipeline;
extern id<MTLComputePipelineState> g_argmax_pipeline;
extern id<MTLComputePipelineState> g_rope_apply_pipeline;
extern id<MTLComputePipelineState> g_kv_cache_copy_pipeline;
extern id<MTLComputePipelineState> g_kv_cache_copy_f16_pipeline;
extern id<MTLComputePipelineState> g_decoder_attention_pipeline;
extern id<MTLComputePipelineState> g_decoder_attention_f16_pipeline;
extern id<MTLComputePipelineState> g_encoder_attention_qstrided_pipeline;
extern id<MTLComputePipelineState> g_encoder_attention_kv_f16_qstrided_pipeline;
extern id<MTLComputePipelineState> g_batched_rope_apply_strided_pipeline;
extern id<MTLComputePipelineState> g_batched_kv_cache_copy_strided_pipeline;
extern id<MTLComputePipelineState> g_batched_kv_cache_copy_strided_f16_pipeline;
extern id<MTLComputePipelineState> g_silu_mul_merged_pipeline;

/* Q8 native matmul pipelines */
extern id<MTLComputePipelineState> g_matmul_q8_pipeline;
extern id<MTLComputePipelineState> g_matmul_q8_residual_pipeline;
extern id<MTLComputePipelineState> g_matmul_q8_tiled_pipeline;
extern id<MTLComputePipelineState> g_matmul_q8_tiled_residual_pipeline;
/* Q8 fused decoder M=1 pipelines (still used by BF16 path's encode_wo_ffn_steps) */
extern id<MTLComputePipelineState> g_decoder_ffn_gate_q8_pipeline;
extern id<MTLComputePipelineState> g_decoder_w2_residual_q8_pipeline;
extern id<MTLComputePipelineState> g_decoder_wo_residual_q8_pipeline;

/* ---- Shared helpers (defined in voxtral_metal.m) ---- */

id<MTLBuffer> pool_get_buffer(size_t size);
void pool_release_buffer(id<MTLBuffer> buf);
id<MTLBuffer> get_cached_weight_buffer(const float *weights, size_t size);
id<MTLBuffer> find_shared_buffer(void *ptr);
size_t mmap_offset(const void *ptr);

/* ---- Q8 dispatch functions (defined in voxtral_metal_q8.m) ---- */

int vox_metal_encoder_full_step_q8(void *ctx, float *x, int new_len,
                                    const float *rope_freqs, int cache_len);
void vox_metal_decoder_prefill_step_q8(void *ctx, float *x, int seq_len,
                                        const float *rope_freqs);
int vox_metal_decoder_full_step_q8(void *ctx, const float *rope_freqs,
                                    float *logits);

#endif /* VOXTRAL_METAL_INTERNAL_H */
