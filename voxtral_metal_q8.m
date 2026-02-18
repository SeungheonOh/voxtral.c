/*
 * voxtral_metal_q8.m - Native Q8 Metal dispatch
 *
 * Implements encoder, decoder prefill, and decoder M=1 steps using
 * matmul_q8 compute kernels that read int8 weights directly from the
 * mmap'd safetensors file (zero-copy).  Eliminates all Q8->F16 conversion
 * and GPU buffer copies, targeting ~5 GB RSS for the Q8 model.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "voxtral_metal_internal.h"
#include "voxtral.h"
#include <stdio.h>
#include <math.h>

extern int vox_verbose;

/* ========================================================================
 * Q8 matmul dispatch helper
 *
 * C[M,N] = A[M,K] @ W_q8[N,K]^T * scales[N]  (or += for residual)
 * A is in bufA with a_stride floats per row.
 * C is in bufC with c_stride floats per row, column offset c_col_off.
 * W_q8 and scales are read from g_mmap_buffer via byte offsets passed
 * as kernel constants (avoids Metal's buffer offset alignment requirement).
 * ======================================================================== */

static void dispatch_matmul_q8(id<MTLComputeCommandEncoder> enc,
                                id<MTLBuffer> bufA, int a_off_bytes, int a_stride,
                                const int8_t *w_q8, const float *scales,
                                id<MTLBuffer> bufC, int c_off_bytes, int c_stride, int c_col_off,
                                int M, int N, int K, bool residual) {
    bool use_tiled = (M > 1) && g_matmul_q8_tiled_pipeline;
    if (use_tiled) {
        [enc setComputePipelineState:residual ? g_matmul_q8_tiled_residual_pipeline
                                              : g_matmul_q8_tiled_pipeline];
    } else {
        [enc setComputePipelineState:residual ? g_matmul_q8_residual_pipeline
                                              : g_matmul_q8_pipeline];
    }
    [enc setBuffer:bufA offset:(NSUInteger)a_off_bytes atIndex:0];
    [enc setBuffer:g_mmap_buffer offset:0 atIndex:1];
    [enc setBuffer:bufC offset:(NSUInteger)c_off_bytes atIndex:3];
    [enc setBytes:&M length:4 atIndex:4];
    [enc setBytes:&N length:4 atIndex:5];
    [enc setBytes:&K length:4 atIndex:6];
    [enc setBytes:&a_stride length:4 atIndex:7];
    [enc setBytes:&c_stride length:4 atIndex:8];
    [enc setBytes:&c_col_off length:4 atIndex:9];
    uint64_t w_off = (uint64_t)mmap_offset(w_q8);
    uint64_t s_off = (uint64_t)mmap_offset(scales);
    [enc setBytes:&w_off length:sizeof(uint64_t) atIndex:10];
    [enc setBytes:&s_off length:sizeof(uint64_t) atIndex:11];
    if (use_tiled) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 31) / 32),
                                              (NSUInteger)((M + 15) / 16), 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(M * N), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

/* Helper: dispatch decoder_ffn_gate_q8 kernel for M=1 */
static void dispatch_ffn_gate_q8(id<MTLComputeCommandEncoder> enc,
                                  id<MTLBuffer> bufXnorm,
                                  const int8_t *w1_q8, const float *w1_scales,
                                  const int8_t *w3_q8, const float *w3_scales,
                                  id<MTLBuffer> bufGate,
                                  int dim, int hidden) {
    [enc setComputePipelineState:g_decoder_ffn_gate_q8_pipeline];
    [enc setBuffer:bufXnorm offset:0 atIndex:0];
    [enc setBuffer:g_mmap_buffer offset:0 atIndex:1];
    [enc setBuffer:bufGate offset:0 atIndex:5];
    [enc setBytes:&dim length:sizeof(int) atIndex:6];
    [enc setBytes:&hidden length:sizeof(int) atIndex:7];
    uint64_t w1_off = (uint64_t)mmap_offset(w1_q8);
    uint64_t w1s_off = (uint64_t)mmap_offset(w1_scales);
    uint64_t w3_off = (uint64_t)mmap_offset(w3_q8);
    uint64_t w3s_off = (uint64_t)mmap_offset(w3_scales);
    [enc setBytes:&w1_off length:sizeof(uint64_t) atIndex:8];
    [enc setBytes:&w1s_off length:sizeof(uint64_t) atIndex:9];
    [enc setBytes:&w3_off length:sizeof(uint64_t) atIndex:10];
    [enc setBytes:&w3s_off length:sizeof(uint64_t) atIndex:11];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)hidden, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

/* ========================================================================
 * Monolithic Encoder Step (Q8 native): 32 layers + final norm
 * ======================================================================== */

int vox_metal_encoder_full_step_q8(void *ctx_ptr, float *x, int new_len,
                                    const float *rope_freqs, int cache_len) {
    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_encoder_t *enc = &ctx->encoder;

    int dim = VOX_ENC_DIM;          /* 1280 */
    int n_heads = VOX_ENC_HEADS;    /* 32 */
    int n_kv_heads = VOX_ENC_KV_HEADS; /* 32 */
    int head_dim = VOX_ENC_HEAD_DIM;/* 64 */
    int hidden = VOX_ENC_HIDDEN;    /* 5120 */
    int qkv_dim = n_heads * head_dim; /* 2048 */
    int kv_dim = n_kv_heads * head_dim; /* 2048 */
    int M = new_len;
    int total_kv = cache_len + new_len;
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    int window = VOX_ENC_WINDOW;

    id<MTLBuffer> gpu_kv_k = find_shared_buffer(ctx->enc_kv_cache_k);
    id<MTLBuffer> gpu_kv_v = find_shared_buffer(ctx->enc_kv_cache_v);
    if (!gpu_kv_k || !gpu_kv_v) return -1;

    @autoreleasepool {
        int qkv_merged = qkv_dim + kv_dim + kv_dim; /* 6144 */
        int ffn_merged = hidden * 2;                  /* 10240 */
        id<MTLBuffer> bufX = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufXnorm = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufQKV = pool_get_buffer((size_t)M * qkv_merged * sizeof(float));
        id<MTLBuffer> bufAttn = pool_get_buffer((size_t)M * qkv_dim * sizeof(float));
        id<MTLBuffer> bufProj = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufGate = pool_get_buffer((size_t)M * ffn_merged * sizeof(float));
        id<MTLBuffer> bufFfnOut = pool_get_buffer((size_t)M * dim * sizeof(float));

        if (bufX) memcpy([bufX contents], x, (size_t)M * dim * sizeof(float));
        size_t rope_size = (size_t)M * (head_dim / 2) * 2 * sizeof(float);
        id<MTLBuffer> bufRope = pool_get_buffer(rope_size);
        if (bufRope) memcpy([bufRope contents], rope_freqs, rope_size);

        if (!bufX || !bufXnorm || !bufQKV ||
            !bufAttn || !bufProj || !bufGate || !bufFfnOut || !bufRope) {
            pool_release_buffer(bufX);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQKV);
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufRope);
            return -1;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
            vox_enc_layer_t *l = &enc->layers[layer];

            /* Step 1: rms_norm(x, attention_norm) -> x_norm */
            {
                id<MTLBuffer> bufNorm = get_cached_weight_buffer(l->attention_norm,
                                                                   dim * sizeof(float));
                float eps = VOX_ENC_NORM_EPS;
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc_cmd endEncoding];
            }

            /* Step 2: Q, K, V projections (strided into packed bufQKV) */
            {
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                dispatch_matmul_q8(enc_cmd,
                                   bufXnorm, 0, dim,
                                   l->wq_weight_q8, l->wq_scale_q8,
                                   bufQKV, 0, qkv_merged, 0,
                                   M, qkv_dim, dim, false);
                dispatch_matmul_q8(enc_cmd,
                                   bufXnorm, 0, dim,
                                   l->wk_weight_q8, l->wk_scale_q8,
                                   bufQKV, 0, qkv_merged, qkv_dim,
                                   M, kv_dim, dim, false);
                dispatch_matmul_q8(enc_cmd,
                                   bufXnorm, 0, dim,
                                   l->wv_weight_q8, l->wv_scale_q8,
                                   bufQKV, 0, qkv_merged, qkv_dim + kv_dim,
                                   M, kv_dim, dim, false);
                [enc_cmd endEncoding];
            }

            /* Step 3: Bias add + RoPE + KV cache write + attention */
            {
                id<MTLBuffer> bufQBias = get_cached_weight_buffer(l->wq_bias,
                                              qkv_dim * sizeof(float));
                id<MTLBuffer> bufVBias = get_cached_weight_buffer(l->wv_bias,
                                              kv_dim * sizeof(float));

                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                int q_stride = qkv_merged;
                int q_col_off = 0;
                int k_col_off = qkv_dim;
                int v_col_off = qkv_dim + kv_dim;

                /* Q slice += wq_bias */
                int total_q = M * qkv_dim;
                [enc_cmd setComputePipelineState:g_bias_add_strided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:bufQBias offset:0 atIndex:1];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&qkv_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&total_q length:sizeof(int) atIndex:5];
                {
                    NSUInteger tg = MIN((NSUInteger)total_q,
                                        g_bias_add_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)total_q, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* V slice += wv_bias */
                int total_v = M * kv_dim;
                [enc_cmd setBuffer:bufVBias offset:0 atIndex:1];
                [enc_cmd setBytes:&kv_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&v_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&total_v length:sizeof(int) atIndex:5];
                {
                    NSUInteger tg = MIN((NSUInteger)total_v,
                                        g_bias_add_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)total_v, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Batched RoPE on packed Q slice */
                [enc_cmd setComputePipelineState:g_batched_rope_apply_strided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:bufRope offset:0 atIndex:1];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Batched RoPE on packed K slice */
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_kv_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Copy K to KV cache */
                int kv_k_offset = (int)((size_t)layer * ctx->enc_kv_cache_max + cache_len) * kv_dim;
                int kv_total = M * kv_dim;
                [enc_cmd setComputePipelineState:g_batched_kv_cache_copy_strided_pipeline];
                [enc_cmd setBuffer:gpu_kv_k offset:0 atIndex:0];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:1];
                [enc_cmd setBytes:&kv_k_offset length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&kv_dim length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&kv_total length:sizeof(int) atIndex:6];
                {
                    NSUInteger tg = MIN((NSUInteger)kv_total,
                                        g_batched_kv_cache_copy_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Copy V to KV cache */
                [enc_cmd setBuffer:gpu_kv_v offset:0 atIndex:0];
                [enc_cmd setBytes:&v_col_off length:sizeof(int) atIndex:4];
                [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(
                        MIN((NSUInteger)kv_total,
                            g_batched_kv_cache_copy_strided_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Encoder attention */
                int q_offset_val = cache_len;
                size_t layer_kv_offset = (size_t)layer * ctx->enc_kv_cache_max * kv_dim * sizeof(float);
                [enc_cmd setComputePipelineState:g_encoder_attention_qstrided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:gpu_kv_k offset:layer_kv_offset atIndex:1];
                [enc_cmd setBuffer:gpu_kv_v offset:layer_kv_offset atIndex:2];
                [enc_cmd setBuffer:bufAttn offset:0 atIndex:3];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:6];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:7];
                [enc_cmd setBytes:&total_kv length:sizeof(int) atIndex:8];
                [enc_cmd setBytes:&attn_scale length:sizeof(float) atIndex:9];
                [enc_cmd setBytes:&window length:sizeof(int) atIndex:10];
                [enc_cmd setBytes:&q_offset_val length:sizeof(int) atIndex:11];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:12];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:13];
                {
                    int bq = 8;
                    int n_q_blocks = (M + bq - 1) / bq;
                    int n_groups = n_heads * n_q_blocks;
                    [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)n_groups, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                }

                [enc_cmd endEncoding];
            }

            /* Step 4: wo projection (attn_out @ wo^T -> proj) */
            {
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                dispatch_matmul_q8(enc_cmd,
                                   bufAttn, 0, qkv_dim,
                                   l->wo_weight_q8, l->wo_scale_q8,
                                   bufProj, 0, dim, 0,
                                   M, dim, qkv_dim, false);
                [enc_cmd endEncoding];
            }

            /* Step 5: wo bias + residual + FFN norm */
            {
                id<MTLBuffer> bufWoBias = get_cached_weight_buffer(l->wo_bias,
                                                dim * sizeof(float));
                id<MTLBuffer> bufFfnNorm = get_cached_weight_buffer(l->ffn_norm,
                                                dim * sizeof(float));
                int n = M * dim;
                float eps = VOX_ENC_NORM_EPS;

                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];

                /* proj_out += wo_bias */
                [enc_cmd setComputePipelineState:g_bias_add_pipeline];
                [enc_cmd setBuffer:bufProj offset:0 atIndex:0];
                [enc_cmd setBuffer:bufWoBias offset:0 atIndex:1];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&n length:sizeof(int) atIndex:3];
                {
                    NSUInteger tg = MIN((NSUInteger)n,
                                        g_bias_add_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* x += proj_out */
                [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufProj offset:0 atIndex:1];
                [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                {
                    NSUInteger tg = MIN((NSUInteger)n,
                                        g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* x_norm = rms_norm(x, ffn_norm) */
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufFfnNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc_cmd endEncoding];
            }

            /* Step 6: FFN (w1, w3, silu*mul, w2) */
            {
                {
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    dispatch_matmul_q8(enc_cmd,
                                       bufXnorm, 0, dim,
                                       l->w1_weight_q8, l->w1_scale_q8,
                                       bufGate, 0, ffn_merged, 0,
                                       M, hidden, dim, false);
                    dispatch_matmul_q8(enc_cmd,
                                       bufXnorm, 0, dim,
                                       l->w3_weight_q8, l->w3_scale_q8,
                                       bufGate, 0, ffn_merged, hidden,
                                       M, hidden, dim, false);
                    [enc_cmd endEncoding];
                }

                {
                    int n_gate = M * hidden;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    [enc_cmd setComputePipelineState:g_silu_mul_merged_pipeline];
                    [enc_cmd setBuffer:bufGate offset:0 atIndex:0];
                    [enc_cmd setBytes:&hidden length:sizeof(int) atIndex:1];
                    [enc_cmd setBytes:&n_gate length:sizeof(int) atIndex:2];
                    NSUInteger tg = MIN((NSUInteger)n_gate,
                                        g_silu_mul_merged_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_gate, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    [enc_cmd endEncoding];
                }

                {
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    dispatch_matmul_q8(enc_cmd,
                                       bufGate, 0, ffn_merged,
                                       l->w2_weight_q8, l->w2_scale_q8,
                                       bufFfnOut, 0, dim, 0,
                                       M, dim, hidden, false);
                    [enc_cmd endEncoding];
                }

                {
                    id<MTLBuffer> bufW2Bias = get_cached_weight_buffer(l->w2_bias,
                                                dim * sizeof(float));
                    int n = M * dim;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];

                    [enc_cmd setComputePipelineState:g_bias_add_pipeline];
                    [enc_cmd setBuffer:bufFfnOut offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufW2Bias offset:0 atIndex:1];
                    [enc_cmd setBytes:&dim length:sizeof(int) atIndex:2];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:3];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_bias_add_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }

                    [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                    [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufFfnOut offset:0 atIndex:1];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }

                    [enc_cmd endEncoding];
                }
            }
        } /* end 32 layers */

        /* Final norm */
        {
            id<MTLBuffer> bufNorm = get_cached_weight_buffer(enc->norm,
                                                               dim * sizeof(float));
            float eps = VOX_ENC_NORM_EPS;
            id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
            [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
            [enc_cmd setBuffer:bufX offset:0 atIndex:0];
            [enc_cmd setBuffer:bufNorm offset:0 atIndex:1];
            [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
            [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
            [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
            [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc_cmd endEncoding];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(x, [bufXnorm contents], (size_t)M * dim * sizeof(float));

        pool_release_buffer(bufX);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQKV);
        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufRope);
    }

    return 0;
}

/* ========================================================================
 * Monolithic Decoder Full Step (Q8 native): 26 layers + logits, M=1
 * ======================================================================== */

int vox_metal_decoder_full_step_q8(void *ctx_ptr, const float *rope_freqs,
                                    float *logits_out) {
    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_decoder_t *dec = &ctx->decoder;

    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int pos = ctx->kv_cache_len;
    int total_seq = pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_fp16 = ctx->kv_cache_fp16;
    size_t kv_elem_size = kv_fp16 ? sizeof(uint16_t) : sizeof(float);

    void *kv_k_ptr = kv_fp16 ? (void *)ctx->kv_cache_k_f16 : (void *)ctx->kv_cache_k;
    void *kv_v_ptr = kv_fp16 ? (void *)ctx->kv_cache_v_f16 : (void *)ctx->kv_cache_v;
    id<MTLBuffer> gpu_kv_k = find_shared_buffer(kv_k_ptr);
    id<MTLBuffer> gpu_kv_v = find_shared_buffer(kv_v_ptr);
    if (!gpu_kv_k || !gpu_kv_v) return -1;

    int result = 0;

    @autoreleasepool {
        id<MTLBuffer> bufXnorm = pool_get_buffer(dim * sizeof(float));
        id<MTLBuffer> bufQKV = pool_get_buffer((q_dim + kv_dim + kv_dim) * sizeof(float));
        id<MTLBuffer> bufAttn = pool_get_buffer(q_dim * sizeof(float));
        id<MTLBuffer> bufProj = pool_get_buffer(dim * sizeof(float));
        id<MTLBuffer> bufGate = pool_get_buffer(hidden * 2 * sizeof(float));
        id<MTLBuffer> bufFfnOut = pool_get_buffer(dim * sizeof(float));
        id<MTLBuffer> bufLogits = pool_get_buffer((size_t)VOX_VOCAB_SIZE * sizeof(float));
        id<MTLBuffer> bufArgmax = pool_get_buffer(sizeof(int));
        id<MTLBuffer> bufRope = pool_get_buffer(head_dim * sizeof(float));
        if (bufRope) memcpy([bufRope contents], rope_freqs, head_dim * sizeof(float));

        if (!bufXnorm || !bufQKV || !bufAttn ||
            !bufProj || !bufGate || !bufFfnOut ||
            !bufLogits || !bufArgmax || !bufRope) {
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQKV);
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufLogits);
            pool_release_buffer(bufArgmax);
            pool_release_buffer(bufRope);
            return -1;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
            vox_dec_layer_t *l = &dec->layers[layer];

            /* If not first layer, encode wo+FFN for previous layer */
            if (layer > 0) {
                vox_dec_layer_t *prev = &dec->layers[layer - 1];
                const float *ada_s = ctx->ada_scale ?
                    ctx->ada_scale + (size_t)(layer - 1) * dim : NULL;

                /* Step 1: x += attn_out @ wo^T (fused residual) */
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                    dispatch_matmul_q8(enc,
                                       bufAttn, 0, q_dim,
                                       prev->wo_weight_q8, prev->wo_scale_q8,
                                       g_dec_x, 0, dim, 0,
                                       1, dim, q_dim, true);
                    [enc endEncoding];
                }

                /* Steps 2+3+4: rms_norm + ada_scale */
                {
                    id<MTLBuffer> bufNorm = get_cached_weight_buffer(prev->ffn_norm, dim * sizeof(float));
                    id<MTLBuffer> bufAda = ada_s ?
                        get_cached_weight_buffer(ada_s, dim * sizeof(float)) : nil;
                    float eps = VOX_DEC_NORM_EPS;
                    id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

                    [enc setComputePipelineState:g_rms_norm_pipeline];
                    [enc setBuffer:g_dec_x offset:0 atIndex:0];
                    [enc setBuffer:bufNorm offset:0 atIndex:1];
                    [enc setBuffer:bufXnorm offset:0 atIndex:2];
                    [enc setBytes:&dim length:sizeof(int) atIndex:3];
                    [enc setBytes:&eps length:sizeof(float) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    if (bufAda) {
                        int n = dim;
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        [enc setComputePipelineState:g_ada_scale_mul_pipeline];
                        [enc setBuffer:bufXnorm offset:0 atIndex:0];
                        [enc setBuffer:bufAda offset:0 atIndex:1];
                        [enc setBytes:&n length:sizeof(int) atIndex:2];
                        [enc setBytes:&dim length:sizeof(int) atIndex:3];
                        NSUInteger tgSize = MIN((NSUInteger)n,
                            g_ada_scale_mul_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
                    }

                    [enc endEncoding];
                }

                /* Steps 5+6+7+8: FFN gate (w1+w3 fused kernel) */
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                    dispatch_ffn_gate_q8(enc, bufXnorm,
                                         prev->w1_weight_q8, prev->w1_scale_q8,
                                         prev->w3_weight_q8, prev->w3_scale_q8,
                                         bufGate, dim, hidden);
                    [enc endEncoding];
                }

                /* Steps 9+10: x += gate @ w2^T */
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                    dispatch_matmul_q8(enc,
                                       bufGate, 0, hidden,
                                       prev->w2_weight_q8, prev->w2_scale_q8,
                                       g_dec_x, 0, dim, 0,
                                       1, dim, hidden, true);
                    [enc endEncoding];
                }
            }

            /* RMSNorm + QKV (separate matmuls via matmul_q8) */
            {
                id<MTLBuffer> bufNorm = get_cached_weight_buffer(l->attention_norm, dim * sizeof(float));
                float eps = VOX_DEC_NORM_EPS;
                int Nqkv = q_dim + kv_dim + kv_dim;

                /* rms_norm */
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                    [enc setComputePipelineState:g_rms_norm_pipeline];
                    [enc setBuffer:g_dec_x offset:0 atIndex:0];
                    [enc setBuffer:bufNorm offset:0 atIndex:1];
                    [enc setBuffer:bufXnorm offset:0 atIndex:2];
                    [enc setBytes:&dim length:sizeof(int) atIndex:3];
                    [enc setBytes:&eps length:sizeof(float) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                /* Q, K, V projections into packed bufQKV */
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                    dispatch_matmul_q8(enc,
                                       bufXnorm, 0, dim,
                                       l->wq_weight_q8, l->wq_scale_q8,
                                       bufQKV, 0, Nqkv, 0,
                                       1, q_dim, dim, false);
                    dispatch_matmul_q8(enc,
                                       bufXnorm, 0, dim,
                                       l->wk_weight_q8, l->wk_scale_q8,
                                       bufQKV, 0, Nqkv, q_dim,
                                       1, kv_dim, dim, false);
                    dispatch_matmul_q8(enc,
                                       bufXnorm, 0, dim,
                                       l->wv_weight_q8, l->wv_scale_q8,
                                       bufQKV, 0, Nqkv, q_dim + kv_dim,
                                       1, kv_dim, dim, false);
                    [enc endEncoding];
                }
            }

            /* RoPE + KV cache write + attention */
            {
                int kv_offset = (int)((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
                size_t layer_kv_offset = (size_t)layer * ctx->kv_cache_max * kv_dim * kv_elem_size;
                int window_dec = VOX_DEC_WINDOW;
                int q_pos_val = ctx->kv_pos_offset + pos;
                size_t off_k = (size_t)q_dim * sizeof(float);
                size_t off_v = (size_t)(q_dim + kv_dim) * sizeof(float);

                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

                /* RoPE on Q */
                int n_threads_q = n_heads * (head_dim / 2);
                [enc setComputePipelineState:g_rope_apply_pipeline];
                [enc setBuffer:bufQKV offset:0 atIndex:0];
                [enc setBuffer:bufRope offset:0 atIndex:1];
                [enc setBytes:&n_heads length:sizeof(int) atIndex:2];
                [enc setBytes:&head_dim length:sizeof(int) atIndex:3];
                {
                    NSUInteger tg = MIN((NSUInteger)n_threads_q,
                                        g_rope_apply_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_threads_q, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* RoPE on K */
                int n_threads_k = n_kv_heads * (head_dim / 2);
                [enc setBuffer:bufQKV offset:off_k atIndex:0];
                [enc setBytes:&n_kv_heads length:sizeof(int) atIndex:2];
                {
                    NSUInteger tg = MIN((NSUInteger)n_threads_k,
                                        g_rope_apply_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_threads_k, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Write K to KV cache */
                id<MTLComputePipelineState> kv_copy_ps = kv_fp16 ?
                    g_kv_cache_copy_f16_pipeline : g_kv_cache_copy_pipeline;
                [enc setComputePipelineState:kv_copy_ps];
                [enc setBuffer:gpu_kv_k offset:0 atIndex:0];
                [enc setBuffer:bufQKV offset:off_k atIndex:1];
                [enc setBytes:&kv_offset length:sizeof(int) atIndex:2];
                [enc setBytes:&kv_dim length:sizeof(int) atIndex:3];
                {
                    NSUInteger tg = MIN((NSUInteger)kv_dim,
                                        kv_copy_ps.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Write V to KV cache */
                [enc setBuffer:gpu_kv_v offset:0 atIndex:0];
                [enc setBuffer:bufQKV offset:off_v atIndex:1];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(
                    MIN((NSUInteger)kv_dim,
                        kv_copy_ps.maxTotalThreadsPerThreadgroup), 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Attention */
                [enc setComputePipelineState:kv_fp16 ?
                    g_decoder_attention_f16_pipeline : g_decoder_attention_pipeline];
                [enc setBuffer:bufQKV offset:0 atIndex:0];
                [enc setBuffer:gpu_kv_k offset:layer_kv_offset atIndex:1];
                [enc setBuffer:gpu_kv_v offset:layer_kv_offset atIndex:2];
                [enc setBuffer:bufAttn offset:0 atIndex:3];
                [enc setBytes:&n_heads length:sizeof(int) atIndex:4];
                [enc setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
                [enc setBytes:&head_dim length:sizeof(int) atIndex:6];
                [enc setBytes:&kv_dim length:sizeof(int) atIndex:7];
                [enc setBytes:&total_seq length:sizeof(int) atIndex:8];
                [enc setBytes:&scale length:sizeof(float) atIndex:9];
                [enc setBytes:&window_dec length:sizeof(int) atIndex:10];
                [enc setBytes:&q_pos_val length:sizeof(int) atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc endEncoding];
            }
        }

        /* Final: wo+FFN for last layer + logits + argmax */
        {
            vox_dec_layer_t *last = &dec->layers[VOX_DEC_LAYERS - 1];
            const float *ada_s = ctx->ada_scale ?
                ctx->ada_scale + (size_t)(VOX_DEC_LAYERS - 1) * dim : NULL;

            /* x += attn_out @ wo^T */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                dispatch_matmul_q8(enc,
                                   bufAttn, 0, q_dim,
                                   last->wo_weight_q8, last->wo_scale_q8,
                                   g_dec_x, 0, dim, 0,
                                   1, dim, q_dim, true);
                [enc endEncoding];
            }

            /* rms_norm + ada_scale */
            {
                id<MTLBuffer> bufNorm = get_cached_weight_buffer(last->ffn_norm, dim * sizeof(float));
                id<MTLBuffer> bufAda = ada_s ?
                    get_cached_weight_buffer(ada_s, dim * sizeof(float)) : nil;
                float eps = VOX_DEC_NORM_EPS;
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

                [enc setComputePipelineState:g_rms_norm_pipeline];
                [enc setBuffer:g_dec_x offset:0 atIndex:0];
                [enc setBuffer:bufNorm offset:0 atIndex:1];
                [enc setBuffer:bufXnorm offset:0 atIndex:2];
                [enc setBytes:&dim length:sizeof(int) atIndex:3];
                [enc setBytes:&eps length:sizeof(float) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                if (bufAda) {
                    int n = dim;
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:g_ada_scale_mul_pipeline];
                    [enc setBuffer:bufXnorm offset:0 atIndex:0];
                    [enc setBuffer:bufAda offset:0 atIndex:1];
                    [enc setBytes:&n length:sizeof(int) atIndex:2];
                    [enc setBytes:&dim length:sizeof(int) atIndex:3];
                    NSUInteger tgSize = MIN((NSUInteger)n,
                        g_ada_scale_mul_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
                }

                [enc endEncoding];
            }

            /* FFN gate (w1+w3 fused kernel) */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                dispatch_ffn_gate_q8(enc, bufXnorm,
                                     last->w1_weight_q8, last->w1_scale_q8,
                                     last->w3_weight_q8, last->w3_scale_q8,
                                     bufGate, dim, hidden);
                [enc endEncoding];
            }

            /* x += gate @ w2^T */
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                dispatch_matmul_q8(enc,
                                   bufGate, 0, hidden,
                                   last->w2_weight_q8, last->w2_scale_q8,
                                   g_dec_x, 0, dim, 0,
                                   1, dim, hidden, true);
                [enc endEncoding];
            }

            /* Final RMSNorm */
            {
                id<MTLBuffer> bufFinalNorm = get_cached_weight_buffer(dec->norm,
                                                                        dim * sizeof(float));
                float eps = VOX_DEC_NORM_EPS;
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                [enc setComputePipelineState:g_rms_norm_pipeline];
                [enc setBuffer:g_dec_x offset:0 atIndex:0];
                [enc setBuffer:bufFinalNorm offset:0 atIndex:1];
                [enc setBuffer:bufXnorm offset:0 atIndex:2];
                [enc setBytes:&dim length:sizeof(int) atIndex:3];
                [enc setBytes:&eps length:sizeof(float) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            /* Logits = x_norm @ tok_emb^T (Q8) */
            {
                int vocab = VOX_VOCAB_SIZE;
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                dispatch_matmul_q8(enc,
                                   bufXnorm, 0, dim,
                                   dec->tok_embeddings_q8, dec->tok_embeddings_scale_q8,
                                   bufLogits, 0, vocab, 0,
                                   1, vocab, dim, false);
                [enc endEncoding];
            }

            /* Argmax on GPU */
            {
                int vocab = VOX_VOCAB_SIZE;
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                [enc setComputePipelineState:g_argmax_pipeline];
                [enc setBuffer:bufLogits offset:0 atIndex:0];
                [enc setBuffer:bufArgmax offset:0 atIndex:1];
                [enc setBytes:&vocab length:sizeof(int) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "[decoder-step-q8] CMD ERROR: %s\n",
                    [[cmdBuffer.error localizedDescription] UTF8String]);
        }

        result = ((int *)[bufArgmax contents])[0];

        if (logits_out)
            memcpy(logits_out, [bufLogits contents], (size_t)VOX_VOCAB_SIZE * sizeof(float));

        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQKV);
        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufLogits);
        pool_release_buffer(bufArgmax);
        pool_release_buffer(bufRope);
    }

    ctx->kv_cache_len = pos + 1;
    return result;
}

/* ========================================================================
 * Monolithic Decoder Prefill (Q8 native): 26 layers, M>1
 * ======================================================================== */

void vox_metal_decoder_prefill_step_q8(void *ctx_ptr, float *x, int seq_len,
                                        const float *rope_freqs) {
    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_decoder_t *dec = &ctx->decoder;

    int dim = VOX_DEC_DIM;          /* 3072 */
    int n_heads = VOX_DEC_HEADS;    /* 32 */
    int n_kv_heads = VOX_DEC_KV_HEADS; /* 8 */
    int head_dim = VOX_DEC_HEAD_DIM;/* 128 */
    int hidden = VOX_DEC_HIDDEN;    /* 9216 */
    int q_dim = n_heads * head_dim; /* 4096 */
    int kv_dim = n_kv_heads * head_dim; /* 1024 */
    int M = seq_len;
    int start_pos = ctx->kv_cache_len;
    int total_kv = start_pos + seq_len;
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    int window = VOX_DEC_WINDOW;
    int kv_fp16 = ctx->kv_cache_fp16;
    size_t kv_elem_size = kv_fp16 ? sizeof(uint16_t) : sizeof(float);

    void *kv_k_ptr = kv_fp16 ? (void *)ctx->kv_cache_k_f16 : (void *)ctx->kv_cache_k;
    void *kv_v_ptr = kv_fp16 ? (void *)ctx->kv_cache_v_f16 : (void *)ctx->kv_cache_v;
    id<MTLBuffer> gpu_kv_k = find_shared_buffer(kv_k_ptr);
    id<MTLBuffer> gpu_kv_v = find_shared_buffer(kv_v_ptr);
    if (!gpu_kv_k || !gpu_kv_v) return;

    @autoreleasepool {
        int qkv_merged = q_dim + kv_dim + kv_dim;  /* 6144 */
        int ffn_merged = hidden * 2;                 /* 18432 */

        id<MTLBuffer> bufX = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufXnorm = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufQKV = pool_get_buffer((size_t)M * qkv_merged * sizeof(float));
        id<MTLBuffer> bufAttn = pool_get_buffer((size_t)M * q_dim * sizeof(float));
        id<MTLBuffer> bufProj = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufGate = pool_get_buffer((size_t)M * ffn_merged * sizeof(float));
        id<MTLBuffer> bufFfnOut = pool_get_buffer((size_t)M * dim * sizeof(float));

        if (bufX) memcpy([bufX contents], x, (size_t)M * dim * sizeof(float));
        size_t rope_size = (size_t)M * (head_dim / 2) * 2 * sizeof(float);
        id<MTLBuffer> bufRope = pool_get_buffer(rope_size);
        if (bufRope) memcpy([bufRope contents], rope_freqs, rope_size);

        if (!bufX || !bufXnorm || !bufQKV ||
            !bufAttn || !bufProj || !bufGate || !bufFfnOut || !bufRope) {
            pool_release_buffer(bufX);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQKV);
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufRope);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
            vox_dec_layer_t *l = &dec->layers[layer];

            /* Step 1: rms_norm(x, attention_norm) -> x_norm */
            {
                id<MTLBuffer> bufNorm = get_cached_weight_buffer(l->attention_norm,
                                                                   dim * sizeof(float));
                float eps = VOX_DEC_NORM_EPS;
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc_cmd endEncoding];
            }

            /* Step 2: Q, K, V projections (strided into packed bufQKV) */
            {
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                dispatch_matmul_q8(enc_cmd,
                                   bufXnorm, 0, dim,
                                   l->wq_weight_q8, l->wq_scale_q8,
                                   bufQKV, 0, qkv_merged, 0,
                                   M, q_dim, dim, false);
                dispatch_matmul_q8(enc_cmd,
                                   bufXnorm, 0, dim,
                                   l->wk_weight_q8, l->wk_scale_q8,
                                   bufQKV, 0, qkv_merged, q_dim,
                                   M, kv_dim, dim, false);
                dispatch_matmul_q8(enc_cmd,
                                   bufXnorm, 0, dim,
                                   l->wv_weight_q8, l->wv_scale_q8,
                                   bufQKV, 0, qkv_merged, q_dim + kv_dim,
                                   M, kv_dim, dim, false);
                [enc_cmd endEncoding];
            }

            /* Step 3: RoPE + KV cache write + attention */
            {
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                int q_stride = qkv_merged;
                int q_col_off = 0;
                int k_col_off = q_dim;
                int v_col_off = q_dim + kv_dim;

                /* Batched RoPE on packed Q slice */
                [enc_cmd setComputePipelineState:g_batched_rope_apply_strided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:bufRope offset:0 atIndex:1];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Batched RoPE on packed K slice */
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_kv_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Copy K to KV cache */
                int kv_offset = (int)((size_t)layer * ctx->kv_cache_max + start_pos) * kv_dim;
                int kv_total = M * kv_dim;
                id<MTLComputePipelineState> kv_copy_ps = kv_fp16 ?
                    g_batched_kv_cache_copy_strided_f16_pipeline :
                    g_batched_kv_cache_copy_strided_pipeline;
                [enc_cmd setComputePipelineState:kv_copy_ps];
                [enc_cmd setBuffer:gpu_kv_k offset:0 atIndex:0];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:1];
                [enc_cmd setBytes:&kv_offset length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&kv_dim length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&kv_total length:sizeof(int) atIndex:6];
                {
                    NSUInteger tg = MIN((NSUInteger)kv_total,
                                        kv_copy_ps.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Copy V to KV cache */
                [enc_cmd setBuffer:gpu_kv_v offset:0 atIndex:0];
                [enc_cmd setBytes:&v_col_off length:sizeof(int) atIndex:4];
                [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(
                        MIN((NSUInteger)kv_total,
                            kv_copy_ps.maxTotalThreadsPerThreadgroup), 1, 1)];

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Batched attention */
                int q_offset_val = ctx->kv_pos_offset + start_pos;
                size_t layer_kv_offset = (size_t)layer * ctx->kv_cache_max * kv_dim * kv_elem_size;
                [enc_cmd setComputePipelineState:kv_fp16 ?
                    g_encoder_attention_kv_f16_qstrided_pipeline :
                    g_encoder_attention_qstrided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:gpu_kv_k offset:layer_kv_offset atIndex:1];
                [enc_cmd setBuffer:gpu_kv_v offset:layer_kv_offset atIndex:2];
                [enc_cmd setBuffer:bufAttn offset:0 atIndex:3];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:6];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:7];
                [enc_cmd setBytes:&total_kv length:sizeof(int) atIndex:8];
                [enc_cmd setBytes:&attn_scale length:sizeof(float) atIndex:9];
                [enc_cmd setBytes:&window length:sizeof(int) atIndex:10];
                [enc_cmd setBytes:&q_offset_val length:sizeof(int) atIndex:11];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:12];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:13];
                {
                    int bq = 8;
                    int n_q_blocks = (M + bq - 1) / bq;
                    int n_groups = n_heads * n_q_blocks;
                    [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)n_groups, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake((NSUInteger)head_dim, 1, 1)];
                }

                [enc_cmd endEncoding];
            }

            /* Step 4: wo projection (attn_out @ wo^T -> proj) */
            {
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                dispatch_matmul_q8(enc_cmd,
                                   bufAttn, 0, q_dim,
                                   l->wo_weight_q8, l->wo_scale_q8,
                                   bufProj, 0, dim, 0,
                                   M, dim, q_dim, false);
                [enc_cmd endEncoding];
            }

            /* Step 5: residual + FFN norm + ada_scale */
            {
                id<MTLBuffer> bufFfnNorm = get_cached_weight_buffer(l->ffn_norm,
                                                dim * sizeof(float));
                id<MTLBuffer> bufAda = ctx->ada_scale ?
                    get_cached_weight_buffer(ctx->ada_scale + (size_t)layer * dim,
                                               dim * sizeof(float)) : nil;
                int n = M * dim;
                float eps = VOX_DEC_NORM_EPS;

                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];

                /* x += proj_out */
                [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufProj offset:0 atIndex:1];
                [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                {
                    NSUInteger tg = MIN((NSUInteger)n,
                                        g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* x_norm = rms_norm(x, ffn_norm) */
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufFfnNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                /* x_norm *= (1 + ada_scale) if present */
                if (bufAda) {
                    [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc_cmd setComputePipelineState:g_ada_scale_mul_pipeline];
                    [enc_cmd setBuffer:bufXnorm offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufAda offset:0 atIndex:1];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                    [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_ada_scale_mul_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }
                }

                [enc_cmd endEncoding];
            }

            /* Step 6: FFN (w1, w3, silu*mul, w2) */
            {
                /* w1 + w3 -> bufGate[M, ffn_merged] */
                {
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    dispatch_matmul_q8(enc_cmd,
                                       bufXnorm, 0, dim,
                                       l->w1_weight_q8, l->w1_scale_q8,
                                       bufGate, 0, ffn_merged, 0,
                                       M, hidden, dim, false);
                    dispatch_matmul_q8(enc_cmd,
                                       bufXnorm, 0, dim,
                                       l->w3_weight_q8, l->w3_scale_q8,
                                       bufGate, 0, ffn_merged, hidden,
                                       M, hidden, dim, false);
                    [enc_cmd endEncoding];
                }

                /* Fused silu + mul */
                {
                    int n_gate = M * hidden;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    [enc_cmd setComputePipelineState:g_silu_mul_merged_pipeline];
                    [enc_cmd setBuffer:bufGate offset:0 atIndex:0];
                    [enc_cmd setBytes:&hidden length:sizeof(int) atIndex:1];
                    [enc_cmd setBytes:&n_gate length:sizeof(int) atIndex:2];
                    NSUInteger tg = MIN((NSUInteger)n_gate,
                                        g_silu_mul_merged_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_gate, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    [enc_cmd endEncoding];
                }

                /* w2: gate @ w2^T -> bufFfnOut */
                {
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    dispatch_matmul_q8(enc_cmd,
                                       bufGate, 0, ffn_merged,
                                       l->w2_weight_q8, l->w2_scale_q8,
                                       bufFfnOut, 0, dim, 0,
                                       M, dim, hidden, false);
                    [enc_cmd endEncoding];
                }

                /* x += ffn_out */
                {
                    int n = M * dim;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                    [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufFfnOut offset:0 atIndex:1];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }
                    [enc_cmd endEncoding];
                }
            }
        } /* end 26 layers */

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(x, [bufX contents], (size_t)M * dim * sizeof(float));

        pool_release_buffer(bufX);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQKV);
        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufRope);
    }

    ctx->kv_cache_len = start_pos + seq_len;
}
