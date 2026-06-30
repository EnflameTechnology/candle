#include "cuda_utils.cuh"
#include<stdint.h>

#define AFFINE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const TYPENAME mul, \
    const TYPENAME add \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = inp ? inp[i] : out[i]; \
            out[i] = x * mul + add; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = x * mul + add; \
        } \
    } \
} \

// BF16 affine with F32 promotion for precision
#if __CUDA_ARCH__ >= 800
extern "C" __global__ void affine_bf16(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const __nv_bfloat16 *inp,
    __nv_bfloat16 *out,
    const __nv_bfloat16 mul,
    const __nv_bfloat16 add
) {
    const float mul_f = __bfloat162float(mul);
    const float add_f = __bfloat162float(add);
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            float x = __bfloat162float(inp ? inp[i] : out[i]);
            out[i] = __float2bfloat16(fmaf(x, mul_f, add_f));
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __bfloat162float(inp ? inp[strided_i] : out[i]);
            out[i] = __float2bfloat16(fmaf(x, mul_f, add_f));
        }
    }
}
#endif

// F16 affine with F32 promotion for precision
#if __CUDA_ARCH__ >= 530
extern "C" __global__ void affine_f16(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const __half *inp,
    __half *out,
    const __half mul,
    const __half add
) {
    const float mul_f = __half2float(mul);
    const float add_f = __half2float(add);
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            float x = __half2float(inp ? inp[i] : out[i]);
            out[i] = __float2half(fmaf(x, mul_f, add_f));
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __half2float(inp ? inp[strided_i] : out[i]);
            out[i] = __float2half(fmaf(x, mul_f, add_f));
        }
    }
}
#endif

AFFINE_OP(float, affine_f32)
AFFINE_OP(double, affine_f64)
AFFINE_OP(uint8_t, affine_u8)
AFFINE_OP(uint32_t, affine_u32)
AFFINE_OP(int64_t, affine_i64)
