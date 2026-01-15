#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================

namespace {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
using namespace nvcuda;
#endif

constexpr int BLOCK_ROWS = 64;
constexpr int BLOCK_COLS = 64;
constexpr int THREADS_PER_BLOCK = 512;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr bool kEnableTensorCore = true;

inline int div_up(int x, int y) {
    return (x + y - 1) / y;
}

template <typename T>
__device__ inline T clamp_read(bool cond, const T& value, T zero_val = T(0)) {
    return cond ? value : zero_val;
}

__global__ void add_kernel(const float* src, float* dst, long long num_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < num_elements) {
        dst[idx] += src[idx];
        idx += stride;
    }
}

__global__ void linear_kernel_simple(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size || col >= out_features) { return; }

    const float* input_row = input + static_cast<long long>(row) * in_features;
    const float* weight_row = weights + static_cast<long long>(col) * in_features;

    float acc = biases[col];
    #pragma unroll 4
    for (int k = 0; k < in_features; ++k) {
        acc += input_row[k] * weight_row[k];
    }

    output[static_cast<long long>(row) * out_features + col] = acc;
}

template <bool FuseIF>
__device__ void linear_wmma_kernel_body(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ accumulator_out,
    float* __restrict__ spikes,
    int batch_size,
    int in_features,
    int out_features)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __shared__ half s_a[2][BLOCK_ROWS * WMMA_K];
    __shared__ half s_b[2][WMMA_K * BLOCK_COLS];
    __shared__ float s_output_tile[BLOCK_ROWS * BLOCK_COLS];

    const int block_row = blockIdx.y * BLOCK_ROWS;
    const int block_col = blockIdx.x * BLOCK_COLS;
    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id >> 2;
    const int warp_col = warp_id & 3;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    auto load_tile = [&](int tile_k, int buffer) {
        int a_elems = BLOCK_ROWS * WMMA_K;
        for (int idx = threadIdx.x; idx < a_elems; idx += THREADS_PER_BLOCK) {
            int row = idx / WMMA_K;
            int col = idx % WMMA_K;
            int g_row = block_row + row;
            int g_col = tile_k + col;
            half val = __float2half((g_row < batch_size && g_col < in_features)
                ? input[static_cast<long long>(g_row) * in_features + g_col]
                : 0.0f);
            s_a[buffer][idx] = val;
        }

        int b_elems = WMMA_K * BLOCK_COLS;
        for (int idx = threadIdx.x; idx < b_elems; idx += THREADS_PER_BLOCK) {
            int row = idx / BLOCK_COLS;
            int col = idx % BLOCK_COLS;
            int g_row = tile_k + row;
            int g_col = block_col + col;
            half val = __float2half((g_col < out_features && g_row < in_features)
                ? weights[static_cast<long long>(g_col) * in_features + g_row]
                : 0.0f);
            s_b[buffer][idx] = val;
        }
    };

    int tiles = (in_features + WMMA_K - 1) / WMMA_K;
    load_tile(0, 0);
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        int cur = tile & 1;
        int next = cur ^ 1;

        if (warp_row < BLOCK_ROWS / WMMA_M && warp_col < BLOCK_COLS / WMMA_N) {
            const half* a_ptr = &s_a[cur][warp_row * WMMA_M * WMMA_K];
            const half* b_ptr = &s_b[cur][warp_col * WMMA_N];
            wmma::load_matrix_sync(frag_a, a_ptr, WMMA_K);
            wmma::load_matrix_sync(frag_b, b_ptr, BLOCK_COLS);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        if (tile + 1 < tiles) {
            load_tile((tile + 1) * WMMA_K, next);
        }
        __syncthreads();
    }

    if (warp_row < BLOCK_ROWS / WMMA_M && warp_col < BLOCK_COLS / WMMA_N) {
        float* tile_ptr = &s_output_tile[(warp_row * WMMA_M) * BLOCK_COLS + warp_col * WMMA_N];
        wmma::store_matrix_sync(tile_ptr, frag_c, BLOCK_COLS, wmma::mem_row_major);
    }
    __syncthreads();

    const int total_tile_elems = BLOCK_ROWS * BLOCK_COLS;
    for (int idx = threadIdx.x; idx < total_tile_elems; idx += THREADS_PER_BLOCK) {
        int tile_row = idx / BLOCK_COLS;
        int tile_col = idx % BLOCK_COLS;
        int g_row = block_row + tile_row;
        int g_col = block_col + tile_col;
        if (g_row >= batch_size || g_col >= out_features) { continue; }
        float val = s_output_tile[idx] + biases[g_col];
        long long out_idx = static_cast<long long>(g_row) * out_features + g_col;
        if constexpr (FuseIF) {
            float v = accumulator_out[out_idx] + val;
            float spike = v >= 1.0f ? 1.0f : 0.0f;
            accumulator_out[out_idx] = v * (1.0f - spike);
            spikes[out_idx] = spike;
        } else {
            accumulator_out[out_idx] = val;
        }
    }
#endif
}

__global__ void linear_if_fused_kernel_wmma(
    const float* input, const float* weights, const float* biases,
    float* membrane_potential, float* spikes,
    int batch_size, int in_features, int out_features)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    linear_wmma_kernel_body<true>(
        input, weights, biases,
        membrane_potential, spikes,
        batch_size, in_features, out_features);
#endif
}

__global__ void linear_kernel_wmma(
    const float* input, const float* weights, const float* biases,
    float* output,
    int batch_size, int in_features, int out_features)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    linear_wmma_kernel_body<false>(
        input, weights, biases,
        output, nullptr,
        batch_size, in_features, out_features);
#endif
}

// ------------ Convolution + IF fused kernels -------------

__global__ void fused_conv_if_kernel1(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ membrane_potential,
    float* __restrict__ spikes,
    int N, int C, int H, int W,
    int K, int R, int S,
    int Oh, int Ow)
{
    constexpr int INPUT_C = 1;
    constexpr int INPUT_H = 28;
    constexpr int INPUT_W = 28;
    constexpr int FILTER_K = 12;
    constexpr int FILTER_R = 5;
    constexpr int FILTER_S = 5;

    __shared__ float s_weights[FILTER_K * INPUT_C * FILTER_R * FILTER_S];
    __shared__ float s_biases[FILTER_K];
    __shared__ float s_input[INPUT_H * INPUT_W];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;

    for (int idx = tid; idx < FILTER_K * INPUT_C * FILTER_R * FILTER_S; idx += threads_per_block) {
        s_weights[idx] = weights[idx];
    }
    if (tid < FILTER_K) {
        s_biases[tid] = biases[tid];
    }

    int b = blockIdx.z;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= N) { return; }

    const float* input_batch = input + static_cast<long long>(b) * C * H * W;

    float accum[FILTER_K] = {0.0f};

    for (int in_c = 0; in_c < INPUT_C; ++in_c) {
        const float* channel_ptr = input_batch + static_cast<long long>(in_c) * H * W;
        for (int idx = tid; idx < INPUT_H * INPUT_W; idx += threads_per_block) {
            s_input[idx] = channel_ptr[idx];
        }
        __syncthreads();

        if (out_y < Oh && out_x < Ow) {
            #pragma unroll
            for (int ky = 0; ky < FILTER_R; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < FILTER_S; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    float input_val = s_input[in_y * INPUT_W + in_x];
                    #pragma unroll
                    for (int oc = 0; oc < FILTER_K; ++oc) {
                        int w_idx = (((oc * INPUT_C) + in_c) * FILTER_R + ky) * FILTER_S + kx;
                        accum[oc] += input_val * s_weights[w_idx];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (out_y < Oh && out_x < Ow) {
        for (int oc = 0; oc < FILTER_K; ++oc) {
            float conv_out = accum[oc] + s_biases[oc];
            long long elem_idx = (static_cast<long long>(b) * K + oc) * Oh * Ow +
                                 static_cast<long long>(out_y) * Ow + out_x;
            float v = membrane_potential[elem_idx] + conv_out;
            float spike = v >= 1.0f ? 1.0f : 0.0f;
            membrane_potential[elem_idx] = v * (1.0f - spike);
            spikes[elem_idx] = spike;
        }
    }
}

__global__ void fused_conv_if_kernel2(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ membrane_potential,
    float* __restrict__ spikes,
    int N, int C, int H, int W,
    int K, int R, int S,
    int Oh, int Ow)
{
    constexpr int INPUT_C = 12;
    constexpr int INPUT_H = 12;
    constexpr int INPUT_W = 12;
    constexpr int FILTER_K = 24;
    constexpr int FILTER_R = 5;
    constexpr int FILTER_S = 5;

    __shared__ float s_weights[FILTER_K * INPUT_C * FILTER_R * FILTER_S];
    __shared__ float s_biases[FILTER_K];
    __shared__ float s_input[INPUT_H * INPUT_W];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;

    for (int idx = tid; idx < FILTER_K * INPUT_C * FILTER_R * FILTER_S; idx += threads_per_block) {
        s_weights[idx] = weights[idx];
    }
    if (tid < FILTER_K) {
        s_biases[tid] = biases[tid];
    }

    int b = blockIdx.z;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= N) { return; }

    const float* input_batch = input + static_cast<long long>(b) * C * H * W;
    float accum[FILTER_K];
    #pragma unroll
    for (int oc = 0; oc < FILTER_K; ++oc) { accum[oc] = 0.0f; }

    for (int in_c = 0; in_c < INPUT_C; ++in_c) {
        const float* channel_ptr = input_batch + static_cast<long long>(in_c) * H * W;
        for (int idx = tid; idx < INPUT_H * INPUT_W; idx += threads_per_block) {
            s_input[idx] = channel_ptr[idx];
        }
        __syncthreads();

        if (out_y < Oh && out_x < Ow) {
            #pragma unroll
            for (int ky = 0; ky < FILTER_R; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < FILTER_S; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    float input_val = s_input[in_y * INPUT_W + in_x];
                    #pragma unroll
                    for (int oc = 0; oc < FILTER_K; ++oc) {
                        int w_idx = (((oc * INPUT_C) + in_c) * FILTER_R + ky) * FILTER_S + kx;
                        accum[oc] += input_val * s_weights[w_idx];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (out_y < Oh && out_x < Ow) {
        for (int oc = 0; oc < FILTER_K; ++oc) {
            float conv_out = accum[oc] + s_biases[oc];
            long long elem_idx = (static_cast<long long>(b) * K + oc) * Oh * Ow +
                                 static_cast<long long>(out_y) * Ow + out_x;
            float v = membrane_potential[elem_idx] + conv_out;
            float spike = v >= 1.0f ? 1.0f : 0.0f;
            membrane_potential[elem_idx] = v * (1.0f - spike);
            spikes[elem_idx] = spike;
        }
    }
}

__global__ void maxpool2d_kernel1(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int pool_size)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= batch_size || out_y >= out_h || out_x >= out_w) { return; }

    const long long input_batch_offset = static_cast<long long>(b) * channels * in_h * in_w;
    const long long output_batch_offset = static_cast<long long>(b) * channels * out_h * out_w;

    for (int c = 0; c < channels; ++c) {
        int in_y_start = out_y * pool_size;
        int in_x_start = out_x * pool_size;
        float max_val = -1e9f;
        for (int py = 0; py < pool_size; ++py) {
            for (int px = 0; px < pool_size; ++px) {
                int in_y = in_y_start + py;
                int in_x = in_x_start + px;
                long long input_idx = input_batch_offset +
                    static_cast<long long>(c) * in_h * in_w +
                    static_cast<long long>(in_y) * in_w + in_x;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
        long long output_idx = output_batch_offset +
            static_cast<long long>(c) * out_h * out_w +
            static_cast<long long>(out_y) * out_w + out_x;
        output[output_idx] = max_val;
    }
}

__global__ void maxpool2d_kernel2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int pool_size)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    int c = threadIdx.z;

    if (b >= batch_size || out_y >= out_h || out_x >= out_w || c >= channels) { return; }

    const long long input_batch_offset = static_cast<long long>(b) * channels * in_h * in_w;
    const long long output_batch_offset = static_cast<long long>(b) * channels * out_h * out_w;

    int in_y_start = out_y * pool_size;
    int in_x_start = out_x * pool_size;

    float max_val = -1e9f;
    for (int py = 0; py < pool_size; ++py) {
        for (int px = 0; px < pool_size; ++px) {
            int in_y = in_y_start + py;
            int in_x = in_x_start + px;
            long long input_idx = input_batch_offset +
                static_cast<long long>(c) * in_h * in_w +
                static_cast<long long>(in_y) * in_w + in_x;
            max_val = fmaxf(max_val, input[input_idx]);
        }
    }

    long long output_idx = output_batch_offset +
        static_cast<long long>(c) * out_h * out_w +
        static_cast<long long>(out_y) * out_w + out_x;
    output[output_idx] = max_val;
}

__global__ void if_node_kernel(
    const float* __restrict__ input,
    float* __restrict__ membrane,
    float* __restrict__ spikes,
    long long num_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < num_elements) {
        float v = membrane[idx] + input[idx];
        float spike = v >= 1.0f ? 1.0f : 0.0f;
        membrane[idx] = v * (1.0f - spike);
        spikes[idx] = spike;
        idx += stride;
    }
}

} // namespace


std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b
    // YOU CAN ADD MORE PARAMETERS HERE!!!
    )
{
    static int compute_major = -1;
    if (compute_major == -1) {
        cudaDeviceProp prop{};
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        compute_major = prop.major;
    }

    const bool use_tensor_core = kEnableTensorCore && compute_major >= 7;

    const int total_images = static_cast<int>(images.size());
    const int batch_size = 512;
    const int num_batches = div_up(total_images, batch_size);
    const int T = 2;

    const int block_size = 256;
    const int in_size = 28 * 28;

    const int c1_in_c = 1, c1_in_h = 28, c1_in_w = 28, c1_k = 5;
    const int c1_out_c = 12, c1_out_h = 24, c1_out_w = 24;
    const int c1_out_size = c1_out_c * c1_out_h * c1_out_w;

    const int p1_k = 2, p1_out_c = c1_out_c, p1_out_h = 12, p1_out_w = 12;
    const int p1_out_size = p1_out_c * p1_out_h * p1_out_w;

    const int c2_in_c = p1_out_c, c2_in_h = p1_out_h, c2_in_w = p1_out_w, c2_k = 5;
    const int c2_out_c = 24, c2_out_h = 8, c2_out_w = 8;
    const int c2_out_size = c2_out_c * c2_out_h * c2_out_w;

    const int p2_k = 2, p2_out_c = c2_out_c, p2_out_h = 4, p2_out_w = 4;
    const int p2_out_size = p2_out_c * p2_out_h * p2_out_w;

    const int fc1_in = p2_out_size, fc1_out = 240;
    const int fc2_in = fc1_out, fc2_out = 120;
    const int fc3_in = fc2_out, fc3_out = 10;

    const long long batch_in_size = static_cast<long long>(batch_size) * in_size;
    const long long c1_out_batch_size = static_cast<long long>(batch_size) * c1_out_size;
    const long long p1_out_batch_size = static_cast<long long>(batch_size) * p1_out_size;
    const long long c2_out_batch_size = static_cast<long long>(batch_size) * c2_out_size;
    const long long p2_out_batch_size = static_cast<long long>(batch_size) * p2_out_size;
    const long long fc1_out_batch_size = static_cast<long long>(batch_size) * fc1_out;
    const long long fc2_out_batch_size = static_cast<long long>(batch_size) * fc2_out;
    const long long fc3_out_batch_size = static_cast<long long>(batch_size) * fc3_out;

    long long workspace_size_elements =
        batch_in_size + c1_out_batch_size + c1_out_batch_size + p1_out_batch_size +
        c2_out_batch_size + c2_out_batch_size + p2_out_batch_size +
        fc1_out_batch_size + fc1_out_batch_size + fc1_out_batch_size +
        fc2_out_batch_size + fc2_out_batch_size + fc2_out_batch_size +
        fc3_out_batch_size + fc3_out_batch_size;

    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    float* h_pinned_images[num_streams];
    float* h_pinned_output[num_streams];
    float* d_workspace[num_streams];

    for (int s = 0; s < num_streams; ++s) {
        checkCudaErrors(cudaMallocHost(&h_pinned_images[s], batch_in_size * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&h_pinned_output[s], fc3_out_batch_size * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_workspace[s], workspace_size_elements * sizeof(float)));
    }

    std::vector<int> predictions(total_images, 0);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int stream_id = batch_idx % num_streams;
        cudaStream_t stream = streams[stream_id];

        if (batch_idx >= num_streams) {
            int prev_batch_idx = batch_idx - num_streams;
            checkCudaErrors(cudaStreamSynchronize(stream));

            int prev_start_idx = prev_batch_idx * batch_size;
            int prev_end_idx = std::min(prev_start_idx + batch_size, total_images);
            int prev_batch_size = prev_end_idx - prev_start_idx;

            for (int i = 0; i < prev_batch_size; ++i) {
                float* scores = h_pinned_output[stream_id] + static_cast<long long>(i) * fc3_out;
                int pred = static_cast<int>(std::max_element(scores, scores + fc3_out) - scores);
                predictions[prev_start_idx + i] = pred;
            }
        }

        int start_idx = batch_idx * batch_size;
        int end_idx = std::min(start_idx + batch_size, total_images);
        int current_batch_size = end_idx - start_idx;

        for (int i = 0; i < current_batch_size; ++i) {
            std::memcpy(
                h_pinned_images[stream_id] + static_cast<long long>(i) * in_size,
                images[start_idx + i].data(),
                in_size * sizeof(float));
        }

        long long offset = 0;
        float* d_image        = d_workspace[stream_id] + offset; offset += batch_in_size;
        float* d_v1           = d_workspace[stream_id] + offset; offset += c1_out_batch_size;
        float* d_if1_spikes   = d_workspace[stream_id] + offset; offset += c1_out_batch_size;
        float* d_pool1_out    = d_workspace[stream_id] + offset; offset += p1_out_batch_size;
        float* d_v2           = d_workspace[stream_id] + offset; offset += c2_out_batch_size;
        float* d_if2_spikes   = d_workspace[stream_id] + offset; offset += c2_out_batch_size;
        float* d_pool2_out    = d_workspace[stream_id] + offset; offset += p2_out_batch_size;
        float* d_fc1_out      = d_workspace[stream_id] + offset; offset += fc1_out_batch_size;
        float* d_v3           = d_workspace[stream_id] + offset; offset += fc1_out_batch_size;
        float* d_if3_spikes   = d_workspace[stream_id] + offset; offset += fc1_out_batch_size;
        float* d_fc2_out      = d_workspace[stream_id] + offset; offset += fc2_out_batch_size;
        float* d_v4           = d_workspace[stream_id] + offset; offset += fc2_out_batch_size;
        float* d_if4_spikes   = d_workspace[stream_id] + offset; offset += fc2_out_batch_size;
        float* d_fc3_out      = d_workspace[stream_id] + offset; offset += fc3_out_batch_size;
        float* d_final_output = d_workspace[stream_id] + offset; offset += fc3_out_batch_size;
        (void)d_fc1_out; (void)d_fc2_out;

        checkCudaErrors(cudaMemcpyAsync(
            d_image,
            h_pinned_images[stream_id],
            static_cast<long long>(current_batch_size) * in_size * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));

        long long current_c1 = static_cast<long long>(current_batch_size) * c1_out_size;
        long long current_c2 = static_cast<long long>(current_batch_size) * c2_out_size;
        long long current_p1 = static_cast<long long>(current_batch_size) * p1_out_size;
        long long current_p2 = static_cast<long long>(current_batch_size) * p2_out_size;
        long long current_fc1 = static_cast<long long>(current_batch_size) * fc1_out;
        long long current_fc2 = static_cast<long long>(current_batch_size) * fc2_out;
        long long current_fc3 = static_cast<long long>(current_batch_size) * fc3_out;

        checkCudaErrors(cudaMemsetAsync(d_v1, 0, current_c1 * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_v2, 0, current_c2 * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_v3, 0, current_fc1 * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_v4, 0, current_fc2 * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_final_output, 0, current_fc3 * sizeof(float), stream));

        for (int t_step = 0; t_step < T; ++t_step) {
            dim3 block_conv(8, 8, 1);
            dim3 grid_conv1(div_up(c1_out_w, block_conv.x),
                            div_up(c1_out_h, block_conv.y),
                            current_batch_size);
            fused_conv_if_kernel1<<<grid_conv1, block_conv, 0, stream>>>(
                d_image, d_conv1_w, d_conv1_b,
                d_v1, d_if1_spikes,
                current_batch_size, c1_in_c, c1_in_h, c1_in_w,
                c1_out_c, c1_k, c1_k,
                c1_out_h, c1_out_w);

            dim3 block_pool1(16, 16, 1);
            dim3 grid_pool1(div_up(p1_out_w, block_pool1.x),
                            div_up(p1_out_h, block_pool1.y),
                            current_batch_size);
            maxpool2d_kernel1<<<grid_pool1, block_pool1, 0, stream>>>(
                d_if1_spikes, d_pool1_out,
                current_batch_size, p1_out_c,
                c1_out_h, c1_out_w,
                p1_out_h, p1_out_w,
                p1_k);

            dim3 grid_conv2(div_up(c2_out_w, block_conv.x),
                            div_up(c2_out_h, block_conv.y),
                            current_batch_size);
            fused_conv_if_kernel2<<<grid_conv2, block_conv, 0, stream>>>(
                d_pool1_out, d_conv2_w, d_conv2_b,
                d_v2, d_if2_spikes,
                current_batch_size, c2_in_c, c2_in_h, c2_in_w,
                c2_out_c, c2_k, c2_k,
                c2_out_h, c2_out_w);

            dim3 block_pool2(4, 4, p2_out_c);
            dim3 grid_pool2(div_up(p2_out_w, block_pool2.x),
                            div_up(p2_out_h, block_pool2.y),
                            current_batch_size);
            maxpool2d_kernel2<<<grid_pool2, block_pool2, 0, stream>>>(
                d_if2_spikes, d_pool2_out,
                current_batch_size, p2_out_c,
                c2_out_h, c2_out_w,
                p2_out_h, p2_out_w,
                p2_k);

            if (use_tensor_core) {
                dim3 grid_wmma1(div_up(fc1_out, BLOCK_COLS),
                                div_up(current_batch_size, BLOCK_ROWS));
                linear_if_fused_kernel_wmma<<<grid_wmma1, THREADS_PER_BLOCK, 0, stream>>>(
                    d_pool2_out, d_fc1_w, d_fc1_b,
                    d_v3, d_if3_spikes,
                    current_batch_size, fc1_in, fc1_out);
            } else {
                dim3 grid_simple1(div_up(fc1_out, block_size),
                                  current_batch_size);
                linear_kernel_simple<<<grid_simple1, block_size, 0, stream>>>(
                    d_pool2_out, d_fc1_w, d_fc1_b,
                    d_fc1_out,
                    current_batch_size, fc1_in, fc1_out);
                dim3 grid_if1(div_up(static_cast<int>(current_fc1), block_size));
                if_node_kernel<<<grid_if1, block_size, 0, stream>>>(
                    d_fc1_out, d_v3, d_if3_spikes, current_fc1);
            }

            if (use_tensor_core) {
                dim3 grid_wmma2(div_up(fc2_out, BLOCK_COLS),
                                div_up(current_batch_size, BLOCK_ROWS));
                linear_if_fused_kernel_wmma<<<grid_wmma2, THREADS_PER_BLOCK, 0, stream>>>(
                    d_if3_spikes, d_fc2_w, d_fc2_b,
                    d_v4, d_if4_spikes,
                    current_batch_size, fc2_in, fc2_out);
            } else {
                dim3 grid_simple2(div_up(fc2_out, block_size),
                                  current_batch_size);
                linear_kernel_simple<<<grid_simple2, block_size, 0, stream>>>(
                    d_if3_spikes, d_fc2_w, d_fc2_b,
                    d_fc2_out,
                    current_batch_size, fc2_in, fc2_out);
                dim3 grid_if2(div_up(static_cast<int>(current_fc2), block_size));
                if_node_kernel<<<grid_if2, block_size, 0, stream>>>(
                    d_fc2_out, d_v4, d_if4_spikes, current_fc2);
            }

            if (use_tensor_core) {
                dim3 grid_wmma3(div_up(fc3_out, BLOCK_COLS),
                                div_up(current_batch_size, BLOCK_ROWS));
                linear_kernel_wmma<<<grid_wmma3, THREADS_PER_BLOCK, 0, stream>>>(
                    d_if4_spikes, d_fc3_w, d_fc3_b,
                    d_fc3_out,
                    current_batch_size, fc3_in, fc3_out);
            } else {
                dim3 grid_simple3(div_up(fc3_out, block_size),
                                  current_batch_size);
                linear_kernel_simple<<<grid_simple3, block_size, 0, stream>>>(
                    d_if4_spikes, d_fc3_w, d_fc3_b,
                    d_fc3_out,
                    current_batch_size, fc3_in, fc3_out);
            }

            dim3 grid_add(div_up(static_cast<int>(current_fc3), block_size));
            add_kernel<<<grid_add, block_size, 0, stream>>>(
                d_fc3_out, d_final_output, current_fc3);
        }

        checkCudaErrors(cudaMemcpyAsync(
            h_pinned_output[stream_id],
            d_final_output,
            static_cast<long long>(current_batch_size) * fc3_out * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
    }

    int num_remaining = std::min(num_streams, num_batches);
    for (int i = 0; i < num_remaining; ++i) {
        int batch_base = num_batches - num_remaining + i;
        int stream_id = batch_base % num_streams;
        checkCudaErrors(cudaStreamSynchronize(streams[stream_id]));

        int start_idx = batch_base * batch_size;
        int end_idx = std::min(start_idx + batch_size, total_images);
        int cur_batch = end_idx - start_idx;

        for (int j = 0; j < cur_batch; ++j) {
            float* scores = h_pinned_output[stream_id] + static_cast<long long>(j) * fc3_out;
            int pred = static_cast<int>(std::max_element(scores, scores + fc3_out) - scores);
            predictions[start_idx + j] = pred;
        }
    }

    for (int s = 0; s < num_streams; ++s) {
        checkCudaErrors(cudaStreamDestroy(streams[s]));
        checkCudaErrors(cudaFreeHost(h_pinned_images[s]));
        checkCudaErrors(cudaFreeHost(h_pinned_output[s]));
        checkCudaErrors(cudaFree(d_workspace[s]));
    }

    checkCudaErrors(cudaGetLastError());
    return predictions;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
	std::string dir = argv[1];
	
    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
        // YOU CAN ADD MORE PARAMETERS HERE!!!
        );
    
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================