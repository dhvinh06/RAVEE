#include <cuda_runtime.h>
#include <cstddef>

namespace ravee {
namespace {

struct Homography {
    double value[9];
};

__global__ void fillWarpBorder(const unsigned char* previous, size_t previous_step,
                               unsigned char* warped, size_t warped_step,
                               int width, int height, Homography homography) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const double* h = homography.value;
    const double w = h[6] * x + h[7] * y + h[8];
    const double sx = (h[0] * x + h[1] * y + h[2]) / w;
    const double sy = (h[3] * x + h[4] * y + h[5]) / w;
    const int margin = 5;
    if (!(sx > margin && sx < width - 1 - margin && sy > margin && sy < height - 1 - margin)) {
        warped[y * warped_step + x] = previous[y * previous_step + x];
    }
}

__global__ void medianFlow5(const unsigned char* source, size_t source_step,
                            unsigned char* output, size_t output_step, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float2 window[25];
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
        const int sy = max(0, min(height - 1, y + dy));
        const auto* row = reinterpret_cast<const float2*>(source + sy * source_step);
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            window[(dy + 2) * 5 + dx + 2] = row[max(0, min(width - 1, x + dx))];
        }
    }

    // Partial selection gives the componentwise median with replicated borders.
    #pragma unroll
    for (int i = 0; i <= 12; ++i) {
        #pragma unroll
        for (int j = i + 1; j < 25; ++j) {
            const float2 a = window[i];
            const float2 b = window[j];
            window[i] = make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
            window[j] = make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
        }
    }
    reinterpret_cast<float2*>(output + y * output_step)[x] = window[12];
}

}  // namespace

cudaError_t restoreWarpBorder(const unsigned char* previous, size_t previous_step,
                              unsigned char* warped, size_t warped_step, int width, int height,
                              const double* matrix, cudaStream_t stream) {
    Homography homography;
    for (int i = 0; i < 9; ++i) homography.value[i] = matrix[i];
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fillWarpBorder<<<grid, block, 0, stream>>>(previous, previous_step, warped, warped_step,
                                               width, height, homography);
    return cudaGetLastError();
}

cudaError_t filterFlowMedian5(const unsigned char* source, size_t source_step,
                              unsigned char* output, size_t output_step, int width, int height,
                              cudaStream_t stream) {
    const dim3 block(16, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    medianFlow5<<<grid, block, 0, stream>>>(source, source_step, output, output_step, width, height);
    return cudaGetLastError();
}

}  // namespace ravee
