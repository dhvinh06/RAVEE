#ifndef RAVEE_FARNEBACK_CUDA_H
#define RAVEE_FARNEBACK_CUDA_H

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "CudaFlowProfile.h"

namespace ravee {

cudaError_t restoreWarpBorder(const unsigned char*, size_t, unsigned char*, size_t,
                              int, int, const double*, cudaStream_t);
cudaError_t filterFlowMedian5(const unsigned char*, size_t, unsigned char*, size_t,
                              int, int, cudaStream_t);

inline bool useCudaFlow() {
    const char* backend = std::getenv("RAVEE_IDT_BACKEND");
    return backend && std::strcmp(backend, "cuda") == 0;
}

inline void initializeCudaFlow() {
    if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
        CV_Error(cv::Error::StsError, "RAVEE CUDA backend cannot see a CUDA device");
    }
    cv::cuda::setDevice(0);
    if (!cv::cuda::DeviceInfo(0).isCompatible()) {
        CV_Error(cv::Error::StsError, "CUDA device is incompatible with the OpenCV build");
    }
    // OpenCV Farneback uses temporary buffers from the CUDA stream pool.
    static std::once_flag pool;
    std::call_once(pool, [] {
        cv::cuda::setBufferPoolUsage(true);
        cv::cuda::setBufferPoolConfig(0, 32 * 1024 * 1024, 6);
    });
}

class CudaFarneback {
public:
    CudaFarneback(const std::vector<cv::Size>& sizes, const std::vector<float>& scales,
                  int poly_n, double poly_sigma, float scale_stride,
                  std::vector<cv::Mat>& raw, std::vector<cv::Mat>& warped)
        : levels_(sizes.size()), previous_pyramid_(sizes.size()), current_pyramid_(sizes.size()),
          warped_pyramid_(warped.size()), raw_(raw), warped_(warped), scale_stride_(scale_stride) {
        const cv::Size full_size = sizes.front();
        upload_buffer_.create(full_size, CV_8UC1);
        upload_view_ = upload_buffer_.createMatHeader();
        previous_.create(full_size, CV_8UC1);
        current_.create(full_size, CV_8UC1);
        full_float_.create(full_size, CV_32FC1);
        blurred_.create(full_size, CV_32FC1);
        if (!warped_.empty()) warped_image_.create(full_size, CV_8UC1);

        for (size_t i = 0; i < sizes.size(); ++i) {
            auto& level = levels_[i];
            const cv::Size size = sizes[i];
            previous_pyramid_[i].create(size, CV_32FC1);
            current_pyramid_[i].create(size, CV_32FC1);
            if (!warped_.empty()) warped_pyramid_[i].create(size, CV_32FC1);
            level.flow.create(size, CV_32FC2);
            level.filtered.create(size, CV_32FC2);
            for (auto& channel : level.channels) channel.create(size, CV_32FC1);
            level.raw_host.create(size, CV_32FC2);
            raw_[i] = level.raw_host.createMatHeader();
            if (!warped_.empty()) {
                level.warped_host.create(size, CV_32FC2);
                warped_[i] = level.warped_host.createMatHeader();
            }

            const double sigma = (scales[i] - 1) * 0.5;
            const int width = std::max(cvRound(sigma * 5) | 1, 3);
            level.smooth = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1,
                cv::Size(width, width), sigma, sigma, cv::BORDER_REFLECT_101);
            int flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN;
            if (i + 1 < sizes.size()) flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
            // The caller builds the IDT pyramid. Fast mode with 0 extra levels
            // uses this exact image without a second blur inside the solver.
            level.solver = cv::cuda::FarnebackOpticalFlow::create(
                0, 0.5, true, 11, 2, poly_n, poly_sigma, flags);
        }
        const char* profile = std::getenv("RAVEE_IDT_CUDA_PROFILE");
        if (profile && std::strcmp(profile, "1") == 0)
            profile_.reset(new CudaFlowProfile(sizes.size()));
    }

    ~CudaFarneback() {
        try {
            stream_.waitForCompletion();
        } catch (const cv::Exception&) {
            // Avoid masking the error already reported by the processing loop.
        }
        // Mat headers refer to page-locked buffers owned by levels_.
        raw_.clear();
        warped_.clear();
    }

    void seed(const cv::Mat& frame) {
        upload(frame, previous_);
        buildPyramid(previous_, previous_pyramid_);
        wait();
    }

    void beginFrame(const cv::Mat& frame) {
        CudaFlowProfile::HostSpan timer(profile_.get(), CudaFlowProfile::Host::RawSubmit);
        if (profile_) profile_->pass(CudaFlowProfile::Pass::Raw);
        upload(frame, current_);
        streamStage(CudaFlowProfile::Gpu::Pyramid, -1, [&] { buildPyramid(current_, current_pyramid_); });
        solve(current_pyramid_, raw_);
        ++raw_pairs_;
    }

    void compensateCamera(const cv::Mat& previous_to_current) {
        CudaFlowProfile::HostSpan timer(profile_.get(), CudaFlowProfile::Host::WarpedSubmit);
        if (profile_) profile_->pass(CudaFlowProfile::Pass::Warped);
        streamStage(CudaFlowProfile::Gpu::Warp, -1, [&] {
            if (!cv::checkRange(previous_to_current) || cv::determinant(previous_to_current) == 0) {
                previous_.copyTo(warped_image_, stream_);
            } else {
                cv::cuda::warpPerspective(current_, warped_image_, previous_to_current, current_.size(),
                    cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(), stream_);
                const cv::Mat matrix = previous_to_current.isContinuous()
                    ? previous_to_current : previous_to_current.clone();
                checkCuda(restoreWarpBorder(previous_.data, previous_.step, warped_image_.data,
                    warped_image_.step, current_.cols, current_.rows, matrix.ptr<double>(), cudaStream()));
            }
        });
        streamStage(CudaFlowProfile::Gpu::Pyramid, -1, [&] { buildPyramid(warped_image_, warped_pyramid_); });
        solve(warped_pyramid_, warped_);
        ++warped_pairs_;
    }

    void wait(bool warped = false) {
        CudaFlowProfile::HostSpan timer(profile_.get(), warped ? CudaFlowProfile::Host::WarpedWait
                                                              : CudaFlowProfile::Host::RawWait);
        stream_.waitForCompletion();
        timer.stop();
        if (profile_) profile_->collect();
    }

    CudaFlowProfile* profiler() const { return profile_.get(); }

    void advance() {
        std::swap(previous_, current_);
        previous_pyramid_.swap(current_pyramid_);
    }

    void report() const {
        std::fprintf(stderr, "IDT CUDA optical flow: uploads=%zu raw_pairs=%zu warped_pairs=%zu\n",
            uploads_, raw_pairs_, warped_pairs_);
        if (profile_) profile_->report();
    }

private:
    struct Level {
        cv::Ptr<cv::cuda::Filter> smooth;
        cv::Ptr<cv::cuda::FarnebackOpticalFlow> solver;
        cv::cuda::GpuMat flow, filtered, channels[2];
        cv::cuda::HostMem raw_host, warped_host;
    };

    void upload(const cv::Mat& frame, cv::cuda::GpuMat& device) {
        frame.copyTo(upload_view_);
        streamStage(CudaFlowProfile::Gpu::Upload, -1, [&] { device.upload(upload_view_, stream_); });
        ++uploads_;
    }

    void buildPyramid(const cv::cuda::GpuMat& image, std::vector<cv::cuda::GpuMat>& pyramid) {
        image.convertTo(full_float_, CV_32FC1, stream_);
        for (size_t i = 0; i < pyramid.size(); ++i) {
            if (i == 0) {
                levels_[i].smooth->apply(full_float_, pyramid[i], stream_);
            } else {
                levels_[i].smooth->apply(full_float_, blurred_, stream_);
                cv::cuda::resize(blurred_, pyramid[i], pyramid[i].size(), 0, 0, cv::INTER_LINEAR, stream_);
            }
        }
    }

    void solve(const std::vector<cv::cuda::GpuMat>& target, std::vector<cv::Mat>& output) {
        using Gpu = CudaFlowProfile::Gpu;
        for (int i = static_cast<int>(levels_.size()) - 1; i >= 0; --i) {
            auto& level = levels_[i];
            if (static_cast<size_t>(i + 1) < levels_.size()) {
                streamStage(Gpu::Seed, i, [&] {
                    const auto& coarse = levels_[i + 1];
                    for (int channel = 0; channel < 2; ++channel) {
                        auto& seed = level.channels[channel];
                        cv::cuda::resize(coarse.channels[channel], seed, level.flow.size(),
                            0, 0, cv::INTER_LINEAR, stream_);
                        seed.convertTo(seed, CV_32FC1, scale_stride_, stream_);
                    }
                    cv::cuda::merge(level.channels, 2, level.flow, stream_);
                });
            }
            streamStage(Gpu::Farneback, i, [&] {
                level.solver->calc(previous_pyramid_[i], target[i], level.flow, stream_);
            });
            streamStage(Gpu::Median, i, [&] {
                checkCuda(filterFlowMedian5(level.flow.data, level.flow.step, level.filtered.data,
                    level.filtered.step, level.flow.cols, level.flow.rows, cudaStream()));
            });
            if (i > 0) streamStage(Gpu::Split, i, [&] { cv::cuda::split(level.filtered, level.channels, stream_); });
            streamStage(Gpu::Download, i, [&] { level.filtered.download(output[i], stream_); });
        }
    }

    template<typename Work>
    void streamStage(CudaFlowProfile::Gpu phase, int level, Work work) {
        if (profile_) profile_->onStream(stream_, phase, level, work);
        else work();
    }

    cudaStream_t cudaStream() { return cv::cuda::StreamAccessor::getStream(stream_); }

    static void checkCuda(cudaError_t error) {
        if (error != cudaSuccess) CV_Error(cv::Error::StsError, cudaGetErrorString(error));
    }

    cv::cuda::Stream stream_;
    std::vector<Level> levels_;
    std::vector<cv::cuda::GpuMat> previous_pyramid_, current_pyramid_, warped_pyramid_;
    cv::cuda::GpuMat previous_, current_, warped_image_, full_float_, blurred_;
    cv::cuda::HostMem upload_buffer_;
    cv::Mat upload_view_;
    std::vector<cv::Mat>& raw_;
    std::vector<cv::Mat>& warped_;
    float scale_stride_;
    size_t uploads_ = 0, raw_pairs_ = 0, warped_pairs_ = 0;
    std::unique_ptr<CudaFlowProfile> profile_;
};

}  // namespace ravee

#endif
