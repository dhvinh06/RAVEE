#ifndef RAVEE_CUDA_FLOW_PROFILE_H
#define RAVEE_CUDA_FLOW_PROFILE_H

#include <array>
#include <chrono>
#include <cstdio>
#include <vector>

#include <opencv2/core/cuda.hpp>

namespace ravee {

class CudaFlowProfile {
public:
    enum class Pass { Raw, Warped };
    enum class Gpu { Upload, Warp, Pyramid, Seed, Farneback, Median, Split, Download, Count };
    enum class Host { Frame, RawSubmit, RawWait, Surf, Homography, WarpedSubmit, WarpedWait,
                      Histograms, TrackingAndDescriptors, CollectEvents, Count };

    class HostSpan {
    public:
        HostSpan(CudaFlowProfile* profile, Host phase)
            : profile_(profile && profile->record_ ? profile : nullptr), phase_(phase) {
            if (profile_) start_ = Clock::now();
        }
        HostSpan(const HostSpan&) = delete;
        HostSpan& operator=(const HostSpan&) = delete;
        ~HostSpan() { stop(); }

        void stop() {
            if (!profile_) return;
            profile_->host_ms_[static_cast<size_t>(phase_)] +=
                std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
            profile_ = nullptr;
        }

    private:
        using Clock = std::chrono::steady_clock;
        CudaFlowProfile* profile_;
        Host phase_;
        Clock::time_point start_;
    };

    explicit CudaFlowProfile(size_t levels) : events_(2 + 5 * levels),
        level_ms_{std::vector<GpuTimes>(levels), std::vector<GpuTimes>(levels)} {}

    void frame(size_t index) {
        record_ = index >= warmup_frames;
        if (record_) ++frames_;
    }

    void pass(Pass pass) { pass_ = static_cast<size_t>(pass); }

    template<typename Work>
    void onStream(cv::cuda::Stream& stream, Gpu phase, int level, Work work) {
        if (!record_) {
            work();
            return;
        }
        auto& event = events_.at(pending_++);
        event.phase = static_cast<size_t>(phase);
        event.level = level;
        event.start.record(stream);
        work();
        event.end.record(stream);
    }

    // The caller already synchronized this stream before reading frames or flow.
    void collect() {
        if (!pending_) return;
        HostSpan timer(this, Host::CollectEvents);
        for (size_t i = 0; i < pending_; ++i) {
            const auto& event = events_[i];
            const double ms = cv::cuda::Event::elapsedTime(event.start, event.end);
            gpu_ms_[pass_][event.phase] += ms;
            if (event.level >= 0) level_ms_[pass_][event.level][event.phase] += ms;
        }
        pending_ = 0;
    }

    void report() const {
        static const char* const gpu_names[] = {"upload_h2d", "warp_and_border", "pyramid",
            "seed_upsample", "farneback", "median_5x5", "flow_split", "download_d2h"};
        static const char* const host_names[] = {"frame_loop", "raw_submit", "raw_wait", "surf_and_match",
            "flow_matches_and_homography", "warped_submit", "warped_wait", "descriptor_histograms",
            "tracking_and_descriptor_reads", "collect_events"};
        static const char* const pass_names[] = {"raw", "warped"};
        std::fprintf(stderr, "IDT CUDA profile: {\"warmup_frames\":%zu,\"measured_frames\":%zu,\"host_ms\":",
            warmup_frames, frames_);
        printTimes(host_names, host_ms_);
        std::fprintf(stderr, ",\"stream_ms\":{");
        for (size_t pass = 0; pass < gpu_ms_.size(); ++pass) {
            std::fprintf(stderr, "%s\"%s\":", pass ? "," : "", pass_names[pass]);
            printTimes(gpu_names, gpu_ms_[pass]);
        }
        std::fprintf(stderr, "},\"stream_ms_by_level\":{");
        for (size_t pass = 0; pass < level_ms_.size(); ++pass) {
            std::fprintf(stderr, "%s\"%s\":[", pass ? "," : "", pass_names[pass]);
            for (size_t level = 0; level < level_ms_[pass].size(); ++level) {
                if (level) std::fputc(',', stderr);
                printTimes(gpu_names, level_ms_[pass][level]);
            }
            std::fputc(']', stderr);
        }
        std::fprintf(stderr, "}}\n");
    }

private:
    using GpuTimes = std::array<double, static_cast<size_t>(Gpu::Count)>;
    static constexpr size_t warmup_frames = 30;

    struct Record {
        cv::cuda::Event start, end;
        size_t phase = 0;
        int level = -1;
    };

    template<size_t Count>
    static void printTimes(const char* const (&names)[Count], const std::array<double, Count>& values) {
        std::fputc('{', stderr);
        for (size_t i = 0; i < Count; ++i)
            std::fprintf(stderr, "%s\"%s\":%.6f", i ? "," : "", names[i], values[i]);
        std::fputc('}', stderr);
    }

    std::vector<Record> events_;
    std::array<std::vector<GpuTimes>, 2> level_ms_;
    std::array<GpuTimes, 2> gpu_ms_{};
    std::array<double, static_cast<size_t>(Host::Count)> host_ms_{};
    bool record_ = false;
    size_t pending_ = 0, pass_ = 0, frames_ = 0;
};

}  // namespace ravee

#endif
