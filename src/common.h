#pragma once

#include <iostream>
#include <chrono>
#include "omp.h"

template<typename... Args>
void print(const Args&... args) {
    ((std::cout << args << " "), ...) << std::endl;  // Fold expression for variadic printing
}

// life time of a function
struct TimeIt {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
    std::string message;

    TimeIt(std::string&& message="") : message(message), start(Clock::now()) {}

    ~TimeIt() {
        auto end = Clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long ns = duration_ns.count();

        if (!message.empty()) {
            print("\n",message);
        }

        if (ns >= 3600000000000) {
            std::cout << "\nTime elapsed: " << ns / 3600000000000.0 << " hours.\n";
   
        }
        else if (ns >= 60000000000) {
            std::cout << "\nTime elapsed: " << ns / 60000000000.0 << " minutes.\n";
        }
        else if (ns >= 1000000000) {
            std::cout << "\nTime elapsed: " << ns / 1000000000.0 << " seconds.\n";
        }
        else if (ns >= 1000000) {
            std::cout << "\nTime elapsed: " << ns / 1000000.0 << " milliseconds.\n";
        }
        else if (ns >= 1000) {
            std::cout << "\nTime elapsed: " << ns / 1000.0 << " microseconds.\n";
        }
        else {
            std::cout << "\nTime elapsed: " << ns << " nanoseconds.\n";
        }
    }

	
};

cv::Mat hwcToChw(const cv::Mat& input) {
    CV_Assert(input.dims == 3 || (input.dims == 2 && input.channels() == 3));

    const int height = input.rows;
    const int width = input.cols;
    const int channels = input.channels();
    const int stride = width * height;

    // Create output matrix with CHW layout
    cv::Mat output(1, channels * height * width, CV_32F);
    float* output_ptr = output.ptr<float>();

    // For better cache utilization, process one channel at a time
    omp_set_num_threads(4);
    print("number of threads its running : ", omp_get_num_threads());
#pragma omp parallel for if(height * width > 100000)
    for (int c = 0; c < channels; ++c) {
        float* channel_ptr = output_ptr + c * stride;

        // Use pointer arithmetic for faster access
        const uint8_t* input_ptr = input.ptr<uint8_t>();

        // Vectorization-friendly loop
        for (int i = 0; i < height * width; ++i) {
            channel_ptr[i] = static_cast<float>(input_ptr[i * channels + c]);
        }
    }

    return output;
}

void hwcToChw(const cv::Mat& input, cv::Mat& output) {
    TimeIt t("hwc->chw conversion time : ");
    //CV_Assert(input.dims == 3 || (input.dims == 2 && input.channels() == 3));
    const int height = input.rows;
    const int width = input.cols;
    const int channels = input.channels();
    const int stride = width * height;

    // Ensure output has correct size and type
    if (output.empty() || output.size() != cv::Size(width * height, channels)) {
        output.create(channels, width * height, CV_32F);
    }

    // Use temp buffer only if input and output are the same matrix
    std::vector<float> temp_buffer;
    float* final_output_ptr = output.ptr<float>();

    if (input.data == output.data) {
        temp_buffer.resize(channels * height * width);
        final_output_ptr = temp_buffer.data();
    }

    omp_set_num_threads(3);    
#pragma omp parallel for if(height * width > 100'000)
    for (int c = 0; c < channels; ++c) {
        //print("Thread ID: ", omp_get_thread_num(), " is processing channel: ", c);
        float* channel_ptr = final_output_ptr + c * stride;
        const uint8_t* input_ptr = input.ptr<uint8_t>();

        // Unrolled loop for better performance
        const int blockSize = 4;
        int i = 0;
        for (; i <= height * width - blockSize; i += blockSize) {
            channel_ptr[i] = static_cast<float>(input_ptr[i * channels + c]);
            channel_ptr[i + 1] = static_cast<float>(input_ptr[(i + 1) * channels + c]);
            channel_ptr[i + 2] = static_cast<float>(input_ptr[(i + 2) * channels + c]);
            channel_ptr[i + 3] = static_cast<float>(input_ptr[(i + 3) * channels + c]);
        }
        // Handle remaining elements
        for (; i < height * width; ++i) {
            channel_ptr[i] = static_cast<float>(input_ptr[i * channels + c]);
        }
    }

    // If we used a temp buffer, copy back to output
    if (!temp_buffer.empty()) {
        std::memcpy(output.data, temp_buffer.data(),
            channels * height * width * sizeof(float));
    }
}

void convertHWCtoCHW(const cv::Mat& input, cv::Mat& output) { // ~2.8 ms
    TimeIt t("hwc->chw conversion time : ");
    int height = input.rows;
    int width = input.cols;
    int channels = input.channels();
    int total_pixels = height * width;

    // Allocate output Mat with the same type but CHW layout
    output.create(channels, height * width, CV_32F);  // Channels first, flatten width * height for each

    const uint8_t* input_ptr = input.ptr<uint8_t>();  // Pointer to input data
    float* output_ptr = output.ptr<float>();          // Pointer to output data

#pragma omp parallel for schedule(static) if(total_pixels > 1000)
    for (int i = 0; i < total_pixels; ++i) {
        for (int c = 0; c < channels; ++c) {
            output_ptr[c * total_pixels + i] = static_cast<float>(input_ptr[i * channels + c]);
        }
    }
}