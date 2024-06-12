#pragma once
#include <cudnn.h>

#include <chrono>
#include <limits>

#define CONV_MODE CUDNN_CROSS_CORRELATION

namespace ch {
using namespace std::chrono;
}

class CUDNNConvolution {
    const cudnnDataType_t DATA_TYPE = CUDNN_DATA_FLOAT;
    cudnnMathType_t MATH_TYPE = CUDNN_DEFAULT_MATH;
    cudnnTensorFormat_t INPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
    cudnnTensorFormat_t OUTPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
    cudnnTensorFormat_t KERNEL_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
    const int warmup = 200, rounds = 200;

    int batch_size, in_channels, in_height, in_width, out_channels, kernel_size,
        stride, padding, dilation, groups, input_channels_per_group, out_height,
        out_width;
    size_t workspace_size = (size_t) 1024 * 1024 * 1024;
    float *d_input, *d_kernel, *d_output, *d_bias, *workspace;
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_descriptor, output_descriptor,
        bias_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnActivationDescriptor_t activation_descriptor;

public:
    CUDNNConvolution(int batch_size,
                     int in_channels,
                     int in_height,
                     int in_width,
                     int out_channels,
                     int kernel_size,
                     int stride,
                     int padding,
                     int dilation,
                     int groups,
                     bool relu_act)
        : batch_size(batch_size),
          in_channels(in_channels),
          in_height(in_height),
          in_width(in_width),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          groups(groups) {
        input_channels_per_group = in_channels / groups;
        out_height =
            (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) /
                stride +
            1;
        out_width =
            (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) /
                stride +
            1;
        cudaSetDevice(0);
        cudnnCreate(&cudnn);
        // input
        size_t input_bytes =
            batch_size * in_channels * in_height * in_width * sizeof(float);
        cudaMalloc(&d_input, input_bytes);
        // kernel
        size_t kernel_bytes = out_channels * in_channels * kernel_size *
                              kernel_size * sizeof(float);
        cudaMalloc(&d_kernel, kernel_bytes);
        // bias
        cudaMalloc(&d_bias, out_channels * sizeof(float));
        // descriptor
        cudnnCreateTensorDescriptor(&input_descriptor);
        cudnnSetTensor4dDescriptor(input_descriptor,
                                   INPUT_TENSOR_FORMAT,
                                   DATA_TYPE,
                                   batch_size,
                                   in_channels,
                                   in_height,
                                   in_width);
        cudnnCreateFilterDescriptor(&kernel_descriptor);
        cudnnSetFilter4dDescriptor(kernel_descriptor,
                                   DATA_TYPE,
                                   KERNEL_TENSOR_FORMAT,
                                   out_channels,
                                   input_channels_per_group,
                                   kernel_size,
                                   kernel_size);
        cudnnCreateConvolutionDescriptor(&convolution_descriptor);
        cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        padding,
                                        padding,
                                        stride,
                                        stride,
                                        dilation,
                                        dilation,
                                        CONV_MODE,
                                        DATA_TYPE);
        cudnnCreateTensorDescriptor(&bias_descriptor);
        cudnnSetTensor4dDescriptor(bias_descriptor,
                                   OUTPUT_TENSOR_FORMAT,
                                   DATA_TYPE,
                                   1,
                                   out_channels,
                                   1,
                                   1);
        cudnnCreateActivationDescriptor(&activation_descriptor);
        cudnnSetActivationDescriptor(
            activation_descriptor,
            relu_act ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY,
            CUDNN_NOT_PROPAGATE_NAN,
            std::numeric_limits<float>::infinity());
        if (groups > 1) {
            cudnnSetConvolutionGroupCount(convolution_descriptor, groups);
        }
        cudnnSetConvolutionMathType(convolution_descriptor, MATH_TYPE);
        cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                              input_descriptor,
                                              kernel_descriptor,
                                              &batch_size,
                                              &out_channels,
                                              &out_height,
                                              &out_width);
        // output
        size_t output_bytes =
            batch_size * out_channels * out_height * out_width * sizeof(float);
        cudaMalloc(&d_output, output_bytes);
        cudnnCreateTensorDescriptor(&output_descriptor);
        cudnnSetTensor4dDescriptor(output_descriptor,
                                   OUTPUT_TENSOR_FORMAT,
                                   DATA_TYPE,
                                   batch_size,
                                   out_channels,
                                   out_height,
                                   out_width);
        cudaMalloc(&workspace, workspace_size);
    }
    ~CUDNNConvolution() {
        cudaDeviceSynchronize();
        cudaFree(d_kernel);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_bias);
        cudaFree(workspace);
        cudnnDestroyTensorDescriptor(input_descriptor);
        cudnnDestroyTensorDescriptor(output_descriptor);
        cudnnDestroyFilterDescriptor(kernel_descriptor);
        cudnnDestroyTensorDescriptor(bias_descriptor);
        cudnnDestroyActivationDescriptor(activation_descriptor);
        cudnnDestroyConvolutionDescriptor(convolution_descriptor);
        cudnnDestroy(cudnn);
        cudaDeviceSynchronize();
    }
    float profile_conv(cudnnConvolutionFwdAlgo_t algo) {
        const float alpha = 1, beta = 0;
        if (cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor,
                                    d_input,
                                    kernel_descriptor,
                                    d_kernel,
                                    convolution_descriptor,
                                    algo,
                                    workspace,
                                    workspace_size,
                                    &beta,
                                    output_descriptor,
                                    d_output) != CUDNN_STATUS_SUCCESS) {
            return -1;
        }
        ch::time_point<ch::high_resolution_clock, ch::nanoseconds> begin, end;
        // warmup
        for (int kk = 0; kk < warmup; ++kk) {
            cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor,
                                    d_input,
                                    kernel_descriptor,
                                    d_kernel,
                                    convolution_descriptor,
                                    algo,
                                    workspace,
                                    workspace_size,
                                    &beta,
                                    output_descriptor,
                                    d_output);
        }
        cudaDeviceSynchronize();
        begin = ch::high_resolution_clock::now();
        for (int i = 0; i < rounds; ++i) {
            cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor,
                                    d_input,
                                    kernel_descriptor,
                                    d_kernel,
                                    convolution_descriptor,
                                    algo,
                                    workspace,
                                    workspace_size,
                                    &beta,
                                    output_descriptor,
                                    d_output);
        }
        cudaDeviceSynchronize();
        end = ch::high_resolution_clock::now();
        return ch::duration_cast<ch::duration<double>>(end - begin).count() *
               1000 / rounds;
    }
    float profile_conv_bias_act(cudnnConvolutionFwdAlgo_t algo) {
        const float alpha1 = 1, alpha2 = 0;
        if (cudnnConvolutionBiasActivationForward(cudnn,
                                                  &alpha1,
                                                  input_descriptor,
                                                  d_input,
                                                  kernel_descriptor,
                                                  d_kernel,
                                                  convolution_descriptor,
                                                  algo,
                                                  workspace,
                                                  workspace_size,
                                                  &alpha2,
                                                  output_descriptor,
                                                  d_output,
                                                  bias_descriptor,
                                                  d_bias,
                                                  activation_descriptor,
                                                  output_descriptor,
                                                  d_output) !=
            CUDNN_STATUS_SUCCESS) {
            return -1;
        }
        ch::time_point<ch::high_resolution_clock, ch::nanoseconds> begin, end;
        // warmup
        for (int kk = 0; kk < warmup; ++kk) {
            cudnnConvolutionBiasActivationForward(cudnn,
                                                  &alpha1,
                                                  input_descriptor,
                                                  d_input,
                                                  kernel_descriptor,
                                                  d_kernel,
                                                  convolution_descriptor,
                                                  algo,
                                                  workspace,
                                                  workspace_size,
                                                  &alpha2,
                                                  output_descriptor,
                                                  d_output,
                                                  bias_descriptor,
                                                  d_bias,
                                                  activation_descriptor,
                                                  output_descriptor,
                                                  d_output);
        }
        cudaDeviceSynchronize();
        begin = ch::high_resolution_clock::now();
        for (int i = 0; i < rounds; ++i) {
            cudnnConvolutionBiasActivationForward(cudnn,
                                                  &alpha1,
                                                  input_descriptor,
                                                  d_input,
                                                  kernel_descriptor,
                                                  d_kernel,
                                                  convolution_descriptor,
                                                  algo,
                                                  workspace,
                                                  workspace_size,
                                                  &alpha2,
                                                  output_descriptor,
                                                  d_output,
                                                  bias_descriptor,
                                                  d_bias,
                                                  activation_descriptor,
                                                  output_descriptor,
                                                  d_output);
        }
        cudaDeviceSynchronize();
        end = ch::high_resolution_clock::now();
        return ch::duration_cast<ch::duration<double>>(end - begin).count() *
               1000 / rounds;
    }
};

float profile_conv(int batch_size,
                   int in_channels,
                   int in_height,
                   int in_width,
                   int out_channels,
                   int kernel_size,
                   int stride,
                   int padding,
                   int dilation,
                   int groups,
                   int algo,
                   int mode) {
    if (mode < 0 || mode > 2) {
        return -1;
    }
    CUDNNConvolution cudnn_conv(batch_size,
                                in_channels,
                                in_height,
                                in_width,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                groups,
                                mode == 2);
    if (mode == 0) {
        return cudnn_conv.profile_conv((cudnnConvolutionFwdAlgo_t) algo);
    } else {
        return cudnn_conv.profile_conv_bias_act(
            (cudnnConvolutionFwdAlgo_t) algo);
    }
}