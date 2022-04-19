#ifndef _IMAGE_DEVICE_H
#define _IMAGE_DEVICE_H_

#include "image.cuh"
#define FILTER_WIDTH 5
__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];

float deviceTime_min_rgb2gray = 10000;
float deviceTime_max_rgb2gray = 0;
float deviceTime_min_calcEnergy = 10000;
float deviceTime_max_calcEnergy = 0;
float deviceTime_min_calcMinCost = 10000;
float deviceTime_max_calcMinCost = 0;
float deviceTime_min_removeSeam = 10000;
float deviceTime_max_removeSeam = 0;

// Device - Convert RGB to grayscale
void convertRgb2Gray_device(uchar3 *inPixels, int width, int height, uint8_t *outPixels, dim3 blkSize = dim3(32, 32));

// Device - Remove seam VERTICAL
void removePixels_device(uchar3 *inPixels, int &width, int &height, int *seam, uchar3 *outPixels);

// Kernel covert rgb to grayscale
__global__ void convertRgb2GrayKernel(uchar3 *inPixels, int width, int height, uint8_t *outPixels);

// Kernel remove seam
__global__ void removePixelsKernel(uchar3 *inPixels, int width, int height, int *seam, uchar3 *outPixels);

void convertRgb2Gray_device(uchar3 *inPixels, int width, int height, uint8_t *outPixels, dim3 blkSize)
{
    GpuTimer timer;
    timer.Start();

    // Allocate device memories
    uchar3 *d_inPixels;
    uint8_t *d_outPixels;
    size_t nInBytes = width * height * sizeof(uchar3);
    size_t nOutBytes = width * height * sizeof(uint8_t);
    CHECK(cudaMalloc(&d_inPixels, nInBytes));
    CHECK(cudaMalloc(&d_outPixels, nOutBytes));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_inPixels, inPixels, nInBytes, cudaMemcpyHostToDevice));

    // Set grid size and call kernel
    dim3 gridSize((width - 1) / blkSize.x + 1, (height - 1) / blkSize.y + 1);
    convertRgb2GrayKernel<<<gridSize, blkSize>>>(d_inPixels, width, height, d_outPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(outPixels, d_outPixels, nOutBytes, cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));

    timer.Stop();
    float time = timer.Elapsed();
    if (time < deviceTime_min_rgb2gray)
        deviceTime_min_rgb2gray = time;
    if (time > deviceTime_max_rgb2gray)
        deviceTime_max_rgb2gray = time;
}

__global__ void convertRgb2GrayKernel(uchar3 *inPixels, int width, int height, uint8_t *outPixels)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    {
        int i = r * width + c;
        uchar3 t = inPixels[i];
        // outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        outPixels[i] = __fadd_rn(__fadd_rn(__fmul_rn(0.299f, t.x), __fmul_rn(0.587f, t.y)), __fmul_rn(0.114f, t.z));
    }
}

void removePixels_device(uchar3 *inPixels, int &width, int &height, int *seam, uchar3 *outPixels)
{
    GpuTimer timer;
    timer.Start();

    // Alocate device memories
    size_t nBytes = width * height * sizeof(uchar3);
    size_t seamBytes = height * sizeof(int);
    uchar3 *d_inPixels, *d_outPixels;
    int *d_seam;
    CHECK(cudaMalloc(&d_inPixels, nBytes));
    CHECK(cudaMalloc(&d_outPixels, nBytes));
    CHECK(cudaMalloc(&d_seam, seamBytes));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_seam, seam, seamBytes, cudaMemcpyHostToDevice));

    // Set grid size and call kernel
    dim3 blkSize(32, 32);
    dim3 gridSize((width - 2) / blkSize.x + 1,
                  (height - 1) / blkSize.y + 1);
    int tWidth = width, tHeight = height;
    removePixelsKernel<<<gridSize, blkSize>>>(d_inPixels, tWidth, tHeight, d_seam, d_outPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(outPixels, d_outPixels, (width - 1) * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_seam));

    width--;

    timer.Stop();
    float time = timer.Elapsed();
    if (time < deviceTime_min_removeSeam)
        deviceTime_min_removeSeam = time;
    if (time > deviceTime_max_removeSeam)
        deviceTime_max_removeSeam = time;
}

__global__ void removePixelsKernel(uchar3 *inPixels, int width, int height, int *seam, uchar3 *outPixels)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c < (width - 1) && r < height)
    {
        if (c < seam[r])
            outPixels[r * (width - 1) + c] = inPixels[r * width + c];
        else
            outPixels[r * (width - 1) + c] = inPixels[r * width + c + 1];
    }
}

#endif