#ifndef _SEAMCARVER_DEVICE_H_
#define _SEAMCARVER_DEVICE_H_

#include "seamcarver.cuh"
#include "image_device.cuh"
#include <cooperative_groups.h>

#define MAX_THREADS 1024

__constant__ float d_Gx[9];
__constant__ float d_Gy[9];

// Device - Compute energies of image (use RGB)
void computeEnergy_device(uchar3 *inPixels, int width, int height, float *outPixels);

// Kernel compute energies of image
__global__ void computeEnergyKernel(uint8_t *inPixels, int width, int height, int filterWidth, float *outPixels);

// Device - Find the vertical seam
int *findSeam_device(float *energyPixels, int width, int height);

int *findSeam_device2(float *energyPixels, int width, int height);

// Kernel to findSeam
__global__ void compute_min_cost_kernel(float *dp, int *trace, float *minsCost, int width, int row);

// Kernel right if width < blockSize
__global__ void compute_min_cost_kernel_smallWidth(float *dp, int *trace, int width, int height);

// Device - Remove the seam, width & height will be change
void seamCarving_device(uchar3 *inPixels, int &width, int &height, int rWidth, int rHeight, uchar3 *outPixels);



void computeEnergy_device(uchar3 *inPixels, int width, int height, float *outPixels)
{
    // Convert RGB to grayScale
    uint8_t *grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convertRgb2Gray_device(inPixels, width, height, grayPixels);

    GpuTimer timer;
    timer.Start();

    // Edge detect by Sobel
    int filterWidth = 3;
    float Gx[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    // Allocate device memories
    uint8_t *d_inPixels;
    float *d_outPixels;
    size_t nInBytes = width * height * sizeof(uint8_t);
    size_t nOutBytes = width * height * sizeof(float);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    CHECK(cudaMalloc(&d_inPixels, nInBytes));
    CHECK(cudaMalloc(&d_outPixels, nOutBytes));

    // Coppy data to device memories
    CHECK(cudaMemcpy(d_inPixels, grayPixels, nInBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_Gx, &Gx, filterSize));
    CHECK(cudaMemcpyToSymbol(d_Gy, &Gy, filterSize));

    // Set grid size and call kernel
    dim3 blkSize = dim3(32, 32); // Occupancy = 100% on C.C 7.5
    dim3 gridSize((width - 1) / blkSize.x + 1, (height - 1) / blkSize.y + 1);
    size_t s_inPixels_size = (blkSize.x + filterWidth - 1) * (blkSize.y + filterWidth - 1) * sizeof(uint8_t);
    computeEnergyKernel<<<gridSize, blkSize, s_inPixels_size>>>(d_inPixels, width, height, filterWidth, d_outPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Coppy result from device memories
    CHECK(cudaMemcpy(outPixels, d_outPixels, nOutBytes, cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));

    // Free memories
    free(grayPixels);

    timer.Stop();
    float time = timer.Elapsed();
    if (time < deviceTime_min_calcEnergy)
        deviceTime_min_calcEnergy = time;
    if (time > deviceTime_max_calcEnergy)
        deviceTime_max_calcEnergy = time;
}

__global__ void computeEnergyKernel(uint8_t *inPixels, int width, int height, int filterWidth, float *outPixels)
{
    extern __shared__ uint8_t s_data[];
    int inPixelsCornerR = blockIdx.y * blockDim.y - filterWidth / 2;
    int inPixelsCornerC = blockIdx.x * blockDim.x - filterWidth / 2;
    int s_width = blockDim.x + filterWidth - 1;
    int s_height = blockDim.y + filterWidth - 1;
    for (int s_R = threadIdx.y; s_R < s_height; s_R += blockDim.y)
    {
        for (int s_C = threadIdx.x; s_C < s_width; s_C += blockDim.x)
        {
            int inPixelsR = inPixelsCornerR + s_R;
            int inPixelsC = inPixelsCornerC + s_C;
            inPixelsR = min(max(inPixelsR, 0), height - 1);
            inPixelsC = min(max(inPixelsC, 0), width - 1);
            s_data[s_R * s_width + s_C] = inPixels[inPixelsR * width + inPixelsC];
        }
    }
    __syncthreads();

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int i = r * width + c;

    if (r >= height || c >= width)
    {
        return;
    }

    float Gx_sum = 0;
    float Gy_sum = 0;
    for (int fr = 0; fr < filterWidth; fr++)
    {
        for (int fc = 0; fc < filterWidth; fc++)
        {
            int imageR = threadIdx.y + fr;
            int imageC = threadIdx.x + fc;
            imageR = min(height - 1, max(0, imageR));
            imageC = min(width - 1, max(0, imageC));
            float inPixel = s_data[imageR * s_width + imageC];

            Gx_sum += d_Gx[fr * filterWidth + fc] * inPixel;
            Gy_sum += d_Gy[fr * filterWidth + fc] * inPixel;
        }
    }

    float G = abs(Gx_sum) + abs(Gy_sum);
    G = G > 255 ? 255 : G;
    outPixels[i] = G;
}

int *findSeam_device(float *energyPixels, int width, int height)
{
    {
        float *dp = (float *)malloc(width * height * sizeof(float)); // Store computed min energies
        int *trace = (int *)malloc(width * height * sizeof(int));

        // 1. CREATE A MIN COST TABLE
        GpuTimer timer;
        timer.Start();
        {
            // Allocate device memories
            size_t nInBytes = width * height * sizeof(float);
            size_t row_size = width * sizeof(float);

            float *d_minsCost; // Use to store a lastest row
            float *d_dp;       // Use to store dp in device
            int *d_trace;

            CHECK(cudaMalloc(&d_minsCost, row_size));
            CHECK(cudaMalloc(&d_dp, nInBytes));
            CHECK(cudaMalloc(&d_trace, width * height * sizeof(int)));

            // Copy data to device memories
            CHECK(cudaMemcpy(d_dp, energyPixels, nInBytes, cudaMemcpyHostToDevice));

            // Invoke the kernel to compute the min cost table
            int blkSize = MAX_THREADS;
            int gridSize = ((width - 1) / blkSize + 1);

            if (width < blkSize)
            {
                size_t s_mem = width * sizeof(float);
                compute_min_cost_kernel_smallWidth<<<gridSize, blkSize, s_mem>>>(d_dp, d_trace, width, height);
                cudaDeviceSynchronize();
                CHECK(cudaGetLastError());
            }
            else
            {
                CHECK(cudaMemcpy(d_minsCost, energyPixels, row_size, cudaMemcpyHostToDevice)); // Copy row 0 to d_temp
                for (int row = 1; row < height; row++)                                         // calculate minimum cost table row by row
                {
                    compute_min_cost_kernel<<<gridSize, blkSize>>>(d_dp, d_trace, d_minsCost, width, row);
                    cudaDeviceSynchronize();
                    CHECK(cudaGetLastError());
                }
            }

            // Copy resust from device memory
            CHECK(cudaMemcpy(dp, d_dp, width * height * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(trace, d_trace, width * height * sizeof(int), cudaMemcpyDeviceToHost));

            // Free device memories
            CHECK(cudaFree(d_minsCost));
            CHECK(cudaFree(d_dp));
            CHECK(cudaFree(d_trace));
        }

        timer.Stop();
        float time = timer.Elapsed();
        if (time < deviceTime_min_calcMinCost)
            deviceTime_min_calcMinCost = time;
        if (time > deviceTime_max_calcMinCost)
            deviceTime_max_calcMinCost = time;

        // 2. FIND THE MINIMUM VALUE IN THE BOTTOM ROW
        float min_value = dp[(height - 1) * width];
        int min_index = 0;
        for (int c = 1; c < width; c++)
        {
            if (dp[(height - 1) * width + c] < min_value)
            {
                min_value = dp[(height - 1) * width + c];
                min_index = c;
            }
        }

        // 3. CREATE THE SEAM IN REVERSE ORDER
        int *minSeam = (int *)malloc(height * sizeof(int));
        minSeam[0] = min_index;
        int nextTrace = min_index;
        int nc = height - 1;
        for (int r = height - 1; r >= 1; r--)
        {
            minSeam[nc] = trace[r * width + nextTrace];
            nextTrace = minSeam[nc];
            nc--;
        }

        free(dp);
        free(trace);

        return minSeam;
    }
}

int *findSeam_device2(float *energyPixels, int width, int height)
{
    {
        float *dp = (float *)malloc(width * height * sizeof(float)); // Store computed min energies
        int *trace = (int *)malloc(width * height * sizeof(int));

        // 1. CREATE A MIN COST TABLE
        GpuTimer timer;
        timer.Start();
        if (width < MAX_THREADS) // compute by device
        {
            // Allocate device memories
            size_t nInBytes = width * height * sizeof(float);
            size_t row_size = width * sizeof(float);

            float *d_minsCost; // Use to store a lastest row
            float *d_dp;       // Use to store dp in device
            int *d_trace;

            CHECK(cudaMalloc(&d_minsCost, row_size));
            CHECK(cudaMalloc(&d_dp, nInBytes));
            CHECK(cudaMalloc(&d_trace, width * height * sizeof(int)));

            // Copy data to device memories
            CHECK(cudaMemcpy(d_dp, energyPixels, nInBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_minsCost, energyPixels, row_size, cudaMemcpyHostToDevice)); // Copy row 0 to d_temp

            // Invoke the kernel to compute the min cost table
            int blkSize = MAX_THREADS;
            int gridSize = ((width - 1) / blkSize + 1);

            size_t s_mem = width * sizeof(float);
            compute_min_cost_kernel_smallWidth<<<gridSize, blkSize, s_mem>>>(d_dp, d_trace, width, height);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());

            // Copy resust from device memory
            CHECK(cudaMemcpy(dp, d_dp, width * height * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(trace, d_trace, width * height * sizeof(int), cudaMemcpyDeviceToHost));

            // Free device memories
            CHECK(cudaFree(d_minsCost));
            CHECK(cudaFree(d_dp));
            CHECK(cudaFree(d_trace));
        }
        else // compute by host
        {
            memcpy(dp, energyPixels, width * height * sizeof(float));
            int prevStep;

            for (int r = 1; r < height; r++)
                for (int c = 0; c < width; c++)
                {
                    float minVal = dp[(r - 1) * width + c];
                    prevStep = c;
                    float left = (c == 0) ? dp[(r - 1) * width] : dp[(r - 1) * width + c - 1];
                    float right = (c == width - 1) ? dp[(r - 1) * width + c] : dp[(r - 1) * width + c + 1];

                    if (left < minVal)
                    {
                        minVal = left;
                        prevStep--;
                    }
                    if (right < minVal)
                    {
                        minVal = right;
                        prevStep++;
                    }

                    dp[r * width + c] += minVal;
                    trace[r * width + c] = prevStep;
                }
        }

        timer.Stop();
        float time = timer.Elapsed();
        if (time < deviceTime_min_calcMinCost)
            deviceTime_min_calcMinCost = time;
        if (time > deviceTime_max_calcMinCost)
            deviceTime_max_calcMinCost = time;

        // 2. FIND THE MINIMUM VALUE IN THE BOTTOM ROW
        float min_value = dp[(height - 1) * width];
        int min_index = 0;
        for (int c = 1; c < width; c++)
        {
            if (dp[(height - 1) * width + c] < min_value)
            {
                min_value = dp[(height - 1) * width + c];
                min_index = c;
            }
        }

        // 3. CREATE THE SEAM IN REVERSE ORDER
        int *minSeam = (int *)malloc(height * sizeof(int));
        minSeam[0] = min_index;
        int nextTrace = min_index;
        int nc = height - 1;
        for (int r = height - 1; r >= 1; r--)
        {
            minSeam[nc] = trace[r * width + nextTrace];
            nextTrace = minSeam[nc];
            nc--;
        }

        free(dp);
        free(trace);

        return minSeam;
    }
}

__global__ void compute_min_cost_kernel(float *dp, int *trace, float *minsCost, int width, int row)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < width)
    {
        float minVal = dp[(row - 1) * width + c];
        int prevStep = c;
        float left = (c == 0) ? dp[(row - 1) * width] : dp[(row - 1) * width + c - 1];
        float right = (c == width - 1) ? dp[(row - 1) * width + c] : dp[(row - 1) * width + c + 1];

        if (left < minVal)
        {
            minVal = left;
            prevStep--;
        }
        if (right < minVal)
        {
            minVal = right;
            prevStep++;
        }

        minVal += dp[row * width + c];
        dp[row * width + c] = minVal;
        trace[row * width + c] = prevStep;
        minsCost[c] = minVal;
    }
}

__global__ void compute_min_cost_kernel_smallWidth(float *dp, int *trace, int width, int height)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_minsCost[];

    if (c < width)
    {
        s_minsCost[threadIdx.x] = dp[c];
        __syncthreads();

        // Compute minimum cost row by row
        for (int r = 1; r < height; r++)
        {
            float minVal = s_minsCost[threadIdx.x];
            int prevStep = threadIdx.x;

            float left = (c == 0) ? s_minsCost[threadIdx.x] : s_minsCost[threadIdx.x - 1];
            float right = (c == width - 1) ? s_minsCost[threadIdx.x] : s_minsCost[threadIdx.x + 1];

            if (left < minVal)
            {
                minVal = left;
                prevStep--;
            }
            if (right < minVal)
            {
                minVal = right;
                prevStep++;
            }

            minVal += dp[r * width + c];
            dp[r * width + c] = minVal;
            trace[r * width + c] = prevStep;
            s_minsCost[c] = minVal;
            __syncthreads();
        }
    }
}

void seamCarving_device(uchar3 *inPixels, int &width, int &height, int rWidth, int rHeight, uchar3 *outPixels)
{
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));
    float *energyPixels = (float *)malloc(width * height * sizeof(float));

    // Find vertical seams and remove seams
    int *seamWidth = (int *)malloc(height * sizeof(int));
    for (int i = 0; i < rWidth; i++)
    {
        // Compute energies
        computeEnergy_device(outPixels, width, height, energyPixels);

        // Find seam
        seamWidth = findSeam_device(energyPixels, width, height);

        // Remove seam
        removePixels_device(outPixels, width, height, seamWidth, outPixels);
    }
    free(seamWidth);

    // Rotate image 90 degrees to find horizontal seams
    uchar3 *temp = (uchar3 *)malloc(width * height * sizeof(uchar3));
    rotateImage90(outPixels, width, height, temp);

    // Find horizontal seams and remove seams
    int *seamHeight = (int *)malloc(width * sizeof(int));
    for (int i = 0; i < rHeight; i++)
    {
        // Compute energies
        computeEnergy_device(temp, width, height, energyPixels);

        // Find seam
        seamHeight = findSeam_device(energyPixels, width, height);

        // Remove seam
        removePixels_device(temp, width, height, seamHeight, temp);
    }
    free(seamHeight);

    // Rotate image -90 degrees
    rotateImage_90(temp, width, height, outPixels);

    // Free memories
    free(energyPixels);
    free(temp);

    timer.Stop();
    printf("Time by Device: %.3f ms\n", timer.Elapsed());
}

#endif