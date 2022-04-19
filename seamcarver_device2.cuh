#ifndef _SEAMCARVER2_DEVICE_H_
#define _SEAMCARVER2_DEVICE_H_

#include "seamcarver_device.cuh"

// Device - Remove the seam, width & height will be change
void seamCarving_device2(uchar3 *inPixels, int &width, int &height, int rWidth, int rHeight, uchar3 *outPixels);

void seamCarving_allInOne(uchar3 *d_pixels, int &width, int &height)
{
    GpuTimer timer2;

    // Allocate device memories
    uint8_t *d_grayPixels;
    float *d_energyPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_energyPixels, width * height * sizeof(float)));

    // 1. CONVERT RGB TO GRAYSCALE
    timer2.Start();

    dim3 blkSize(32, 32);
    dim3 gridSize((width - 1) / blkSize.x + 1, (height - 1) / blkSize.y + 1);
    convertRgb2GrayKernel<<<gridSize, blkSize>>>(d_pixels, width, height, d_grayPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    timer2.Stop();
    float time = timer2.Elapsed();
    if (time < deviceTime_min_rgb2gray)
        deviceTime_min_rgb2gray = time;
    if (time > deviceTime_max_rgb2gray)
        deviceTime_max_rgb2gray = time;

    // 2. COMPUTE ENERGIES
    timer2.Start();

    int filterWidth = 3;
    float Gx[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    // Coppy data to device memories
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    CHECK(cudaMemcpyToSymbol(d_Gx, &Gx, filterSize));
    CHECK(cudaMemcpyToSymbol(d_Gy, &Gy, filterSize));

    // Set grid size and call kernel
    blkSize = dim3(32, 32);
    gridSize = dim3((width - 1) / blkSize.x + 1, (height - 1) / blkSize.y + 1);
    size_t s_inPixels_size = (blkSize.x + filterWidth - 1) * (blkSize.y + filterWidth - 1) * sizeof(uint8_t);
    computeEnergyKernel<<<gridSize, blkSize, s_inPixels_size>>>(d_grayPixels, width, height, filterWidth, d_energyPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Free device memories
    CHECK(cudaFree(d_grayPixels));

    timer2.Stop();
    time = timer2.Elapsed();
    if (time < deviceTime_min_calcEnergy)
        deviceTime_min_calcEnergy = time;
    if (time > deviceTime_max_calcEnergy)
        deviceTime_max_calcEnergy = time;

    // 3. FIND SEAM
    timer2.Start();

    // Allocate device memories
    float *d_minsCost;
    int *d_trace;
    size_t row_size = width * sizeof(float);

    CHECK(cudaMalloc(&d_minsCost, width * sizeof(float)));
    CHECK(cudaMalloc(&d_trace, width * height * sizeof(int)));

    // Invoke the kernel to compute the min cost table
    blkSize = dim3(MAX_THREADS);
    gridSize = dim3((width - 1) / blkSize.x + 1);

    if (width < blkSize.x)
    {
        size_t s_mem = width * sizeof(float);
        compute_min_cost_kernel_smallWidth<<<gridSize, blkSize, s_mem>>>(d_energyPixels, d_trace, width, height);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
    }
    else
    {
        // Copy data to device memories
        CHECK(cudaMemcpy(d_minsCost, d_energyPixels, row_size, cudaMemcpyDeviceToDevice));
        for (int row = 1; row < height; row++) // calculate minimum cost table row by row
        {
            compute_min_cost_kernel<<<gridSize, blkSize>>>(d_energyPixels, d_trace, d_minsCost, width, row);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
        }
    }
    float *dp = (float *)malloc(width * height * sizeof(float)); // Store computed min energies
    int *trace = (int *)malloc(width * height * sizeof(int));
    CHECK(cudaMemcpy(dp, d_energyPixels, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(trace, d_trace, width * height * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_minsCost));
    CHECK(cudaFree(d_energyPixels));
    CHECK(cudaFree(d_trace));

    timer2.Stop();
    time = timer2.Elapsed();
    if (time < deviceTime_min_calcMinCost)
        deviceTime_min_calcMinCost = time;
    if (time > deviceTime_max_calcMinCost)
        deviceTime_max_calcMinCost = time;

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

    // 4. REMOVE SEAM
    timer2.Start();

    // Alocate device memories
    uchar3 *d_outPixels;
    int *d_seam;
    CHECK(cudaMalloc(&d_outPixels, width * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_seam, height * sizeof(int)));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_seam, minSeam, height * sizeof(int), cudaMemcpyHostToDevice));

    // Set grid size and call kernel
    blkSize = dim3(32, 32);
    gridSize = dim3((width - 2) / blkSize.x + 1,
                    (height - 1) / blkSize.y + 1);
    int tWidth = width, tHeight = height;
    removePixelsKernel<<<gridSize, blkSize>>>(d_pixels, tWidth, tHeight, d_seam, d_outPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(d_pixels, d_outPixels, (width - 1) * height * sizeof(uchar3), cudaMemcpyDeviceToDevice));

    // Free device memories
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_seam));

    width--;

    timer2.Stop();
    time = timer2.Elapsed();
    if (time < deviceTime_min_removeSeam)
        deviceTime_min_removeSeam = time;
    if (time > deviceTime_max_removeSeam)
        deviceTime_max_removeSeam = time;
}

void seamCarving_device2(uchar3 *inPixels, int &width, int &height, int rWidth, int rHeight, uchar3 *outPixels)
{
    GpuTimer timer;
    timer.Start();

    uchar3 *d_pixels;
    CHECK(cudaMalloc(&d_pixels, width * height * sizeof(uchar3)));
    CHECK(cudaMemcpy(d_pixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Find vertical seams and remove seams
    for (int i = 0; i < rWidth; i++)
    {
        seamCarving_allInOne(d_pixels, width, height);
    }

    CHECK(cudaMemcpy(outPixels, d_pixels, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    // Rotate image 90 degrees to find horizontal seams
    uchar3 *temp = (uchar3 *)malloc(width * height * sizeof(uchar3));
    rotateImage90(outPixels, width, height, temp);

    CHECK(cudaMemcpy(d_pixels, temp, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Find horizontal seams and remove seams
    for (int i = 0; i < rHeight; i++)
    {
        seamCarving_allInOne(d_pixels, width, height);
    }

    CHECK(cudaMemcpy(temp, d_pixels, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    // Rotate image -90 degrees
    rotateImage_90(temp, width, height, outPixels);

    // Free memories
    free(temp);

    timer.Stop();
    printf("Time by Device: %.3f ms\n", timer.Elapsed());
}

#endif