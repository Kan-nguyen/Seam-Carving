#ifndef _SEAMCARVER_H_
#define _SEAMCARVER_H_

#include "image.cuh"

// Compute energies of image (use RGB)
void computeEnergy(uchar3 *inPixels, int width, int height, float *outPixels);

// Find the vertical seam
int *findSeam(float *energyPixels, int width, int height);

// Remove the seam, width & height will be change
void seamCarving(uchar3 *inPixels, int &width, int &height, int rWidth, int rHeight, uchar3 *outPixels);

void computeEnergy(uchar3 *inPixels, int width, int height, float *outPixels)
{
    // Convert RGB to grayScale
    uint8_t *grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convertRgb2Gray(inPixels, width, height, grayPixels);

    GpuTimer timer;
    timer.Start();
    // Edge detect by Sobel
    int filterWidth = 3;
    float Gx[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    float Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    float Gx_sum = 0;
    float Gy_sum = 0;
    float G = 0;
    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
    {
        for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
        {
            for (int filterR = 0; filterR < filterWidth; filterR++)
            {
                for (int filterC = 0; filterC < filterWidth; filterC++)
                {
                    int inPixelsR = outPixelsR - filterWidth / 2 + filterR;
                    int inPixelsC = outPixelsC - filterWidth / 2 + filterC;
                    inPixelsR = min(max(0, inPixelsR), height - 1);
                    inPixelsC = min(max(0, inPixelsC), width - 1);
                    Gx_sum += Gx[filterR][filterC] * grayPixels[inPixelsR * width + inPixelsC];
                    Gy_sum += Gy[filterR][filterC] * grayPixels[inPixelsR * width + inPixelsC];
                }
            }
            G = abs(Gx_sum) + abs(Gy_sum);
            G = G > 255 ? 255 : G;
            outPixels[outPixelsR * width + outPixelsC] = G;
            Gx_sum = 0;
            Gy_sum = 0;
        }
    }

    // Free memories
    free(grayPixels);

    timer.Stop();
    float time = timer.Elapsed();
    if (time < hostTime_min_calcEnergy)
        hostTime_min_calcEnergy = time;
    if (time > hostTime_max_calcEnergy)
        hostTime_max_calcEnergy = time;
}

int *findSeam(float *energyPixels, int width, int height)
{
    float *dp = (float *)malloc(width * height * sizeof(float));

    // 1. CREATE A MIN COST TABLE
    GpuTimer timer;
    timer.Start();
    memcpy(dp, energyPixels, width * height * sizeof(float));
    int *trace = (int *)malloc(width * height * sizeof(int));
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

    timer.Stop();
    float time = timer.Elapsed();
    if (time < hostTime_min_calcMinCost)
        hostTime_min_calcMinCost = time;
    if (time > hostTime_max_calcMinCost)
        hostTime_max_calcMinCost = time;

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

void seamCarving(uchar3 *inPixels, int &width, int &height, int rWidth, int rHeight, uchar3 *outPixels)
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
        computeEnergy(outPixels, width, height, energyPixels);

        // Find seam
        seamWidth = findSeam(energyPixels, width, height);

        // Remove seam
        removePixels(outPixels, width, height, seamWidth, outPixels);
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
        computeEnergy(temp, width, height, energyPixels);

        // Find seam
        seamHeight = findSeam(energyPixels, width, height);

        // Remove seam
        removePixels(temp, width, height, seamHeight, temp);
    }
    free(seamHeight);

    // Rotate image -90 degrees
    rotateImage_90(temp, width, height, outPixels);

    // Free memories
    free(energyPixels);
    free(temp);

    timer.Stop();
    printf("Time by Host: %.3f ms\n", timer.Elapsed());
}

#endif