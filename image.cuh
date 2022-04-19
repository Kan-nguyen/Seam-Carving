#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <stdio.h>
#include <stdint.h>

#include "device.cuh"

float hostTime_min_rgb2gray = 10000;
float hostTime_max_rgb2gray = 0;
float hostTime_min_calcEnergy = 10000;
float hostTime_max_calcEnergy = 0;
float hostTime_min_calcMinCost = 10000;
float hostTime_max_calcMinCost = 0;
float hostTime_min_removeSeam = 10000;
float hostTime_max_removeSeam = 0;
#define PI 3.141592653589793238462643383279502884L /* pi */

// Read file .pnm P3 (RGB)
void readPnm(char *fileName, int &width, int &height, uchar3 *&pixels);

// Write file .pnm P3 (RGB)
void writePnm(uchar3 *pixels, int width, int height, char *fileName);

// Str = str1 + str2
char *concatStr(const char *s1, const char *s2);

// Compute difference between 2 image
float computeError(uchar3 *a1, uchar3 *a2, int n);

// Print Error
void printError(uchar3 *deviceResult, uchar3 *hostResult, int width, int height);

// Convert RGB to grayscale
void convertRgb2Gray(uchar3 *inPixels, int width, int height, uint8_t *outPixels);

// Remove seam VERTICAL
void removePixels(uchar3 *inPixels, int &width, int &height, int *seam, uchar3 *outPixels);

// Rotate image 90 degrees, width & height will be swap
void rotateImage90(uchar3 *inPixels, int &width, int &height, uchar3 *outPixels);

// Rotate image -90 degrees, width & height will be swap
void rotateImage_90(uchar3 *inPixels, int &width, int &height, uchar3 *outPixels);

// Bold seam vertical (seam will be red)
void boldSeams(uchar3 *pixels, int width, int height, int *seam);

void readPnm(char *fileName, int &width, int &height, uchar3 *&pixels)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);

    if (strcmp(type, "P3") != 0)
    {
        fclose(f);
        printf("Cannot read %s", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255)
    {
        fclose(f);
        printf("Cannot read %s", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, char *fileName)
{
    FILE *f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P3\n%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height; i++)
        fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

    fclose(f);
}

char *concatStr(const char *s1, const char *s2)
{
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

float computeError(uchar3 *a1, uchar3 *a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

void printError(uchar3 *deviceResult, uchar3 *hostResult, int width, int height)
{
    float err = computeError(deviceResult, hostResult, width * height);
    printf("Error: %f\n", err);
}

void convertRgb2Gray(uchar3 *inPixels, int width, int height, uint8_t *outPixels)
{
    GpuTimer timer;
    timer.Start();
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
    timer.Stop();
    float time = timer.Elapsed();
    if (time < hostTime_min_rgb2gray)
        hostTime_min_rgb2gray = time;
    if (time > hostTime_max_rgb2gray)
        hostTime_max_rgb2gray = time;
}

void removePixels(uchar3 *inPixels, int &width, int &height, int *seam, uchar3 *outPixels)
{
    GpuTimer timer;
    timer.Start();

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width - 1; c++)
        {
            if (c < seam[r])
                outPixels[r * (width - 1) + c] = inPixels[r * width + c];
            else
                outPixels[r * (width - 1) + c] = inPixels[r * width + c + 1];
        }
    }
    width--;

    timer.Stop();
    float time = timer.Elapsed();
    if (time < hostTime_min_removeSeam)
        hostTime_min_removeSeam = time;
    if (time > hostTime_max_removeSeam)
        hostTime_max_removeSeam = time;
}

void rotateImage90(uchar3 *inPixels, int &width, int &height, uchar3 *outPixels)
{
    int temp = width;
    width = height;
    height = temp;
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            outPixels[r * width + c] = inPixels[c * height + (height - 1 - r)];
        }
    }
}

void rotateImage_90(uchar3 *inPixels, int &width, int &height, uchar3 *outPixels)
{
    int temp = width;
    width = height;
    height = temp;
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            outPixels[r * width + c] = inPixels[(width - 1 - c) * height + r];
        }
    }
}

void boldSeams(uchar3 *pixels, int width, int height, int *seam)
{
    for (int r = 0; r < height; r++)
    {
        pixels[r * width + seam[r]] = make_uchar3(255, 0, 0);
    }
}

#endif