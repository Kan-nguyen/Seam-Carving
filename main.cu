// #include "seamcarver.cuh"
// #include "seamcarver_device.cuh"
#include "seamcarver_device2.cuh"

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5 && argc != 7)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    // Read input image file
    int width, height;
    uchar3 *inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);

    // Read number width and height be removed
    int rWidth = atoi(argv[3]);
    int rHeight = 0;
    if (argc == 5)
        rHeight = atoi(argv[4]);

    // Seam Carving not using device
    uchar3 *correctOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    int hostWidth = width, hostHeight = height;
    printf("---------------------------------------------------------\n");
    printf("Seam Carving by Host\n");
    seamCarving(inPixels, hostWidth, hostHeight, rWidth, rHeight, correctOutPixels);
    // Print min max time of function by host

    printf("Convert rgb to gray time (%s): min: %f ms, max: %f\n", "use host", hostTime_min_rgb2gray, hostTime_max_rgb2gray);
    printf("Calculate energy map time (%s): min: %f ms, max: %f\n", "use host", hostTime_min_calcEnergy, hostTime_max_calcEnergy);
    printf("Compute cumulative energy map(min cost map) time (%s): min: %f ms, max: %f\n", "use host", hostTime_min_calcMinCost, hostTime_max_calcMinCost);
    printf("Remove seam time (%s): min: %f ms, max: %f\n", "use host", hostTime_min_removeSeam, hostTime_max_removeSeam);
    printf("---------------------------------------------------------\n");

    // Seam Carving using device, kernel 1
    uchar3 *outPixels1 = (uchar3 *)malloc(width * height * sizeof(uchar3));
    int deviceWidth = width, deviceHeight = height;

    printf("---------------------------------------------------------\n");
    printf("Seam Carving by Device\n");

    seamCarving_device(inPixels, deviceWidth, deviceHeight, rWidth, rHeight, outPixels1);
    printError(outPixels1, correctOutPixels, deviceWidth, deviceHeight);

    printf("Convert rgb to gray time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_rgb2gray, deviceTime_max_rgb2gray);
    printf("Calculate energy map time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_calcEnergy, deviceTime_max_calcEnergy);
    printf("Compute cumulative energy map(min cost map) time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_calcMinCost, deviceTime_max_calcMinCost);
    printf("Remove seam time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_removeSeam, deviceTime_max_removeSeam);
    printf("---------------------------------------------------------\n");

    deviceTime_min_rgb2gray = 10000;
    deviceTime_max_rgb2gray = 0;
    deviceTime_min_calcEnergy = 10000;
    deviceTime_max_calcEnergy = 0;
    deviceTime_min_calcMinCost = 10000;
    deviceTime_max_calcMinCost = 0;
    deviceTime_min_removeSeam = 10000;
    deviceTime_max_removeSeam = 0;

    // Seam Carving using device, kernel 2
    uchar3 *outPixels2 = (uchar3 *)malloc(width * height * sizeof(uchar3));
    int deviceWidth2 = width, deviceHeight2 = height;

    printf("---------------------------------------------------------\n");
    printf("Seam Carving by Device with function all in one\n");

    seamCarving_device2(inPixels, deviceWidth2, deviceHeight2, rWidth, rHeight, outPixels2);
    printError(outPixels2, correctOutPixels, deviceWidth2, deviceHeight2);

    printf("Convert rgb to gray time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_rgb2gray, deviceTime_max_rgb2gray);
    printf("Calculate energy map time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_calcEnergy, deviceTime_max_calcEnergy);
    printf("Compute cumulative energy map(min cost map) time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_calcMinCost, deviceTime_max_calcMinCost);
    printf("Remove seam time (%s): min: %f ms, max: %f\n", "use device", deviceTime_min_removeSeam, deviceTime_max_removeSeam);
    printf("---------------------------------------------------------\n");

    // Write results to files
    printf("Image resized (width x height): %i x %i\n\n", hostWidth, hostHeight);
    char *outFileNameBase = strtok(argv[2], ".");
    writePnm(correctOutPixels, hostWidth, hostHeight, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(outPixels1, deviceWidth, deviceHeight, concatStr(outFileNameBase, "_device1.pnm"));
    writePnm(outPixels2, deviceWidth2, deviceHeight2, concatStr(outFileNameBase, "_device2.pnm"));

    // Free memories
    free(inPixels);
    free(correctOutPixels);
    free(outFileNameBase);
    free(outPixels1);
    free(outPixels2);
}