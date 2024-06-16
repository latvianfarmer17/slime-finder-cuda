// NOTE: I tried to make header files to nicely structure the project but I just could not figure out how to link CUDA and C/C++ code...
//       So enjoy going through this if you ever need to (or want to)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>

#define SEED_RANGE(x) (x < -281474976710656 ? 0 : (x > 281474976710655 ? 0 : 1))
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

typedef struct vec2PatternStruct {
    int x, z, type;
} vec2Pattern;

typedef struct sfDataStruct {
    int mode;
    int64_t startSeed, endSeed;
    int64_t worldSeed;
    int rx, rz, rw, rh;
    int pw, ph, pl;
    vec2Pattern pattern[32];
    int frequency;
} sfData;

enum ErrorCode {
    errArgumentCount,
    errSeedNegativeRange,
    errSeedOutOfRange,
    errNotInteger,
    errInvalidMode,
    errPatternSize,
    errInvalidPattern,
    errInvalidFrequency,
    errCudaError,
    errUnexpectedError
};

enum FinderMode {
    modePattern,
    modeFrequency
};

int isStringInt(char *string);
int64_t stringToInt(char *string);
int stringEquate(char *a, char *b);
int stringLength(char *string);
int parsePattern(sfData *data, char *pattern);
int parseFrequency(sfData *data, char *frequency);
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line);
void exitError(int errorCode);
void launchKernel(sfData *data);

__device__ int isSlimeChunk(int64_t seed, int x, int z);
__device__ int checkPattern(sfData *data, int64_t seed, int xOff, int zOff);
__device__ int checkFrequency(sfData *data, int64_t seed, int xOff, int zOff);
__global__ void deviceTask(sfData *data);

// Code which emulates Minecraft's slime chunk determination
__device__ int isSlimeChunk(int64_t seed, int x, int z) {
    seed += (int64_t)(x * x * 0x4c1906);
    seed += (int64_t)(x * 0x5ac0db);
    seed += (int64_t)(z * z) * 0x4307a7L;
    seed += (int64_t)(z * 0x5f24f);
    seed ^= 0x3ad8025fL;

    seed ^= 0x5deece66dL;
    seed &= 0xffffffffffff;

    int bits, val;

    do {
        seed = (seed * 0x5deece66dL + 0xbL) & 0xffffffffffff;
        bits = (int)(seed >> 17);
        val = bits % 10;
    } while (bits - val + 9 < 0);

    return val == 0;
}

// Checks if the given pattern is valid for the world seed and the current region co-ordinate offset
__device__ int checkPattern(sfData *data, int64_t seed, int xOff, int zOff) {
    for (int i = 0; i < data->pl; i++) {
        vec2Pattern p = data->pattern[i];
        if (isSlimeChunk(seed, xOff + p.x, zOff + p.z) != p.type) return 0;
    }

    return 1;
}

// Counts the total slime chunks within a subset of the region
__device__ int checkFrequency(sfData *data, int64_t seed, int xOff, int zOff) {
    int chunksFound = 0;

    for (int x = 0; x < data->pw; x++) {
        for (int z = 0; z < data->ph; z++) {
            if (isSlimeChunk(seed, xOff + x, zOff + z)) chunksFound++;
        }
    }

    return chunksFound;
}

// 'Main' kernel function, evaluates the slime finder mode and looks for slime chunks accordingly
__global__ void deviceTask(sfData *data) {
    int64_t seed = blockIdx.x * blockDim.x + threadIdx.x + data->startSeed;
    int x = blockIdx.y * blockDim.y + threadIdx.y + data->rx;
    int z = blockIdx.z * blockDim.z + threadIdx.z + data->rz;

    if (seed <= data->endSeed && x < data->rx + data->rw - 1 && z < data->rz + data->rh - 1) {
        switch (data->mode) {
        case modePattern:
            if (checkPattern(data, seed, x, z)) {
                printf("    (+) Found seed -> %lld at (%d, %d) / (%d, %d)\a\n", seed, x, z, x << 4, z << 4);
            }

            break;
        case modeFrequency:
            int frequency = checkFrequency(data, seed, x, z);

            if (frequency >= data->frequency) {
                printf("    (+) Found seed -> %lld at (%d, %d) / (%d, %d) with frequency %d\a\n", seed, x, z, x << 4, z << 4, frequency);
            }

            break;
        }
    }
}

int main(int argc, char **argv) {
    sfData data;

    if (argc == 9) {
        // Set the slime finder mode
        if (stringEquate(argv[1], "pattern")) {
            data.mode = modePattern;
        }
        else if (stringEquate(argv[1], "frequency")) {
            data.mode = modeFrequency;
        }
        else {
            exitError(errInvalidMode);
        }

        // Check all arguments are integers (except the last one and first two)
        for (int i = 2; i < argc - 1; i++) {
            if (!isStringInt(argv[i])) {
                exitError(errNotInteger);
            }
        }

        // Populate the slime finder data
        data.startSeed = stringToInt(argv[2]);
        data.endSeed = stringToInt(argv[3]);
        data.rx = (int)stringToInt(argv[4]);
        data.rz = (int)stringToInt(argv[5]);
        data.rw = (int)stringToInt(argv[6]);
        data.rh = (int)stringToInt(argv[7]);

        if (data.startSeed > data.endSeed) exitError(errSeedNegativeRange);
        if (!SEED_RANGE(data.startSeed) || !SEED_RANGE(data.endSeed)) exitError(errSeedOutOfRange);

        if (data.mode == modeFrequency) {
            if (!parseFrequency(&data, argv[8])) exitError(errInvalidFrequency);
        }
        else {
            if (!parsePattern(&data, argv[8])) exitError(errInvalidPattern);
            if (data.rw < data.pw || data.rh < data.ph) exitError(errPatternSize);
        }
    }
    else if (argc == 2) {
        const char helpString[5] = "help";
        int length = stringLength(argv[1]);

        if (length != 4) exitError(errInvalidMode);

        for (int i = 0; i < 4; i++) {
            if (helpString[i] != argv[1][i]) exitError(errInvalidMode);
        }

        printf("\nSlimeFinder.exe <mode=pattern>   <start-seed> <end-seed> <rx> <rz> <rw> <rh> <pattern>\n");
        printf("SlimeFinder.exe <mode=frequency> <start-seed> <end-seed> <rx> <rz> <rw> <rh> <frequency.srw.srh>\n\n");

        return 0;
    }
    else {
        exitError(errArgumentCount);
    }

    launchKernel(&data);

    cudaDeviceReset();

    return 0;
}

// Check if a string is in a valid integer format
int isStringInt(char *string) {
    const char validChars[11] = "0123456789";
    int i = -1;

    while (string[++i] != '\0') {
        char c = string[i];
        int valid = 0;

        for (int j = 0; j < 11; j++) {
            if (c == validChars[j] || (i == 0 && c == '-')) {
                valid = 1;
                break;
            }
        }

        if (!valid) return 0;
    }

    return 1;
}

// Convert a string to an integer
int64_t stringToInt(char *string) {
    const int negative = string[0] == '-';
    int length = stringLength(string);

    int64_t n = 0;

    for (int i = negative ? 1 : 0; i < length; i++) {
        n += (int64_t)(pow(10, length - 1 - i) * (string[i] - '0'));
    }

    return negative ? -n : n;
}

// Compare if strings a and b are the same
int stringEquate(char *a, char *b) {
    int aLen = stringLength(a);
    int bLen = stringLength(b);

    if (aLen != bLen) return 0;

    for (int i = 0; i < aLen; i++) {
        if (a[i] != b[i]) return 0;
    }

    return 1;
}

// Compute the string length
int stringLength(char *string) {
    int length = -1;
    while (string[++length] != '\0');
    return length;
}

// Process the user pattern input
int parsePattern(sfData *data, char *pattern) {
    int patternLength = stringLength(pattern);
    int patternWidth = 0;
    int patternHeight = 0;
    int patternIndex = 0;

    for (int i = 0; i < patternLength; i++) {
        if (pattern[i] == '.') {
            break;
        }
        else {
            patternWidth++;
        }
    }

    char *token = strtok(pattern, ".");

    while (token != NULL) {
        if (stringLength(token) != patternWidth) return 0;
        patternHeight++;
        token = strtok(NULL, ".");
    }

    data->pw = patternWidth;
    data->ph = patternHeight;
    data->pl = 0;

    for (int z = 0; z < patternHeight; z++) {
        for (int x = 0; x < patternWidth; x++) {
            int patternValue = pattern[x + z * patternHeight + z] - '0';

            if (patternValue != 0 && patternValue != 1 && patternValue != 2) return 0;
            
            if (patternValue == 0 || patternValue == 1) {
                data->pattern[patternIndex++] = {x, z, patternValue};
                data->pl++;
            }
        }
    }

    return 1;
}

// Process the user frequency input
int parseFrequency(sfData *data, char *frequency) {
    int f, pw, ph;
    int index = 0;

    char *token = strtok(frequency, ".");

    while (token != NULL) {
        switch (index) {
        case 0:
            if (!isStringInt(token)) return 0;
            f = (int)stringToInt(token);
            break;
        case 1:
            if (!isStringInt(token)) return 0;
            pw = (int)stringToInt(token);
            break;
        case 2:
            if (!isStringInt(token)) return 0;
            ph = (int)stringToInt(token);
            break;
        }

        index++;
        token = strtok(NULL, ".");
    }

    data->frequency = f;
    data->pw = pw;
    data->ph = ph;

    return f <= pw * ph;
}

// Processes any CUDA function and evaluates the returned error (if any)
inline void __cudaSafeCall(cudaError_t cError, const char *file, const int line) {
    if (cError != cudaSuccess) {
        printf("(%d) Error! '%s' CUDA error %d\n", errCudaError, cudaGetErrorName(cError), (int)cError);
        cudaDeviceReset();
        exit(errCudaError);
    }
}

// Exit the program with a specific error message
void exitError(int errorCode) {
    switch (errorCode) {
    case errArgumentCount:
        printf("(%d) Error! Invalid argument count. Try using 'SlimeFinder.exe help'\n", errorCode);
        break;
    case errSeedNegativeRange:
        printf("(%d) Error! Start seed must be smaller than the end seed\n", errorCode);
        break;
    case errSeedOutOfRange:
        printf("(%d) Error! Seed must be between -281,474,976,710,656 and 281,474,976,710,655\n", errorCode);
        break;
    case errNotInteger:
        printf("(%d) Error! Argument must be an integer\n", errorCode);
        break;
    case errInvalidMode:
        printf("(%d) Error! Invalid mode selected. Try using 'SlimeFinder.exe help'\n", errorCode);
        break;
    case errInvalidPattern:
        printf("(%d) Error! Invalid pattern\n", errorCode);
        break;
    case errInvalidFrequency:
        printf("(%d) Error! Invalid frequency\n", errorCode);
        break;
    case errPatternSize:
        printf("(%d) Error! Pattern must fit in the region\n", errorCode);
        break;
    default:
        printf("(?) Error! Unexpected error\n");
        exit(errUnexpectedError);
    }

    cudaDeviceReset();

    exit(errorCode);
}

// Launches the kernel
void launchKernel(sfData *data) {
    int deviceId;
    cudaDeviceProp prop;
    
    cudaSafeCall(cudaGetDevice(&deviceId));
    cudaSafeCall(cudaGetDeviceProperties(&prop, deviceId));

    const int64_t startSeed = data->startSeed;
    const int64_t endSeed = data->endSeed;
    const int xStart = data->rx;
    const int zStart = data->rz;
    const int xRange = data->rw;
    const int zRange = data->rh;

    // Calculate the highest amount of threads per block possible
    int tpb = (int)pow(prop.maxThreadsPerBlock, 0.3333333333333333f);
    
    dim3 threadsPerBlock(tpb, tpb, tpb);

    // Calculate the number of blocks for each grid dimension
    uint64_t numBlocksSeed = ((endSeed - startSeed + 1) + threadsPerBlock.x - 1) / threadsPerBlock.x;
    uint64_t numBlocksX = (xRange + threadsPerBlock.y - 1) / threadsPerBlock.y;
    uint64_t numBlocksZ = (zRange + threadsPerBlock.z - 1) / threadsPerBlock.z;
    
    // Calculate the total amount of chunks and ranges
    //  Seed
    uint64_t seedMaxBlocks = prop.maxGridSize[0];
    uint64_t seedRemainder = numBlocksSeed % seedMaxBlocks;
    int seedTotalChunks = (int)((numBlocksSeed - seedRemainder) / seedMaxBlocks);
    
    uint64_t seedRange = endSeed - startSeed;
    uint64_t seedRangeRemainder = seedRange % (seedMaxBlocks * tpb);
    uint64_t seedsPerChunk = (seedRange - seedRangeRemainder) / (seedTotalChunks + 1);

    //  X
    uint64_t xMaxBlocks = prop.maxGridSize[1];
    uint64_t xRemainder = numBlocksX % xMaxBlocks;
    int xTotalChunks = (int)((numBlocksX - xRemainder) / xMaxBlocks);

    uint64_t xRangeRemainder = xRange % (xMaxBlocks * tpb);
    uint64_t xPerChunk = (xRange - xRangeRemainder) / (xMaxBlocks + 1);

    //  Z
    uint64_t zMaxBlocks = prop.maxGridSize[2];
    uint64_t zRemainder = numBlocksZ % zMaxBlocks;
    int zTotalChunks = (int)((numBlocksZ - zRemainder) / zMaxBlocks);

    uint64_t zRangeRemainder = zRange % (zMaxBlocks * tpb);
    uint64_t zPerChunk = (zRange - zRangeRemainder) / (zMaxBlocks + 1);

    // Distribute the task across "data chunks" if there is enough data to process
    sfData *dataDevice;

    for (int sc = 0; sc <= seedTotalChunks; sc++) {
        for (int xc = 0; xc <= xTotalChunks; xc++) {
            for (int zc = 0; zc <= zTotalChunks; zc++) {
                printf("(?) Computing data chunk %d|%d|%d\n", sc, xc, zc);

                dim3 numBlocks(sc == seedTotalChunks ? seedRemainder : seedMaxBlocks, xc == xTotalChunks ? xRemainder : xMaxBlocks, zc == zTotalChunks ? zRemainder : zMaxBlocks);
                
                data->startSeed = (sc * seedsPerChunk) + startSeed;
                data->endSeed = data->startSeed + seedsPerChunk + (sc == seedTotalChunks ? seedRangeRemainder : 0);

                data->rx = (xc * xPerChunk) + xStart;
                data->rz = (zc * zPerChunk) + zStart;
                data->rw = (xc * xPerChunk) + (xc == xTotalChunks ? xRangeRemainder : 0);
                data->rh = (zc * zPerChunk) + (zc == zTotalChunks ? zRangeRemainder : 0);

                cudaSafeCall(cudaMalloc((void **)&dataDevice, sizeof(sfData)));
                cudaSafeCall(cudaMemcpy(dataDevice, data, sizeof(sfData), cudaMemcpyHostToDevice));

                deviceTask<<< numBlocks, threadsPerBlock >>>(dataDevice);
                
                cudaSafeCall(cudaGetLastError());
                cudaSafeCall(cudaDeviceSynchronize());

                cudaSafeCall(cudaFree(dataDevice));
            }
        }
    }

    cudaDeviceReset();
}