#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define SEED_RANGE(x) (x < -281474976710656 ? 0 : (x > 281474976710655 ? 0 : 1))

typedef long long int I64;
typedef long long unsigned U64;

typedef struct ChunkStruct {
    int x, z, isSlime;
} Chunk;

typedef struct DataStruct {
    int mode;
    I64 startSeed, endSeed;
    U64 totalSeeds;
    int rx, rz, rw, rh;
    int pw, ph, pl;
    Chunk pattern[32];
    int frequency;
} Data;

enum ErrorCode {
    errArgumentCount,
    errSeedNegativeRange,
    errSeedOutOfRange,
    errNotInteger,
    errInvalidMode,
    errPatternSize,
    errInvalidPattern,
    errInvalidFrequency,
    errNegativeInt,
    errCudaError,
    errUnexpectedError,
    errDLL,
    errFuncDLL
};

enum FinderMode {
    modePattern,
    modeFrequency,
    modeBenchmark
};

bool isStringInt(char *string);
I64 stringToInt(char *string);
bool stringEquate(char *a, char *b);
int stringLength(char *string);
bool parsePattern(Data *data, char *pattern);
bool parseFrequency(Data *data, char *frequency);
void exitError(int errorCode);

__device__ int isSlimeChunk(I64 seed, int x, int z);
__global__ void deviceTask(Data *data);

inline void __cudaSafeCall(cudaError_t cError, const char *file, const int line);
void launchKernel(Data *data);

int main(int argc, char **argv) {
	Data data;

    if (argc == 9) {
        if (stringEquate(argv[1], "pattern")) {
            data.mode = modePattern;
        }
        else if (stringEquate(argv[1], "frequency")) {
            data.mode = modeFrequency;
        }
        else {
            exitError(errInvalidMode);
        }

        for (int i = 2; i < argc - 1; i++) {
            if (!isStringInt(argv[i])) {
                exitError(errNotInteger);
            }
        }

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
        if (stringEquate(argv[1], "help")) {
            printf("SlimeFinder.exe help\n");
            printf("SlimeFinder.exe benchmark <total-seeds>\n");
            printf("SlimeFinder.exe pattern <start-seed> <end-seed> <rx> <rz> <rw> <rh> <pattern>\n");
            printf("SlimeFinder.exe frequency <start-seed> <end-seed> <rx> <rz> <rw> <rh> <frequency.srw.srh>\n");

            return 0;
        }
        else {
            exitError(errInvalidMode);
        }
    }
    else if (argc == 3) {
        if (stringEquate(argv[1], "benchmark")) {
            if (!isStringInt(argv[2])) exitError(errNotInteger);
            I64 totalSeeds = stringToInt(argv[2]);
            if (totalSeeds <= 0) exitError(errNegativeInt);

            srand(time(nullptr));

            data.mode = modeBenchmark;
            data.startSeed = rand();
            data.endSeed = data.startSeed + totalSeeds;
            data.totalSeeds = (U64)totalSeeds;
            data.rx = -250;
            data.rz = -250;
            data.rw = 500;
            data.rh = 500;
            data.pl = 0;

            int patternIndex = 0;

            for (int z = 0; z < 4; z++) {
                for (int x = 0; x < 4; x++) {
                    if ((x + z) % 2 == 0) {
                        data.pattern[patternIndex++] = { x, z, 1 };
                        data.pl++;
                    }
                }
            }
        }
        else {
            exitError(errInvalidMode);
        }
    }
    else {
        exitError(errArgumentCount);
    }

    launchKernel(&data);

    return 0;
}

bool isStringInt(char *string) {
    const char validChars[11] = "0123456789";
    int i = -1;

    while (string[++i] != '\0') {
        char c = string[i];
        bool valid = false;

        for (int j = 0; j < 11; j++) {
            if (c == validChars[j] || (i == 0 && c == '-')) {
                valid = true;
                break;
            }
        }

        if (!valid) return false;
    }

    return true;
}

I64 stringToInt(char *string) {
    const int negative = string[0] == '-';
    int length = stringLength(string);

    I64 n = 0;

    for (int i = negative ? 1 : 0; i < length; i++) {
        n += (I64)(pow(10, length - 1 - i) * (string[i] - '0'));
    }

    return negative ? -n : n;
}

bool stringEquate(char *a, char *b) {
    int aLen = stringLength(a);
    int bLen = stringLength(b);

    if (aLen != bLen) return false;

    for (int i = 0; i < aLen; i++) {
        if (a[i] != b[i]) return false;
    }

    return true;
}

int stringLength(char *string) {
    int length = -1;
    while (string[++length] != '\0');
    return length;
}

bool parsePattern(Data *data, char *pattern) {
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
        if (stringLength(token) != patternWidth) return false;
        patternHeight++;
        token = strtok(NULL, ".");
    }

    data->pw = patternWidth;
    data->ph = patternHeight;
    data->pl = 0;

    int *patternLinear = (int *)malloc((patternWidth * patternHeight) * sizeof(int));
    
    if (patternLinear == NULL) return false;

    int patterLinearIndex = 0;

    for (int i = 0; i < patternLength; i++) {
        if (pattern[i] == '0' || pattern[i] == '1' || pattern[i] == '2') {
            patternLinear[patterLinearIndex++] = pattern[i] - '0';
        }
    }

    for (int z = 0; z < patternHeight; z++) {
        for (int x = 0; x < patternWidth; x++) {
            int patternValue = patternLinear[x + z * patternWidth];

            if (patternValue != 0 && patternValue != 1 && patternValue != 2) return false;
                   
            if (patternValue == 0 || patternValue == 1) {
                data->pattern[patternIndex++] = {x, z, patternValue};
                data->pl++;
            }
        }
    }

    free(patternLinear);

    return true;
}

bool parseFrequency(Data *data, char *frequency) {
    int f, pw, ph;
    int index = 0;

    char *token = strtok(frequency, ".");

    while (token != NULL) {
        switch (index) {
        case 0:
            if (!isStringInt(token)) return false;
            f = (int)stringToInt(token);
            if (f <= 0) return false;
            break;
        case 1:
            if (!isStringInt(token)) return false;
            pw = (int)stringToInt(token);
            if (pw <= 0) return false;
            break;
        case 2:
            if (!isStringInt(token)) return false;
            ph = (int)stringToInt(token);
            if (ph <= 0) return false;
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
    case errNegativeInt:
        printf("(%d) Error! Integer must be positive\n", errorCode);
        break;
    case errPatternSize:
        printf("(%d) Error! Pattern must fit in the region\n", errorCode);
        break;
    case errDLL:
        printf("(%d) Error! DLL could not be loaded\n", errorCode);
        break;
    case errFuncDLL:
        printf("(%d) Error! DLL function could not be loaded\n", errorCode);
        break;
    default:
        printf("(?) Error! Unexpected error\n");
        break;
    }

    exit(EXIT_FAILURE);
}

__device__ int isSlimeChunk(I64 seed, int x, int z) {
    seed += (I64)(x * x * 0x4c1906);
    seed += (I64)(x * 0x5ac0db);
    seed += (I64)(z * z) * 0x4307a7L;
    seed += (I64)(z * 0x5f24f);
    seed ^= 0x3ad8025fL;
    seed ^= 0x5deece66dL;
    seed &= 0xffffffffffff;
    return (((seed * 0x5deece66dL + 0xbL) & 0xffffffffffff) >> 17) % 10 == 0;
}

__global__ void deviceTask(Data *data) {
    int64_t seed = blockIdx.x * blockDim.x + threadIdx.x + data->startSeed;
    int x = blockIdx.y * blockDim.y + threadIdx.y + data->rx;
    int z = blockIdx.z * blockDim.z + threadIdx.z + data->rz;

    int validPattern, frequency;

    if (seed <= data->endSeed && x < data->rx + data->rw - 1 && z < data->rz + data->rh - 1) {
        switch (data->mode) {
        case modePattern:
            validPattern = 1;

            for (int i = 0; i < data->pl; i++) {
                Chunk p = data->pattern[i];
                if (isSlimeChunk(seed, x + p.x, z + p.z) != p.isSlime) {
                    validPattern = 0;
                    break;
                };
            }

            if (validPattern) {
                printf("        (+) Found seed -> %lld at (%d, %d) / (%d, %d)\a\n", seed, x, z, x << 4, z << 4);
            }

            break;
        case modeFrequency:
            frequency = 0;

            for (int px = 0; px < data->pw; px++) {
                for (int pz = 0; pz < data->ph; pz++) {
                    if (isSlimeChunk(seed, x + px, z + pz)) frequency++;
                }
            }

            if (frequency >= data->frequency) {
                printf("        (+) Found seed -> %lld at (%d, %d) / (%d, %d) with frequency %d\a\n", seed, x, z, x << 4, z << 4, frequency);
            }

            break;
        case modeBenchmark:
            validPattern = 1;

            for (int i = 0; i < data->pl; i++) {
                Chunk p = data->pattern[i];
                if (isSlimeChunk(seed, x + p.x, z + p.z) != p.isSlime) {
                    validPattern = 0;
                    break;
                }
            }

            break;
        }
    }
}

inline void __cudaSafeCall(cudaError_t cError, const char *file, const int line) {
    if (cError != cudaSuccess) {
        printf("(%d) Error! '%s' CUDA error %d\n", errCudaError, cudaGetErrorName(cError), (int)cError);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void launchKernel(Data *data) {
    int deviceId;
    cudaDeviceProp prop;

    cudaSafeCall(cudaGetDevice(&deviceId));
    cudaSafeCall(cudaGetDeviceProperties(&prop, deviceId));

    const I64 startSeed = data->startSeed;
    const I64 endSeed = data->endSeed;
    const int xStart = data->rx;
    const int zStart = data->rz;
    const int xRange = data->rw;
    const int zRange = data->rh;

    int tpb = 1 << (int)log2(pow(prop.maxThreadsPerBlock, 0.3333333333333333f));

    dim3 threadsPerBlock(tpb, tpb, tpb);

    U64 numBlocksSeed = ((endSeed - startSeed) + threadsPerBlock.x - 1) / threadsPerBlock.x;
    U64 numBlocksX = (xRange + threadsPerBlock.y - 1) / threadsPerBlock.y;
    U64 numBlocksZ = (zRange + threadsPerBlock.z - 1) / threadsPerBlock.z;

    U64 seedMaxBlocks = prop.maxGridSize[0] >> 4;
    U64 seedRemainder = numBlocksSeed % seedMaxBlocks;
    int seedTotalChunks = (int)((numBlocksSeed - seedRemainder) / seedMaxBlocks);

    U64 seedRange = endSeed - startSeed;
    U64 seedRangeRemainder = seedRange % (seedMaxBlocks * tpb);
    U64 seedsPerChunk = (seedRange - seedRangeRemainder) / (seedTotalChunks + 1);

    U64 xMaxBlocks = prop.maxGridSize[1] >> 4;
    U64 xRemainder = numBlocksX % xMaxBlocks;
    int xTotalChunks = (int)((numBlocksX - xRemainder) / xMaxBlocks);

    U64 xRangeRemainder = xRange % (xMaxBlocks * tpb);
    U64 xPerChunk = xRange / (xTotalChunks + 1);

    U64 zMaxBlocks = prop.maxGridSize[2] >> 4;
    U64 zRemainder = numBlocksZ % zMaxBlocks;
    int zTotalChunks = (int)((numBlocksZ - zRemainder) / zMaxBlocks);

    U64 zRangeRemainder = zRange % (zMaxBlocks * tpb);
    U64 zPerChunk = zRange / (zTotalChunks + 1);

    Data *dataDevice;
    cudaEvent_t startEvent, stopEvent;

    if (data->mode == modeBenchmark) {
        cudaSafeCall(cudaEventCreate(&startEvent));
        cudaSafeCall(cudaEventCreate(&stopEvent));
    }
    else {
        printf("(?) Device      | %s\n", prop.name);
        printf("(?) Mode        | %s\n", data->mode == modePattern ? "Pattern" : "Frequency");
        printf("(?) Seed range  | %lld to %lld\n", data->startSeed, data->endSeed);
        printf("(?) Total chunks| (%d, %d, %d)\n", seedTotalChunks + 1, xTotalChunks + 1, zTotalChunks + 1);
    }

    for (int sc = 0; sc <= seedTotalChunks; sc++) {
        for (int xc = 0; xc <= xTotalChunks; xc++) {
            for (int zc = 0; zc <= zTotalChunks; zc++) {
                if (data->mode != modeBenchmark) {
                    printf("    (!) Computing data chunk (%d, %d, %d)\n", sc, xc, zc);
                }
                else {
                    printf("(!) Benchmarking...\n");
                }

                dim3 numBlocks(sc == seedTotalChunks ? seedRemainder : seedMaxBlocks, xc == xTotalChunks ? xRemainder : xMaxBlocks, zc == zTotalChunks ? zRemainder : zMaxBlocks);

                data->startSeed = (sc * seedsPerChunk) + startSeed;
                data->endSeed = data->startSeed + seedsPerChunk + (sc == seedTotalChunks ? seedRangeRemainder : 0);

                data->rx = (xc * xPerChunk) + xStart;
                data->rz = (zc * zPerChunk) + zStart;
                data->rw = (xc * xPerChunk) + (xc == xTotalChunks ? xRangeRemainder : xPerChunk);
                data->rh = (zc * zPerChunk) + (zc == zTotalChunks ? zRangeRemainder : zPerChunk);

                cudaSafeCall(cudaMalloc((void **)&dataDevice, sizeof(Data)));
                cudaSafeCall(cudaMemcpy(dataDevice, data, sizeof(Data), cudaMemcpyHostToDevice));

                if (data->mode == modeBenchmark) cudaSafeCall(cudaEventRecord(startEvent));

                deviceTask <<< numBlocks, threadsPerBlock >>> (dataDevice);

                if (data->mode == modeBenchmark) {
                    cudaSafeCall(cudaEventRecord(stopEvent));
                    cudaSafeCall(cudaEventSynchronize(stopEvent));

                    float timeTaken = 0;

                    cudaSafeCall(cudaEventElapsedTime(&timeTaken, startEvent, stopEvent));

                    uint64_t seedRate = (uint64_t)(data->totalSeeds * 247009000) / (uint64_t)timeTaken;

                    printf("(?) Benchmark took %f ms which is approximately %llu pattern checks per second\n", timeTaken, seedRate);
                    printf("    (?) The pattern has dimensions of 4x4 with 50%% of chunks being slime chunks\n");
                    printf("    (?) A region of (-250, -250, 500, 500) was checked with %llu seeds\n", data->totalSeeds);
                    printf("    (?) So a region of (-50, -50, 100, 100) could check %llu seeds instead\n", seedRate / 9409);
                    printf("        with the same pattern in the same amount of time (roughly speaking)\n");
                    printf("\n    Slime chunks | Avg. Time (min)\n");

                    for (int i = 1; i <= 20; i++) {
                        float t = pow(10, i) / ((float)seedRate * 60.0f);
                        if (t <= 30000.0f) {
                            printf("         %-7d |    %f\n", i, t);
                        }
                        else {
                            printf("         %-7d |    Long...\n", i);
                        }
                    }
                }
                else {
                    cudaSafeCall(cudaDeviceSynchronize());
                }

                cudaSafeCall(cudaFree(dataDevice));
            }
        }
    }

    cudaDeviceReset();
}
