#include "../include/SlimeFinder.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cmath>

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

extern "C" {

__declspec(dllexport) __device__ int isSlimeChunk(I64 seed, int x, int z) {
    seed += (I64)(x * x * 0x4c1906);
    seed += (I64)(x * 0x5ac0db);
    seed += (I64)(z * z) * 0x4307a7L;
    seed += (I64)(z * 0x5f24f);
    seed ^= 0x3ad8025fL;
    seed ^= 0x5deece66dL;
    seed &= 0xffffffffffff;
    return (((seed * 0x5deece66dL + 0xbL) & 0xffffffffffff) >> 17) % 10 == 0;
}

__declspec(dllexport) __global__ void deviceTask(Data *data) {
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

__declspec(dllexport) inline void __cudaSafeCall(cudaError_t cError, const char *file, const int line) {
    if (cError != cudaSuccess) {
        printf("(%d) Error! '%s' CUDA error %d\n", errCudaError, cudaGetErrorName(cError), (int)cError);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__declspec(dllexport) void launchKernel(Data *data) {
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

}
