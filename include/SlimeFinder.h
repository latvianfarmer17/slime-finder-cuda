#ifndef SLIME_FINDER_H
#define SLIME_FINDER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <windows.h>

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

#endif
