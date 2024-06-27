#include "../include/SlimeFinder.h"
#include "../include/UserIO.h"

typedef void (*launchKernelType)(Data*);

int main(int argc, char **argv) {
	Data data;
	launchKernelType launchKernel;
	
	HMODULE hModule = LoadLibraryA("SlimeFinder.dll");

	if (hModule != NULL) {
		launchKernel = (launchKernelType)GetProcAddress(hModule, "launchKernel");
		if (launchKernel == NULL) exitError(errFuncDLL);
	}
	else {
		exitError(errDLL);
	}

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
