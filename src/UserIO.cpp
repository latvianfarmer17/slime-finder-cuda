#include "../include/UserIO.h"

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
