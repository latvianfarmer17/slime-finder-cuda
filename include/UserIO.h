#ifndef USER_IO_H
#define USER_IO_H

#include "SlimeFinder.h"

#define SEED_RANGE(x) (x < -281474976710656 ? 0 : (x > 281474976710655 ? 0 : 1))

bool isStringInt(char *string);
I64 stringToInt(char *string);
bool stringEquate(char *a, char *b);
int stringLength(char *string);
bool parsePattern(Data *data, char *pattern);
bool parseFrequency(Data *data, char *frequency);
void exitError(int errorCode);

#endif
