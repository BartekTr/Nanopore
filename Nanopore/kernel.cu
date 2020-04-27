#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
//code from https://stackoverflow.com/questions/735126/are-there-alternate-implementations-of-gnu-getline-interface/735472#735472
typedef intptr_t ssize_t;

ssize_t getline(char** lineptr, size_t* n, FILE* stream) {
    size_t pos;
    int c;

    if (lineptr == NULL || stream == NULL || n == NULL) {
        errno = EINVAL;
        return -1;
    }

    c = getc(stream);
    if (c == EOF) {
        return -1;
    }

    if (*lineptr == NULL) {
        *lineptr = (char*)malloc(128);
        if (*lineptr == NULL) {
            return -1;
        }
        *n = 128;
    }

    pos = 0;
    while (c != EOF) {
        if (pos + 1 >= *n) {
            size_t new_size = *n + (*n >> 2);
            if (new_size < 128) {
                new_size = 128;
            }
            char* new_ptr = (char*)realloc(*lineptr, new_size);
            if (new_ptr == NULL) {
                return -1;
            }
            *n = new_size;
            *lineptr = new_ptr;
        }

        ((unsigned char*)(*lineptr))[pos++] = c;
        if (c == '\n') {
            break;
        }
        c = getc(stream);
    }

    (*lineptr)[pos] = '\0';
    return pos;
}
int main()
{
    char* buf = NULL;   
    size_t bufSize = 10;
    FILE* f;
    size_t line_size;
    f = fopen("D:/Pobrane_D/chr1.fastq", "r");   
    int i = 0;
    do
    {
        i++;
        line_size = getline(&buf, &bufSize, f);
        if(i%4 == 0)
            printf("%s\n", buf);

    } while (line_size > 0);


    if (f != NULL) {
        printf("KK");
        fclose(f);   // zamkniêcie pliku i zapisanie zmian
    }
    else
    {
        printf("sth wrong");
    }

    return 0;
}
