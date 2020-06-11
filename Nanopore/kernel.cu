#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include<bitset>
#include<list>
//code from https://stackoverflow.com/questions/735126/are-there-alternate-implementations-of-gnu-getline-interface/735472#735472

#define K 5

typedef intptr_t ssize_t;
typedef std::bitset<4> BYTE;

typedef struct K_MER_NODE
{
    BYTE K_MER[K];
    short K_MER_QUALITY;

} K_MER_NODE;

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

BYTE char_to_byte(char c)
{
    BYTE byte;
    if (c == 'A')
        byte = 8;
    else if (c == 'C')
        byte = 4;
    else if (c == 'T')
        byte = 2;
    else if (c == 'G')
        byte = 1;
    else
        byte = 0;

    return byte;
}

int main()
{
    std::list<K_MER_NODE> K_MER_NODE_LIST = {};

    char* buf = NULL;  
    char* genotype = NULL;
    char* probability = NULL;
    size_t bufSize = 10;
    FILE* f;
    size_t line_size;
    f = fopen("G:/chr2.fastq", "r"); 

    int i = 0;
    do
    {
        i++;
        line_size = getline(&buf, &bufSize, f);
        if (i % 4 == 2) // Set A, C, T, G as BYTE
        {
            int length = strlen(buf);
            if (genotype != NULL)
            {
                free(genotype);
            }
            genotype = (char*)malloc(length * sizeof(char));
            memcpy(genotype, buf, length);
        }
        else if (i % 4 == 0) // Set probability
        {
            probability = buf;
        }

        if (i % 4 == 0)
        {
            int length = strlen(genotype);
            for (int x = 0; x < length - K; x++)
            {
                K_MER_NODE k_mer;
                long prob = 0;
                for (int k_len = 0; k_len < K; k_len++)
                {
                    k_mer.K_MER[k_len] = char_to_byte(genotype[x + k_len]);
                    prob += (int)probability[x + k_len];
                }

                k_mer.K_MER_QUALITY = prob / length;
                K_MER_NODE_LIST.push_back(k_mer);
                //printf("%c\n", genotype[x]);
                //printf("%c\n", probability[x]);
                //printf("\n");
                //char char_to_byte
                
            }
            printf("K_MER");
        }

        //printf("4 lines!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            //if (i == 4)
              //  break;
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
