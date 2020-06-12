#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include<bitset>
#include<list>
#include<map>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
//code from https://stackoverflow.com/questions/735126/are-there-alternate-implementations-of-gnu-getline-interface/735472#735472

#define K 20

typedef intptr_t ssize_t;
typedef std::bitset<4> BYTE;

typedef struct K_MER_NODE
{
    long long value;
    short K_MER_QUALITY;

} K_MER_NODE;

size_t getline(char** lineptr, size_t* n, FILE* stream) {
    size_t pos;
    int c;

    if (lineptr == NULL || stream == NULL || n == NULL) {
        errno = EINVAL;
        return 0;
    }

    c = getc(stream);
    if (c == EOF) {
        return 0;
    }

    if (*lineptr == NULL) {
        *lineptr = (char*)malloc(128);
        if (*lineptr == NULL) {
            return 0;
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
                return 0;
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

long long get_value(char c)
{
    long long value;
    if (c == 'A')
        value = 0;
    else if (c == 'C')
        value = 1;
    else if (c == 'T')
        value = 2;
    else if (c == 'G')
        value = 3;
    else
        value = 0;

    return value;
}

int main()
{
    std::list<K_MER_NODE> K_MER_NODE_LIST = {};
    //std::map<K_MER_NODE, int> 

    char* buf = NULL;  
    char* genotype = NULL;
    char* probability = NULL;
    size_t bufSize = 10;
    FILE* f;
    size_t line_size;
    f = fopen("G:/chr100mb.fastq", "r"); 

    int i = 0;
    do
    {
        i++;
        line_size = 0;
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
                    k_mer.value += get_value(genotype[x + k_len]) * (int)pow(4, K - k_len - 1);
                    prob += (int)probability[x + k_len];
                }

                K_MER_NODE_LIST.push_back(k_mer);               
            }
            printf("K_MER");
        }
    } while (line_size > 0);

    std::list<K_MER_NODE>::iterator it = K_MER_NODE_LIST.begin();

    /*for (int i = 0; i < allElements; i++)
    {


        std::advance(it, 1);
    }*/

    int allElements = K_MER_NODE_LIST.size();
    int hashTableLength = 10;
    int* id_of_all_kmers_CPU = (int*)malloc(sizeof(int) * allElements);


    int* id_of_all_kmers_GPU;
    cudaMalloc((void**)&id_of_all_kmers_GPU, sizeof(int) * allElements);
    cudaMemcpy(id_of_all_kmers_GPU, id_of_all_kmers_CPU, sizeof(int) * allElements, cudaMemcpyHostToDevice);

    int* id_of_kmer;
    int* amount_of_kmer;
    cudaMalloc((void**)&id_of_kmer, sizeof(int) * hashTableLength);
    cudaMalloc((void**)&amount_of_kmer, sizeof(int) * hashTableLength);
    thrust::pair<int*, int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + allElements, thrust::make_constant_iterator(1), id_of_kmer, amount_of_kmer);


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
