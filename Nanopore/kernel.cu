#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <bitset>
#include <list>
#include <map>
#include <set>

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/unique.h>
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
                genotype = NULL;
            }
            genotype = (char*)malloc(length * sizeof(char));
            memcpy(genotype, buf, length);
        }

        if (i % 4 == 0) // Set probability
        {
            int length = strlen(genotype);
            for (int x = 0; x < length - K; x++)
            {
                K_MER_NODE k_mer;
                k_mer.value = 0;
                k_mer.K_MER_QUALITY = 0;
                long prob = 0;
                for (int k_len = 0; k_len < K; k_len++)
                {
                    k_mer.value += get_value(genotype[x + k_len]) * pow(4, K - k_len - 1);
                    prob += (int)buf[x + k_len];
                }

                k_mer.K_MER_QUALITY = prob / K;
                K_MER_NODE_LIST.push_back(k_mer);               
            }
            printf("K_MER");
        }

        //free(buf);

    } while (line_size > 0);

    std::list<K_MER_NODE>::iterator it = K_MER_NODE_LIST.begin();
    std::set<long long> setOf_K_Mers;
    int allElements = K_MER_NODE_LIST.size();
    long long* id_of_all_kmers_CPU = (long long*)malloc(sizeof(long long) * allElements);
    int hashTableLength = 0;
    for (int i = 0; i < allElements; i++)
    {
        long long val = (*it).value;
        if (setOf_K_Mers.find(val) == setOf_K_Mers.end()) // TODO: do it with thrust
        {
            hashTableLength++;
            setOf_K_Mers.insert(val);
        }

        id_of_all_kmers_CPU[i] = val;
        std::advance(it, 1);
    }
    setOf_K_Mers.clear();
    K_MER_NODE_LIST.clear();

    long long* id_of_all_kmers_GPU;
    cudaMalloc((void**)&id_of_all_kmers_GPU, sizeof(long long) * allElements);
    cudaMemcpy(id_of_all_kmers_GPU, id_of_all_kmers_CPU, sizeof(long long) * allElements, cudaMemcpyHostToDevice);
    thrust::sort(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + allElements);
    //int* new_end = thrust::unique(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + allElements);
    free(id_of_all_kmers_CPU);

    long long* id_of_kmer_GPU;
    int* amount_of_kmer_GPU;
    cudaMalloc((void**)&id_of_kmer_GPU, sizeof(long long) * hashTableLength);
    cudaMalloc((void**)&amount_of_kmer_GPU, sizeof(int) * hashTableLength);
    thrust::pair<long long*, int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + allElements, thrust::make_constant_iterator(1), id_of_kmer_GPU, amount_of_kmer_GPU);


    cudaFree(id_of_all_kmers_GPU);
    cudaFree(id_of_kmer_GPU);
    cudaFree(amount_of_kmer_GPU);

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
