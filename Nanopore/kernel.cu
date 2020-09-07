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
#include <thrust/device_vector.h>

//code from https://stackoverflow.com/questions/735126/are-there-alternate-implementations-of-gnu-getline-interface/735472#735472

#define K 5

typedef intptr_t ssize_t;
typedef std::bitset<4> BYTE;

typedef struct K_MER_NODE
{
    long long value;
    short K_MER_QUALITY;

} K_MER_NODE;


void print_in_4(long long value, int k)
{
    for (int i = k-1; i >= 0; --i)
        printf("%lld", (value >> (2 * i)) % 4);
    printf("\n");
}

struct last_mer
{
    const long long a;

    last_mer(long long _a) : a(_a) {}

    __host__ __device__
        long long operator()(const long long &x) const {
        return x % a;
    }
};

struct first_mer
{
    __host__ __device__
        long long operator()(const long long& x) const {
        return x >> 2;
    }
};



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
    size_t bufSize = 0;
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

    } while (line_size > 0);

    if (f != NULL) {
        printf("KK");
        fclose(f);   // zamkniêcie pliku i zapisanie zmian
    }
    else
    {
        printf("sth wrong");
    }
    free(buf);
    std::list<K_MER_NODE>::iterator it = K_MER_NODE_LIST.begin();
    std::set<long long> setOf_K_Mers;
    int allElements = K_MER_NODE_LIST.size();
    long long* id_of_all_kmers_CPU = (long long*)malloc(sizeof(long long) * allElements);
    int hashTableLength = 0;
    for (int i = 0; i < allElements; i++)
    {
        long long val = (*it).value;
        //if (setOf_K_Mers.find(val) == setOf_K_Mers.end()) // TODO: do it with thrust
        //{
        //    hashTableLength++;
        //    setOf_K_Mers.insert(val);
        //}

        id_of_all_kmers_CPU[i] = val;
        std::advance(it, 1);
    }
    for (int i = 0; i < 1000; i++)
        printf("%lld\n", id_of_all_kmers_CPU[i]);

    setOf_K_Mers.clear();
    K_MER_NODE_LIST.clear();
    printf("ok1\n");
    long long* id_of_all_kmers_GPU;
    cudaMalloc((void**)&id_of_all_kmers_GPU, sizeof(long long) * allElements);
    cudaMemcpy(id_of_all_kmers_GPU, id_of_all_kmers_CPU, sizeof(long long) * allElements, cudaMemcpyHostToDevice);
    thrust::sort(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + allElements);
    
    //TODO: check if 199-203 code (hashTableLength) is same as hashTableLengthv2
    long long* hashTableLengthv2;
    cudaMalloc((void**)&hashTableLengthv2, sizeof(long long) * allElements);
    long long* new_end_for_unique = thrust::unique_copy(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + allElements, hashTableLengthv2);
    hashTableLength = new_end_for_unique - hashTableLengthv2;
    cudaFree(hashTableLengthv2);
    //END
    free(id_of_all_kmers_CPU);
    printf("ok2\n");
    long long* id_of_kmer_GPU;
    int* amount_of_kmer_GPU;
    cudaMalloc((void**)&id_of_kmer_GPU, sizeof(long long) * hashTableLength);
    cudaMalloc((void**)&amount_of_kmer_GPU, sizeof(int) * hashTableLength);
    thrust::pair<long long*, int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + allElements, thrust::make_constant_iterator(1), id_of_kmer_GPU, amount_of_kmer_GPU);

    long long to_mod = pow(4, K - 1);
    //C array,  weights = amount_of_kmer
    long long* C;
    cudaMalloc((void**)&C, sizeof(long long) * hashTableLength);
    thrust::transform(thrust::device, id_of_kmer_GPU, id_of_kmer_GPU + hashTableLength, C, last_mer(to_mod));
    
    //to do R array, transform id_of_kmer_GPU, reduce_by_key and transform again:
    long long* temp;
    cudaMalloc((void**)&temp, sizeof(long long) * hashTableLength);
    thrust::transform(thrust::device, id_of_kmer_GPU, id_of_kmer_GPU + hashTableLength, temp, first_mer());

    long long* first;
    int* second;
    cudaMalloc((void**)&first, sizeof(long long) * hashTableLength);
    cudaMalloc((void**)&second, sizeof(int) * hashTableLength);
    thrust::pair<long long*, int*> end;
    end = thrust::reduce_by_key(thrust::device, temp, temp + hashTableLength, thrust::make_constant_iterator(1), first, second);
    
    long long* a = (long long*)malloc(sizeof(long long) * hashTableLength);
    long long* b = (long long*)malloc(sizeof(long long) * hashTableLength);
    cudaMemcpy(a, second, sizeof(long long)* hashTableLength, cudaMemcpyDeviceToHost);
    b[0] = 0;
    for (int i = 1; i < hashTableLength; i++)
        b[i] = b[i - 1] + a[i - 1];

    //first kmer : h_data[i] >> 2, last kmer: h_data[i] % to_mod
    long long* h = (long long*)malloc(sizeof(long long) * hashTableLength);
    cudaMemcpy(h, C, sizeof(long long) * hashTableLength, cudaMemcpyDeviceToHost);
    for (int i = 0; i < hashTableLength; i++)
        printf("%lld\n", h[i]);

    printf("ok3\n");
    cudaFree(id_of_all_kmers_GPU);
    cudaFree(id_of_kmer_GPU);
    cudaFree(amount_of_kmer_GPU);


    return 0;
}
