#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <iostream>
#include <fstream>      // std::filebuf
#include <time.h>

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


#define K 5
#define MAX_SIZE 1024*1024

typedef intptr_t ssize_t;
typedef std::bitset<4> BYTE;

typedef struct K_MER_NODE
{
    long long value;
    short K_MER_QUALITY;

} K_MER_NODE;


__device__ __host__ long long get_value(char c)
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

__global__ void SetKMerValues(K_MER_NODE* out, char *genotype, char* buf, int length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x + 1; i <= length; i += blockDim.x * gridDim.x)
    {
        K_MER_NODE k_mer;
        k_mer.value = 0;
        k_mer.K_MER_QUALITY = 0;
        int prob = 0;
        for (int k_len = 0; k_len < K; k_len++)
        {
            k_mer.value += get_value(genotype[i + k_len]) * pow((float)4,(float) K - k_len - 1);
            prob += (int)buf[i + k_len];
        }
        k_mer.K_MER_QUALITY = prob / K;
        out[i] = k_mer;
    }
}

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

int main()
{
    cudaSetDevice(0);
    std::list<K_MER_NODE> K_MER_NODE_LIST = {};
    K_MER_NODE *K_MER_NODES = NULL;
    int allElements = 0;
    char* buf = (char*)malloc(sizeof(char) * MAX_SIZE);  
    char* genotype = (char*)malloc(MAX_SIZE * sizeof(char));
    std::filebuf f;
    int length = 0;
    if (f.open("G:/chr100mb.fastq", std::ios::binary | std::ios::in))
    {
        std::istream is(&f);
        int i = 0;
        while (is.getline(buf, MAX_SIZE))
        {
            i++;
            if (i % 4 == 2) // Set A, C, T, G as BYTE
            {
                length = strlen(buf);
                memcpy(genotype, buf, length);
            }

            if (i % 4 == 0) // Set probability
            {
                K_MER_NODE *arrGPU, *arr;
                char* bufGPU, *genotypeGPU;
                cudaMalloc((void**)&bufGPU, sizeof(K_MER_NODE) * length);
                cudaMemcpy(bufGPU, buf, sizeof(char) * length, cudaMemcpyHostToDevice);
                cudaMalloc((void**)&genotypeGPU, sizeof(K_MER_NODE) * length);
                cudaMemcpy(genotypeGPU, genotype, sizeof(char) * length, cudaMemcpyHostToDevice);
                cudaMalloc((void**)&arrGPU, sizeof(K_MER_NODE) * (length - K));
                SetKMerValues << <1024, 1024 >> > (arrGPU, genotypeGPU, bufGPU, length - K);
                cudaDeviceSynchronize();
                allElements += length - K;
                if (K_MER_NODES == NULL)
                    K_MER_NODES = (K_MER_NODE*)malloc(sizeof(K_MER_NODE) * (length - K));
                else
                    K_MER_NODES = (K_MER_NODE*)realloc(K_MER_NODES, allElements * sizeof(K_MER_NODE));
                cudaMemcpy(K_MER_NODES + allElements - length + K, arrGPU, sizeof(K_MER_NODE) * (length - K), cudaMemcpyDeviceToHost);
                cudaFree(bufGPU);
                cudaFree(genotypeGPU);
                cudaFree(arrGPU);
                printf("K_MER");
            }
        }
    }
    if (f.is_open()) {
        printf("KK");
        f.close();   // zamkniêcie pliku i zapisanie zmian
    }
    else
    {
        printf("sth wrong");
    }
    free(buf);
    free(genotype);
    std::set<long long> setOf_K_Mers;
    long long* id_of_all_kmers_CPU = (long long*)malloc(sizeof(long long) * allElements);
    int hashTableLength = 0;
    for (int i = 0; i < allElements; i++)
        id_of_all_kmers_CPU[i] = K_MER_NODES[i].value;
    for (int i = 0; i < hashTableLength; i++)
    {
        printf("%lld\n", id_of_all_kmers_CPU[i]);
        printf("in 4:");
        print_in_4(id_of_all_kmers_CPU[i], K);
    }
    free(K_MER_NODES);
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
        print_in_4(h[i], K);
        //printf("%lld\n", h[i]);

    printf("ok3\n");
    cudaFree(id_of_all_kmers_GPU);
    cudaFree(id_of_kmer_GPU);
    cudaFree(amount_of_kmer_GPU);

    return 0;
}
