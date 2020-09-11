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


// TODO For testing
#include <algorithm>


#define K 31
#define MAX_SIZE 1024*1024
#define MIN_K_MER_QUALITY 60
#define MAX_K_MERS_TO_ALLOCATE 500000000

typedef intptr_t ssize_t;

typedef struct K_MER_NODE
{
    unsigned long long value;
    short K_MER_QUALITY;

} K_MER_NODE;

// Changing DNA code to numbers
__device__ __host__ unsigned long long get_value(char c)
{
    unsigned long long value;
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

// Decoding K-mers from numbers to special code
__global__ void SetKMerValues(char* genotype, char* buf, int length, unsigned long long* id_of_good_kmers_GPU, int* goodQualityElementsGpu)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x)
    {
        unsigned long long decodedValue = 0;
        int quality = 32767;
        for (int k_len = 0; k_len < K; k_len++)
        {
            decodedValue += get_value(genotype[i + k_len]) * (unsigned long long)pow((float)4, (float)K - k_len - 1);

            // Setting K-mer quality as minimum value of single reading
            if (quality > (int)buf[i + k_len])
            {
                quality = (int)buf[i + k_len];
            }
        }

        // Using only K-Mers with good quality
        if (quality > MIN_K_MER_QUALITY)
        {
            int index = atomicAdd(goodQualityElementsGpu, 1);
            id_of_good_kmers_GPU[index] = decodedValue;
        }
    }
}

// Reading encoded K-mer
void print_in_4(unsigned long long value, int k)
{
    for (int i = k-1; i >= 0; --i)
        printf("%lld", (value >> (2 * i)) % 4);
    printf("\n");
}

struct last_mer
{
    const unsigned long long a;

    last_mer(unsigned long long _a) : a(_a) {}

    __host__ __device__
        unsigned long long operator()(const unsigned long long &x) const {
        return x % a;
    }
};

struct first_mer
{
    __host__ __device__
        unsigned long long operator()(const unsigned long long& x) const {
        return x >> 2;
    }
};

int main()
{
    cudaSetDevice(0);

    // All K-mers
    int allElements = 0;
    
    // Temporary data for reading file
    std::filebuf f;
    int length = 0;
    char* buf = (char*)malloc(sizeof(char) * MAX_SIZE);
    char* genotype = (char*)malloc(sizeof(char) * MAX_SIZE);
    char* bufGPU, * genotypeGPU;
    cudaMalloc((void**)&bufGPU, sizeof(char) * MAX_SIZE);
    cudaMalloc((void**)&genotypeGPU, sizeof(char) * MAX_SIZE);

    // Data for processing file
    int elementsWithEnoughQuality = 0;
    int* goodQualityElementsGpu;
    cudaMalloc((void**)&goodQualityElementsGpu, sizeof(int));
    cudaMemcpy(goodQualityElementsGpu, &elementsWithEnoughQuality, sizeof(int), cudaMemcpyHostToDevice);

    // K-Mers with enough quality
    unsigned long long* id_of_all_kmers_GPU;
    cudaMalloc((void**)&id_of_all_kmers_GPU, sizeof(unsigned long long) * MAX_K_MERS_TO_ALLOCATE);

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
                // Copying currently readed data to Device
                cudaMemcpy(bufGPU, buf, sizeof(char) * length, cudaMemcpyHostToDevice);
                cudaMemcpy(genotypeGPU, genotype, sizeof(char) * length, cudaMemcpyHostToDevice);

                // Processing readed data into K-mers with own way
                SetKMerValues << <(length - K)/256 + 1, 256 >> > (genotypeGPU, bufGPU, length - K, id_of_all_kmers_GPU, goodQualityElementsGpu);
                cudaDeviceSynchronize();

                allElements += length - K;
                printf("K_MER");
            }
        }
    }

    // Delete data for reading file
    free(buf);
    free(genotype);
    cudaFree(bufGPU);
    cudaFree(genotypeGPU);

    // File close and error handling
    if (f.is_open()) {
        printf("File closed\n");
        f.close();
    }
    else
    {
        printf("Error during file closing\n");
    }

    // Readed data summary
    cudaMemcpy(&elementsWithEnoughQuality, goodQualityElementsGpu, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n All K-MERS: %d", allElements);
    printf("\n K-MERS with enough quality: %d", elementsWithEnoughQuality);
    printf("ok1\n");
    
    //Sorting K-mers
    thrust::sort(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + elementsWithEnoughQuality);
    
    // hashTableLengthv2 - amount of different K-mers
    unsigned long long* hashTableLengthv2;
    cudaMalloc((void**)&hashTableLengthv2, sizeof(unsigned long long) * elementsWithEnoughQuality);
    unsigned long long* new_end_for_unique = thrust::unique_copy(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + elementsWithEnoughQuality, hashTableLengthv2);
    int hashTableLength = new_end_for_unique - hashTableLengthv2;
    cudaFree(hashTableLengthv2);

    // Getting amount of different K-mers
    unsigned long long* id_of_kmer_GPU;
    int* amount_of_kmer_GPU;
    cudaMalloc((void**)&id_of_kmer_GPU, sizeof(unsigned long long) * hashTableLength);
    cudaMalloc((void**)&amount_of_kmer_GPU, sizeof(int) * hashTableLength);
    thrust::pair<unsigned long long*, int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, id_of_all_kmers_GPU, id_of_all_kmers_GPU + elementsWithEnoughQuality, thrust::make_constant_iterator(1), id_of_kmer_GPU, amount_of_kmer_GPU);

    unsigned long long to_mod = pow(4, K - 1);
    //C array,  weights = amount_of_kmer
    unsigned long long* C;
    cudaMalloc((void**)&C, sizeof(unsigned long long) * hashTableLength);
    thrust::transform(thrust::device, id_of_kmer_GPU, id_of_kmer_GPU + hashTableLength, C, last_mer(to_mod));
    
    //to do R array, transform id_of_kmer_GPU, reduce_by_key and transform again:
    unsigned long long* temp;
    cudaMalloc((void**)&temp, sizeof(unsigned long long) * hashTableLength);
    thrust::transform(thrust::device, id_of_kmer_GPU, id_of_kmer_GPU + hashTableLength, temp, first_mer());

    unsigned long long* first;
    int* second;
    cudaMalloc((void**)&first, sizeof(unsigned long long) * hashTableLength);
    cudaMalloc((void**)&second, sizeof(int) * hashTableLength);
    thrust::pair<unsigned long long*, int*> end;
    end = thrust::reduce_by_key(thrust::device, temp, temp + hashTableLength, thrust::make_constant_iterator(1), first, second);
    
    unsigned long long* a = (unsigned long long*)malloc(sizeof(unsigned long long) * hashTableLength);
    unsigned long long* b = (unsigned long long*)malloc(sizeof(unsigned long long) * hashTableLength);
    cudaMemcpy(a, second, sizeof(unsigned long long)* hashTableLength, cudaMemcpyDeviceToHost);
    b[0] = 0;
    for (int i = 1; i < hashTableLength; i++)
        b[i] = b[i - 1] + a[i - 1];

    //first kmer : h_data[i] >> 2, last kmer: h_data[i] % to_mod
    unsigned long long* h = (unsigned long long*)malloc(sizeof(unsigned long long) * hashTableLength);
    cudaMemcpy(h, C, sizeof(unsigned long long) * hashTableLength, cudaMemcpyDeviceToHost);
    //C = edges array on GPU
    //h = edges array on CPU
    //b = array of starting positions
    for (int i = 0; i < hashTableLength; i++)
        print_in_4(h[i], K);

    printf("ok3\n");
    free(a);
    free(b);
    free(h);
    cudaFree(temp);
    cudaFree(first);
    cudaFree(second);
    cudaFree(C);
    cudaFree(id_of_all_kmers_GPU);
    cudaFree(id_of_kmer_GPU);
    cudaFree(amount_of_kmer_GPU);
    cudaFree(goodQualityElementsGpu);

    return 0;
}
