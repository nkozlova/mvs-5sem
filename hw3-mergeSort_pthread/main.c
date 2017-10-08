#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "omp_realization.h"


typedef struct someArgs_tag {
    int** arr;
    int* arr1;
    int* arr2;
    int it1;
    int it2;
    int s1;
    int s2;
} someArgs_t;

typedef struct someParams_tag {
    int** arr1;
    int* arr2;
    int s1;
    int s2;
    int p;
} someParams_t;

int compare(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

void* just_merge(void* args);
void* merge(void* params);
void mergeSort(int* array, int size, int step, int p);
void writeResults(int* arr, int* sort_arr, double time, int n, int m, int p);


int main(int argc, char* argv[]) {
    if (argc != 4) {
        exit(1);
    }
    srand(time(NULL));

    int n = atoi(argv[1]), m = atoi(argv[2]);
    int p = atoi(argv[3]);

    int* a = (int*)calloc(n, sizeof(int));
    int* b = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 10000;
        b[i] = a[i];
    }

    mergeSort(a, n, m, p);
    mergeSortOMP(b, n, m, p);

    free(a);
    free(b);
    return 0;
}


void* just_merge(void* args) {
    someArgs_t* arg = (someArgs_t*) args;
    while (arg->it1 < arg->s1 && arg->it2 < arg->s2) {
        int i1 = arg->it1, i2 = arg->it2;
        while (arg->it1 < arg->s1 && arg->it2 < arg->s2 && arg->arr1[arg->it1] <= arg->arr2[arg->it2]) {
            arg->it1++;
        }
        memcpy((*(arg->arr)) + i1 + i2, arg->arr1 + i1, (arg->it1 - i1) * sizeof(int));
        while (arg->it1 < arg->s1 && arg->it2 < arg->s2 && arg->arr1[arg->it1] > arg->arr2[arg->it2]) {
            arg->it2++;
        }
        memcpy((*(arg->arr)) + arg->it1 + i2, arg->arr2 + i2, (arg->it2 - i2) * sizeof(int));
    }
    if (arg->it1 < arg->s1) {
        memcpy((*(arg->arr)) + arg->it1 + arg->it2, arg->arr1 + arg->it1, (arg->s1 - arg->it1) * sizeof(int));
    } else {
        memcpy((*(arg->arr)) + arg->it1 + arg->it2, arg->arr2 + arg->it2, (arg->s2 - arg->it2) * sizeof(int));
    }
}

void* merge(void* params) {
    someParams_t* param = (someParams_t*) params;

    int mid2 = param->s2 / 2;

    int* arr = (int*)calloc(param->s1 + param->s2, sizeof(int));

    int left = 0;
    int right = param->s1;
    int mid1;
    while (left <= right) {
        mid1 = (left + right) / 2;
        if (param->arr2[mid2] <= (*(param->arr1))[mid1] && (mid1 == 0 || (*(param->arr1))[mid1 - 1] <= param->arr2[mid2])) {
            break;
        }
        if (param->arr2[mid2] > (*(param->arr1))[mid1]) {
            left = mid1 + 1;
        } else {
            right = mid1 - 1;
        }
    }
    arr[mid2 + mid1] = param->arr2[mid2];

    someArgs_t args[2];
    for (int i = 0; i < 2; i++) {
        args[i].arr = &arr;
        args[i].arr1 = *(param->arr1);
        args[i].arr2 = param->arr2;
        if (i == 0) {
            args[i].it1 = 0;
            args[i].it2 = 0;
            args[i].s1 = mid1;
            args[i].s2 = mid2;
        } else {
            args[i].it1 = mid1;
            args[i].it2 = mid2;
            args[i].s1 = param->s1;
            args[i].s2 = param->s2;
        }
    }

    if (param->p == 1) {
        just_merge((void *) &args[0]);
        just_merge((void *) &args[1]);
    } else {
        pthread_t threads[2];
        for (int i = 0; i < 2; i++) {
            pthread_create(&threads[i], NULL, just_merge, (void *) &args[i]);
        }
        for (int i = 0; i < 2; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    memcpy(*(param->arr1), arr, (param->s1 + param->s2) * sizeof(int));
    free(arr);
}

void mergeSort(int* array, int size, int step, int p) {

    int ts1 = clock();
    int num_chunk = (size + step - 1) / step;

    int** result = (int**)calloc(num_chunk, sizeof(int*));
    int* sizes = (int*)calloc(num_chunk, sizeof(int));

    for (int i = 0; i < num_chunk; i++) {
        if (i < num_chunk - 1) {
            sizes[i] = step;
        } else {
            sizes[i] = size - step * i;
        }

        result[i] = (int*)calloc(size, sizeof(int));
        memcpy(result[i], array + i * step, sizes[i] * sizeof(int));
        qsort(result[i], sizes[i], sizeof(int), compare);
    }

    pthread_t threads[p];
    for (int j = 1; j < num_chunk; j *= 2) {
        int k = (num_chunk + j - 1) / (2 * j);
        someParams_t param[k];
        for (int i = 0; i < k; i++) {
            int r1 = 2 * j * i;
            int r2 = r1 + j;
            param[i].arr1 = &result[r1];
            param[i].arr2 = result[r2];
            param[i].s1 = sizes[r1];
            param[i].s2 = sizes[r2];
            param[i].p = 1;
            sizes[r1] += sizes[r2];
            sizes[r2] = 0;
        }

        if (p == 1) {
            for (int i = 0; i < k; i++) {
                merge((void*)&param[i]);
            }
        } else {
            for (int i = 0; i < k; i += p) {
                for (int l = 0; l < p; l++) {
                    if (i + l < k) {
                        pthread_create(&threads[l], NULL, merge, (void *) &param[i + l]);
                    }
                }
                for (int l = 0; l < p; l++) {
                    if (i + l < k) {
                        pthread_join(threads[l], NULL);
                    }
                }
            }
        }

        int i = 2 * k * j;
        if (i < num_chunk) {
            param[0].arr1 = &result[0];
            param[0].arr2 = result[i];
            param[0].s1 = sizes[0];
            param[0].s2 = sizes[i];
            param[0].p = p;
            sizes[0] += sizes[i];
            sizes[i] = 0;
            merge((void*)&param[0]);
        }
    }
    int ts2 = clock();

    writeResults(array, result[0], ((double)(ts2 - ts1)) / CLOCKS_PER_SEC, size, step, p);

    for (int i = 0; i < num_chunk; i++) {
        free(result[i]);
    }
    free(result);
    free(sizes);
}

void writeResults(int* arr, int* sort_arr, double time, int n, int m, int p) {
    FILE* file_stats;
    FILE* file_data;
    file_stats = fopen("stats.txt", "a");
    file_data = fopen("data.txt", "w");

    for (int i = 0; i < n; i++) {
        fprintf(file_data, "%d ", arr[i]);
    }
    fprintf(file_data, "\n");

    for (int i = 0; i < n; i++) {
        fprintf(file_data, "%d ", sort_arr[i]);
    }
    fprintf(file_data, "\n");

    fprintf(file_stats, "%fs %d %d %d\n", time, n, m, p);

    int ts1 = clock();
    qsort(arr, n, sizeof(int), compare);
    int ts2 = clock();
    fprintf(file_stats, "%fs\n", ((double)(ts2 - ts1)) / CLOCKS_PER_SEC);

    fclose(file_data);
    fclose(file_stats);
}
