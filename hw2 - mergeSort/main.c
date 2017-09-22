#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>


int compare(const void* a, const void* b) {
    return *(int*)a >= *(int*)b;
}

void just_merge(int** arr, int* arr1, int* arr2, int it1, int it2, int s1, int s2);
void merge(int** arr1, int* arr2, int s1, int s2, int p);
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
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 10000;
    }

    mergeSort(a, n, m, p);

    free(a);
    return 0;
}


void just_merge(int** arr, int* arr1, int* arr2, int i1, int i2, int s1, int s2) {
    int it1 = i1, it2 = i2;
    while (it1 < s1 && it2 < s2) {
        if (arr1[it1] < arr2[it2]) {
            (*arr)[it1 + it2] = arr1[it1];
            it1 += 1;
        } else {
            (*arr)[it1 + it2] = arr2[it2];
            it2 += 1;
        }
    }
    while (it1 < s1) {
        (*arr)[it1 + it2] = arr1[it1];
        it1 += 1;
    }
    while (it2 < s2) {
        (*arr)[it1 + it2] = arr2[it2];
        it2 += 1;
    }
}

void merge(int** arr1, int* arr2, int s1, int s2, int p) {
    omp_set_num_threads(p);
    int mid2 = s2 / 2;

    int* arr = (int*)calloc(s1 + s2, sizeof(int));

    int left = 0;
    int right = s1;
    int mid1;
    while (left <= right) {
        mid1 = (left + right) / 2;
        if (arr2[mid2] <= (*arr1)[mid1] && (mid1 == 0 || (*arr1)[mid1 - 1] <= arr2[mid2])) {
            break;
        }
        if (arr2[mid2] > (*arr1)[mid1]) {
            left = mid1 + 1;
        } else {
            right = mid1 - 1;
        }
    }
    arr[mid2 + mid1] = arr2[mid2];

#pragma omp parallel sections
    {
#pragma omp section
        {
            just_merge(&arr, *arr1, arr2, 0, 0, mid1, mid2);
        }
#pragma omp section
        {
            just_merge(&arr, *arr1, arr2, mid1, mid2, s1, s2);
        }
    }

    for (int i = 0; i < s1 + s2; i++) {
        (*arr1)[i] = arr[i];
    }
    free(arr);
}

void mergeSort(int* array, int size, int step, int p) {
    omp_set_num_threads(p);

    int num_chunk = size / step;
    if (size % step > 0) {
        num_chunk += 1;
    }

    int** result = (int**)calloc(num_chunk, sizeof(int*));
    int* sizes = (int*)calloc(num_chunk, sizeof(int));

#pragma parallel for
    for (int i = 0; i < num_chunk; i++) {
        sizes[i] = step;
        if (size - step * (i + 1) < 0) {
            sizes[i] = size - step * i;
        }

        result[i] = (int*)calloc(size, sizeof(int));

        for (int j = 0; j < sizes[i]; j++) {
            result[i][j] = array[i * step + j];
        }
    }

    double ts1 = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < num_chunk; i++) {
        qsort(result[i], sizes[i], sizeof(int), compare);
    }

    for (int j = 1; j < num_chunk; j *= 2) {
        for (int i = j; i < num_chunk + j; i += 2 * j) {
            if (i < num_chunk) {
                if (sizes[i] > 0) {
                    merge(&result[i - j], result[i], sizes[i - j], sizes[i], p);
                    sizes[i - j] += sizes[i];
                    sizes[i] = 0;
                }
            } else {
                if (sizes[i - j] > 0) {
                    merge(&result[0], result[i - j], sizes[0], sizes[i - j], p);
                    sizes[0] += sizes[i - j];
                    sizes[i - j] = 0;
                }
            }
        }
    }
    double ts2 = omp_get_wtime();

    writeResults(array, result[0], ts2 - ts1, size, step, p);

#pragma omp parallel for
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

    fclose(file_data);
    fclose(file_stats);
}
