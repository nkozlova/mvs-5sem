#include <omp.h>

int compareOMP(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

void writeResultsOMP(double time) {
    FILE* file_stats;
    file_stats = fopen("stats.txt", "a");

    fprintf(file_stats, "%fs\n", time);

    fclose(file_stats);
}

void just_mergeOMP(int** arr, int* arr1, int* arr2, int it1, int it2, int s1, int s2) {
    while (it1 < s1 && it2 < s2) {
        int i1 = it1, i2 = it2;
        while (it1 < s1 && it2 < s2 && arr1[it1] <= arr2[it2]) {
            it1++;
        }
        memcpy((*arr) + i1 + i2, arr1 + i1, (it1 - i1) * sizeof(int));
        while (it1 < s1 && it2 < s2 && arr1[it1] > arr2[it2]) {
            it2++;
        }
        memcpy((*arr) + it1 + i2, arr2 + i2, (it2 - i2) * sizeof(int));
    }
    if (it1 < s1) {
        memcpy((*arr) + it1 + it2, arr1 + it1, (s1 - it1) * sizeof(int));
    } else {
        memcpy((*arr) + it1 + it2, arr2 + it2, (s2 - it2) * sizeof(int));
    }
}

void mergeOMP(int** arr1, int* arr2, int s1, int s2) {
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
            just_mergeOMP(&arr, *arr1, arr2, 0, 0, mid1, mid2);
        }
#pragma omp section
        {
            just_mergeOMP(&arr, *arr1, arr2, mid1, mid2, s1, s2);
        }
    }

    memcpy(*arr1, arr, (s1 + s2) * sizeof(int));
    free(arr);
}

void mergeSortOMP(int* array, int size, int step, int p) {
    int ts1 = clock();

    int num_chunk = (size + step - 1) / step;

    int** result = (int**)calloc(num_chunk, sizeof(int*));
    int* sizes = (int*)calloc(num_chunk, sizeof(int));

#pragma omp parallel for num_threads(p)
    for (int i = 0; i < num_chunk; i++) {
        if (i < num_chunk - 1) {
            sizes[i] = step;
        } else {
            sizes[i] = size - step * i;
        }

        result[i] = (int*)calloc(size, sizeof(int));

        memcpy(result[i], array + i * step, sizes[i] * sizeof(int));

        qsort(result[i], sizes[i], sizeof(int), compareOMP);
    }

    for (int j = 1; j < num_chunk; j *= 2) {
        int k = 0;
#pragma omp parallel for reduction(+: k)  num_threads(p)
        for (int i = j; i < num_chunk; i += 2 * j) {
            mergeOMP(&result[i - j], result[i], sizes[i - j], sizes[i]);
            sizes[i - j] += sizes[i];
            sizes[i] = 0;
            k++;
        }
        int i = 2 * k * j;
        if (i < num_chunk) {
            mergeOMP(&result[0], result[i], sizes[0], sizes[i]);
            sizes[0] += sizes[i];
            sizes[i] = 0;
        }
    }
    int ts2 = clock();

    writeResultsOMP(((double)(ts2 - ts1)) / CLOCKS_PER_SEC);

    for (int i = 0; i < num_chunk; i++) {
        free(result[i]);
    }
    free(result);
    free(sizes);
}
