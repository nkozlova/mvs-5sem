#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <malloc.h>


void merge(int **arr3, int *arr2, int *sizes, int s1, int s2) {
    if (sizes[s2] > 0) {
        int *arr1 = *arr3;
        int mid2 = sizes[s2] / 2;

        int *arr = (int *)calloc(sizes[s1] + sizes[s2], sizeof(int));

        int left = 0;
        int right = sizes[s1];
        int mid1;
        while (left <= right) {
            mid1 = (left + right) / 2;
            if (arr2[mid2] <= arr1[mid1] && (mid1 == 0 || arr1[mid1 - 1] <= arr2[mid2])) {
                break;
            }
            if (arr2[mid2] > arr1[mid1]) {
                left = mid1 + 1;
            } else {
                right = mid1 - 1;
            }
        }
        arr[mid2 + mid1] = arr2[mid2];

#pragma omp sections
        {
#pragma omp section
            {
                int it1 = 0, it2 = 0;
                while (it1 < mid1 && it2 < mid2) {
                    if (arr1[it1] < arr2[it2]) {
                        arr[it1 + it2] = arr1[it1];
                        it1 += 1;
                    } else {
                        arr[it1 + it2] = arr2[it2];
                        it2 += 1;
                    }
                }
                while (it1 < mid1) {
                    arr[it1 + it2] = arr1[it1];
                    it1 += 1;
                }
                while (it2 < mid2) {
                    arr[it1 + it2] = arr2[it2];
                    it2 += 1;
                }
            }

#pragma omp section
            {
                int it1 = mid1, it2 = mid2;
                while (it1 < sizes[s1] && it2 < sizes[s2]) {
                    if (arr1[it1] < arr2[it2]) {
                        arr[it1 + it2] = arr1[it1];
                        it1 += 1;
                    } else {
                        arr[it1 + it2] = arr2[it2];
                        it2 += 1;
                    }
                }
                while (it1 < sizes[s1]) {
                    arr[it1 + it2] = arr1[it1];
                    it1 += 1;
                }
                while (it2 < sizes[s2]) {
                    arr[it1 + it2] = arr2[it2];
                    it2 += 1;
                }
            }
        }

        sizes[s1] += sizes[s2];
        sizes[s2] = 0;

        free(arr1);
        free(arr2);
        *arr3 = arr;
    }
}

int compare(const void *a, const void *b) {
    return *(int *)a > *(int *)b;
}

double mergeSort(int *array, int size, int step, int p) {
    int k = 0;
    int num_chunk = size / step;
    if (size % step != 0) {
        num_chunk += 1;
    }

    int **result = (int **)calloc(num_chunk, sizeof(int *));
    int *sizes = (int *)calloc(num_chunk, sizeof(int));

    double ts1 = omp_get_wtime( );
#pragma omp parallel for num_threads(p)
    for (int i = 0; i < num_chunk; i++) {
        sizes[i] = step;
        if (size - step * (i + 1) < 0) {
            sizes[i] = size - step * i;
        }

        result[i] = (int *)calloc(sizes[i], sizeof(int));
        for (int j = 0; j < sizes[i]; j++) {
            result[i][j] = array[k++];
        }

        qsort(result[i], sizes[i], sizeof(int), compare);
    }


    for (int j = 1; j < num_chunk; j *= 2) {
        for (int i = j; i < num_chunk + j; i += 2 * j) {
            if (i < num_chunk) {
                merge(&result[i - j], result[i], sizes, i - j, i);
            } else {
                merge(&result[0], result[i - j], sizes, 0, i - j);
            }
        }
    }

#pragma omp parallel for num_threads(p)
    for (int i = 0; i < sizes[0]; i++) {
        array[i] = result[0][i];
    }
    double ts2 = omp_get_wtime( );

    free(result[0]);
    free(result);
    free(sizes);

    return ts2 - ts1;
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        exit(1);
    }
    srand(time(NULL));

    int n = atoi(argv[1]), m = atoi(argv[2]);
    int p = atoi(argv[3]);

    FILE *file_stats, *file_data;
    file_stats = fopen("stats.txt", "a");
    file_data = fopen("data.txt", "w");

    int *a = (int *)calloc(n, sizeof(int));
    int *b = (int *)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 10000;
        b[i] = a[i];
        fprintf(file_data, "%d ", a[i]);
    }
    fprintf(file_data, "\n");
    fprintf(file_stats, "%fs %d %d %d\n", mergeSort(a, n, m, p), n, m, p);

    for (int i = 0; i < n; i++) {
        fprintf(file_data, "%d ", a[i]);
    }
    fprintf(file_data, "\n");

    double ts1 = omp_get_wtime( );
    qsort(b, n, sizeof(int), compare);
    double ts2 = omp_get_wtime( );
    fprintf(file_stats, "%fs - qsort\n", ts2 - ts1);

    fclose(file_data);
    fclose(file_stats);
    free(a);
    free(b);
    return 0;
}
