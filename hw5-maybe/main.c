#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


typedef struct Particular_t {
    int number; // номер частицы
    int node;   // номер узла, который рассматривает квадрат, в котором находится частица
    int pos;    // позиция в самом квадрате
    int step;   // номер шага, который совершает частица
} Particular;


int main(int argc, char* argv[]) {
    if (argc != 10) {
        exit(1);
    }
    int l = atoi(argv[1]);
    int a = atoi(argv[2]), b = atoi(argv[3]);
    int n = atoi(argv[4]);
    int N = atoi(argv[5]);
    double p_l = atof(argv[6]);
    double p_r = atof(argv[7]);
    double p_u = atof(argv[8]);
    double p_d = atof(argv[9]);

    srand(time(NULL));

    FILE *file;
    file = fopen("stats.txt", "a");

    MPI_Init(&argc, &argv);

    int arr_send[a * b][N];
    int arr_recv[a * b][N];
    double p_send[a * b][N];
    double p_recv[a * b][N];
    int rank, size;

    int count[a * b];
    Particular parts[N * a * b];

    double ts1 = MPI_Wtime();

    MPI_Request Request;
    MPI_Status Status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Генерация начального положения частиц в каждой ячейке */
    if (rank == a * b) {
        for (int i = 0; i < a * b; i++) {
            for (int j = 0; j < N; j++) {
                arr_send[i][j] = rand() % (l * l);
            }
        }
        for (int i = 0; i < a * b; i++) {
            MPI_Isend(&arr_send[i], N, MPI_INT, i, 1, MPI_COMM_WORLD, &Request);
        }
    }

    /* Раскладываем частицы по местам */
    if (rank != a * b) {
        count[rank] = N;
        MPI_Recv(arr_recv[rank], N, MPI_INT, a * b, 1, MPI_COMM_WORLD, &Status);
        for (int i = 0; i < N; i++) {
            parts[rank * N + i].number = rank * N + i;
            parts[rank * N + i].node = rank;
            parts[rank * N + i].pos = arr_recv[rank][i];
            parts[rank * N + i].step = 0;
        }
    }

    /* Генерация n шагов для каждой частицы */
 //   for (int k = 0; k < n; k++)
    if (rank == a * b) {
        for (int i = 0; i < a * b; i++) {
            for (int j = 0; j < N; j++) {
                p_send[i][j] = 1. * rand() / RAND_MAX;
            }
            MPI_Isend(&p_send[i], N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &Request);
        }
    }

    if (rank != a * b) {
        MPI_Recv(p_recv[rank], N, MPI_DOUBLE, a * b, 1, MPI_COMM_WORLD, &Status);
        for (int i = 0; i < N; i++) {
            printf("%d - %f\n", rank, p_recv[rank][i]);
        }
    }


    double ts2 = MPI_Wtime();

    if (rank == 0) {
        fprintf(file, "%d %d %d %d %d %f %f %f %f %fs\n", l, a, b, n, N, p_l, p_r, p_u, p_d, ts2 - ts1);
    }
    if (rank < a * b) {
        fprintf(file, "%d: %d\n", rank, count[rank]);
    }

    fclose(file);
    MPI_Finalize();
    return 0;
}