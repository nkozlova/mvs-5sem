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

    int array[10];
    int rank, size;

/*    int pos[a * b][l * l];
    int particular[a * b][N * a * b];
*/    int count[a * b];
    Particular parts[N * a * b];

    double ts1 = MPI_Wtime();

    MPI_Request Request;
    MPI_Status Status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == a * b) {
        for (int i = 0; i < N * a * b; i++) {
            int z = rand() % (l * l);
            MPI_Isend(&z, 1, MPI_INT, i % (a * b), 1, MPI_COMM_WORLD, &Request);
        }
    }


    count[rank] = N;
    for (int i = 0; i < N; i++) {
        parts[rank * (a + b) + i].number = rank * (a + b) + i;
        parts[rank * (a + b) + i].node = rank;
        parts[rank * (a + b) + i].step = 0;

        MPI_Recv(&parts[rank * (a + b) + i].pos, 1, MPI_INT, a * b, rank, MPI_COMM_WORLD,  &Status);
    }
  /*   printf("I am %d of %d\n", rank, size);
    for (int j = 0; j < l * l; j++) {
        pos[rank][j] = rank;
        printf("%d ", pos[rank][j]);
    }
    printf("\n");
*/



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