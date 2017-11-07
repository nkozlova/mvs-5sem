#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>


#define MASTER 0
#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3


typedef struct Ctx_t {
    int l;
    int a, b;
    int n;
    int N;
    double p_l, p_r, p_u, p_d;
} Ctx;

typedef struct Particle_t {
    int x;
    int y;
    int n;
    int process;
} Particle;


void swap(int* x, int* y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}

void swapP(Particle** x, Particle** y) {
    Particle* tmp = *x;
    *x = *y;
    *y = tmp;
}


int step(Ctx* ctx);
void writeResult(Ctx* ctx, Particle* result, int res_size, int rank, int size);
void randomWalk(Ctx* ctx, int rank, int size);


int main (int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 10) {
        exit(1);
    }
    Ctx ctx = {
            .l = atoi(argv[1]),
            .a = atoi(argv[2]),
            .b = atoi(argv[3]),
            .n = atoi(argv[4]),
            .N = atoi(argv[5]),
            .p_l = atof(argv[6]),
            .p_r = atof(argv[7]),
            .p_u = atof(argv[8]),
            .p_d = atof(argv[9]),
    };

    if (ctx.a * ctx.b != size) {
        exit(1);
    }

    omp_set_num_threads(2);

    double ts1, ts2;
    if (rank == MASTER) {
        ts1 = MPI_Wtime();
    }

    randomWalk(&ctx, rank, size);

    if (rank == MASTER) {
        ts2 = MPI_Wtime();
        FILE* file;
        file = fopen("stats.txt", "w");
        fprintf(file, "%fs %d %d %d %d %d %f %f %f %f\n", ts2 - ts1, ctx.l, ctx.a, ctx.b, ctx.n, ctx.N,
                ctx.p_l, ctx.p_r, ctx.p_u, ctx.p_d);
        fclose(file);
    }

    MPI_Finalize();
    return 0;
}


int step(Ctx* ctx) {
    double p = rand() / RAND_MAX;
    if (p <= ctx->p_l) {
        return LEFT;
    } else if (p <= ctx->p_u) {
        return UP;
    } else if (p <= ctx->p_r) {
        return RIGHT;
    } else {
        return DOWN;
    }
}

void randomWalk(Ctx* ctx, int rank, int size) {
    int count = ctx->N;
    int max_count = 2 * count;
    Particle* particles = (Particle*) calloc(max_count, sizeof(Particle));

    int left_rank = rank - 1, right_rank = rank + 1, up_rank = rank - ctx->a, down_rank = rank + ctx->a;
    if (rank % ctx->a == 0) {
        left_rank += ctx->a;
    }
    if (rank % ctx->a == ctx->a - 1) {
        right_rank -= ctx->a;
    }
    if (rank / ctx->a == 0) {
        up_rank += size;
    }
    if (rank / ctx->a == ctx->b - 1) {
        down_rank = rank % ctx->a;
    }

    int left_size = 0, right_size = 0, up_size = 0, down_size = 0;
    int left_max_count = count, right_max_count = count, up_max_count = count, down_max_count = count;

    Particle* to_left = (Particle*) calloc(left_max_count, sizeof(Particle));
    Particle* to_right = (Particle*) calloc(right_max_count, sizeof(Particle));
    Particle* to_up = (Particle*) calloc(up_max_count, sizeof(Particle));
    Particle* to_down = (Particle*) calloc(down_max_count, sizeof(Particle));


    int fin_size = 0;
    int fin_max_count = count;
    Particle* finished = (Particle*) calloc(fin_max_count, sizeof(Particle));

    omp_lock_t lock;
    int is_running = 1;
    omp_init_lock(&lock);
    omp_set_lock(&lock);

#pragma omp parallel sections default(shared)
    {
#pragma omp section
        {
            while(is_running) {
                omp_set_lock(&lock);
                for (int j = 0; j < 100; j++) {
                    int i = 0;
                    while(i < count) {
                        if (particles[i].n == 0) {
                            if (fin_size >= fin_max_count) {
                                fin_max_count *= 2;
                                finished = (Particle*) realloc(finished, fin_max_count * sizeof(Particle));
                            }
                            finished[fin_size] = particles[i];
                            fin_size++;
                            particles[i] = particles[count - 1];
                            count--;
                            continue;
                        }
                        particles[i].n--;

                        switch(step(ctx)) {
                            case LEFT:
                                particles[i].x--;
                                if (particles[i].x < 0) {
                                    particles[i].x = ctx->l + particles[i].x;

                                    if (left_size >= left_max_count) {
                                        left_max_count *= 2;
                                        to_left = (Particle* )realloc(to_left, left_max_count * sizeof(Particle));
                                    }
                                    to_left[left_size] = particles[i];
                                    left_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;

                            case RIGHT:
                                particles[i].x++;
                                if (particles[i].x >= ctx->l) {
                                    particles[i].x = ctx->l - particles[i].x;

                                    if (right_size >= right_max_count) {
                                        left_max_count *= 2;
                                        to_right = (Particle* )realloc(to_right, right_max_count * sizeof(Particle));
                                    }
                                    to_right[right_size] = particles[i];
                                    right_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;
                            case DOWN:
                                particles[i].y--;
                                if (particles[i].y < 0) {
                                    particles[i].y = ctx->l + particles[i].y;

                                    if (down_size >= down_max_count) {
                                        down_max_count *= 2;
                                        to_down = (Particle* )realloc(to_down, down_max_count * sizeof(Particle));
                                    }
                                    to_down[down_size] = particles[i];
                                    down_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;
                            case UP:
                                particles[i].y++;
                                if (particles[i].y >= ctx->l) {
                                    particles[i].y = ctx->l - particles[i].y;

                                    if (up_size >= up_max_count) {
                                        up_max_count *= 2;
                                        to_up = (Particle* )realloc(to_up, up_max_count * sizeof(Particle));
                                    }
                                    to_up[up_size] = particles[i];
                                    up_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;
                            default:
                                break;
                        }
                        i++;
                    }
                }
                omp_unset_lock(&lock);
            }
        }

#pragma omp section
        {
            int* seeds;
            int seed;
            if (rank == MASTER) {
                srand(time(NULL));
                seeds = (int*) calloc(size, sizeof(int));
                for (int i = 0; i < size; i++) {
                    seeds[i] = rand();
                }
            }
            MPI_Scatter(seeds, 1, MPI_INT, &seed, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
            if (rank == MASTER) {
                free(seeds);
            }
            srand(seed);

            for (int i = 0; i < count; i++) {
                int x = rand() % ctx->l;
                int y = rand() % ctx->l;
                Particle tmp_particle = {
                        .x = x,
                        .y = y,
                        .n = ctx->n,
                        .process = rank,
                };
                particles[i] = tmp_particle;
            }

            int tmp_left_size = left_size, tmp_right_size = right_size, tmp_up_size = up_size, tmp_down_size = down_size;
            int tmp_left_max_count = left_max_count, tmp_right_max_count = right_max_count,
                    tmp_up_max_count = up_max_count, tmp_down_max_count = down_max_count;

            Particle* tmp_left = (Particle*) calloc(tmp_left_max_count, sizeof(Particle));
            Particle* tmp_right = (Particle*) calloc(tmp_right_max_count, sizeof(Particle));
            Particle* tmp_up = (Particle*) calloc(tmp_up_max_count, sizeof(Particle));
            Particle* tmp_down = (Particle*) calloc(tmp_down_max_count, sizeof(Particle));
            int tmp_finished_size = fin_size;

            omp_unset_lock(&lock);

            while (is_running) {
                omp_set_lock(&lock);
                swap(&tmp_right_size, &right_size);
                swap(&tmp_up_size, &up_size);
                swap(&tmp_down_size, &down_size);
                swap(&tmp_left_max_count, &left_max_count);
                swap(&tmp_right_max_count, &right_max_count);
                swap(&tmp_up_max_count, &up_max_count);
                swap(&tmp_down_max_count, &down_max_count);
                left_size = 0, right_size = 0, up_size = 0, down_size = 0;
                swapP(&tmp_left, &to_left);
                swapP(&tmp_right, &to_right);
                swapP(&tmp_up, &to_up);
                swapP(&tmp_down, &to_down);
                tmp_finished_size = fin_size;

                omp_unset_lock(&lock);

                MPI_Request requests[8];

                int from_left_size, from_right_size, from_up_size, from_down_size;
                MPI_Issend(&tmp_left_size, 1, MPI_INT, left_rank, LEFT, MPI_COMM_WORLD, requests);
                MPI_Issend(&tmp_right_size, 1, MPI_INT, right_rank, RIGHT, MPI_COMM_WORLD, requests + 1);
                MPI_Issend(&tmp_up_size, 1, MPI_INT, up_rank, UP, MPI_COMM_WORLD, requests + 2);
                MPI_Issend(&tmp_down_size, 1, MPI_INT, down_rank, DOWN, MPI_COMM_WORLD, requests + 3);

                MPI_Irecv(&from_left_size, 1, MPI_INT, left_rank, RIGHT, MPI_COMM_WORLD, requests + 4);
                MPI_Irecv(&from_right_size, 1, MPI_INT, right_rank, LEFT, MPI_COMM_WORLD, requests + 5);
                MPI_Irecv(&from_up_size, 1, MPI_INT, up_rank, DOWN, MPI_COMM_WORLD, requests + 6);
                MPI_Irecv(&from_down_size, 1, MPI_INT, down_rank, UP, MPI_COMM_WORLD, requests + 7);

                MPI_Waitall(8, requests, MPI_STATUS_IGNORE);

                Particle* from_left = (Particle*) calloc(from_left_size, sizeof(Particle));
                Particle* from_right = (Particle*) calloc(from_right_size, sizeof(Particle));
                Particle* from_up = (Particle*) calloc(from_up_size, sizeof(Particle));
                Particle* from_down = (Particle*) calloc(from_down_size, sizeof(Particle));

                MPI_Issend(tmp_left, tmp_left_size * sizeof(Particle), MPI_BYTE,
                           left_rank, LEFT, MPI_COMM_WORLD, requests);
                MPI_Issend(tmp_right, tmp_right_size * sizeof(Particle), MPI_BYTE,
                           right_rank, RIGHT, MPI_COMM_WORLD, requests + 1);
                MPI_Issend(tmp_up, tmp_up_size * sizeof(Particle), MPI_BYTE,
                           up_rank, UP, MPI_COMM_WORLD, requests + 2);
                MPI_Issend(tmp_down, tmp_down_size * sizeof(Particle), MPI_BYTE,
                           down_rank, DOWN, MPI_COMM_WORLD, requests + 3);

                MPI_Irecv(from_left, from_left_size * sizeof(Particle), MPI_BYTE,
                          left_rank, RIGHT, MPI_COMM_WORLD, requests + 4);
                MPI_Irecv(from_right, from_right_size * sizeof(Particle), MPI_BYTE,
                          right_rank, LEFT, MPI_COMM_WORLD, requests + 5);
                MPI_Irecv(from_up, from_up_size * sizeof(Particle), MPI_BYTE,
                          up_rank, DOWN, MPI_COMM_WORLD, requests + 6);
                MPI_Irecv(from_down, from_down_size * sizeof(Particle), MPI_BYTE,
                          down_rank, UP, MPI_COMM_WORLD, requests + 7);

                MPI_Waitall(8, requests, MPI_STATUS_IGNORE);

                int all_finished[size];
                MPI_Gather(&tmp_finished_size, 1, MPI_INT, all_finished, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

                omp_set_lock(&lock);

                for (int i = 0; i < from_left_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_left[i];
                    count++;
                }

                for (int i = 0; i < from_right_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_right[i];
                    count++;
                }

                for (int i = 0; i < from_up_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_up[i];
                    count++;
                }

                for (int i = 0; i < from_down_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_down[i];
                    count++;
                }

                int is_actives[size];

                if (rank == MASTER) {

                    int sum_finished = 0;
                    for (int i = 0; i < size; i++) {
                        sum_finished += all_finished[i];
                    }

                    if (sum_finished == size * ctx->N) {
                        for (int i = 0; i < size; i++) {
                            is_actives[i] = 0;
                        }
                    } else {
                        for (int i = 0; i < size; i++) {
                            is_actives[i] = 1;
                        }
                    }
                }

                MPI_Scatter(is_actives, 1, MPI_INT, &is_running, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

                omp_unset_lock(&lock);

                free(from_left);
                free(from_right);
                free(from_up);
                free(from_down);
            }

            writeResult(ctx, finished, fin_size, rank, size);

            free(tmp_left);
            free(tmp_right);
            free(tmp_up);
            free(tmp_down);
        }
    }

    free(particles);
    free(to_left);
    free(to_right);
    free(to_up);
    free(to_down);
    free(finished);
    omp_destroy_lock(&lock);
}

void writeResult(Ctx* ctx, Particle* result, int res_size, int rank, int size) {
    MPI_File data;
    MPI_File_delete("data.bin", MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, "data.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &data);

    int pos[ctx->l][ctx->l * size];
    for (int y = 0; y < ctx->l; y++) {
        for (int x = 0; x < ctx->l * size; x++) {
            pos[y][x] = 0;
        }
    }

    for (int i = 0; i < res_size; i++) {
        pos[result[i].y][result[i].x * size + result[i].process] += 1;
    }

    int start_seek = ((ctx->l * ctx->l) * (rank / ctx->a) * ctx->a + ctx->l * (rank % ctx->a)) * size * sizeof (int);
    int line_seek = ctx->l * ctx->a * size * sizeof(int);

    for (int y = 0; y < ctx->l; y++) {
        MPI_File_set_view(data, start_seek + line_seek * y, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
        MPI_File_write(data, pos[y], ctx->l * size, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&data);
}
